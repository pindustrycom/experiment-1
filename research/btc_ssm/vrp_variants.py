"""Round 4: can the tradeable BTC vol trade be made to work? Same 66 first-of-month Deribit chains as vrp_trade.py,
now with Deribit's option fees, hourly hedge paths and three families of changes:

  hedging  daily (round 3), every 4 hours, or only when the book's delta drifts past a band (checked hourly)
  strikes  full dK/K^2 strip 0.6-1.6 F, narrow strip 0.8-1.25 F, put or call wing only, ATM straddle,
           25-delta strangle, and an iron condor (25-delta strangle sold, 10-delta wings bought as crash insurance)
  timing   always sell, sell only when ATM implied variance beats the rough-vol 30-day forecast (round 2's best
           long-horizon model, walk-forward), or size each month by the gap (IV^2 - forecast) / IV^2

    python vrp_variants.py DATA_DIR OUT_DIR SV_DIR     (SV_DIR = sv_run.py output, for the walk-forward vol forecasts)

Fees: 0.03% of the underlying per option contract capped at 12.5% of the option price, plus a 0.015% delivery fee on
options that finish in the money (same cap), both maker and taker. Perpetual funding on the hedge is ignored.
P&L is in strike units, P&L / (net vega * sigma_atm / 2), so (K - RV)/K for a variance swap; t and Sharpe do not depend
on that scale. Every variant is one hypothesis; p-values are Holm-corrected across all of them.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as tdist

import evaluate as ev
import run
from vrp_trade import bs, delivery_prices, load_chain

HEDGE_BP = 5.0
FEE, DLV_FEE, FEE_CAP = 3e-4, 1.5e-4, 0.125
HEDGES = {"daily": ("step", 24), "4h": ("step", 4), "band10": ("band", 0.10), "band20": ("band", 0.20), "band30": ("band", 0.30)}
BOOKS = ["strip", "strip_narrow", "put_strip", "call_strip", "straddle", "strangle25", "condor"]


def entry(c, day, d, dlv):
    """Everything about one month's trade that does not depend on the book: chain, forward, expiry, hourly path."""
    snap = c[c.date == day]
    exps = snap.groupby("exp")["T"].first()
    exps = exps[(exps * 365 >= 18) & (exps * 365 <= 45)]
    if exps.empty:
        return None
    exp = (exps * 365 - 30).abs().idxmin()
    ch = snap[(snap.exp == exp) & (snap.bid_price > 0) & (snap.ask_price > 0)].copy()
    if len(ch) < 6 or pd.Timestamp(exp) >= d.index[-1]:
        return None
    S0 = float(ch.underlying_price.median())
    T = float(ch["T"].iloc[0])
    idx = float(d.close.asof(day - pd.Timedelta(hours=1)))
    for k in ("bid_price", "ask_price", "mark_price"):
        ch[k + "_usd"] = ch[k] * idx
    ch["mid_usd"] = 0.5 * (ch.bid_price_usd + ch.ask_price_usd)
    ch["delta"] = pd.to_numeric(ch.delta, errors="coerce")
    atm_k = ch.strike_price.iloc[(ch.strike_price - S0).abs().argmin()]
    atm = ch[ch.strike_price == atm_k]
    if set(atm.type) != {"call", "put"}:
        return None
    sig_atm = float(atm.mark_iv.mean()) / 100
    ST = float(dlv.get(exp.normalize(), np.nan)) if len(dlv) else np.nan
    if not np.isfinite(ST):
        ST = float(d.close.asof(exp - pd.Timedelta(hours=1)))
    times = pd.date_range(day, exp, freq="h")
    path = d.close.reindex(times - pd.Timedelta(hours=1)).to_numpy()   # close of the bar ending at each hour
    if not np.isfinite(path).all():
        return None
    hr = d.r.loc[day:exp - pd.Timedelta(hours=1)]
    return {"day": day, "exp": exp, "ch": ch, "S0": S0, "T": T, "idx": idx, "atm_k": atm_k, "sig_atm": sig_atm, "ST": ST,
            "times": times, "path": path, "rv_ann": float((hr / 100).pow(2).sum() / T)}


def _nearest(ch, kind, target):
    s = ch[ch.type == kind]
    return s.iloc[[(s.delta - target).abs().argmin()]] if len(s) else s


def legs(e, book):
    """Legs with weights w (> 0 sold, < 0 bought)."""
    ch, S0 = e["ch"], e["S0"]
    otm = ch[((ch.type == "put") & (ch.strike_price < S0)) | ((ch.type == "call") & (ch.strike_price >= S0))]
    rng = {"strip": (0.6, 1.6), "strip_narrow": (0.8, 1.25), "put_strip": (0.6, 1.0), "call_strip": (1.0, 1.6)}
    if book in rng:
        lo, hi = rng[book]
        b = otm[(otm.strike_price >= lo * S0) & (otm.strike_price <= hi * S0)].sort_values("strike_price")
        if len(b) < 3:
            return None
        ks = b.strike_price.to_numpy()
        return b.assign(w=np.gradient(ks) / ks ** 2 * S0 ** 2 / 2)
    if book == "straddle":
        return ch[ch.strike_price == e["atm_k"]].assign(w=1.0)
    if book in ("rr25", "rr25_rev"):            # risk reversal: sell the 25-delta call, buy the 25-delta put (or the reverse)
        c25, p25 = _nearest(otm, "call", 0.25), _nearest(otm, "put", -0.25)
        if len(c25) != 1 or len(p25) != 1:
            return None
        sgn = 1.0 if book == "rr25" else -1.0
        return pd.concat([c25.assign(w=sgn), p25.assign(w=-sgn)])
    sold = pd.concat([_nearest(otm, "put", -0.25), _nearest(otm, "call", 0.25)]).assign(w=1.0)
    if book == "strangle25":
        return sold if len(sold) == 2 else None
    wings = pd.concat([_nearest(otm, "put", -0.10), _nearest(otm, "call", 0.10)]).assign(w=-1.0)
    if len(sold) != 2 or len(wings) != 2 or set(wings.strike_price) & set(sold.strike_price):
        return None
    return pd.concat([sold, wings])


def book_pnl(e, b, hedge, fill="bid", scale_by="net"):
    """scale_by: "net" vega (short-vol books) or "gross" vega (books near vega-neutral, like a risk reversal).
    Hedge "ww...": Whalley-Wilmott (1997) band, rebalance when the delta leaves +-(1.5 * cost * F * Gamma^2 / g)^(1/3)
    around its target, g = k / (book notional in USD); the band narrows as gamma falls and widens near expiry."""
    K, kind, w = b.strike_price.to_numpy(), b.type.to_numpy(), b.w.to_numpy()
    sig = b.mark_iv.to_numpy() / 100
    S0, T, ST, path, times, exp = e["S0"], e["T"], e["ST"], e["path"], e["times"], e["exp"]
    _, _, vega = bs(S0, K, T, sig, kind)
    nv = float((w * vega).sum()) if scale_by == "net" else float((np.abs(w) * vega).sum())
    if nv <= 0:
        return None
    unit = nv / (2 * S0 * norm.pdf(0.5 * e["sig_atm"] * np.sqrt(T)) * np.sqrt(T))    # straddle-equivalents of vega
    if fill == "bid":
        px = np.where(w > 0, b.bid_price_usd, b.ask_price_usd)
    else:
        px = b.mid_usd.to_numpy()
    prem = float((w * px).sum())
    spread = float((np.abs(w) * np.abs(b.mid_usd.to_numpy() - np.where(w > 0, b.bid_price_usd, b.ask_price_usd))).sum())
    payoff = np.where(kind == "call", np.maximum(ST - K, 0), np.maximum(K - ST, 0))
    fees = float((np.abs(w) * np.minimum(FEE * e["idx"], FEE_CAP * px)).sum())
    fees += float((np.abs(w) * np.where(payoff > 0, np.minimum(DLV_FEE * ST, FEE_CAP * payoff), 0)).sum())
    # target hedge each hour: +sum(w * delta) BTC (short options)
    tau = np.maximum(((exp - times) / pd.Timedelta(days=365)).to_numpy()[:-1], 1e-6)
    _, dl, _ = bs(path[:-1, None], K[None, :], tau[:, None], sig[None, :], kind[None, :])
    target = (dl * w[None, :]).sum(1)
    mode, x = HEDGES[hedge]
    if mode == "ww":
        F = path[:-1, None]
        sq = sig[None, :] * np.sqrt(tau[:, None])
        d1 = (np.log(F / K[None, :]) + 0.5 * sq ** 2) / sq
        gam = np.abs((w[None, :] * norm.pdf(d1) / (F * sq)).sum(1))
        band = (1.5 * HEDGE_BP / 1e4 * path[:-1] * gam ** 2 / (x / (S0 * unit))) ** (1 / 3)
    cur, held = 0.0, np.empty(len(target))
    traded = 0.0
    for i, h in enumerate(target):
        if i == 0 or (mode == "step" and i % x == 0) or (mode == "band" and abs(h - cur) > x * unit) or (mode == "ww" and abs(h - cur) > band[i]):
            traded += abs(h - cur) * path[i]
            cur = h
        held[i] = cur
    traded += abs(cur) * path[-1]
    hcost = traded * HEDGE_BP / 1e4
    hedge_pnl = float((held * np.diff(path)).sum())
    scale = nv * e["sig_atm"] / 2
    pnl = prem - float((w * payoff).sum()) + hedge_pnl - hcost - fees
    return {"pnl": pnl / scale, "ret": pnl / (unit * S0), "spread": spread / scale, "fees": fees / scale, "hedge_cost": hcost / scale, "n_legs": len(b)}


def f30(sv_dir, model="rough"):
    """A model's forecast of annualized variance over the next 30 days, made at each 00:00 UTC (walk-forward files)."""
    f = pd.concat([pd.read_pickle(p)["daily"]["f30d"] for p in sorted(glob.glob(f"{sv_dir}/{model}_*.pkl"))]).sort_index()
    return f / 1e4 * 365 / 30


def _t(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def checks(t, d, sv_dir, book, hedge="band10", n_placebo=5000):
    """Is it the forecast that helps? Gate the same book with other forecasters, with trailing realized vol (no model),
    and with random gates that trade as many months as the rough-vol gate (placebo distribution of t)."""
    g = t[(t.book == book) & (t.hedge == hedge) & (t.fill == "bid")].set_index("date")
    ix = pd.to_datetime(g.index)
    out = {}
    for m in ("rough", "har", "garch", "heston_j", "ewma"):
        on = g.iv2.to_numpy() > f30(sv_dir, m).asof(ix).to_numpy()
        out[m] = {"months": int(on.sum()), "t": _t(g.pnl.to_numpy() * on)}
    trv = ((d.r / 100) ** 2).rolling(720).sum() * 365 / 30
    on = g.iv2.to_numpy() > trv.asof(ix - pd.Timedelta(hours=1)).to_numpy()
    out["trailing_rv30"] = {"months": int(on.sum()), "t": _t(g.pnl.to_numpy() * on)}
    k = out["rough"]["months"]
    rng = np.random.default_rng(0)
    ts = []
    for _ in range(n_placebo):
        m = np.zeros(len(g))
        m[rng.choice(len(g), k, replace=False)] = 1
        ts.append(_t(g.pnl.to_numpy() * m))
    out["placebo"] = {"p95": float(np.percentile(ts, 95)), "p99": float(np.percentile(ts, 99)),
                      "share_above_rough": float(np.mean(np.array(ts) >= out["rough"]["t"]))}
    return out


def capital(t, d, book, hedge, timing_w):
    """One straddle-equivalent of vega per BTC of capital (round 3's convention), compounding monthly."""
    g = t[(t.book == book) & (t.hedge == hedge) & (t.fill == "bid")].set_index("date")
    r = g.ret * timing_w.reindex(g.index).fillna(0.0)
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    don, _ = ev.donchian_daily(d, 25, r.index.min(), "2026-10-01")
    dm = pd.Series([float(np.prod(1 + don.loc[a:b])) - 1 for a, b in zip(r.index, list(r.index[1:]) + ["2026-10-01"])], index=r.index)
    by_year = r.groupby(r.index.str[:4]).apply(lambda x: float(np.prod(1 + x) - 1))
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(12)),
            "max_dd": float((eq / eq.cummax() - 1).min()), "worst_month": float(r.min()), "corr_donchian": float(np.corrcoef(r, dm)[0, 1]),
            "by_year": by_year.round(4).to_dict()}


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    t = m / s * np.sqrt(n)
    return {"n": n, "mean": m, "t": t, "p": float(2 * tdist.sf(abs(t), n - 1)), "hit": float((x > 0).mean()),
            "worst": float(x.min()), "sharpe_ann": float(m / s * np.sqrt(12))}


def main(folder, out, sv_dir):
    d, _ = run.prepare(folder)
    c = load_chain(folder)
    dlv = delivery_prices(folder, d)
    fr = f30(sv_dir)
    rows = []
    for day in sorted(c.date.unique()):
        e = entry(c, pd.Timestamp(day), d, dlv)
        if e is None:
            continue
        fc = float(fr.asof(e["day"])) if e["day"] >= fr.index[0] else np.nan
        base = {"date": str(e["day"].date()), "iv2": e["sig_atm"] ** 2, "rough_f30": fc, "rv2": e["rv_ann"]}
        for book in BOOKS:
            b = legs(e, book)
            if b is None:
                continue
            for hedge in HEDGES:
                for fill in ("bid", "mid"):
                    r = book_pnl(e, b, hedge, fill)
                    if r is not None:
                        rows.append({**base, "book": book, "hedge": hedge, "fill": fill, **r})
    t = pd.DataFrame(rows)
    t.to_csv(f"{out}/vrp_variants_trades.csv", index=False)
    months = t.date.unique()
    gap = t.drop_duplicates("date").set_index("date")
    gap = (gap.iv2 - gap.rough_f30) / gap.iv2
    sized = (gap.clip(lower=0) / gap.clip(lower=0).mean()).fillna(0.0)
    timing = {"always": pd.Series(1.0, index=months), "gate": (gap > 0).astype(float),
              "gate25": (gap > 0.2).astype(float), "sized": sized}
    res = []
    for (book, hedge, fill), g in t.groupby(["book", "hedge", "fill"]):
        g = g.set_index("date")
        for tm, wgt in timing.items():
            x = g.pnl * wgt.reindex(g.index).fillna(0.0)
            first = x[x.index < "2024-01-01"]
            s = stats(x)
            res.append({"book": book, "hedge": hedge, "fill": fill, "timing": tm, **s, "traded": int((wgt.reindex(g.index) > 0).sum()),
                        "mean_2021_23": float(first.mean()), "mean_2024_26": float(x[x.index >= "2024-01-01"].mean()),
                        "spread": float(g.spread.mean()), "fees": float(g.fees.mean()), "hedge_cost": float(g.hedge_cost.mean())})
    r = pd.DataFrame(res)
    r = r.reset_index(drop=True)
    bid = r.fill == "bid"
    r.loc[bid, "p_holm"] = ev.holm(r.loc[bid, "p"].to_numpy())
    r = r.sort_values("t", ascending=False).reset_index(drop=True)
    bid = r.fill == "bid"
    r.to_csv(f"{out}/vrp_variants.csv", index=False)
    summary = {"n_months": len(months), "n_variants_bid": int(bid.sum()),
               "gap_stats": {"mean": float(gap.mean()), "share_positive": float((gap > 0).mean())},
               "checks": {b: checks(t, d, sv_dir, b) for b in ("call_strip", "strip", "strangle25", "straddle")},
               "capital": {f"{b}_{h}_{tm}": capital(t, d, b, h, timing[tm]) for b, h, tm in
                           [("call_strip", "band10", "gate"), ("call_strip", "band10", "always"), ("strip", "band10", "gate"),
                            ("strip", "daily", "always"), ("strangle25", "band10", "gate")]},
               "top_bid": r[bid].head(15).to_dict("records")}
    with open(f"{out}/vrp_variants.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(json.dumps({k: summary[k] for k in ("checks", "capital")}, indent=1, default=float))
    pd.set_option("display.width", 250)
    cols = ["book", "hedge", "fill", "timing", "n", "traded", "mean", "t", "p_holm", "hit", "worst", "sharpe_ann",
            "mean_2021_23", "mean_2024_26", "spread", "fees", "hedge_cost"]
    print(r[bid][cols].head(30).round(3).to_string())
    print(r[(r.timing == "always") & bid][cols].round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
