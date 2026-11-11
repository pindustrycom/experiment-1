"""Round 7: the gated call wing entered every day, 2021-04 to 2026-09, with the Greeks split and exit rules of round 6.

Real Deribit chains exist only for the 1st of each month (Tardis free files) and, from 2026-10-30, every day
(collect_chains.py). Until the daily collection has a few months, each day's book is rebuilt from real data:
  - strikes, smile and spreads: the latest first-of-month chain on or before the day. Each call-wing leg (1.0-1.6 F,
    dK/K^2 weights) keeps its moneyness K/F, its IV relative to ATM IV, and its relative half-spread at the bid.
  - level: ATM IV = DVOL that hour x (ATM IV / DVOL on the chain's date), so the level moves daily with Deribit's
    30-day index while the shape comes from the month's real chain.
  - path: Binance hourly BTC, 30 days, settled at the close of the last hour (no Deribit delivery price).
Fees, band-0.1 hourly hedge and P&L units are as in rounds 4-6. Gates: ATM IV^2 above the rough-vol (or HAR) 30-day
forecast made at that 00:00 UTC (walk-forward, sv_run.py).

The approximation is checked first: on the 66 first-of-month days it is compared with round 4's real-chain P&L.
Overlapping entries are scored on non-overlapping 30-day blocks (mean of the block's entries), with Newey-West
(1 lag) on the blocks as well.

    python daily_greeks.py DATA_DIR OUT_DIR SV_DIR
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import evaluate as ev
import run
from greeks import RULES, decompose, run_exit, setup
from vrp_trade import bs, delivery_prices, load_chain
from vrp_variants import entry, f30, legs, stats

DAYS = 30
DAILY_RULES = ["hold", "tp50", "iv_jump10", "stop2", "rv_breach"]


def calibrations(c, d, dlv):
    """Per first-of-month chain: call-wing legs as (K/F, IV/ATM IV, relative half-spread, weight), and ATM IV / DVOL."""
    cal = {}
    for day in sorted(c.date.unique()):
        e = entry(c, pd.Timestamp(day), d, dlv)
        if e is None:
            continue
        b = legs(e, "call_strip")
        dv = float(d.dvol.asof(e["day"] - pd.Timedelta(hours=1)))
        if b is None or not np.isfinite(dv):
            continue
        mid = b.mid_usd.to_numpy()
        cal[e["day"]] = {"m": b.strike_price.to_numpy() / e["S0"], "ivr": b.mark_iv.to_numpy() / 100 / e["sig_atm"],
                         "half": (mid - b.bid_price_usd.to_numpy()) / mid, "w": b.w.to_numpy(), "atm_dvol": e["sig_atm"] / (dv / 100)}
    return pd.Series(cal).sort_index()


def synth(day, cal, d):
    """The book and path for one entry day, in the shapes greeks.setup expects."""
    exp = day + pd.Timedelta(days=DAYS)
    times = pd.date_range(day, exp, freq="h")
    path = d.close.reindex(times - pd.Timedelta(hours=1)).to_numpy()
    dv = float(d.dvol.asof(day - pd.Timedelta(hours=1)))
    if not np.isfinite(path).all() or not np.isfinite(dv):
        return None, None
    S0, T = float(path[0]), DAYS / 365
    sig_atm = dv / 100 * cal["atm_dvol"]
    K = cal["m"] * S0
    iv = sig_atm * cal["ivr"]
    mid = bs(S0, K, T, iv, np.array(["call"] * len(K)))[0]
    b = pd.DataFrame({"strike_price": K, "type": "call", "w": cal["w"], "mark_iv": iv * 100, "mid_usd": mid,
                      "bid_price_usd": mid * (1 - cal["half"]), "ask_price_usd": mid * (1 + cal["half"])})
    hr = d.r.loc[day:exp - pd.Timedelta(hours=1)]
    e = {"day": day, "exp": exp, "S0": S0, "T": T, "idx": S0, "sig_atm": sig_atm, "ST": float(path[-1]), "times": times,
         "path": path, "rv_ann": float((hr / 100).pow(2).sum() / T)}
    return e, b


def blocks_t(x):
    blk = x.groupby(np.arange(len(x)) // DAYS).mean()
    return float(blk.mean() / blk.std(ddof=1) * np.sqrt(len(blk))), float(ev.nw_t(blk.to_numpy(), 1)), len(blk)


def main(folder, out, sv_dir):
    d, _ = run.prepare(folder)
    c = load_chain(folder)
    dlv = delivery_prices(folder, d)
    cal = calibrations(c, d, dlv)
    fr, fh = f30(sv_dir, "rough"), f30(sv_dir, "har")
    days = pd.date_range(cal.index[0], d.index[-1] - pd.Timedelta(days=DAYS + 1), freq="D")
    dec, ex = [], []
    for day in days:
        e, b = synth(day, cal.iloc[cal.index.searchsorted(day, side="right") - 1], d)
        if e is None:
            continue
        s = setup(e, b, d.dvol)
        if s is None:
            continue
        base = {"date": day, "iv2": e["sig_atm"] ** 2, "rough": float(fr.asof(day)), "har": float(fh.asof(day)),
                "rv": np.sqrt(e["rv_ann"]), "first": day.day == 1}
        dec.append({**base, **decompose(e, s)})
        for name in DAILY_RULES:
            ex.append({**base, "rule": name, **run_exit(e, s, RULES[name])})
    dec, ex = pd.DataFrame(dec).set_index("date"), pd.DataFrame(ex)
    for t in (dec, ex):
        t["gate"] = (t.iv2 > t.rough).astype(float)
        t["gate_har"] = (t.iv2 > t.har).astype(float)
    dec.to_csv(f"{out}/daily_greeks_decomp.csv")
    ex.to_csv(f"{out}/daily_greeks_exits.csv", index=False)

    # 1. does the rebuilt book track the real one on the 66 first-of-month days?
    real = pd.read_csv(f"{out}/greeks_decomp.csv")
    real = real[real.book == "call_strip"].set_index(pd.to_datetime(real[real.book == "call_strip"].date))
    both = dec[dec["first"]].join(real[["total", "theta", "gamma"]], rsuffix="_real", how="inner")
    res = {"check_vs_real_chains": {"months": len(both), "corr_total": float(np.corrcoef(both.total, both.total_real)[0, 1]),
                                    "mean_synth": float(both.total.mean()), "mean_real": float(both.total_real.mean()),
                                    "mean_synth_gated": float((both.total * both.gate).mean()),
                                    "mean_real_gated": float((both.total_real * both.gate).mean()),
                                    "corr_theta": float(np.corrcoef(both.theta, both.theta_real)[0, 1]),
                                    "corr_gamma": float(np.corrcoef(both.gamma, both.gamma_real)[0, 1])}}
    # 2. daily entries: always / rough gate / HAR gate, hold to expiry
    parts = ["theta", "gamma", "slip", "edge", "fees", "hedge_cost", "residual", "total"]
    res["daily"] = {}
    for tm, w in (("always", 1.0), ("rough_gate", dec.gate), ("har_gate", dec.gate_har)):
        x = dec.total * w
        tb, tnw, nb = blocks_t(x)
        first = x[dec["first"]]
        on = dec[w > 0] if tm != "always" else dec
        res["daily"][tm] = {"entries": len(x), "share_on": float((np.asarray(w) > 0).mean()) if tm != "always" else 1.0,
                            "mean": float(x.mean()), "t_blocks": tb, "t_blocks_nw": tnw, "blocks": nb,
                            "t_first_of_month_only": stats(first.to_numpy())["t"],
                            "by_year": x.groupby(x.index.year).mean().round(4).to_dict(),
                            "greeks_when_on": on[parts].mean().round(4).to_dict()}
    # gate on vs off, Greeks per entry
    for g in (1.0, 0.0):
        res["daily"][f"rough_gate_{'on' if g else 'off'}_greeks"] = dec[dec.gate == g][parts].mean().round(4).to_dict()
    # random-gate placebo with the same share of days on, block structure kept
    rng = np.random.default_rng(0)
    n_on = int(dec.gate.sum())
    pl = []
    for _ in range(2000):
        m = np.zeros(len(dec))
        m[rng.choice(len(dec), n_on, replace=False)] = 1
        pl.append(blocks_t(dec.total * m)[0])
    res["daily"]["placebo_share_above_rough"] = float(np.mean(np.array(pl) >= res["daily"]["rough_gate"]["t_blocks"]))
    # 3. exit rules on daily entries (rough gate), paired against holding
    res["exits"] = {}
    h = ex[ex.rule == "hold"].set_index("date")
    for rule, x in ex.groupby("rule"):
        x = x.set_index("date")
        for tm in ("always", "gate"):
            w = x.gate if tm == "gate" else 1.0
            p = x.pnl * w
            diff = (x.pnl - h.pnl) * w
            res["exits"][f"{rule}_{tm}"] = {"mean": float(p.mean()), "t_blocks": blocks_t(p)[0],
                                            "vs_hold_mean": float(diff.mean()), "vs_hold_t_blocks": blocks_t(diff)[0] if diff.std() > 0 else 0.0,
                                            "worst_mtm": float(x[np.asarray(w) > 0].worst_mtm.min()) if tm == "gate" else float(x.worst_mtm.min())}
    # 4. capital: ladder, 1/30 of one straddle-vega per BTC entered each day, daily P&L booked at expiry
    for tm, w in (("always", 1.0), ("rough_gate", ex[ex.rule == "hold"].set_index("date").gate)):
        r = h.ret * w / DAYS
        r = r.groupby(r.index + pd.Timedelta(days=DAYS)).sum()
        m = r.groupby(r.index.to_period("M")).sum()
        eq = (1 + m).cumprod()
        res.setdefault("capital", {})[tm] = {"cagr": float(eq.iloc[-1] ** (12 / len(m)) - 1), "sharpe": float(m.mean() / m.std() * np.sqrt(12)),
                                             "max_dd": float((eq / eq.cummax() - 1).min()), "worst_month": float(m.min()),
                                             "by_year": m.groupby(m.index.year).sum().round(4).to_dict()}
    with open(f"{out}/daily_greeks.json", "w") as fh_:
        json.dump(res, fh_, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:4])
