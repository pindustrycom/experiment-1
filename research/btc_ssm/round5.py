"""Round 5: the ranked next tests from the volatility literature review (every one paper only).

    python round5.py DATA_DIR OUT_DIR POTS_CSV

  1  vol-size the other pots: each pot's daily returns (the point-in-time replays, POTS_CSV) scaled by
     (its long-run vol, expanding) / (its recent vol forecast), capped at 2x, 25% no-trade band, 5 bp per unit resized
  3  risk reversal on the 66 Deribit chains: sell the 25-delta call, buy the 25-delta put, delta-hedged (and the reverse)
  4  Whalley-Wilmott hedge band instead of the fixed 0.1 band (vrp_variants.HEDGES "ww")
  5  the BTC variance risk premium (DVOL^2 - trailing 30-day RV, and DVOL^2 - rough-vol forecast) as a predictor of
     the next 30 days' BTC and Donchian returns
  6  half-Kelly leverage for the Donchian book from a variance forecast (30-day hist vol, or the Student-t GRU), with a
     cap on the 1% daily value at risk

Item 2 (daily Deribit chains) is collect_chains.py + deribit-chains.yml; item 7 (daily entries) needs that data.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

import evaluate as ev
import run
import sv_eval
import vrp_variants as vv
from vrp_trade import delivery_prices, load_chain

H = 24
WW_K = (0.5, 2, 8, 32, 128)
for _k in WW_K:
    vv.HEDGES[f"ww{_k:g}"] = ("ww", _k)


def _tt(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def _perf(r, freq=252):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return {"cagr": float(eq.iloc[-1] ** (freq / len(r)) - 1), "vol": float(r.std() * np.sqrt(freq)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(freq)), "max_dd": float((eq / eq.cummax() - 1).min())}


# ------------------------------------------------------------------ 1. vol-size the pots

def size_series(r, halflife=21, cap=2.0, band=0.25, cost_bp=5.0, warm=252):
    """Weight for day t from returns up to t-1 only: long-run vol (expanding) over an EWMA vol forecast."""
    ew = np.sqrt((r ** 2).ewm(halflife=halflife, min_periods=halflife).mean()).shift(1)
    lr = r.expanding(min_periods=warm).std().shift(1)
    want = (lr / ew).clip(upper=cap).to_numpy()
    w, cur = np.full(len(r), np.nan), np.nan
    for i, v in enumerate(want):
        if np.isfinite(v) and (not np.isfinite(cur) or abs(v - cur) > band * cur):
            cur = v
        w[i] = cur
    w = pd.Series(w, index=r.index)
    turn = w.diff().abs().fillna(0.0)
    return w * r - turn * cost_bp / 1e4, w


def item1(pots_csv):
    p = pd.read_csv(pots_csv, index_col=0, parse_dates=True)
    out = {}
    for k in p.columns:
        r = p[k].fillna(0.0)
        s, w = size_series(r)
        ok = w.notna()
        base, sized = r[ok], s[ok]
        diff = (sized / sized.std() - base / base.std()).to_numpy()      # same-risk return difference
        out[k] = {"from": str(base.index[0].date()), "plain": _perf(base), "sized": _perf(sized), "avg_w": float(w[ok].mean()),
                  "t_sharpe_gain": float(ev.nw_t(diff, 5)),
                  "years": {str(y): [float((1 + g).prod() - 1), float((1 + sized.loc[g.index]).prod() - 1)]
                            for y, g in base.groupby(base.index.year)}}
    return out


# ------------------------------------------------------------------ 3 + 4. risk reversal and Whalley-Wilmott

def items34(folder, sv_dir):
    d, _ = run.prepare(folder)
    c = load_chain(folder)
    dlv = delivery_prices(folder, d)
    fr = vv.f30(sv_dir)
    rows = []
    for day in sorted(c.date.unique()):
        e = vv.entry(c, pd.Timestamp(day), d, dlv)
        if e is None:
            continue
        fc = float(fr.asof(e["day"]))
        for book in ("call_strip", "strip", "strangle25", "rr25", "rr25_rev"):
            b = vv.legs(e, book)
            if b is None:
                continue
            for h in ["daily", "band10", *[f"ww{k:g}" for k in WW_K]]:
                r = vv.book_pnl(e, b, h, "bid", "gross" if book.startswith("rr") else "net")
                if r:
                    rows.append({"date": str(e["day"].date()), "gate": float(e["sig_atm"] ** 2 > fc), "book": book, "hedge": h, **r})
    t = pd.DataFrame(rows)
    res = {}
    for (b, h), g in t.groupby(["book", "hedge"]):
        x, xg = g.pnl.to_numpy(), (g.pnl * g.gate).to_numpy()
        res[f"{b}|{h}"] = {"n": len(g), "always_mean": float(x.mean()), "always_t": _tt(x), "gated_mean": float(xg.mean()),
                           "gated_t": _tt(xg), "hedge_cost": float(g.hedge_cost.mean()), "worst": float(x.min())}
    return res, t


# ------------------------------------------------------------------ 5. variance premium as a return predictor

def nw_slope(y, x, lags):
    """OLS slope of y on x with its Newey-West t (overlapping 30-day targets need about 30 lags)."""
    X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n = len(y)
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for k in range(1, lags + 1):
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None]) / n
        S += (1 - k / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    V = Q @ S @ Q / n
    return float(b[1]), float(b[1] / np.sqrt(V[1, 1]))


def item5(d, sv_dir):
    day = d.close.resample("D").last()
    dv = (d.dvol / 100).resample("D").last()                              # annualized implied vol at 00:00 UTC (+1h shifted in data.py)
    rv30 = ((d.r / 100) ** 2).rolling(30 * H).sum().resample("D").last() * 365 / 30
    rough = vv.f30(sv_dir)
    rough.index = rough.index.floor("D")
    rough = rough[~rough.index.duplicated()].reindex(day.index)
    fwd = np.log(day.shift(-30) / day)                                     # next 30 days' BTC log return
    don, _ = ev.donchian_daily(d, 25, "2019-01-01", str(day.index[-1].date()))
    don = don.reindex(day.index).fillna(0.0)
    dfwd = np.log1p(don).rolling(30).sum().shift(-30)
    x = pd.DataFrame({"vrp_hist": dv ** 2 - rv30, "vrp_rough": dv ** 2 - rough, "btc": fwd, "don": dfwd}).loc["2021-04-01":]
    res = {}
    for sig in ("vrp_hist", "vrp_rough"):
        for tgt in ("btc", "don"):
            z = x[[sig, tgt]].dropna()
            b, t = nw_slope(z[tgt].to_numpy(), z[sig].to_numpy(), 30)
            m = z[z.index.day == 1]
            bm = np.polyfit(m[sig], m[tgt], 1)[0]
            tm = float(np.corrcoef(m[sig], m[tgt])[0, 1] * np.sqrt(len(m) - 2) / np.sqrt(1 - np.corrcoef(m[sig], m[tgt])[0, 1] ** 2))
            res[f"{sig}->{tgt}"] = {"daily_overlapping_beta": float(b), "nw_t": float(t), "monthly_beta": float(bm), "monthly_t": tm,
                                    "monthly_n": len(m), "corr": float(z[sig].corr(z[tgt]))}
    # a timing rule: hold BTC (or run Donchian) only on days when the premium is above its expanding median
    r1 = day.pct_change().reindex(x.index).fillna(0.0)
    s = x.vrp_hist.shift(1)
    on = (s > s.expanding(90).median()).astype(float)
    res["timing_hold"] = {"always": _perf(r1, 365), "vrp_on": _perf(r1 * on - on.diff().abs().fillna(0) * 10 / 1e4, 365),
                          "share_on": float(on.mean())}
    dd = don.reindex(x.index).fillna(0.0)
    res["timing_donchian"] = {"always": _perf(dd, 365), "vrp_on": _perf(dd * on, 365)}
    return res


# ------------------------------------------------------------------ 6. half-Kelly leverage with a VaR cap

def item6(d, out, var_cap=0.08, cap=3.0, band=0.25, cost_bp=10.0):
    ret = np.expm1(np.log(d.close.resample("D").last()).diff()).fillna(0.0)
    don, _ = ev.donchian_daily(d, 25, "2019-01-01", str(ret.index[-1].date()))
    don = don.reindex(ret.index).fillna(0.0)
    held = don != 0
    mu = don.where(held).expanding(min_periods=200).mean().shift(1)         # drift while the rule holds BTC, past days only
    hist = ((d.r / 100) ** 2).rolling(30 * H).mean().resample("D").last() * H
    hist.index = hist.index + pd.Timedelta(days=1)
    _, daily = sv_eval.neural(d, out)
    fc = {"hist30": hist}
    if "gru_t" in daily:
        g = daily["gru_t"]["f1d"] / 1e4
        g.index = g.index.floor("D")
        fc["gru_t"] = g[~g.index.duplicated()]
    q = norm.ppf(0.99)
    res, series = {}, {"donchian_1x": don}
    start = "2021-01-01"
    for name, f in fc.items():
        v = f.reindex(ret.index)
        want = {"kelly_half": (0.5 * mu / v).clip(lower=0, upper=cap),
                "vol_2.5pct": (0.025 / np.sqrt(v)).clip(upper=2.0)}
        for rule, wv in want.items():
            wv = np.minimum(wv, var_cap / (q * np.sqrt(v)))                    # 1% daily VaR at most var_cap of the pot
            w, cur = np.zeros(len(wv)), 0.0
            for i, x in enumerate(wv.to_numpy()):
                if np.isfinite(x) and (cur == 0 or abs(x - cur) > band * cur):
                    cur = x
                w[i] = cur
            w = pd.Series(w, index=ret.index)
            turn = w.diff().abs().fillna(0.0) * held
            r = w * don - turn * cost_bp / 1e4
            key = f"{rule}|{name}"
            series[key] = r
            rr = r.loc[start:]
            res[key] = _perf(rr, 365) | {"avg_w_held": float(w[held].loc[start:].mean()),
                                         "var_breach": float((rr < -var_cap).mean()),
                                         "t_vs_1x": float(ev.nw_t((rr / rr.std() - don.loc[start:] / don.loc[start:].std()).to_numpy(), 5))}
    res["donchian_1x"] = _perf(don.loc[start:], 365)
    res["hold"] = _perf(ret.loc[start:], 365)
    return res, pd.DataFrame(series)


def main(folder, out, pots_csv):
    d, _ = run.prepare(folder)
    res = {"item1_pots": item1(pots_csv)}
    res["item34_options"], t = items34(folder, f"{out}/sv")
    t.to_csv(f"{out}/round5_options.csv", index=False)
    res["item5_vrp_predict"] = item5(d, f"{out}/sv")
    res["item6_kelly"], s = item6(d, out)
    s.to_csv(f"{out}/round5_kelly_daily.csv")
    with open(f"{out}/round5.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res, indent=1, default=lambda v: round(float(v), 4)))


if __name__ == "__main__":
    main(*sys.argv[1:4])
