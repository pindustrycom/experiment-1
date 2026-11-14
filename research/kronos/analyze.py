"""Score the Kronos forecast paths (forecast.py) against what actually happened, and against simple baselines.

Usage: python analyze.py RESULTS_DIR OUT_DIR
RESULTS_DIR holds paths_*.npz and the <SYMBOL>_1h.csv files the forecasts were made from (the kronos-results release).

1. Direction: does the mean of the sampled paths call the sign of the next 1, 2, 4, 8, 24 hours better than always
   guessing "up" (holding) or than momentum (the sign of the previous same-length return)? Hit rate, AUC, rank IC with a
   Newey-West t-stat, and Holm-adjusted p-values over every horizon and coin tested.
2. Volatility: is the spread of the sampled paths a good forecast of the next 24 hours' realized variance (sum of
   squared hourly log returns)? QLIKE loss against EWMA, GARCH(1,1) and HAR-RV, each using only data before the
   origin (GARCH and HAR are fit once on 2020-01 to 2025-08, before the test).
3. Calibration: where the realized 24-hour return falls among the 8 sampled paths (uniform if calibrated).
4. Trading: at each origin hold the coin for the next step (BTC 4h, ETH 8h) when the mean path is up (long/flat), or
   long/short on its sign, at 0/5/10/25 bp a side, against holding, momentum and (BTC) the live pot's 4h Donchian rule.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

HORIZONS = (1, 2, 4, 8, 24)
STEP = {"BTCUSDT": 4, "ETHUSDT": 8}
COSTS = (0.0, 0.0005, 0.0010, 0.0025)
FIT = (pd.Timestamp("2020-01-01"), pd.Timestamp("2025-09-01"))
HOURS_YEAR = 24 * 365


def nw_t(x: np.ndarray, lags: int) -> float:
    """t-stat of the mean of x with a Newey-West (Bartlett) long-run variance."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else float("nan")


def auc(score: np.ndarray, up: np.ndarray) -> float:
    r = stats.rankdata(score)
    n1 = up.sum()
    n0 = len(up) - n1
    return float((r[up].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def load(res: str) -> tuple[dict, dict]:
    parts = [np.load(f) for f in sorted(glob.glob(f"{res}/paths_*.npz"))]
    z = {k: np.concatenate([p[k] for p in parts]) for k in ("symbol", "origin", "close0", "c", "h", "l")}
    data = {}
    for sym in STEP:
        if not glob.glob(f"{res}/{sym}_1h.csv"):
            continue
        d = pd.read_csv(f"{res}/{sym}_1h.csv", index_col="t")
        d.index = pd.to_datetime(d.index, format="ISO8601")
        data[sym] = d.astype(float)
    return z, data


# ---------------------------------------------------------------- volatility baselines (causal by construction)
def garch_fit(r: np.ndarray) -> tuple[float, float, float]:
    r = r - r.mean()

    def nll(p):
        w, a, b = np.exp(p[0]), 1 / (1 + np.exp(-p[1])), 1 / (1 + np.exp(-p[2]))
        if a + b >= 0.9999:
            return 1e12
        h = np.empty_like(r)
        h[0] = r.var()
        for t in range(1, len(r)):
            h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
        return 0.5 * np.sum(np.log(h) + r ** 2 / h)

    p0 = [np.log(r.var() * 0.05), np.log(0.1 / 0.9), np.log(0.85 / 0.15)]
    p = optimize.minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 400}).x
    return float(np.exp(p[0])), float(1 / (1 + np.exp(-p[1]))), float(1 / (1 + np.exp(-p[2])))


def garch_var(r: np.ndarray, w: float, a: float, b: float) -> np.ndarray:
    """h[t] = the conditional variance of r[t] given r[:t] (known at the close of bar t-1)."""
    h = np.empty(len(r) + 1)
    h[0] = np.nanvar(r)
    for t in range(len(r)):
        h[t + 1] = w + a * r[t] ** 2 + b * h[t]
    return h


def vol_baselines(d: pd.DataFrame, idx: np.ndarray) -> dict:
    """Forecasts of the next-24h variance for origins at row positions idx (the bar at idx is the first unseen)."""
    r = np.log(d.close).diff().fillna(0).to_numpy()
    r2 = r ** 2
    cs = np.concatenate([[0], np.cumsum(r2)])
    past = lambda i, n: (cs[i] - cs[i - n]) / n * 24  # noqa: E731   (mean hourly var over n bars, scaled to a day)
    out = {}
    # EWMA of squared hourly returns, half-life 1 day (no lookahead: the value at i uses r[:i])
    ew = pd.Series(r2).ewm(halflife=24, adjust=False).mean().shift(1).to_numpy()
    out["ewma"] = ew[idx] * 24
    # GARCH(1,1) on hourly returns, fit on the pre-test window only
    t = d.index
    fit = (t >= FIT[0]) & (t < FIT[1])
    w, a, b = garch_fit(r[fit])
    h = garch_var(r, w, a, b)
    lr = w / (1 - a - b)
    fc = np.zeros(len(idx))
    for k in range(24):
        fc += lr + (a + b) ** k * (h[idx] - lr)
    out["garch"] = fc
    out["_garch_params"] = (w, a, b)
    # HAR-RV on log daily RV: last 1, 7, 30 days -> next day; OLS on pre-test origins every 4h
    def feats(ii):
        return np.column_stack([np.ones(len(ii)), np.log(past(ii, 24)), np.log(past(ii, 168)), np.log(past(ii, 720))])

    ii = np.arange(720, len(r) - 24, 4)
    ii = ii[(t[ii] >= FIT[0]) & (t[ii] < FIT[1])]
    y = np.log(cs[ii + 24] - cs[ii])
    ok = np.isfinite(y) & np.isfinite(feats(ii)).all(1)
    beta, *_ = np.linalg.lstsq(feats(ii)[ok], y[ok], rcond=None)
    s2 = np.var(y[ok] - feats(ii)[ok] @ beta)
    out["har"] = np.exp(feats(idx) @ beta + s2 / 2)
    return out


def qlike(rv: np.ndarray, f: np.ndarray) -> np.ndarray:
    q = rv / f
    return q - np.log(q) - 1


# ---------------------------------------------------------------- trading
def perf(ret: np.ndarray, per_year: float) -> dict:
    eq = np.cumprod(1 + ret)
    dd = eq / np.maximum.accumulate(eq) - 1
    yrs = len(ret) / per_year
    sd = ret.std() * np.sqrt(per_year)
    return {"total_%": round(100 * (eq[-1] - 1), 1), "cagr_%": round(100 * (eq[-1] ** (1 / yrs) - 1), 1),
            "vol_%": round(100 * sd, 1), "sharpe": round(ret.mean() * per_year / sd, 2) if sd else None,
            "max_dd_%": round(100 * dd.min(), 1)}


def run_positions(pos: np.ndarray, gross: np.ndarray, cost: float) -> np.ndarray:
    turn = np.abs(np.diff(np.concatenate([[0], pos])))
    return pos * gross - turn * cost


def donchian_4h(d: pd.DataFrame) -> pd.Series:
    """The live pot's rule on 4h bars (experiments/strategies/strats/crypto.py, 20-bar channel, 200-bar trend):
    position held over each 4h bar, decided at the previous bar's close."""
    b = d.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    hi = b.high.rolling(20).max().shift()
    lo = b.low.rolling(20).min().shift()
    sma = b.close.rolling(200).mean()
    want, pos = False, []
    for c, h_, l_, s in zip(b.close, hi, lo, sma):
        if not want and c > h_ and c > s:
            want = True
        elif want and c < l_:
            want = False
        pos.append(want)
    return pd.Series(np.array(pos, float), index=b.index).shift(1).fillna(0)   # held during the bar


def main() -> None:
    res, out = sys.argv[1], sys.argv[2]
    z, data = load(res)
    summary = {"n_origins": {}, "direction": [], "vol": [], "calibration": {}, "trading": {}}
    for sym, step in STEP.items():
        m = z["symbol"] == sym
        if not m.any():
            continue
        d = data[sym]
        org = pd.to_datetime(z["origin"][m])
        order = np.argsort(org.values)
        org = org[order]
        C, c0 = z["c"][m][order], z["close0"][m][order]          # (n, S, 24) log(close/close0)
        idx = d.index.get_indexer(org)
        assert (idx >= 0).all()
        lc = np.log(d.close.to_numpy())
        assert np.allclose(np.log(c0), lc[idx - 1]), "close0 must be the close of the bar before the origin"
        n = len(idx)
        summary["n_origins"][sym] = {"n": int(n), "first": str(org[0]), "last": str(org[-1])}
        real = {h: lc[idx + h - 1] - lc[idx - 1] for h in HORIZONS}
        prev = {h: lc[idx - 1] - lc[idx - 1 - h] for h in HORIZONS}

        # 1. direction
        for h in HORIZONS:
            mu = C[:, :, h - 1].mean(1)
            pup = (C[:, :, h - 1] > 0).mean(1)
            up = real[h] > 0
            lags = max(0, int(np.ceil(h / step)) - 1) + 2
            rk = stats.rankdata(mu) / n - 0.5
            rr = stats.rankdata(real[h]) / n - 0.5
            ic = float(np.corrcoef(rk, rr)[0, 1])
            t = nw_t(rk * rr * 12, lags)        # rank-IC products, NW for overlapping horizons
            row = {"symbol": sym, "h": h, "n": int(n),
                   "hit_kronos": round(float(np.mean(np.sign(mu) == np.sign(real[h]))), 4),
                   "hit_always_up": round(float(up.mean()), 4),
                   "hit_momentum": round(float(np.mean(np.sign(prev[h]) == np.sign(real[h]))), 4),
                   "share_called_up": round(float((mu > 0).mean()), 3),
                   "auc_mean": round(auc(mu, up), 4), "auc_prob_up": round(auc(pup, up), 4),
                   "rank_ic": round(ic, 4), "ic_t": round(t, 2), "p": float(2 * stats.norm.sf(abs(t))),
                   "ic_momentum": round(float(stats.spearmanr(prev[h], real[h])[0]), 4),
                   "corr_kronos_vs_past_return": round(float(stats.spearmanr(mu, prev[h])[0]), 3)}
            summary["direction"].append(row)

        # 2. volatility (next 24h realized variance)
        path_r = np.diff(np.concatenate([np.zeros(C.shape[:2] + (1,)), C], axis=2), axis=2)
        rv_k = (path_r ** 2).sum(2).mean(1)
        cs = np.concatenate([[0], np.cumsum(np.log(d.close).diff().fillna(0).to_numpy() ** 2)])
        rv = cs[idx + 24] - cs[idx]
        base = vol_baselines(d, idx)
        sc = (np.exp(np.mean(np.log(rv))) / np.exp(np.mean(np.log(rv_k))))
        fcs = {"kronos": rv_k, "kronos_x_level_fixed": rv_k * sc, "ewma": base["ewma"], "garch": base["garch"],
               "har": base["har"]}
        ref = qlike(rv, base["garch"])
        for name, f in fcs.items():
            ql = qlike(rv, f)
            mz = stats.linregress(np.log(f), np.log(rv))
            summary["vol"].append({"symbol": sym, "model": name, "qlike": round(float(ql.mean()), 4),
                                   "vs_garch": round(float((ql - ref).mean()), 4),
                                   "t_vs_garch": round(nw_t(ql - ref, 8), 2),
                                   "log_r2": round(float(mz.rvalue ** 2), 3), "log_slope": round(float(mz.slope), 2),
                                   "bias_ratio": round(float(np.exp(np.mean(np.log(f) - np.log(rv)))), 2)})
        summary["vol"].append({"symbol": sym, "note": "kronos_x_level_fixed rescales Kronos by the in-sample ratio "
                               "of geometric means (look-ahead in the level, kept only to see the ranking skill)"})

        # 3. calibration of the 24h return among the S sampled paths
        S = C.shape[1]
        rank = (C[:, :, 23] < real[24][:, None]).sum(1)
        counts = np.bincount(rank, minlength=S + 1)
        lo_, hi_ = C[:, :, 23].min(1), C[:, :, 23].max(1)
        summary["calibration"][sym] = {
            "rank_counts_0_to_S": counts.tolist(), "expected_each": round(n / (S + 1), 1),
            "chi2_p": float(stats.chisquare(counts).pvalue),
            "inside_min_max": round(float(np.mean((real[24] >= lo_) & (real[24] <= hi_))), 3),
            "inside_expected": round((S - 1) / (S + 1), 3),
            "path_sd_24h_%": round(100 * float(np.median(C[:, :, 23].std(1))), 2),
            "realized_abs_24h_%": round(100 * float(np.median(np.abs(real[24]))), 2)}

        # 4. trading: consecutive non-overlapping holds of `step` hours
        keep = np.concatenate([[True], np.diff(org.values).astype("timedelta64[h]").astype(int) == step])
        if not keep.all():
            print(sym, "gaps in the origin grid:", int((~keep).sum()))
        gross = np.exp(real[step]) - 1
        mu = C[:, :, step - 1].mean(1)
        per_year = HOURS_YEAR / step
        sig = {"kronos_long_flat": (mu > 0).astype(float), "kronos_long_short": np.sign(mu),
               "momentum_long_flat": (prev[step] > 0).astype(float), "hold": np.ones(n)}
        if sym == "BTCUSDT":
            dc = donchian_4h(d)
            # the Donchian position held over each 4h bar; compound bars within one step
            ret4 = dc * (d.close.resample("4h").last().pct_change())
            gr = []
            for o in org:
                seg = ret4.loc[o:o + pd.Timedelta(hours=step - 1)]
                gr.append(float(np.prod(1 + seg.fillna(0)) - 1))
            sig["_donchian_gross"] = np.array(gr)
            sig["_donchian_pos"] = dc.reindex(org).fillna(0).to_numpy()
        tr = {}
        for cost in COSTS:
            key = f"{int(cost * 1e4)}bp"
            tr[key] = {}
            for name, p in sig.items():
                if name.startswith("_"):
                    continue
                tr[key][name] = perf(run_positions(p, gross, cost), per_year)
                if name.startswith("kronos"):
                    tr[key][name]["share_long"] = round(float((p > 0).mean()), 3)
                    tr[key][name]["trades_per_year"] = round(float(np.abs(np.diff(p)).sum() / (n / per_year)), 0)
            if "_donchian_gross" in sig:
                pos = sig["_donchian_pos"]
                turn = np.abs(np.diff(np.concatenate([[0], pos])))
                tr[key]["donchian_4h"] = perf(sig["_donchian_gross"] - turn * cost, per_year)
            ex = run_positions(sig["kronos_long_flat"], gross, cost) - sig["kronos_long_flat"].mean() * gross
            tr[key]["kronos_long_flat_vs_same_exposure_hold_t"] = round(nw_t(ex, 2), 2)
        summary["trading"][sym] = tr
        eq = pd.DataFrame({"origin": org, "hold": np.cumprod(1 + gross),
                           "kronos_long_flat_5bp": np.cumprod(1 + run_positions(sig["kronos_long_flat"], gross, 5e-4)),
                           "kronos_long_short_5bp": np.cumprod(1 + run_positions(sig["kronos_long_short"], gross,
                                                                                  5e-4))})
        eq.to_csv(f"{out}/equity_{sym}.csv", index=False)

    # multiple testing over all direction tests (Holm)
    ps = np.array([r["p"] for r in summary["direction"]])
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (len(ps) - k)))
        adj[i] = run
    for r, a in zip(summary["direction"], adj):
        r["p"] = round(r["p"], 4)
        r["p_holm"] = round(float(a), 4)
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    pd.DataFrame(summary["direction"]).to_csv(f"{out}/direction.csv", index=False)
    pd.DataFrame([v for v in summary["vol"] if "model" in v]).to_csv(f"{out}/vol.csv", index=False)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
