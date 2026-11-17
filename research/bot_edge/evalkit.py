"""Shared scoring for every bot: performance after costs, random-timing baseline, halves, alpha vs holding.

Inputs are bars (open/high/low/close indexed by bar open time, UTC or naive) and either
  * a list of trades (entry bar, exit bar, side, entry price, exit price), or
  * a position series (target position held over each bar, decided at the previous close).
Everything is turned into per-bar net returns, compounded into calendar-day returns.

Random baseline: the strategy's own position series shifted in time (circularly) by every possible offset.
A shift keeps exposure, trade count, holding times and costs and only destroys the timing, so the share of shifts
that do at least as well as the real series is a permutation p-value for "better than random timing".
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SPLIT = pd.Timestamp("2023-01-01")


def bar_returns_from_trades(bars: pd.DataFrame, trades, fee: float):
    """trades: iterable of (i_entry, i_exit, side, px_in, px_out) with bar positions; fee per side as a fraction.
    Returns (net per-bar returns, gross per-bar returns, position series used for the random test)."""
    c = bars.close.to_numpy(float)
    n = len(c)
    lr_net = np.zeros(n)
    lr_gross = np.zeros(n)
    pos = np.zeros(n)
    for i0, i1, side, p_in, p_out in trades:
        i0, i1 = int(i0), int(i1)
        if i1 < i0:
            continue
        # value path of the trade: entry price -> closes -> exit price
        path = np.empty(i1 - i0 + 2)
        path[0] = p_in
        path[1:-1] = c[i0:i1]
        path[-1] = p_out
        rel = path[1:] / path[:-1]
        if side < 0:      # short: value 2 - price/entry, step-by-step
            v = 2 - path / p_in
            v = np.maximum(v, 1e-9)
            rel = v[1:] / v[:-1]
        g = np.log(rel)
        lr_gross[i0:i1 + 1] += g
        lr_net[i0:i1 + 1] += g
        lr_net[i0] += math.log(1 - fee)
        lr_net[i1] += math.log(1 - fee)
        pos[i0 + 1:i1 + 1] = side      # close-to-close bars fully inside the trade
    idx = bars.index
    return pd.Series(np.expm1(lr_net), idx), pd.Series(np.expm1(lr_gross), idx), pd.Series(pos, idx)


def bar_returns_from_position(bars: pd.DataFrame, pos: pd.Series, fee: float, flat_ret: pd.Series | None = None):
    """pos[t] is held over bar t (close t-1 -> close t); it must be decided with data up to close t-1."""
    r = bars.close.pct_change().fillna(0.0)
    p = pos.reindex(bars.index).fillna(0.0).astype(float)
    turn = p.diff().abs().fillna(p.abs())
    gross = p * r
    net = gross - turn * fee
    if flat_ret is not None:
        net = net + (1 - p.abs()) * flat_ret.reindex(bars.index).fillna(0.0)
    return net, gross, p


def daily(r: pd.Series) -> pd.Series:
    idx = r.index.tz_convert(None) if getattr(r.index, "tz", None) is not None else r.index
    s = pd.Series(np.log1p(r.to_numpy()), index=idx)
    return np.expm1(s.groupby(s.index.normalize()).sum())


def nw_t(y: np.ndarray, X: np.ndarray, lags: int = 5):
    """OLS coefficients and Newey-West t-stats."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 1e-30))
    return b, b / se


def perf(d: pd.Series, per_year: int) -> dict:
    d = d.dropna()
    if len(d) < 20:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan, "vol": np.nan}
    eq = (1 + d).cumprod()
    yrs = len(d) / per_year
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "sharpe": float(d.mean() / d.std() * math.sqrt(per_year)) if d.std() > 0 else 0.0,
            "max_dd": float((eq / eq.cummax() - 1).min()), "vol": float(d.std() * math.sqrt(per_year))}


def shift_test(pos: np.ndarray, r: np.ndarray, min_shift: int = 0):
    """Sum of pos*r for every circular shift of pos (FFT). Returns (actual, p one-sided, z)."""
    n = len(r)
    actual = float(pos @ r)
    if not np.any(pos) or n < 50:
        return actual, 1.0, 0.0
    # S[k] = sum_t pos[t] * r[(t+k) mod n]: the same positions moved k bars later, wrapping around
    S = np.fft.irfft(np.conj(np.fft.rfft(pos)) * np.fft.rfft(r), n)
    k = np.arange(n)
    keep = (k >= min_shift) & (k <= n - min_shift) & (k != 0)
    S = S[keep]
    p = (1 + np.sum(S >= actual - 1e-12)) / (1 + len(S))
    z = (actual - S.mean()) / S.std() if S.std() > 0 else 0.0
    return actual, float(p), float(z)


def score(bars: pd.DataFrame, net: pd.Series, gross: pd.Series, pos: pd.Series, per_year: int, n_trades: int,
          bench_daily: pd.Series | None = None, start: str = "2019-01-01", min_shift_bars: int | None = None) -> dict:
    """All the numbers for one strategy on one market."""
    t0 = pd.Timestamp(start)
    idx_naive = bars.index.tz_convert(None) if getattr(bars.index, "tz", None) is not None else bars.index
    m = idx_naive >= t0
    r_cc = bars.close.pct_change().fillna(0.0).to_numpy()
    p = pos.to_numpy(float)
    span_days = max((idx_naive[m][-1] - idx_naive[m][0]).days, 1) * per_year / 365      # trading days in the sample
    bpd = max(1, int(round(m.sum() / span_days)))                                       # bars per trading day
    ms = min_shift_bars if min_shift_bars is not None else 20 * bpd     # shifts of under ~20 days are too close to the original
    out = {}
    _, out["p_random"], out["z_random"] = shift_test(p[m], r_cc[m], ms)
    for name, mm in (("h1", m & (idx_naive < SPLIT)), ("h2", m & (idx_naive >= SPLIT))):
        _, out[f"p_random_{name}"], out[f"z_random_{name}"] = shift_test(p[mm], r_cc[mm], ms)
    dn, dg = daily(net[m]), daily(gross[m])
    hold = bench_daily if bench_daily is not None else daily(bars.close.pct_change().fillna(0.0)[m])
    hold = hold.reindex(dn.index).fillna(0.0)
    out.update(perf(dn, per_year))
    out["gross_cagr"] = perf(dg, per_year)["cagr"]
    out["gross_sharpe"] = perf(dg, per_year)["sharpe"]
    hp = perf(hold, per_year)
    out.update({"hold_cagr": hp["cagr"], "hold_sharpe": hp["sharpe"], "hold_dd": hp["max_dd"]})
    X = np.column_stack([np.ones(len(dn)), hold.to_numpy()])
    b, t = nw_t(dn.to_numpy(), X)
    out.update({"alpha_ann": float(b[0] * per_year), "alpha_t": float(t[0]), "beta": float(b[1])})
    _, tm = nw_t(dn.to_numpy(), np.ones((len(dn), 1)))
    out["mean_t"] = float(tm[0])
    for name, mm in (("h1", dn.index < SPLIT), ("h2", dn.index >= SPLIT)):
        out[f"cagr_{name}"] = perf(dn[mm], per_year)["cagr"]
        out[f"hold_cagr_{name}"] = perf(hold[mm], per_year)["cagr"]
    out["trades"] = int(n_trades)
    out["exposure"] = float(np.mean(np.abs(p[m]) > 0))
    out["years"] = float(len(dn) / per_year)
    return out


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def verdict(row: dict, holm_p: float) -> str:
    """edge = beats random after Holm, beats holding on Sharpe after costs, alpha t > 2, and better than random in both halves."""
    if row["trades"] < 10:
        return "too few trades"
    if (holm_p < 0.05 and row["sharpe"] > row["hold_sharpe"] and row["alpha_t"] > 2
            and row["z_random_h1"] > 0 and row["z_random_h2"] > 0):
        return "EDGE"
    if row["p_random"] < 0.05:
        return "beats random (raw p) but fails a later check"
    return "no edge"
