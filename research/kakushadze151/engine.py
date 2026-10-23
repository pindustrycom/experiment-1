"""Weight backtester for the 151 Trading Strategies tests.

A strategy hands back target weights decided at a day's close: a row of numbers means "rebalance to this", a row of
NaN means "do nothing, let the positions drift". Weights are fractions of capital, shorts negative; whatever is not
invested (1 - sum of weights, negative when levered) earns or pays the T-bill rate (BIL). Shorts also pay a borrow fee.
lag=0 trades at that same close (a market-on-close order), lag=1 at the next close."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def backtest(target: pd.DataFrame, ret: pd.DataFrame, rf: pd.Series, cost_bp, borrow_yr=0.005, lag: int = 0) -> pd.DataFrame:
    """ret: daily total returns per asset (NaN where the asset does not trade yet). Returns daily ret, turnover."""
    cols = list(target.columns)
    R = ret.reindex(columns=cols).fillna(0.0).to_numpy()
    idx = ret.index
    T = target.reindex(idx).shift(lag).to_numpy()
    c = np.broadcast_to(np.asarray(_vec(cost_bp, cols), float) / 1e4, (len(cols),))
    b = np.broadcast_to(np.asarray(_vec(borrow_yr, cols), float) / 252, (len(cols),))
    rfv = rf.reindex(idx).fillna(0.0).to_numpy()
    h = np.zeros(len(cols))
    out = np.zeros(len(idx))
    turn = np.zeros(len(idx))
    for t in range(len(idx)):
        if t:
            p = h @ R[t] + (1 - h.sum()) * rfv[t] - np.abs(np.minimum(h, 0)) @ b
            h = h * (1 + R[t]) / (1 + p) if 1 + p > 0 else h * 0
            out[t] = p
        row = T[t]
        if not np.all(np.isnan(row)):
            new = np.nan_to_num(row)
            dw = np.abs(new - h)
            out[t] -= dw @ c
            turn[t] = dw.sum()
            h = new
    return pd.DataFrame({"ret": out, "turnover": turn}, index=idx)


def _vec(x, cols):
    if isinstance(x, dict):
        return [x[k] for k in cols]
    if callable(x):
        return [x(k) for k in cols]
    return x


def stats(r: pd.Series, bench: pd.Series, per_year: int = 252) -> dict:
    r = r.dropna()
    b = bench.reindex(r.index).fillna(0.0)
    yrs = len(r) / per_year
    eq = (1 + r).cumprod()
    out = {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * math.sqrt(per_year),
           "sharpe": r.mean() / r.std() * math.sqrt(per_year) if r.std() > 0 else 0.0,
           "max_dd": float((eq / eq.cummax() - 1).min())}
    X = np.column_stack([np.ones(len(b)), b.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, r.to_numpy(), rcond=None)
    resid = r.to_numpy() - X @ coef
    se = math.sqrt((resid @ resid) / max(len(r) - 2, 1) * np.linalg.pinv(X.T @ X)[0, 0])
    out.update(beta=coef[1], alpha_ann=coef[0] * per_year, alpha_t=coef[0] / se if se > 0 else 0.0)
    half = r.index < pd.Timestamp("2023-01-01")
    for name, m in (("cagr_2019_22", half), ("cagr_2023_26", ~half)):
        x = r[m]
        out[name] = float((1 + x).prod() ** (per_year / max(len(x), 1)) - 1) if len(x) else float("nan")
    out["by_year"] = {int(k): float(v) for k, v in ((1 + r).groupby(r.index.year).prod() - 1).items()}
    return {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in out.items()}
