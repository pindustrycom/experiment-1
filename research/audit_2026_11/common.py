"""Shared helpers for the November 2026 audit: data loading, performance stats, Newey-West alpha."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

TD = 252


def load_px(prices: Path, ticker: str, col: str = "adj_close") -> pd.Series:
    d = pd.read_csv(prices / f"{ticker}.csv", parse_dates=["date"]).set_index("date")
    return d[col].astype(float).rename(ticker)


def rf_daily(prices: Path, index: pd.DatetimeIndex) -> pd.Series:
    """Daily cash return from the 13-week T-bill yield (^IRX, percent a year), using the previous day's yield."""
    irx = load_px(prices, "^IRX", "close").reindex(index).ffill().shift(1).bfill()
    return (irx / 100 / TD).rename("rf")


def nw_t(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West t statistics (Bartlett kernel)."""
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtX @ S @ XtX
    return b, b / np.sqrt(np.diag(V))


def perf(r: pd.Series, rf: pd.Series | None = None, per: int = TD) -> dict:
    r = r.dropna()
    x = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    eq = (1 + r).cumprod()
    yrs = len(r) / per
    vol = r.std() * math.sqrt(per)
    return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "vol": vol, "sharpe": x.mean() * per / vol if vol else np.nan,
            "max_dd": (eq / eq.cummax() - 1).min(), "worst": r.min(), "years": yrs, "end": eq.iloc[-1]}


def alpha(r: pd.Series, mkt: pd.Series, rf: pd.Series, per: int = TD, lags: int | None = None) -> dict:
    """Excess-return regression on the market: beta, yearly alpha, Newey-West t of alpha."""
    df = pd.concat([r, mkt, rf], axis=1, keys=["r", "m", "rf"]).dropna()
    y = (df["r"] - df["rf"]).to_numpy()
    m = (df["m"] - df["rf"]).to_numpy()
    X = np.c_[np.ones(len(y)), m]
    lags = lags if lags is not None else (5 if per == TD else 3)
    b, t = nw_t(y, X, lags)
    return {"beta": b[1], "alpha": b[0] * per, "alpha_t": t[0], "n": len(y),
            "corr": float(np.corrcoef(y, m)[0, 1])}


def beta_matched(mkt: pd.Series, rf: pd.Series, beta: float, spread: float = 0.01) -> pd.Series:
    """Holding beta x the market, the rest in cash (or borrowed at T-bill + spread a year when beta > 1), rebalanced daily."""
    cash = 1 - beta
    fin = rf + (spread / TD if beta > 1 else 0)
    return (beta * mkt + cash * fin).rename(f"SPY x{beta:.2f}")


def fmt(d: dict, keys: list[str]) -> str:
    out = []
    for k in keys:
        v = d.get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            out.append(f"{'':>8}")
        elif k in ("cagr", "vol", "max_dd", "alpha", "worst"):
            out.append(f"{100 * v:8.1f}")
        else:
            out.append(f"{v:8.2f}")
    return " ".join(out)
