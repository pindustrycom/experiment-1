"""Edge statistics shared by every test here: Newey-West t-stats, alpha vs SPY, and the deflated Sharpe ratio
(Bailey and Lopez de Prado 2014), the same method as the project's earlier edge tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649


def nw_se(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    s = x @ x / T
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / T
    return float(np.sqrt(max(s, 1e-30) / T))


def default_lags(n: int) -> int:
    return int(4 * (n / 100) ** (2 / 9))


def alpha(r: pd.Series, b: pd.Series, lags: int) -> tuple[float, float, float]:
    """OLS r = a + beta * b with Newey-West errors: (alpha per period, its t, beta)."""
    X = np.c_[np.ones(len(r)), b.values]
    coef = np.linalg.lstsq(X, r.values, rcond=None)[0]
    u = r.values - X @ coef
    Xu = X * u[:, None]
    T = len(r)
    S = Xu.T @ Xu / T
    for k in range(1, lags + 1):
        G = Xu[k:].T @ Xu[:-k] / T
        S += (1 - k / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / T)
    V = Q @ S @ Q / T
    return float(coef[0]), float(coef[0] / np.sqrt(V[0, 0])), float(coef[1])


def summary(r: pd.Series, bench: pd.Series | None = None, ppy: int = 12) -> dict:
    r = r.dropna()
    lags = default_lags(len(r))
    m, sd = r.mean(), r.std()
    out = {"start": str(r.index[0].date()), "end": str(r.index[-1].date()), "n": len(r), "ann": m * ppy,
           "vol": sd * np.sqrt(ppy), "sharpe": m / sd * np.sqrt(ppy) if sd > 0 else 0.0, "t": m / nw_se(r.values, lags),
           "max_dd": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()),
           "skew": float(stats.skew(r)), "kurt": float(stats.kurtosis(r, fisher=False))}
    if bench is not None:
        b = bench.reindex(r.index).fillna(0.0)
        a, at, beta = alpha(r, b, lags)
        out.update(alpha_ann=a * ppy, alpha_t=at, beta=beta, corr=float(np.corrcoef(r, b)[0, 1]))
    return out


def expected_max_sr(n_trials: int, sr_var: float) -> float:
    """Expected best per-period Sharpe among n_trials skill-less strategies whose Sharpes vary with variance sr_var."""
    if n_trials <= 1:
        return 0.0
    z = stats.norm.ppf
    return float(np.sqrt(sr_var) * ((1 - EULER) * z(1 - 1 / n_trials) + EULER * z(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(r: pd.Series, n_trials: int, sr_var: float) -> dict:
    """Probability that the true Sharpe is above the best a skill-less search over n_trials would show.
    Per-period Sharpe; skew and fat tails widen the error (Bailey and Lopez de Prado 2014, eq. 9)."""
    r = r.dropna()
    sr = r.mean() / r.std()
    T = len(r)
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    sr0 = expected_max_sr(n_trials, sr_var)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return {"sr0_per_period": sr0, "dsr": float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / den)),
            "psr_vs_0": float(stats.norm.cdf(sr * np.sqrt(T - 1) / den))}


def holm(p: dict[str, float]) -> dict[str, float]:
    keys = sorted(p, key=p.get)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (len(keys) - i) * p[k]))
        out[k] = run
    return out
