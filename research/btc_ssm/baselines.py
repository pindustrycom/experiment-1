"""Closed-form volatility baselines (random walk mean, so mu = 0). Each returns the variance forecast for row t+1 made at
row t's close, aligned to row t+1. Parameters are fitted on the training rows only."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

H = 24


def nll(r, v):
    return 0.5 * (np.log(2 * np.pi * v) + r ** 2 / v)


def hist_var(r, win=30 * H):
    """Random walk + historical vol: mean of the last 30 days of squared returns."""
    c = np.cumsum(np.r_[0.0, r ** 2])
    v = np.full(len(r), np.nan)
    n = np.arange(len(r))
    lo = np.maximum(n - win + 1, 0)
    v[1:] = ((c[n + 1] - c[lo]) / (n + 1 - lo))[:-1]
    return v


def seasonal(r, hours, train):
    """Hour-of-day variance factors from the training rows (mean 1)."""
    s = np.array([np.mean(r[train & (hours == h)] ** 2) for h in range(H)])
    return s / s.mean()


def ewma_var(r, lam, s_now, s_next):
    """RiskMetrics EWMA on de-seasonalized squared returns, re-seasonalized for the next hour."""
    x = r ** 2 / s_now
    z = lfilter([1 - lam], [1, -lam], x, zi=[lam * np.mean(x[: 30 * H])])[0]
    v = np.full(len(r), np.nan)
    v[1:] = z[:-1] * s_next[:-1]
    return v


def fit_ewma(r, hours, train, valid):
    s = seasonal(r, hours, train)
    s_now, s_next = s[hours], s[(hours + 1) % H]
    best = None
    for lam in (0.9, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99, 0.993, 0.995):
        v = ewma_var(r, lam, s_now, s_next)
        m = train & valid & np.isfinite(v)
        score = nll(r[m], v[m]).mean()
        if best is None or score < best[0]:
            best = (score, lam)
    return ewma_var(r, best[1], s_now, s_next), {"lambda": best[1]}


def garch_var(r, p, s_now, s_next, var0):
    a, b = p
    x = r ** 2 / s_now
    w = (1 - a - b) * var0
    z = lfilter([a], [1, -b], x, zi=[b * var0])[0] + w / (1 - b)   # sigma2_{t+1} = w + a x_t + b sigma2_t
    v = np.full(len(r), np.nan)
    v[1:] = z[:-1] * s_next[:-1]
    return v


def fit_garch(r, hours, train, valid):
    """Gaussian GARCH(1,1) with variance targeting on de-seasonalized hourly returns (MLE on training rows)."""
    s = seasonal(r, hours, train)
    s_now, s_next = s[hours], s[(hours + 1) % H]
    var0 = np.mean(r[train] ** 2 / s_now[train])
    m = train & valid

    def obj(q):
        a, b = 1 / (1 + np.exp(-q[0])), 1 / (1 + np.exp(-q[1]))
        if a + b >= 0.9995:
            return 1e9
        v = garch_var(r, (a, b), s_now, s_next, var0)
        mm = m & np.isfinite(v)
        return nll(r[mm], v[mm]).mean()

    q = minimize(obj, [np.log(0.08 / 0.92), np.log(0.9 / 0.1)], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7}).x
    a, b = 1 / (1 + np.exp(-q[0])), 1 / (1 + np.exp(-q[1]))
    return garch_var(r, (a, b), s_now, s_next, var0), {"alpha": a, "beta": b}
