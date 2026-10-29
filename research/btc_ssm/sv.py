"""Continuous-time stochastic-volatility models for Bitcoin, discretized hourly and filtered with a particle filter.

Joaquin's model list (Oct 29, 2026): GBM, Heston, Heston + jumps, rough volatility + jumps, plus the log-volatility
Ornstein-Uhlenbeck model. All run on de-seasonalized hourly log returns x_t = r_t / sqrt(s_hour) (percent) with zero
drift (the drift is not predictable at this horizon, see the README), and Euler steps of dt = 1 hour:

  GBM       x_t = sigma z_t
  log-OU    x_t = exp(h_t / 2) z_t,  h_{t+1} = h_t + kappa (m - h_t) + eta (rho z_t + sqrt(1 - rho^2) e_t)
  Heston    x_t = sqrt(v_t) z_t,     v_{t+1} = v_t + kappa (theta - v_t) + xi sqrt(v_t) (rho z_t + sqrt(1 - rho^2) e_t)
  Heston+J  x_t = sqrt(v_t) z_t + B_t J_t,  B_t ~ Bernoulli(lam), J_t ~ N(mu_j, s_j^2)   (Bates-style compound Poisson)
  Rough+J   rough fractional log-vol forecast (Gatheral, Jaisson & Rosenbaum 2018) of the next day's realized variance
            from the history of daily log RV, with H from the variogram, plus the same jump mixture

rho couples the price and variance shocks (leverage). Full truncation keeps Heston's variance positive. Latent states
are filtered by a bootstrap particle filter; parameters are fitted by its likelihood on the training years only
(common random numbers so the optimizer sees a smooth-ish surface). Every forecast uses data up to its time only.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import ndtr

H = 24
LOG2PI = math.log(2 * math.pi)


# ------------------------------------------------------------------------------------------------ particle filter

@njit(cache=True)
def _pf(x, ok, kind, p, eps, uni, ujump, out):
    """kind 0 = log-OU, 1 = Heston, 2 = Heston + jumps. p = [kappa, level, volvol, rho, lam, mu_j, s_j].
    For each t writes out[t] = (log predictive density of x_t, PIT of x_t, predictive mean variance of x_t (incl. jumps),
    particle mean of the diffusive variance for t+1 (state for multi-step forecasts), mean of h or v for t+1,
    mean of the squared state for t+1 (its dispersion, needed by the log-OU multi-step mean))."""
    T, N = eps.shape
    kappa, level, vv, rho, lam, mj, sj = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    sj2 = sj * sj
    rr = math.sqrt(1.0 - rho * rho)
    st = np.empty(N)
    if kind == 0:
        for i in range(N):
            st[i] = level + 0.5 * eps[0, i] * vv / math.sqrt(max(1 - (1 - kappa) ** 2, 1e-6))
    else:
        for i in range(N):
            st[i] = level
    w = np.empty(N)
    cdf = np.empty(N)
    var = np.empty(N)
    idx = np.empty(N, np.int64)
    tmp = np.empty(N)
    total = 0.0
    for t in range(T):
        xt = x[t]
        for i in range(N):
            v = math.exp(st[i]) if kind == 0 else max(st[i], 1e-8)
            var[i] = v
        # predictive density, PIT and mean variance of x_t
        sw = 0.0
        pit = 0.0
        mv = 0.0
        for i in range(N):
            v = var[i]
            d0 = math.exp(-0.5 * (LOG2PI + math.log(v) + xt * xt / v))
            c0 = 0.5 * math.erfc(-xt / math.sqrt(2 * v))
            if kind == 2:
                vj = v + sj2
                d1 = math.exp(-0.5 * (LOG2PI + math.log(vj) + (xt - mj) ** 2 / vj))
                c1 = 0.5 * math.erfc(-(xt - mj) / math.sqrt(2 * vj))
                w[i] = (1 - lam) * d0 + lam * d1
                cdf[i] = (1 - lam) * c0 + lam * c1
                mv += v + lam * (sj2 + mj * mj)
            else:
                w[i] = d0
                cdf[i] = c0
                mv += v
            sw += w[i]
            pit += cdf[i]
        if not ok[t]:
            # no observation (exchange outage): just propagate without leverage
            for i in range(N):
                e = eps[t, i]
                if kind == 0:
                    st[i] = st[i] + kappa * (level - st[i]) + vv * e
                else:
                    vp = max(st[i], 0.0)
                    st[i] = st[i] + kappa * (level - vp) + vv * math.sqrt(vp) * e
            out[t, 0] = np.nan
            out[t, 1] = np.nan
            out[t, 2] = mv / N
        else:
            ll = math.log(max(sw / N, 1e-300))
            total += ll
            out[t, 0] = ll
            out[t, 1] = pit / N
            out[t, 2] = mv / N
            # systematic resampling
            acc = 0.0
            j = 0
            u0 = uni[t] / N
            for i in range(N):
                tmp[i] = w[i] / sw
            acc = tmp[0]
            for i in range(N):
                u = u0 + i / N
                while u > acc and j < N - 1:
                    j += 1
                    acc += tmp[j]
                idx[i] = j
            for i in range(N):
                tmp[i] = st[idx[i]]
            for i in range(N):
                st[i] = tmp[i]
            # propagate with leverage: diffusive shock z implied by x_t (net of a sampled jump)
            for i in range(N):
                v = math.exp(st[i]) if kind == 0 else max(st[i], 1e-8)
                d = xt
                if kind == 2:
                    vj = v + sj2
                    d0 = (1 - lam) * math.exp(-0.5 * (math.log(v) + xt * xt / v))
                    d1 = lam * math.exp(-0.5 * (math.log(vj) + (xt - mj) ** 2 / vj))
                    if ujump[t, i] * (d0 + d1) < d1:      # sampled "a jump happened": remove its posterior mean
                        d = xt - (mj + sj2 / vj * (xt - mj))
                z = d / math.sqrt(v)
                if z > 6.0:
                    z = 6.0
                elif z < -6.0:
                    z = -6.0
                e = rho * z + rr * eps[t, i]
                if kind == 0:
                    st[i] = st[i] + kappa * (level - st[i]) + vv * e
                else:
                    vp = max(st[i], 0.0)
                    st[i] = st[i] + kappa * (level - vp) + vv * math.sqrt(vp) * e
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for i in range(N):
            v = math.exp(st[i]) if kind == 0 else max(st[i], 0.0)
            s1 += v
            s2 += st[i]
            s3 += st[i] * st[i]
        out[t, 3] = s1 / N
        out[t, 4] = s2 / N
        out[t, 5] = s3 / N
    return total


KINDS = {"logou": 0, "heston": 1, "heston_j": 2}


def unpack(kind, q, var0):
    """Unconstrained q -> p = [kappa, level, volvol, rho, lam, mu_j, s_j] (hourly units, de-seasonalized)."""
    kappa = 1 / (1 + math.exp(-q[0])) * 0.5
    rho = math.tanh(q[3])
    if kind == "logou":
        level, vv = math.log(var0) + q[1], math.exp(q[2])
    else:
        level, vv = var0 * math.exp(q[1]), math.exp(q[2])
    lam = mj = 0.0
    sj = 1.0
    if kind == "heston_j":
        lam = 0.1 / (1 + math.exp(-q[4]))
        mj = q[5]
        sj = math.exp(q[6])
    return np.array([kappa, level, vv, rho, lam, mj, sj])


def q0(kind):
    q = [math.log(0.03 / 0.47), -0.4, math.log(0.25), -0.05]
    if kind == "heston":
        q[2] = math.log(0.12)
    if kind == "heston_j":
        q[1], q[2] = -0.5, math.log(0.10)
        q += [math.log(0.01 / 0.09), 0.0, math.log(2.5)]
    return np.array(q)


def run_pf(kind, x, ok, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    T = len(x)
    eps = rng.standard_normal((T, n))
    uni = rng.random(T)
    ujump = rng.random((T, n)) if kind == "heston_j" else np.full((T, 1), 0.5)
    out = np.zeros((T, 6))
    ll = _pf(x, ok, KINDS[kind], p, eps, uni, ujump, out)
    return ll, out


def fit_pf(kind, x, ok, n=400, seed=0, maxiter=300):
    """Particle-filter MLE with common random numbers (same shocks at every evaluation)."""
    var0 = float(np.mean(x[ok] ** 2))
    rng = np.random.default_rng(seed)
    T = len(x)
    eps = rng.standard_normal((T, n))
    uni = rng.random(T)
    ujump = rng.random((T, n)) if kind == "heston_j" else np.zeros((T, 1)) + 0.5
    out = np.zeros((T, 6))
    k = KINDS[kind]

    def obj(q):
        try:
            p = unpack(kind, q, var0)
        except OverflowError:
            return 1e12
        ll = _pf(x, ok, k, p, eps, uni, ujump, out)
        return -ll / ok.sum() if np.isfinite(ll) else 1e12

    q = q0(kind)
    best = minimize(obj, q, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-6})
    return unpack(kind, best.x, var0), -best.fun


def expected_var_path(kind, p, state_mean_v, state_mean_h, state_mean_h2, k):
    """E[diffusive variance] k = 1..K steps ahead given the filtered particle cloud at t+1.
    Heston: exact for the Euler scheme. log-OU: h_{t+k} is Gaussian given h_{t+1} with mean level + (h - level) phi^(k-1)
    and variance eta^2 (1 - phi^(2(k-1))) / (1 - phi^2); the cloud's own spread (var of h_{t+1}) shrinks by phi^(2(k-1))."""
    kappa, level, vv = p[0], p[1], p[2]
    phi = 1 - kappa
    if kind == "logou":
        var_h = np.maximum(state_mean_h2 - state_mean_h ** 2, 0.0)[:, None]
        s2 = vv ** 2 * (1 - phi ** (2 * (k - 1))) / (1 - phi ** 2)
        return np.exp(level + (state_mean_h[:, None] - level) * phi ** (k - 1) + 0.5 * (s2 + var_h * phi ** (2 * (k - 1))))
    return level + (state_mean_v[:, None] - level) * phi ** (k - 1)


# ------------------------------------------------------------------------------------------------ rough volatility

def hurst(logrv_daily):
    """H from the variogram m(2, D) = E|log RV_{t+D} - log RV_t|^2 ~ D^{2H} (Gatheral et al. 2018), D = 1..20 days."""
    y = np.asarray(logrv_daily, float)
    lags = np.arange(1, 21)
    m = [np.nanmean((y[L:] - y[:-L]) ** 2) for L in lags]
    slope = np.polyfit(np.log(lags), np.log(m), 1)[0]
    return slope / 2


def rough_weights(Hh, delta_days, J=400):
    """Discretized RFSV predictor kernel on past daily log RV (lag u = j + 1/2 days), normalized to sum to one:
    E[log v_{t+D}] ~ sum_j w_j log RV_{t-j},  w_j ~ 1 / ((u + D) u^{H + 1/2})."""
    u = np.arange(J) + 0.5
    w = 1.0 / ((u + delta_days) * u ** (Hh + 0.5))
    return w / w.sum()


def rough_logforecast(r, Hh, delta_days, J=400):
    """At every hour t: forecast of log(variance over the next delta_days days) from the trailing daily log RVs
    (windows of 24 hours ending at t, t-24, ...). Uses returns up to row t only."""
    rv24 = np.convolve(r ** 2, np.ones(H), "full")[: len(r)]           # sum of r^2 over rows t-23..t
    lrv = np.log(rv24 + 1e-4)
    w = rough_weights(Hh, delta_days, J)
    out = np.full(len(r), np.nan)
    lags = np.arange(J) * H
    for t in range(J * H, len(r)):
        out[t] = w @ lrv[t - lags]
    return out


def rough_jump_fit(x, base_var, ok):
    """Fit the level correction c and the jump mixture (lam, mu_j, s_j) of the rough model on training rows."""
    m = ok & np.isfinite(base_var)

    def nll(q):
        c, lam, mj, sj = q[0], 0.1 / (1 + np.exp(-q[1])), q[2], np.exp(q[3])
        v = base_var[m] * np.exp(c)
        d = (1 - lam) * np.exp(-0.5 * (LOG2PI + np.log(v) + x[m] ** 2 / v)) + \
            lam * np.exp(-0.5 * (LOG2PI + np.log(v + sj ** 2) + (x[m] - mj) ** 2 / (v + sj ** 2)))
        return -np.mean(np.log(d + 1e-300))

    q = minimize(nll, [0.0, np.log(0.01 / 0.09), 0.0, np.log(2.5)], method="Nelder-Mead", options={"maxiter": 2000}).x
    return {"c": q[0], "lam": 0.1 / (1 + np.exp(-q[1])), "mu_j": q[2], "s_j": float(np.exp(q[3]))}


def mixture_scores(x, v, lam, mj, sj):
    """Log density, PIT and mean variance of x under (1 - lam) N(0, v) + lam N(mj, v + sj^2)."""
    vj = v + sj ** 2
    d = (1 - lam) * np.exp(-0.5 * (LOG2PI + np.log(v) + x ** 2 / v)) + lam * np.exp(-0.5 * (LOG2PI + np.log(vj) + (x - mj) ** 2 / vj))
    pit = (1 - lam) * ndtr(x / np.sqrt(v)) + lam * ndtr((x - mj) / np.sqrt(vj))
    return np.log(d + 1e-300), pit, v + lam * (sj ** 2 + mj ** 2)
