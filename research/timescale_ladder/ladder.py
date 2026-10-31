"""The time-scale ladder: month -> week -> day -> hour -> minute, each scale a filter, linked both ways.

Each scale keeps a two-part state for its own periods: drift (mu, the expected return) and log variance (h).
1. Filter (per scale, causal). h is a local-level Kalman filter on the period's log realized variance
   (AR(1) around a long-run mean, parameters by maximum likelihood on the training years). mu is a Kalman
   filter on the period's return whose measurement noise is the variance the h state predicts for that period:
   that variance-dependent noise is the nonlinear (EKF) link between the two parts. Intraday scales remove the
   time-of-day pattern (slot means of log rv, fitted on training years) before filtering.
2. Ladder (forecasts). For the next period of each scale, the forecast of log rv and of the return is a linear
   combination, fitted by least squares on the training years, of:
     own   - the scale's own filtered state at the end of the current period;
     down  - the parent's forecast for the parent period that the next period falls in (top-down: the future);
     up    - the child scale's filtered state at the end of the current period (bottom-up: the present).
   Scales are processed month first, so a parent's forecast is ready before its children need it, and every
   input is known before the period being forecast starts.
3. GRU (gru.py) then learns each scale's remaining forecast error from the recent sequence of these inputs.

Train 2019-01..2022-12, test 2023-01..2026-10.

    python ladder.py BARS.pkl OUT_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from scipy.optimize import minimize

LEVELS = ["month", "week", "day", "hour", "minute"]
TEST = pd.Timestamp("2023-01-01")
FLOOR = (0.25e-4) ** 2  # rv floor per minute: about one SPY tick, so a flat minute is not log(0)
FLOORS = {"month": 0.0, "week": 0.0, "day": 0.0, "hour": 60 * FLOOR, "minute": FLOOR}


@numba.njit(cache=True)
def kf_local(y, phi, m, q, r):
    """Local-level filter: h_t = m + phi (h_{t-1} - m) + w, y_t = h_t + v. Returns predicted, filtered, loglik."""
    n = len(y)
    hp, hf = np.empty(n), np.empty(n)
    x, p, ll = m, q / max(1 - phi * phi, 1e-6), 0.0
    for t in range(n):
        x = m + phi * (x - m)
        p = phi * phi * p + q
        hp[t] = x
        s = p + r
        e = y[t] - x
        ll += -0.5 * (np.log(2 * np.pi * s) + e * e / s)
        k = p / s
        x += k * e
        p *= 1 - k
        hf[t] = x
    return hp, hf, ll


@numba.njit(cache=True)
def kf_drift(rr, var_pred, phi, q):
    """Drift filter: mu_t = phi mu_{t-1} + w, r_t = mu_t + e, var(e) = predicted variance of the period (EKF link)."""
    n = len(rr)
    mf = np.empty(n)
    x, p = 0.0, q / max(1 - phi * phi, 1e-6)
    for t in range(n):
        x = phi * x
        p = phi * phi * p + q
        s = p + var_pred[t]
        k = p / s
        x += k * (rr[t] - x)
        p *= 1 - k
        mf[t] = x
    return mf


def fit_local(y: np.ndarray) -> tuple:
    def nll(th):
        phi = 0.999 / (1 + np.exp(-th[0]))
        return -kf_local(y, phi, th[1], np.exp(th[2]), np.exp(th[3]))[2]
    v = np.var(y)
    res = minimize(nll, [3.0, y.mean(), np.log(0.1 * v), np.log(0.5 * v)], method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-3})
    th = res.x
    return 0.999 / (1 + np.exp(-th[0])), th[1], np.exp(th[2]), np.exp(th[3])


def period_start(B: dict, lvl: str) -> pd.Series:
    k = B[lvl]["key"]
    if lvl == "month":
        return pd.to_datetime((k // 100).astype(str) + "-" + (k % 100).astype(str) + "-01")
    if lvl == "week":
        return pd.to_datetime(k.astype(str) + "1", format="%G%V%u")
    return pd.to_datetime(k)


def filters(B: dict) -> dict:
    """Step 1 for every scale: deseasonalized log rv, its filter, the drift filter."""
    S = {}
    for lvl in LEVELS:
        f = B[lvl].copy()
        f["start"] = period_start(B, lvl)
        train = f["start"] < TEST
        y = np.log(f["rv"].to_numpy() + FLOORS[lvl])
        seas = pd.Series(y[train.to_numpy()]).groupby(f.loc[train, "slot"].to_numpy()).mean()
        seas -= seas.mean()
        s = f["slot"].map(seas).fillna(0).to_numpy()
        yd = y - s
        phi, m, q, r = fit_local(yd[train.to_numpy()][-200_000:])
        hp, hf, _ = kf_local(yd, phi, m, q, r)
        # Calibrate exp(h) to variance on training years (log rv is a biased measure of log variance)
        kappa = np.log(np.mean(f.loc[train, "rv"].to_numpy() / np.exp(hp[train.to_numpy()] + s[train.to_numpy()])))
        var_pred = np.exp(hp + s + kappa)
        rr = f["r"].to_numpy()
        best = None
        for qd in [1e-4, 1e-3, 1e-2, 3e-2]:  # drift noise as a share of the period variance, picked on train
            mf = kf_drift(rr, var_pred, 0.99, qd * np.median(var_pred))
            prev = np.r_[0, mf[:-1]]
            ic = np.corrcoef(prev[train.to_numpy()][1:], rr[train.to_numpy()][1:])[0, 1]
            if best is None or ic > best[0]:
                best = (ic, qd, mf)
        f["y"], f["seas"], f["h_f"], f["mu_f"] = yd, s, hf, best[2]
        f.attrs.update({"phi": phi, "m": m, "q": q, "r": r, "kappa": kappa, "drift_q": best[1]})
        S[lvl] = f
        print(f"{lvl:>6}: phi {phi:.4f}  q {q:.4f}  r {r:.3f}  kappa {kappa:+.3f}  drift q {best[1]}")
    return S


def har_cols(f: pd.DataFrame) -> list:
    rv = pd.Series(np.exp(f["y"].to_numpy()))
    return [np.log(rv.rolling(w, min_periods=1).mean()).shift(1).to_numpy() for w in (1, 5, 22)]


def ladder(S: dict, use_down: bool, use_up: bool, tag: str, own_har: bool = False) -> dict:
    """Step 2: next-period forecasts of log rv (deseasonalized) and of the return, from own/down/up inputs."""
    F = {}
    for i, lvl in enumerate(LEVELS):
        f = S[lvl]
        n = len(f)
        X_h = [np.ones(n), np.r_[np.nan, f["h_f"].to_numpy()[:-1]]] + (har_cols(f) if own_har else [])
        X_m = [np.ones(n), np.r_[np.nan, f["mu_f"].to_numpy()[:-1]]]
        names = ["const", "own"] + (["har1", "har5", "har22"] if own_har else [])
        names_m = ["const", "own"]
        if use_down and i > 0:
            # The parent's forecast for the parent period this period falls in, per child (scaled by children count)
            par, P = S[LEVELS[i - 1]], F[LEVELS[i - 1]]
            nkids = f.groupby("parent").size().mean()
            idx = pd.Index(par["key"]).get_indexer(f["parent"])
            X_h.append(P["h_hat_full"][idx] - np.log(nkids))
            X_m.append(P["mu_hat"][idx] / nkids)
            names.append("down")
            names_m.append("down")
        if use_up and i < len(LEVELS) - 1:
            # The child scale's filtered state at the end of each of this scale's periods (scaled to this scale)
            ch = S[LEVELS[i + 1]]
            last = ch.groupby("parent").tail(1).set_index("parent")
            nk = ch.groupby("parent").size().mean()
            hl = (last["h_f"] + last["seas"]).reindex(f["key"]).to_numpy() + np.log(nk)
            ml = last["mu_f"].reindex(f["key"]).to_numpy() * nk
            X_h.append(np.r_[np.nan, hl[:-1]])
            X_m.append(np.r_[np.nan, ml[:-1]])
            names.append("up")
            names_m.append("up")
        Xh, Xm = np.column_stack(X_h), np.column_stack(X_m)
        ok = np.isfinite(Xh).all(1) & np.isfinite(Xm).all(1)
        tr = ok & (f["start"] < TEST).to_numpy()
        bh = np.linalg.lstsq(Xh[tr], f["y"].to_numpy()[tr], rcond=None)[0]
        bm = np.linalg.lstsq(Xm[tr], f["r"].to_numpy()[tr], rcond=None)[0]
        h_hat = np.where(ok, np.nan_to_num(Xh) @ bh, np.nan)
        mu_hat = np.where(ok, np.nan_to_num(Xm) @ bm, np.nan)
        h_full = h_hat + f["seas"].to_numpy()
        kap = np.log(np.nanmean((f["rv"] / np.exp(h_full))[tr]))
        # A period missing an input (first period, or no parent forecast yet) falls back to the own filter
        fallback = f["h_f"].shift(1).to_numpy() + f["seas"].to_numpy()
        h_full = np.where(ok, h_full + kap, fallback + f.attrs["kappa"])
        mu_hat = np.where(ok, mu_hat, 0.0)
        F[lvl] = {"h_hat_full": h_full, "mu_hat": mu_hat, "coef_h": dict(zip(names, bh.round(3))),
                  "coef_mu": dict(zip(names_m, bm.round(4))), "Xh": Xh, "Xm": Xm, "ok": ok}
    return F


def har(S: dict) -> dict:
    """Heterogeneous autoregression (Corsi 2009) on each scale's own log rv: last 1, 5 and 22 periods."""
    F = {}
    for lvl in LEVELS:
        f = S[lvl]
        rv = pd.Series(np.exp(f["y"].to_numpy()))  # deseasonalized
        X = np.column_stack([np.ones(len(f))] + [np.log(rv.rolling(w, min_periods=1).mean()).shift(1).to_numpy() for w in (1, 5, 22)])
        ok = np.isfinite(X).all(1)
        tr = ok & (f["start"] < TEST).to_numpy()
        b = np.linalg.lstsq(X[tr], f["y"].to_numpy()[tr], rcond=None)[0]
        h = np.where(ok, np.nan_to_num(X) @ b, np.nan) + f["seas"].to_numpy()
        h += np.log(np.nanmean((f["rv"] / np.exp(h))[tr]))
        F[lvl] = {"h_hat_full": h, "mu_hat": np.zeros(len(f))}
    return F


def nw_t(d: np.ndarray, lags: int) -> float:
    """t-stat of the mean of d with Newey-West standard errors."""
    n, m = len(d), d.mean()
    e = d - m
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return m / np.sqrt(v / n)


def qlike(rv, v):
    return rv / v - np.log(rv / v) - 1


def evaluate(S: dict, models: dict, ref: str = "HAR (own 1/5/22 periods)") -> pd.DataFrame:
    rows = []
    for lvl in LEVELS:
        f = S[lvl]
        te = (f["start"] >= TEST).to_numpy()
        rv = f["rv"].to_numpy() + FLOORS[lvl]
        r = f["r"].to_numpy()
        for name, F in models.items():
            v = np.exp(F[lvl]["h_hat_full"])
            m = F[lvl]["mu_hat"]
            ok = te & np.isfinite(v)
            qlike_ = np.mean(qlike(rv[ok], v[ok]))
            nll = np.mean(0.5 * (np.log(2 * np.pi * v[ok]) + r[ok] ** 2 / v[ok]))
            hit = np.mean(np.sign(m[ok]) == np.sign(r[ok])) if np.any(m[ok] != 0) else np.nan
            ic = np.corrcoef(m[ok], r[ok])[0, 1] if np.std(m[ok]) > 0 else np.nan
            vr = np.exp(models[ref][lvl]["h_hat_full"])
            both = ok & np.isfinite(vr)
            dm = nw_t(qlike(rv[both], v[both]) - qlike(rv[both], vr[both]), 30 if lvl == "minute" else 5) if name != ref else np.nan
            rows.append({"scale": lvl, "model": name, "test_periods": int(ok.sum()), "qlike": qlike_, "nll": nll,
                         "qlike_vs_har_%": 100 * (np.mean(qlike(rv[both], v[both])) / np.mean(qlike(rv[both], vr[both])) - 1),
                         "dm_t_vs_har": dm,
                         "dir_hit_%": 100 * hit, "ic": ic, "up_share_%": 100 * np.mean(r[ok] > 0)})
    return pd.DataFrame(rows)


def main(bars: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    B = pd.read_pickle(bars)
    S = filters(B)
    models = {"own scale only": ladder(S, False, False, "own"),
              "own + parent (down)": ladder(S, True, False, "down"),
              "own + child (up)": ladder(S, False, True, "up"),
              "full ladder (down + up)": ladder(S, True, True, "both"),
              "HAR + full ladder": ladder(S, True, True, "har_both", own_har=True)}
    # Baseline: last period's variance (random walk), seasonal for intraday scales
    rw = {}
    for lvl in LEVELS:
        f = S[lvl]
        y = np.log(f["rv"].to_numpy() + FLOORS[lvl])
        tr = (f["start"] < TEST).to_numpy()
        h = np.r_[np.nan, (y - f["seas"].to_numpy())[:-1]] + f["seas"].to_numpy()
        k = np.log(np.nanmean((f["rv"] / np.exp(h))[tr]))
        rw[lvl] = {"h_hat_full": h + k, "mu_hat": np.zeros(len(f))}
    models = {"last period (random walk)": rw, "HAR (own 1/5/22 periods)": har(S), **models}
    E = evaluate(S, models)
    pd.set_option("display.width", 200)
    print(E.round(4).to_string(index=False))
    for lvl in LEVELS:
        F = models["HAR + full ladder"][lvl]
        print(f"{lvl:>6} coefficients  log-var {F['coef_h']}  return {F['coef_mu']}")
    E.to_csv(out / "ladder_eval.csv", index=False)
    pd.to_pickle({"S": S, "models": models}, out / "ladder.pkl")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
