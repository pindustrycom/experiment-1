"""Walk-forward run of the stochastic-volatility models (sv.py) plus the multi-horizon variance forecasts every model
needs for the volatility trade: next hour (distribution), next day and next 30 days (realized variance).

    python sv_run.py DATA_DIR OUT_DIR

For each test year: hour-of-day factors and every parameter come from training rows only (the SV models' particle
filter likelihood uses the last two training years to bound the run time). Writes OUT_DIR/sv/<model>_<year>.pkl with
hourly columns ll, pit, var and, at each 00:00 UTC decision row, f1d / f30d = forecast of the sum of squared hourly
returns over the next 24 / 720 hours (percent^2).
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.special import ndtr

import baselines as bl
import run
import sv

H = 24
HZ = {"f1d": H, "f30d": 30 * H}
FIT_HOURS = 2 * 365 * H
_G = {}


def seasonal_sum(s, hours_next, k):
    """sum of the hour factors of the next k rows (rows t+1..t+k) for every row t."""
    hs = (hours_next[:, None] + np.arange(k)[None, :]) % H
    return s[hs].sum(1)


def horizon_rows(d, te0, te1):
    """Decision rows: the bar opening at 23:00 UTC closes at 00:00; its forecasts cover the next calendar day."""
    rows = np.arange(te0, te1)
    return rows[d.index.hour[rows] == 23]


def gaussian_cols(r, mu, v):
    z = (r - mu) / np.sqrt(v)
    return bl.nll(r - mu, v) * -1, ndtr(z), v


def job(args):
    model, year = args
    d = _G["d"]
    out = _G["out"]
    path = f"{out}/sv/{model}_{year}.pkl"
    if os.path.exists(path):
        return path, "cached"
    t0 = time.time()
    r = d.r.to_numpy()
    ok = (~d.gap).to_numpy() & np.isfinite(r)
    hours = d.index.hour.to_numpy()
    tr0, _, te0, te1 = run.fold(d, year)
    train = np.zeros(len(r), bool)
    train[tr0:te0] = True
    s = bl.seasonal(r, hours, train)
    sc = s[hours]                                          # factor of row t (its own hour)
    x = r / np.sqrt(sc)
    nxt = (hours + 1) % H                                  # hour of row t+1
    rows = horizon_rows(d, te0, te1)
    ssum = {k: seasonal_sum(s, nxt[rows], n) for k, n in HZ.items()}
    var0 = float(np.mean(x[train & ok] ** 2))
    a = max(tr0, te0 - FIT_HOURS)
    info = {}
    hourly = pd.DataFrame(index=d.index[te0:te1])
    daily = pd.DataFrame(index=d.index[rows] + pd.Timedelta(hours=1))   # = the 00:00 UTC decision time
    if model == "gbm":
        sig2 = float(np.mean(x[a:te0][ok[a:te0]] ** 2))    # constant volatility, fitted on the same two years
        ll, pit, v = gaussian_cols(x[te0:te1], 0.0, np.full(te1 - te0, sig2))
        hourly["ll"], hourly["pit"], hourly["var"] = ll - 0.5 * np.log(sc[te0:te1]), pit, v * sc[te0:te1]
        for k in HZ:
            daily[k] = sig2 * ssum[k]
        info = {"sigma2": sig2}
    elif model in sv.KINDS:
        p, ll_fit = sv.fit_pf(model, x[a:te0], ok[a:te0])
        w0 = max(tr0, te0 - 60 * H)                          # warm the filter up on the last 60 training days
        _, o = sv.run_pf(model, x[w0:te1], ok[w0:te1], p, n=2000, seed=year)
        o = o[te0 - w0:]
        hourly["ll"] = o[:, 0] - 0.5 * np.log(sc[te0:te1])  # density of r = density of x / sqrt(s)
        hourly["pit"] = o[:, 1]
        hourly["var"] = o[:, 2] * sc[te0:te1]
        jump = p[4] * (p[6] ** 2 + p[5] ** 2) if model == "heston_j" else 0.0
        orow = o[rows - te0]
        for k, n in HZ.items():
            path_v = sv.expected_var_path(model, p, orow[:, 3], orow[:, 4], orow[:, 5], np.arange(1, n + 1))
            hs = (nxt[rows][:, None] + np.arange(n)[None, :]) % H
            daily[k] = (path_v * s[hs]).sum(1) + jump * ssum[k]
        info = {"params": p.tolist(), "fit_ll": ll_fit}
    elif model == "rough":
        days = d.r.iloc[tr0:te0].pow(2).groupby(d.index[tr0:te0].floor("D")).sum()
        Hh = sv.hurst(np.log(days.to_numpy() + 1e-4))
        base = np.exp(sv.rough_logforecast(r, Hh, 1.0)) / H     # hourly x-variance implied by the next-day forecast
        prev = np.r_[np.nan, base[:-1]]                         # row t is scored with the forecast made at t-1
        jp = sv.rough_jump_fit(x[a:te0], prev[a:te0], ok[a:te0])
        v = base * np.exp(jp["c"])
        ll, pit, mv = sv.mixture_scores(x[te0:te1], prev[te0:te1] * np.exp(jp["c"]), jp["lam"], jp["mu_j"], jp["s_j"])
        hourly["ll"], hourly["pit"], hourly["var"] = ll - 0.5 * np.log(sc[te0:te1]), pit, mv * sc[te0:te1]
        jump = jp["lam"] * (jp["s_j"] ** 2 + jp["mu_j"] ** 2)
        v30 = np.exp(sv.rough_logforecast(r, Hh, 30.0) + jp["c"]) / H
        daily["f1d"] = v[rows] * H + jump * ssum["f1d"]
        daily["f30d"] = v30[rows] * 30 * H + jump * ssum["f30d"]
        info = dict(jp, H=Hh)
    elif model in ("ewma", "garch"):
        if model == "ewma":
            vh, prm = bl.fit_ewma(r, hours, train, ok)
        else:
            vh, prm = bl.fit_garch(r, hours, train, ok)
        ll, pit, v = gaussian_cols(r[te0:te1], 0.0, vh[te0:te1])
        hourly["ll"], hourly["pit"], hourly["var"] = ll, pit, v
        v1 = vh[np.minimum(rows + 1, len(r) - 1)] / s[nxt[rows]]                     # de-seasonalized variance forecast for row t+1
        for k, n in HZ.items():
            if model == "ewma":
                daily[k] = v1 * ssum[k]                      # EWMA: flat term structure
            else:
                pers = prm["alpha"] + prm["beta"]
                hs = (nxt[rows][:, None] + np.arange(n)[None, :]) % H
                path_v = var0 + (v1[:, None] - var0) * pers ** np.arange(n)[None, :]
                daily[k] = (path_v * s[hs]).sum(1)
        info = prm
    elif model == "har":
        # HAR-RV (Corsi 2009) on daily realized variance: log RV over the next D days on log RV of the last 1, 7, 30 days
        rv24 = pd.Series(r ** 2, index=d.index).rolling(H).sum().to_numpy()
        feats = np.c_[np.log(rv24 + 1e-4), np.log(pd.Series(rv24).rolling(7 * H).mean().to_numpy() + 1e-4),
                      np.log(pd.Series(rv24).rolling(30 * H).mean().to_numpy() + 1e-4)]
        for k, n in HZ.items():
            fut = pd.Series(r ** 2).rolling(n).sum().shift(-n).to_numpy()   # label: next n rows (training only)
            trn = np.arange(max(tr0, 31 * H), te0 - n)
            trn = trn[(d.index.hour[trn] == 23) & np.isfinite(fut[trn]) & np.isfinite(feats[trn]).all(1)]
            X = np.c_[np.ones(len(trn)), feats[trn]]
            y = np.log(fut[trn] + 1e-4)
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            e2 = np.var(y - X @ b)
            daily[k] = np.exp(np.c_[np.ones(len(rows)), feats[rows]] @ b + 0.5 * e2)
            info[k] = b.tolist()
        hourly["ll"] = hourly["pit"] = hourly["var"] = np.nan
    hourly.attrs = {"model": model, "year": year, **{k: v for k, v in info.items() if not isinstance(v, np.ndarray)}}
    os.makedirs(f"{out}/sv", exist_ok=True)
    pd.to_pickle({"hourly": hourly, "daily": daily, "info": info}, path)
    return path, f"{time.time() - t0:.0f}s"


def init(folder, out):
    d, _ = run.prepare(folder)
    _G.update(d=d, out=out)


MODELS = ["heston_j", "logou", "heston", "rough", "gbm", "garch", "ewma", "har"]

if __name__ == "__main__":
    folder, out = sys.argv[1], sys.argv[2]
    js = [(m, y) for m in MODELS for y in run.YEARS]
    with Pool(int(os.environ.get("WORKERS", os.cpu_count())), initializer=init, initargs=(folder, out)) as pool:
        for i, (p, msg) in enumerate(pool.imap_unordered(job, js), 1):
            print(f"[{i}/{len(js)}] {os.path.basename(p)}: {msg}", flush=True)
