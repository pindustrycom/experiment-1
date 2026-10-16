"""Fit on 2018-2021, freeze, test on 2022-2025.

Models (all predict NVDA's 5-session log return after the close, with an uncertainty):
- ekf          EKF + SSM over price, market cap index and sentiment by category
- ekf_price    the same filter without the sentiment measurements (does sentiment add anything?)
- ridge        linear regression on the same measurements
- ridge_price  linear regression without sentiment
- momentum     the 20-session return, scaled on the training years
- drift        the training years' average 5-session return
- zero         no change

The EKF's parameters are frozen after training; during the test it keeps filtering, i.e. its state
is updated by each day's measurements, exactly as it would run live.
"""
from __future__ import annotations

import copy
import logging
import math
import time

import numpy as np
import pandas as pd
import torch

from .config import Config
from .ekf import EKFSSM, target_nll
from .features import MARKET_COLS, PRICE_COLS, SENT_COLS, TARGET, Scaler

log = logging.getLogger("exp3.model")
Z80 = 1.2815515655446004
FULL = PRICE_COLS + MARKET_COLS + SENT_COLS
PRICE_ONLY = PRICE_COLS + MARKET_COLS


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def split(df: pd.DataFrame, cfg: Config) -> dict:
    """Row positions: first usable row, training, validation (end of training, for early stopping;
    the last `horizon` rows are dropped so no target reaches into the test years) and test."""
    ok = df[PRICE_ONLY].notna().all(axis=1)
    first = int(np.argmax(ok.to_numpy()))
    idx = df.index
    tr0 = max(first, int(idx.searchsorted(pd.Timestamp(cfg.train_start))))
    tr1 = int(idx.searchsorted(pd.Timestamp(cfg.train_end), side="right")) - 1
    te0, te1 = tr1 + 1, int(idx.searchsorted(pd.Timestamp(cfg.test_end), side="right")) - 1
    te1 = min(te1, int(np.where(df[TARGET].notna())[0].max()))
    va1 = tr1 - cfg.horizon
    va0 = va1 - cfg.val_sessions + 1
    return {"first": first, "tr0": tr0, "fit1": va0 - cfg.horizon - 1, "va0": va0, "va1": va1, "tr1": tr1, "te0": te0, "te1": te1}


class Data:
    def __init__(self, df: pd.DataFrame, cols: list[str], sp: dict):
        self.cols, self.sp = cols, sp
        X = df[cols].to_numpy(float)
        tr = slice(sp["tr0"], sp["tr1"] + 1)
        self.scaler = Scaler().fit(X[tr])
        Z = self.scaler.transform(X)
        self.mask = torch.tensor(~np.isnan(Z), dtype=torch.float32)
        self.z = torch.tensor(np.nan_to_num(Z), dtype=torch.float32)
        y = df[TARGET].to_numpy(float)
        ty = slice(sp["tr0"], sp["va1"] + 1)      # targets that end inside the training years
        self.y_mu, self.y_sd = float(np.mean(y[ty])), float(np.std(y[ty]))
        self.y = torch.tensor((y - self.y_mu) / self.y_sd, dtype=torch.float32)[:, None]


def fit_ekf(d: Data, cfg: Config, seed: int) -> tuple[EKFSSM, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sp = d.sp
    m = EKFSSM(d.z.shape[1], 1, cfg.d_state, cfg.d_hidden)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    burn = 32
    lo, hi = sp["first"], sp["fit1"]
    w = min(cfg.window, hi - lo + 1)

    def val_loss():
        with torch.no_grad():
            mu, var, _, _ = m(d.z[None, lo:sp["va1"] + 1], d.mask[None, lo:sp["va1"] + 1])
            k = sp["va1"] - sp["va0"] + 1
            return float(target_nll(mu[0, -k:], var[0, -k:], d.y[sp["va0"]:sp["va1"] + 1]))

    best, best_state, since = math.inf, copy.deepcopy(m.state_dict()), 0
    for step in range(cfg.steps):
        starts = rng.integers(lo, max(lo + 1, hi + 2 - w), size=cfg.batch)
        zb = torch.stack([d.z[s:s + w] for s in starts])
        mb = torch.stack([d.mask[s:s + w] for s in starts])
        yb = torch.stack([d.y[s:s + w] for s in starts])
        # targets of rows before the training start are not scored
        yb = torch.where(torch.stack([torch.arange(s, s + w) for s in starts])[..., None] >= sp["tr0"], yb, torch.nan)
        mu, var, obs_nll, _ = m(zb, mb)
        loss = obs_nll + target_nll(mu[:, burn:], var[:, burn:], yb[:, burn:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 10 == 9 or step == cfg.steps - 1:
            vl = val_loss()
            if vl < best - 1e-5:
                best, best_state, since = vl, copy.deepcopy(m.state_dict()), 0
            else:
                since += 10
            if since >= cfg.patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, best


def ekf_predict(models: list[EKFSSM], d: Data) -> tuple[np.ndarray, np.ndarray]:
    """Run each frozen filter over the whole history; mixture of the seeds' predictions (target units)."""
    lo = d.sp["first"]
    mus, vars_ = [], []
    with torch.no_grad():
        for m in models:
            mu, var, _, _ = m(d.z[None, lo:], d.mask[None, lo:])
            mus.append(mu[0, :, 0].numpy())
            vars_.append(var[0, :, 0].numpy())
    mus, vars_ = np.array(mus), np.array(vars_)
    mu = mus.mean(0)
    var = vars_.mean(0) + mus.var(0)
    full_mu = np.full(len(d.y), np.nan)
    full_sd = np.full(len(d.y), np.nan)
    full_mu[lo:] = mu * d.y_sd + d.y_mu
    full_sd[lo:] = np.sqrt(var) * d.y_sd
    return full_mu, full_sd


def fit_ridge(d: Data, df: pd.DataFrame) -> tuple[np.ndarray, float, dict]:
    sp = d.sp
    Z = np.nan_to_num(d.scaler.transform(df[d.cols].to_numpy(float)))
    y = df[TARGET].to_numpy(float)
    fit = slice(sp["tr0"], sp["fit1"] + 1)
    va = slice(sp["va0"], sp["va1"] + 1)
    best = None
    for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        A = Z[fit]
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y[fit] - y[fit].mean()))
        err = np.mean((y[va] - (Z[va] @ w + y[fit].mean())) ** 2)
        if best is None or err < best[0]:
            best = (err, alpha)
    alpha = best[1]
    ty = slice(sp["tr0"], sp["va1"] + 1)
    A, b = Z[ty], y[ty].mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y[ty] - b))
    sd = float(np.std(y[ty] - (A @ w + b)))
    return Z @ w + b, sd, {"alpha": alpha, "weights": dict(zip(d.cols, np.round(w, 4).tolist()))}


def run(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    sp = split(df, cfg)
    y = df[TARGET].to_numpy(float)
    tr = slice(sp["tr0"], sp["va1"] + 1)      # targets that end inside the training years
    preds, info = {}, {"split": {k: df.index[v].strftime("%Y-%m-%d") for k, v in sp.items()}, "ekf": {}}
    for name, cols in (("ekf", FULL), ("ekf_price", PRICE_ONLY)):
        d = Data(df, cols, sp)
        models, vals = [], []
        for s in cfg.seeds:
            m, v = fit_ekf(d, cfg, s)
            models.append(m)
            vals.append(v)
            log.info("%s seed %d: val nll %.4f", name, s, v)
        preds[name] = ekf_predict(models, d)
        info["ekf"][name] = {"val_nll": vals, "periods": [float(2 * math.pi / th) for th in models[0].theta.detach().abs().clamp_min(1e-6).tolist()]}
    for name, cols in (("ridge", FULL), ("ridge_price", PRICE_ONLY)):
        d = Data(df, cols, sp)
        p, sd, meta = fit_ridge(d, df)
        preds[name] = (p, np.full(len(p), sd))
        info[name] = meta
    ytr = y[tr]
    preds["zero"] = (np.zeros(len(y)), np.full(len(y), float(np.sqrt(np.mean(ytr ** 2)))))
    preds["drift"] = (np.full(len(y), ytr.mean()), np.full(len(y), ytr.std()))
    r20 = df["r20"].to_numpy(float)
    A = np.c_[np.ones(len(ytr)), r20[tr]]
    coef = np.linalg.lstsq(A, ytr, rcond=None)[0]
    preds["momentum"] = (coef[0] + coef[1] * np.nan_to_num(r20), np.full(len(y), float(np.std(ytr - A @ coef))))
    info["momentum"] = {"coef": coef.tolist()}

    rows = []
    for i in range(sp["tr0"], sp["te1"] + 1):
        period = "train" if i <= sp["tr1"] else "test"
        for name, (mu, sd) in preds.items():
            m, s = float(mu[i]), float(sd[i])
            rows.append({"date": df.index[i].strftime("%Y-%m-%d"), "period": period, "model": name, "pred": m, "sd": s,
                         "p_up": _norm_cdf(m / s) if s > 0 else 0.5, "lo80": m - Z80 * s, "hi80": m + Z80 * s, "realized": float(y[i])})
    out = pd.DataFrame(rows)
    info["seconds"] = round(time.time() - t0, 1)
    return out, info


# ---------------------------------------------------------------- scoring

def metrics(d: pd.DataFrame) -> dict:
    d = d.dropna(subset=["realized", "pred"])
    n = len(d)
    if n == 0:
        return {"n": 0}
    p, y = d["pred"].to_numpy(float), d["realized"].to_numpy(float)
    err = y - p
    nz = (y != 0) & (p != 0)
    return {"n": n, "rmse": float(np.sqrt(np.mean(err ** 2))), "skill": float(1 - np.mean(err ** 2) / np.mean(y ** 2)),
            "hit": float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))) if nz.any() else None,
            "ic": float(pd.Series(p).rank().corr(pd.Series(y).rank())) if np.std(p) > 0 else None,
            "cover80": float(np.mean((y >= d["lo80"].to_numpy()) & (y <= d["hi80"].to_numpy()))),
            "brier": float(np.mean((d["p_up"].to_numpy() - (y > 0)) ** 2))}


def dm(d: pd.DataFrame, a: str, b: str) -> dict | None:
    """Is model a's squared error lower than model b's beyond noise? Negative t = a better; |t| > 2 is roughly significant."""
    pa = d[d["model"] == a].set_index("date")
    pb = d[d["model"] == b].set_index("date")
    j = pa.index.intersection(pb.index)
    diff = ((pa.loc[j, "realized"] - pa.loc[j, "pred"]) ** 2 - (pb.loc[j, "realized"] - pb.loc[j, "pred"]) ** 2).dropna()
    if len(diff) < 10 or diff.std() == 0:
        return None
    return {"n": len(diff), "mse_diff": float(diff.mean()), "t": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))}


def summarize(out: pd.DataFrame, horizon: int) -> dict:
    s = {}
    for period in ("train", "test"):
        p = out[out["period"] == period]
        days = sorted(p["date"].unique())[::horizon]
        no = p[p["date"].isin(days)]
        s[period] = {m: metrics(g) for m, g in no.groupby("model")}
        if period == "test":
            s["test_vs_zero"] = {m: dm(no, m, "zero") for m in no["model"].unique() if m != "zero"}
            s["sentiment_adds"] = {"ekf_vs_ekf_price": dm(no, "ekf", "ekf_price"), "ridge_vs_ridge_price": dm(no, "ridge", "ridge_price")}
            s["test_by_year"] = {}
            for yr, g in no.assign(year=no["date"].str[:4]).groupby("year"):
                s["test_by_year"][yr] = {m: round(metrics(gm)["skill"], 4) for m, gm in g.groupby("model")}
    return s
