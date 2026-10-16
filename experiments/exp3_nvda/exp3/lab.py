"""Quant-ideas lab: every idea from quant_ideas.md that runs on the data already collected, scored the same way.

    python -m exp3.lab --data data --out lab

Targets (each over the 5 sessions after the close):
    ret    NVDA's log return (as in the main run)
    vol    log realised variance, log(mean r^2 + 0.01)
    radj   return / (20-session vol x sqrt 5), a z-score
    rel    NVDA minus QQQ
    resid  NVDA's return net of its time-varying beta to QQQ (beta from a Kalman filter, known at the close)
    tb     triple barrier: +1 / -1 if the cumulative return first crosses +-0.6 x weekly vol, else 0
Feature sets:
    P    price and market cap index (as in the main run)
    PV   P + volatility (log r^2 daily, weekly, monthly), a 2-state regime probability (Gaussian HMM) and the beta
    PVS  PV + sentiment levels by category (as in the main run)
    PVX  PV + sentiment surprise (week - long memory), disagreement (weekly dispersion), counts, and major outlets only
Models: the naive forecast (0, or last month's vol for vol), HAR for vol, ridge on each set (one split and a
yearly walk-forward refit), and the EKF + SSM predicting all six targets jointly, with and without a stochastic
volatility state (log variance as a state, measured by log r^2, scaling the return's noise).
Scores on non-overlapping test weeks (2022-2025): skill and DM t vs naive, rank IC, CRPS skill, QLIKE and
Mincer-Zarnowitz R^2 for vol, a weekly paper P&L with 10 bp costs (Sharpe, deflated for the number of strategies
tried), and a placebo (sentiment shifted in time by a year or more) for the sentiment sets.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch

from . import cli, collect, config, features, model, score
from .features import MARKET_COLS, PRICE_COLS, SENT_COLS, Scaler

log = logging.getLogger("exp3.lab")
N = NormalDist()
TARGETS = ["ret", "vol", "radj", "rel", "resid", "tb"]
RETURN_LIKE = ["ret", "radj", "rel", "resid", "tb"]
MAJOR = {"Reuters", "Bloomberg.com", "Bloomberg", "CNBC", "WSJ", "The Wall Street Journal", "Barron's", "MarketWatch", "Yahoo Finance",
         "Financial Times", "Business Insider", "Fortune", "Forbes"}
CATS = list(config.CATEGORIES) + ["major"]
VOL_COLS = ["lrv1", "lrv5", "lrv22", "p_hi", "beta"]
SURP_COLS = [f"{k}_{c}" for c in CATS for k in ("surp", "disp", "count")] + ["week_major", "long_major"]
SETS = {"P": PRICE_COLS + MARKET_COLS, "PV": PRICE_COLS + MARKET_COLS + VOL_COLS}
SETS["PVS"] = SETS["PV"] + SENT_COLS
SETS["PVX"] = SETS["PV"] + SURP_COLS
EKF_RUNS = {"ekf_PV_sv": ("PV", True), "ekf_PVS_sv": ("PVS", True), "ekf_PVX_sv": ("PVX", True), "ekf_PVX": ("PVX", False)}
COST = 0.10          # % per unit of position traded


# ---------------------------------------------------------------- extra measurements

def lrv(r: pd.Series, n: int) -> pd.Series:
    return np.log((r ** 2).rolling(n).mean() + 0.01)


def hmm2(x: np.ndarray, fit: np.ndarray, iters: int = 60) -> np.ndarray:
    """Two-state Gaussian HMM (calm / stressed) fitted by EM on the `fit` rows; returns the filtered
    probability of the high-variance state at each row (uses data up to that row only)."""
    y = x[fit]
    mu = np.array([y.mean(), y.mean()])
    sd = np.array([0.7 * y.std(), 1.6 * y.std()])
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    pi = np.array([0.8, 0.2])
    for _ in range(iters):
        B = np.exp(-0.5 * ((y[:, None] - mu) / sd) ** 2) / sd
        a = np.zeros((len(y), 2))
        c = np.zeros(len(y))
        a[0] = pi * B[0]
        c[0] = a[0].sum()
        a[0] /= c[0]
        for t in range(1, len(y)):
            a[t] = (a[t - 1] @ A) * B[t]
            c[t] = a[t].sum()
            a[t] /= c[t]
        b = np.ones((len(y), 2))
        for t in range(len(y) - 2, -1, -1):
            b[t] = A @ (B[t + 1] * b[t + 1]) / c[t + 1]
        g = a * b
        g /= g.sum(1, keepdims=True)
        xi = (a[:-1, :, None] * A[None] * (B[1:] * b[1:])[:, None, :]) / c[1:, None, None]
        A = xi.sum(0) / xi.sum((0, 2))[:, None]
        pi = g[0]
        mu = (g * y[:, None]).sum(0) / g.sum(0)
        sd = np.sqrt((g * (y[:, None] - mu) ** 2).sum(0) / g.sum(0)) + 1e-6
    hi = int(np.argmax(sd))
    p = np.full(len(x), np.nan)
    f = pi.copy()
    for t in range(len(x)):
        if np.isnan(x[t]):
            p[t] = f[hi]
            continue
        f = (f @ A if t else f) * np.exp(-0.5 * ((x[t] - mu) / sd) ** 2) / sd
        f /= f.sum()
        p[t] = f[hi]
    return p


def kalman_beta(r: np.ndarray, m: np.ndarray, q: float = 1e-4) -> np.ndarray:
    """Random-walk beta of r on m, filtered (beta_t uses data up to t)."""
    beta, P = 1.0, 1.0
    s2 = np.nanvar(r) * 0.5
    out = np.full(len(r), np.nan)
    for t in range(len(r)):
        P += q
        if not (np.isnan(r[t]) or np.isnan(m[t])):
            S = m[t] ** 2 * P + s2
            K = P * m[t] / S
            e = r[t] - beta * m[t]
            beta += K * e
            P -= K * m[t] * P
            s2 = 0.99 * s2 + 0.01 * e ** 2
        out[t] = beta
    return out


def dispersion(items: pd.DataFrame, sessions: pd.DatetimeIndex, week: int) -> pd.DataFrame:
    out = pd.DataFrame(index=sessions)
    for c in CATS:
        it = items[items["cat"] == c]
        w, wv, wv2 = (s.groupby(it["session"]).sum().reindex(sessions, fill_value=0.0).rolling(week, min_periods=1).sum()
                      for s in (it["w"], it["w"] * it["v"], it["w"] * it["v"] ** 2))
        m = wv / w
        out[f"disp_{c}"] = np.sqrt((wv2 / w - m ** 2).clip(lower=0)).where(w > 0)
    return out


def load(data: Path, cfg: config.Config) -> pd.DataFrame:
    news = collect.load_jsonl(data / "news.jsonl")
    reddit = collect.load_jsonl(data / "reddit.jsonl")
    to_score = score.items_to_score(news, reddit)
    scores = score.load_scores(data)
    prices = cli.load_prices(data)
    shares = pd.read_csv(data / "shares.csv")
    an = pd.read_csv(data / "analysts.csv") if (data / "analysts.csv").exists() else pd.DataFrame()
    sessions = prices[cfg.ticker].index
    news = news[news["id"].isin(to_score["id"])] if len(news) else news
    reddit = reddit[reddit["id"].isin(to_score["id"])] if len(reddit) else reddit
    items = features.category_items(news, reddit, an, scores, sessions, cfg.relevance_min)
    maj = features.category_items(news[news["source"].isin(MAJOR)], pd.DataFrame(), pd.DataFrame(), scores, sessions, cfg.relevance_min)
    maj = maj[maj["cat"] == "news"].assign(cat="major")
    items = pd.concat([items, maj], ignore_index=True)
    df = features.build(prices, shares, items, cfg)
    sf = features.sentiment_features(maj.assign(cat="news"), sessions, cfg.week, cfg.long_halflife)
    df["week_major"], df["long_major"], df["count_major"] = sf["week_news"], sf["long_news"], sf["count_news"]
    df = df.join(dispersion(items, sessions, cfg.week))
    for c in CATS:
        df[f"surp_{c}"] = df[f"week_{c}"] - df[f"long_{c}"]
    r, q = df["r1"], df["qqq_r1"]
    df["lrv1"] = np.log(r ** 2 + 0.01)
    df["lrv5"], df["lrv22"] = lrv(r, 5), lrv(r, 22)
    fit = np.asarray((df.index >= pd.Timestamp(cfg.train_start)) & (df.index <= pd.Timestamp(cfg.train_end)) & r.notna())
    df["p_hi"] = hmm2(r.to_numpy(float), fit)
    df["beta"] = kalman_beta(r.to_numpy(float), q.to_numpy(float))
    h = cfg.horizon
    fwd = lambda s: s.rolling(h).sum().shift(-h)  # noqa: E731
    df["t_ret"] = df["y"]
    df["t_vol"] = np.log((r ** 2).rolling(h).mean().shift(-h) + 0.01)
    df["t_radj"] = df["y"] / (df["rv20"] * math.sqrt(h))
    df["t_rel"] = df["y"] - fwd(q)
    df["t_resid"] = df["y"] - df["beta"] * fwd(q)
    b = 0.6 * df["rv20"] * math.sqrt(h)
    path = np.stack([r.shift(-k).to_numpy() for k in range(1, h + 1)], 1).cumsum(1)
    tb = np.zeros(len(df))
    for i in range(len(df)):
        bi = b.iloc[i]
        if np.isnan(path[i]).any() or np.isnan(bi):
            tb[i] = np.nan
            continue
        up, dn = np.argmax(path[i] >= bi) if (path[i] >= bi).any() else 99, np.argmax(path[i] <= -bi) if (path[i] <= -bi).any() else 99
        tb[i] = 0.0 if up == dn == 99 else (1.0 if up < dn else -1.0)
    df["t_tb"] = tb
    # traded payoff of each target's position (for the paper P&L)
    df["pay_ret"] = df["y"]
    df["pay_rel"] = df["t_rel"]
    df["pay_resid"] = df["t_resid"]
    return df


# ---------------------------------------------------------------- models

def fit_ridge(Z: np.ndarray, y: np.ndarray, fit: np.ndarray, va: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, float]:
    best = None
    for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        ok = fit & ~np.isnan(y)
        A = Z[ok]
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y[ok] - y[ok].mean()))
        v = va & ~np.isnan(y)
        err = np.mean((y[v] - (Z[v] @ w + y[ok].mean())) ** 2)
        if best is None or err < best[0]:
            best = (err, alpha)
    ok = full & ~np.isnan(y)
    A = Z[ok]
    w = np.linalg.solve(A.T @ A + best[1] * np.eye(A.shape[1]), A.T @ (y[ok] - y[ok].mean()))
    pred = Z @ w + y[ok].mean()
    return pred, float(np.std(y[ok] - pred[ok]))


def ridge_runs(df: pd.DataFrame, sp: dict, cfg: config.Config, cols: list[str], target: str, walk: bool) -> tuple[np.ndarray, np.ndarray]:
    """One split (fit on training years, alpha on validation) or yearly walk-forward: refit each January on every
    target that has ended by then, alpha chosen on the last 126 such rows."""
    n = len(df)
    idx = np.arange(n)
    y = df[f"t_{target}"].to_numpy(float)
    X = df[cols].to_numpy(float)
    sc = Scaler().fit(X[sp["tr0"]:sp["tr1"] + 1])
    Z = np.nan_to_num(sc.transform(X))
    mu, sd = np.full(n, np.nan), np.full(n, np.nan)
    if not walk:
        p, s = fit_ridge(Z, y, (idx >= sp["tr0"]) & (idx <= sp["fit1"]), (idx >= sp["va0"]) & (idx <= sp["va1"]), (idx >= sp["tr0"]) & (idx <= sp["va1"]))
        return p, np.full(n, s)
    years = sorted({d.year for d in df.index[sp["te0"]:sp["te1"] + 1]})
    for yr in years:
        a = int(df.index.searchsorted(pd.Timestamp(f"{yr}-01-01")))
        b = int(df.index.searchsorted(pd.Timestamp(f"{yr + 1}-01-01"))) - 1
        last = a - cfg.horizon - 1                       # last row whose target has ended before the refit
        p, s = fit_ridge(Z, y, (idx >= sp["tr0"]) & (idx <= last - 126 - cfg.horizon), (idx > last - 126) & (idx <= last), (idx >= sp["tr0"]) & (idx <= last))
        mu[a:b + 1], sd[a:b + 1] = p[a:b + 1], s
    p, s = fit_ridge(Z, y, (idx >= sp["tr0"]) & (idx <= sp["fit1"]), (idx >= sp["va0"]) & (idx <= sp["va1"]), (idx >= sp["tr0"]) & (idx <= sp["va1"]))
    mu[:sp["te0"]], sd[:sp["te0"]] = p[:sp["te0"]], s
    return mu, sd


class LabData(model.Data):
    """Measurements for the EKF and all six targets (each standardised on the training targets)."""

    def __init__(self, df: pd.DataFrame, cols: list[str], sp: dict, sv: bool):
        self.cols, self.sp = cols, sp
        X = df[cols].to_numpy(float)
        self.scaler = Scaler().fit(X[sp["tr0"]:sp["tr1"] + 1])
        if sv:      # log r^2 enters centred but unscaled, so the volatility state measures it with slope 1
            j = cols.index("lrv1")
            self.scaler.scale[j] = 1.0
        Z = self.scaler.transform(X)
        self.mask = torch.tensor(~np.isnan(Z), dtype=torch.float32)
        self.z = torch.tensor(np.nan_to_num(Z), dtype=torch.float32)
        Y = df[[f"t_{t}" for t in TARGETS]].to_numpy(float)
        ty = slice(sp["tr0"], sp["va1"] + 1)
        self.y_mu, self.y_sd = np.nanmean(Y[ty], 0), np.nanstd(Y[ty], 0)
        self.y = torch.tensor((Y - self.y_mu) / self.y_sd, dtype=torch.float32)


def ekf_runs(df: pd.DataFrame, sp: dict, cfg: config.Config, cols: list[str], sv: bool, seeds=(0, 1)) -> tuple[dict, dict]:
    d = LabData(df, cols, sp, sv)
    svi = (cols.index("r1"), cols.index("lrv1")) if sv else None
    mus, vars_, info = [], [], {"val_nll": []}
    for s in seeds:
        m, v = model.fit_ekf(d, cfg, s, "resonance", "exp", True, sv=svi)
        info["val_nll"].append(v)
        if sv:
            info.setdefault("phi", []).append(round(float(m.phi()), 4))
            info.setdefault("kappa", []).append(m.kappa.detach().round(decimals=3).tolist())
        lo = sp["first"]
        with torch.no_grad():
            mu, var, _, X = m(d.z[None, lo:], d.mask[None, lo:])
        mus.append(mu[0].numpy())
        vars_.append(var[0].numpy())
        if sv:
            h = np.full(len(df), np.nan)
            h[lo:] = X[0, :, -1].numpy()
            info.setdefault("h", []).append(h)
    mus, vars_ = np.array(mus), np.array(vars_)
    mu = mus.mean(0) * d.y_sd + d.y_mu
    sd = np.sqrt(vars_.mean(0) + mus.var(0)) * d.y_sd
    out = {}
    for k, t in enumerate(TARGETS):
        a, b = np.full(len(df), np.nan), np.full(len(df), np.nan)
        a[sp["first"]:], b[sp["first"]:] = mu[:, k], sd[:, k]
        out[t] = (a, b)
    if sv:
        info["h_mean"] = np.nanmean(np.array(info.pop("h")), 0)
    return out, info


# ---------------------------------------------------------------- scoring

def crps(mu, sd, y):
    z = (y - mu) / sd
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = np.array([N.cdf(v) for v in z])
    return sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def score_target(rows: pd.DataFrame, naive: str) -> dict:
    """rows: date, model, pred, sd, realized (non-overlapping test weeks)."""
    out = {}
    base = rows[rows["model"] == naive].set_index("date")
    for m, g in rows.groupby("model"):
        g = g.set_index("date").dropna(subset=["pred", "realized"])
        j = g.index.intersection(base.index)
        y, p = g.loc[j, "realized"].to_numpy(), g.loc[j, "pred"].to_numpy()
        pb = base.loc[j, "pred"].to_numpy()
        e, eb = (y - p) ** 2, (y - pb) ** 2
        d = e - eb
        c, cb = crps(p, g.loc[j, "sd"].to_numpy(), y), crps(pb, base.loc[j, "sd"].to_numpy(), y)
        r = {"n": len(j), "skill": float(1 - e.mean() / eb.mean()), "t": float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))) if d.std() > 0 else None,
             "ic": float(pd.Series(p).rank().corr(pd.Series(y).rank())) if np.std(p) > 0 else None, "crps_skill": float(1 - c.mean() / cb.mean())}
        out[m] = r
    return out


def vol_scores(rows: pd.DataFrame) -> dict:
    out = {}
    for m, g in rows.groupby("model"):
        g = g.dropna(subset=["pred", "realized"])
        y, p = g["realized"].to_numpy(), g["pred"].to_numpy()
        ratio = np.exp(y - p)
        A = np.c_[np.ones(len(p)), p]
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - float(np.sum((y - A @ coef) ** 2) / np.sum((y - y.mean()) ** 2))
        out[m] = {"qlike": float(np.mean(ratio - np.log(ratio) - 1)), "mz_r2": r2, "mz_slope": float(coef[1])}
    return out


def pnl(rows: pd.DataFrame, pay: pd.Series, scale: float) -> pd.Series:
    """Weekly paper P&L in %: position clip(pred / scale, -1, 1), held over the next 5 sessions, 10 bp per unit traded."""
    g = rows.set_index("date").sort_index()
    pos = np.clip(g["pred"] / scale, -1, 1)
    gross = pos * pay.reindex(pd.to_datetime(g.index)).to_numpy()
    cost = COST * np.abs(np.diff(pos.to_numpy(), prepend=0.0))
    return pd.Series(gross - cost, index=g.index)


def sharpe(p: pd.Series) -> float:
    p = p.dropna()
    return float(p.mean() / p.std() * math.sqrt(52)) if len(p) > 10 and p.std() > 0 else float("nan")


def deflated(sr_w: float, n: int, trials: list[float], skew: float, kurt: float) -> float:
    k = max(len(trials), 2)
    var = float(np.nanvar(trials, ddof=1))
    g = 0.5772156649
    sr0 = math.sqrt(var) * ((1 - g) * N.inv_cdf(1 - 1 / k) + g * N.inv_cdf(1 - 1 / (k * math.e)))
    den = math.sqrt(max(1e-12, 1 - skew * sr_w + (kurt - 1) / 4 * sr_w ** 2))
    return float(N.cdf((sr_w - sr0) * math.sqrt(n - 1) / den))


def placebo(df: pd.DataFrame, sp: dict, cfg: config.Config, cols: list[str], target: str, n: int = 200, seed: int = 0) -> dict:
    """Sentiment columns shifted (circularly, by 250+ sessions) so their timing is wrong; how often does the
    shifted model do as well in the test years as the real one?"""
    rng = np.random.default_rng(seed)
    sent = [c for c in cols if c not in SETS["PV"]]
    weeks = np.arange(sp["te0"], sp["te1"] + 1, cfg.horizon)
    y = df[f"t_{target}"].to_numpy(float)[weeks]

    def skill(frame):
        p, _ = ridge_runs(frame, sp, cfg, cols, target, walk=False)
        base = 0.0 if target != "vol" else frame["lrv22"].to_numpy(float)[weeks]
        ok = ~np.isnan(y)
        return float(1 - np.mean((y[ok] - p[weeks][ok]) ** 2) / np.mean((y[ok] - (base[ok] if target == "vol" else 0.0)) ** 2))

    real = skill(df)
    sh = []
    lo = sp["first"]
    for _ in range(n):
        k = int(rng.integers(250, len(df) - lo - 250))
        f = df.copy()
        f.iloc[lo:, [df.columns.get_loc(c) for c in sent]] = np.roll(df[sent].to_numpy()[lo:], k, axis=0)
        sh.append(skill(f))
    sh = np.array(sh)
    return {"real_skill": real, "placebo_mean": float(sh.mean()), "placebo_p95": float(np.percentile(sh, 95)), "p_value": float((np.sum(sh >= real) + 1) / (n + 1))}


# ---------------------------------------------------------------- run

def run(data: Path, out: Path, cfg: config.Config = config.DEFAULT, seeds=(0, 1), n_placebo: int = 200, ekf_runs_sel=tuple(EKF_RUNS)) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    df = load(data, cfg)
    sp = model.split(df, cfg)
    rep = {"split": {k: df.index[v].strftime("%Y-%m-%d") for k, v in sp.items()}, "ekf": {}}
    preds = {t: {} for t in TARGETS}
    n = len(df)
    ytr = slice(sp["tr0"], sp["va1"] + 1)
    for t in TARGETS:
        y = df[f"t_{t}"].to_numpy(float)
        if t == "vol":
            base = df["lrv22"].to_numpy(float)
            preds[t]["naive"] = (base, np.full(n, float(np.nanstd((y - base)[ytr]))))
            har = ["lrv1", "lrv5", "lrv22"]
            preds[t]["har"] = ridge_runs(df, sp, cfg, har, t, walk=False)
            preds[t]["har_wf"] = ridge_runs(df, sp, cfg, har, t, walk=True)
        else:
            preds[t]["naive"] = (np.zeros(n), np.full(n, float(np.sqrt(np.nanmean(y[ytr] ** 2)))))
        for name, cols in SETS.items():
            preds[t][f"ridge_{name}"] = ridge_runs(df, sp, cfg, cols, t, walk=False)
            preds[t][f"ridge_{name}_wf"] = ridge_runs(df, sp, cfg, cols, t, walk=True)
    log.info("ridge and HAR done (%.0fs)", time.time() - t0)
    for name in ekf_runs_sel:
        fs, sv = EKF_RUNS[name]
        o, info = ekf_runs(df, sp, cfg, SETS[fs], sv, seeds)
        for t in TARGETS:
            preds[t][name] = o[t]
        if sv:
            h = info.pop("h_mean")
            fv = df["t_vol"].to_numpy(float)
            te = np.arange(sp["te0"], sp["te1"] + 1, cfg.horizon)
            ok = ~np.isnan(h[te]) & ~np.isnan(fv[te])
            info["h_vs_next_week_vol_ic"] = float(pd.Series(h[te][ok]).rank().corr(pd.Series(fv[te][ok]).rank()))
            df[f"h_{name}"] = h
        rep["ekf"][name] = info
        log.info("%s done (%.0fs)", name, time.time() - t0)

    weeks = df.index[np.arange(sp["te0"], sp["te1"] + 1, cfg.horizon)]
    trweeks = df.index[np.arange(sp["tr0"], sp["va1"] + 1, cfg.horizon)]
    res, books = {}, {}
    rows_all = []
    for t in TARGETS:
        y = df[f"t_{t}"]
        rows = []
        for m, (mu, sd) in preds[t].items():
            s = pd.DataFrame({"date": df.index, "model": m, "pred": mu, "sd": sd, "realized": y.to_numpy()}).set_index("date")
            rows.append(s.loc[weeks].reset_index())
            rows_all.append(s.loc[df.index[sp["tr0"]:sp["te1"] + 1]].reset_index().assign(target=t))
        R = pd.concat(rows)
        res[t] = {"scores": score_target(R, "naive")}
        if t == "vol":
            res[t]["vol"] = vol_scores(R)
        if t in RETURN_LIKE:
            pay = df["pay_ret"] if t in ("ret", "radj", "tb") else df[f"pay_{t}"]
            for m, (mu, _) in preds[t].items():
                if m == "naive":
                    continue
                scale = float(np.nanstd(pd.Series(mu, index=df.index).loc[trweeks])) or 1.0
                books[f"{t}:{m}"] = pnl(R[R["model"] == m][["date", "pred"]], pay, scale)
    books["buy_and_hold_nvda"] = pd.Series(df["pay_ret"].reindex(weeks).to_numpy() - 0.0, index=weeks)
    trials = [sharpe(p) / math.sqrt(52) for p in books.values()]
    pl = {}
    for k, p in books.items():
        p = p.dropna()
        sr = sharpe(p)
        pl[k] = {"sharpe": sr, "ann_return_pct": float(p.mean() * 52), "deflated_prob": deflated(sr / math.sqrt(52), len(p), trials, float(p.skew()), float(p.kurt() + 3)),
                 "max_dd_pct": float((p.cumsum() - p.cumsum().cummax()).min())}
    rep["pnl"] = pl
    rep["n_strategies"] = len(books)
    rep["targets"] = res
    rep["placebo"] = {f"{t}:{c}": placebo(df, sp, cfg, SETS[c], t, n=n_placebo) for t in ("ret", "vol") for c in ("PVS", "PVX")}
    rep["seconds"] = round(time.time() - t0, 1)
    pd.concat(rows_all).to_csv(out / "predictions.csv", index=False, float_format="%.5g")
    pd.DataFrame(books).to_csv(out / "pnl_weekly.csv", float_format="%.5g")
    (out / "summary.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="exp3.lab")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("lab"))
    ap.add_argument("--placebo", type=int, default=200)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    rep = run(a.data, a.out, n_placebo=a.placebo)
    print(json.dumps({k: rep[k] for k in ("targets", "pnl", "placebo", "ekf", "seconds")}, indent=1, default=str))


if __name__ == "__main__":
    main()

