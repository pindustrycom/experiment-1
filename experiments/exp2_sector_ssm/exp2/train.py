"""Training (with pruning), the comparison baselines, and prediction."""
from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from .config import Config
from .features import TARGETS, Scaler, feature_columns
from .model import SectorSSM, gaussian_nll

log = logging.getLogger("exp2.train")
Z80 = 1.2815515655446004
BASELINES = ["zero", "drift", "momentum", "ridge"]


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class Fitted:
    """Everything needed to predict: the ensemble, the scalers and the baselines."""

    def __init__(self, cfg: Config, cols, scaler, y_mu, y_sd, models, baselines, info):
        self.cfg, self.cols, self.scaler, self.y_mu, self.y_sd = cfg, cols, scaler, np.asarray(y_mu), np.asarray(y_sd)
        self.models, self.baselines, self.info = models, baselines, info

    def to_state(self) -> dict:
        return {"cols": self.cols, "scaler": self.scaler.state(), "y_mu": self.y_mu.tolist(), "y_sd": self.y_sd.tolist(),
                "models": [m.state_dict() for m in self.models], "n_features": len(self.cols),
                "baselines": self.baselines, "info": self.info}

    @classmethod
    def from_state(cls, s: dict, cfg: Config) -> Fitted:
        models = []
        for sd in s["models"]:
            m = SectorSSM(s["n_features"], len(TARGETS), cfg.model)
            m.load_state_dict(sd)
            m.eval()
            models.append(m)
        return cls(cfg, s["cols"], Scaler.from_state(s["scaler"]), s["y_mu"], s["y_sd"], models, s["baselines"], s["info"])

    # ------------------------------------------------------------ prediction

    @torch.no_grad()
    def ssm_outputs(self, df: pd.DataFrame, end: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble mean and sd (in target units) for rows end-length+1..end, each using the
        same context length the model was trained with."""
        w = self.cfg.model.window
        lo = max(0, end - length + 1 - (w - self.cfg.model.burn_in))
        x = torch.from_numpy(self.scaler.transform(df[self.cols].to_numpy(float)[lo:end + 1]))[None]
        mus, vars_ = [], []
        for m in self.models:
            mu, lv = m(x)
            mus.append(mu[0, -length:].numpy())
            vars_.append(lv[0, -length:].exp().numpy())
        mus, vars_ = np.stack(mus), np.stack(vars_)
        mu = mus.mean(0) * self.y_sd + self.y_mu
        sd = np.sqrt(vars_.mean(0) + mus.var(0)) * self.y_sd
        return mu, sd

    def predict_row(self, df: pd.DataFrame, i: int) -> list[dict]:
        out = []
        mu, sd = self.ssm_outputs(df, i, 1)
        x = df[self.cols].iloc[i].to_numpy(float)
        for k, tgt in enumerate(TARGETS):
            preds = {"ssm": (float(mu[0, k]), float(sd[0, k]))}
            for name, p in self.baselines[tgt].items():
                preds[name] = (baseline_predict(name, p, x, self.cols), p["sd"])
            for model, (m, s) in preds.items():
                out.append({"model": model, "target": tgt, "pred": m, "sd": s, "p_up": _norm_cdf(m / s) if s > 0 else 0.5,
                            "lo80": m - Z80 * s, "hi80": m + Z80 * s})
        return out


# ---------------------------------------------------------------- baselines

def fit_baselines(df: pd.DataFrame, cols: list[str], tr: np.ndarray, va: np.ndarray) -> dict:
    out = {}
    X = df[cols].to_numpy(float)
    xs = Scaler().fit(X[tr])
    for tgt in TARGETS:
        y = df[tgt].to_numpy(float)
        ytr = y[tr]
        b = {"zero": {"sd": float(np.sqrt(np.mean(ytr ** 2)))},
             "drift": {"mean": float(ytr.mean()), "sd": float(ytr.std())}}
        m = df["mom20"].to_numpy(float)
        A = np.c_[np.ones(tr.sum()), m[tr]]
        coef = np.linalg.lstsq(A, ytr, rcond=None)[0]
        b["momentum"] = {"coef": coef.tolist(), "sd": float((ytr - A @ coef).std())}
        Xt, Xv = xs.transform(X[tr]), xs.transform(X[va])
        best = None
        for alpha in (10.0, 100.0, 1000.0, 10000.0):
            w = np.linalg.solve(Xt.T @ Xt + alpha * np.eye(Xt.shape[1]), Xt.T @ (ytr - ytr.mean()))
            err = np.mean((y[va] - (Xv @ w + ytr.mean())) ** 2)
            if best is None or err < best[0]:
                best = (err, alpha, w)
        _, alpha, w = best
        resid = ytr - (Xt @ w + ytr.mean())
        b["ridge"] = {"alpha": alpha, "w": w.tolist(), "b": float(ytr.mean()), "scaler": xs.state(), "sd": float(resid.std())}
        out[tgt] = b
    return out


def baseline_predict(name: str, p: dict, x: np.ndarray, cols: list[str]) -> float:
    if name == "zero":
        return 0.0
    if name == "drift":
        return p["mean"]
    if name == "momentum":
        return p["coef"][0] + p["coef"][1] * x[cols.index("mom20")]
    if name == "ridge":
        xs = Scaler.from_state(p["scaler"]).transform(x[None])[0]
        return float(xs @ np.asarray(p["w"]) + p["b"])
    raise KeyError(name)


# ---------------------------------------------------------------- SSM training

def _batches(n_rows: int, window: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    hi = max(1, n_rows - window + 1)
    return rng.integers(0, hi, size=batch)


def _val_loss(model, X, Y, v0, v1, window, burn):
    model.eval()
    with torch.no_grad():
        lo = max(0, v0 - (window - burn))
        mu, lv = model(X[None, lo:v1 + 1])
        k = v1 - v0 + 1
        loss = gaussian_nll(mu[0, -k:], lv[0, -k:], Y[v0:v1 + 1]).item()
    model.train()
    return loss


def _train(model, X, Y, tr_end, v0, v1, mc, steps, seed, log_every=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.param_groups(mc))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    w = min(mc.window, tr_end + 1)
    best, best_state, since = math.inf, copy.deepcopy(model.state_dict()), 0
    model.train()
    for step in range(steps):
        starts = _batches(tr_end + 1, w, mc.batch, rng)
        xb = torch.stack([X[s:s + w] for s in starts])
        yb = torch.stack([Y[s:s + w] for s in starts])
        burn = min(mc.burn_in, w // 2)
        mu, lv = model(xb)
        loss = gaussian_nll(mu[:, burn:], lv[:, burn:], yb[:, burn:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 10 == 9 or step == steps - 1:
            vl = _val_loss(model, X, Y, v0, v1, mc.window, mc.burn_in)
            if vl < best - 1e-5:
                best, best_state, since = vl, copy.deepcopy(model.state_dict()), 0
            else:
                since += 10
            if since >= mc.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return best


def mean_compare(items: list[dict]) -> dict:
    """Average {criterion: {keep: val_nll}} over seeds or backtest blocks."""
    out: dict = {}
    for it in items:
        for crit, levels in (it or {}).items():
            for keep, v in levels.items():
                out.setdefault(crit, {}).setdefault(keep, []).append(v)
    return {c: {k: float(np.mean(v)) for k, v in lv.items()} for c, lv in out.items()}


def fit(df: pd.DataFrame, as_of: int, cfg: Config, seeds=None, steps: int | None = None, prune: bool = True) -> Fitted:
    """Train on rows whose targets were already known at row `as_of` (i.e. rows <= as_of - horizon).
    The last `val_sessions` of those are held out for early stopping and the pruning check, with a
    `horizon`-row gap so no training target overlaps a validation one."""
    t0 = time.time()
    mc, h = cfg.model, cfg.horizon
    cols = feature_columns(df)
    end = as_of - h                       # last row with a known target
    v1, v0 = end, end - mc.val_sessions + 1
    tr_end = v0 - h - 1
    if tr_end < mc.window:
        raise ValueError(f"not enough history to train ({tr_end + 1} rows)")
    tr = np.zeros(len(df), bool)
    tr[: tr_end + 1] = True
    va = np.zeros(len(df), bool)
    va[v0: v1 + 1] = True

    scaler = Scaler().fit(df[cols].to_numpy(float)[tr])
    Yraw = df[TARGETS].to_numpy(float)
    y_mu, y_sd = Yraw[tr].mean(0), Yraw[tr].std(0)
    X = torch.from_numpy(scaler.transform(df[cols].to_numpy(float)))
    Y = torch.from_numpy(((Yraw - y_mu) / y_sd).astype(np.float32))
    Y[as_of - h + 1:] = float("nan")       # never train on targets unknown at as_of (they're NaN anyway in live use)

    models, per_seed = [], []
    for seed in (seeds if seeds is not None else mc.seeds):
        torch.manual_seed(seed)
        m = SectorSSM(len(cols), len(TARGETS), mc)
        full = _train(m, X, Y, tr_end, v0, v1, mc, steps or mc.steps, seed)
        chosen, chosen_loss, keep_used = m, full, 1.0
        trials, compare = [], {}
        if prune:
            # Input spectra of each layer over the training rows, for the resonance index.
            spectra = m.input_spectra(X[None, : tr_end + 1], seg=mc.spectrum_seg)
            # Compare pruning criteria the way the resonance paper does: prune without retraining and
            # score the validation loss at each level. Costs one forward pass per (criterion, level).
            for crit in mc.compare_criteria:
                for keep in mc.prune_keep:
                    pm = copy.deepcopy(m)
                    if crit == "random":
                        g = torch.Generator().manual_seed(seed)
                        for layer in pm.layers:
                            sc = torch.rand(layer.mask.shape, generator=g)
                            layer.mask.copy_((sc >= sc.flatten().topk(max(1, round(keep * sc.numel()))).values.min()).float())
                    else:
                        pm.prune(keep, crit, spectra)
                    compare.setdefault(crit, {})[str(keep)] = _val_loss(pm, X, Y, v0, v1, mc.window, mc.burn_in)
            # The experiment is a pruned model: prune with the chosen criterion, fine-tune, keep the level
            # that validates best, and record the unpruned network's score next to it.
            best = None
            for keep in mc.prune_keep:
                pm = copy.deepcopy(m)
                pm.prune(keep, mc.prune_criterion, spectra)
                pl = _train(pm, X, Y, tr_end, v0, v1, mc, mc.finetune_steps, seed + 1000)
                trials.append({"keep": keep, "val_nll": pl})
                if best is None or pl < best[0] - mc.prune_tolerance:
                    best = (pl, pm, keep)
            chosen_loss, chosen, keep_used = best
        models.append(chosen)
        active, total = chosen.modes()
        per_seed.append({"seed": seed, "val_nll_full": full, "val_nll": chosen_loss, "keep": keep_used,
                         "modes_active": active, "modes_total": total, "prune_trials": trials,
                         "criterion": mc.prune_criterion, "prune_compare": compare})

    baselines = fit_baselines(df, cols, tr, va)
    info = {"prune_compare": mean_compare([p["prune_compare"] for p in per_seed]),
            "val_nll_full": float(np.mean([p["val_nll_full"] for p in per_seed])), "as_of": df.index[as_of].strftime("%Y-%m-%d"), "train_start": df.index[0].strftime("%Y-%m-%d"),
            "train_end": df.index[tr_end].strftime("%Y-%m-%d"), "val_start": df.index[v0].strftime("%Y-%m-%d"),
            "val_end": df.index[v1].strftime("%Y-%m-%d"), "n_train": int(tr.sum()), "n_val": int(va.sum()),
            "n_features": len(cols), "seeds": per_seed, "config": asdict(cfg.model), "config_digest": cfg.digest(),
            "seconds": round(time.time() - t0, 1)}
    fitted = Fitted(cfg, cols, scaler, y_mu, y_sd, models, baselines, info)
    # Validation scores of every model, on the same rows, in target units.
    mu, _ = fitted.ssm_outputs(df, v1, v1 - v0 + 1)
    yv = Yraw[v0:v1 + 1]
    X_np = df[cols].to_numpy(float)
    val = {}
    for k, tgt in enumerate(TARGETS):
        val[tgt] = {"ssm": float(np.mean((yv[:, k] - mu[:, k]) ** 2))}
        for name, p in baselines[tgt].items():
            pr = np.array([baseline_predict(name, p, X_np[i], cols) for i in range(v0, v1 + 1)])
            val[tgt][name] = float(np.mean((yv[:, k] - pr) ** 2))
    info["val_mse"] = val
    log.info("trained through %s in %.0fs; modes %s", info["train_end"], info["seconds"],
             [(s["modes_active"], s["modes_total"]) for s in per_seed])
    return fitted
