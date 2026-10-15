"""Walk-forward backtest: retrain every `every` sessions on the data known at that point, predict
the following block, repeat. Labelled as a backtest everywhere it is shown: it is a check that the
method works on past data, not a forward test."""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .config import Config
from .features import TARGETS
from .ledger import scorecard
from .train import Z80, _norm_cdf, baseline_predict, fit

log = logging.getLogger("exp2.backtest")


def run(df: pd.DataFrame, cfg: Config, start: str = "2016-01-04", every: int = 126, steps: int = 400,
        max_seconds: float = 1800, prune: bool = True) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    rows = []
    idx = df.index
    i = int(np.searchsorted(idx, pd.Timestamp(start)))
    last = len(df) - 1 - cfg.horizon       # last row whose outcome is known
    blocks, complete = 0, True
    while i <= last:
        if time.time() - t0 > max_seconds:
            complete = False
            break
        f = fit(df, i, cfg, seeds=(0,), steps=steps, prune=prune)
        j1 = min(i + every - 1, last)
        mu, sd = f.ssm_outputs(df, j1, j1 - i + 1)
        X = df[f.cols].to_numpy(float)
        for r, j in enumerate(range(i, j1 + 1)):
            for k, tgt in enumerate(TARGETS):
                y = float(df[tgt].iloc[j])
                preds = {"ssm": (float(mu[r, k]), float(sd[r, k]))}
                for name, p in f.baselines[tgt].items():
                    preds[name] = (baseline_predict(name, p, X[j], f.cols), p["sd"])
                for model, (m, s) in preds.items():
                    rows.append({"as_of": idx[j].strftime("%Y-%m-%d"), "model": model, "target": tgt, "pred": m, "sd": s,
                                 "p_up": _norm_cdf(m / s) if s > 0 else 0.5, "lo80": m - Z80 * s, "hi80": m + Z80 * s,
                                 "realized": y, "trained_through": f.info["train_end"],
                                 "modes": f.info["seeds"][0]["modes_active"]})
        blocks += 1
        log.info("backtest block %d: %s..%s", blocks, idx[i].date(), idx[j1].date())
        i = j1 + 1
    bt = pd.DataFrame(rows)
    summary = {"start": start, "every": every, "blocks": blocks, "complete": complete, "seconds": round(time.time() - t0),
               "first": bt["as_of"].min() if len(bt) else None, "last": bt["as_of"].max() if len(bt) else None,
               "scorecard": scorecard(bt) if len(bt) else {}}
    # Non-overlapping view: every horizon-th session, so outcomes don't share days.
    if len(bt):
        days = sorted(bt["as_of"].unique())[:: cfg.horizon]
        summary["scorecard_nonoverlap"] = scorecard(bt[bt["as_of"].isin(days)])
    return bt, summary
