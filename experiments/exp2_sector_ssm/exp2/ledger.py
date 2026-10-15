"""The forward-test ledger: predictions are appended once, before their outcome exists, and
scored when it does. A row is never rewritten except to fill in its outcome."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["made_at", "as_of", "target_date", "horizon", "model_id", "model", "target", "pred", "sd", "p_up", "lo80", "hi80",
        "close_as_of", "realized", "scored_at"]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(path, dtype={"as_of": str, "target_date": str, "made_at": str, "scored_at": str, "model_id": str})


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(path, index=False, float_format="%.6g")


def append(ledger: pd.DataFrame, as_of: str, rows: list[dict], model_id: str, horizon: int, close: float) -> tuple[pd.DataFrame, bool]:
    """Add predictions for `as_of` unless that session already has them (reruns don't overwrite)."""
    if (ledger["as_of"] == as_of).any():
        return ledger, False
    made = datetime.now(UTC).isoformat(timespec="seconds")
    new = pd.DataFrame([{**r, "made_at": made, "as_of": as_of, "target_date": "", "horizon": horizon, "model_id": model_id,
                         "close_as_of": close, "realized": np.nan, "scored_at": ""} for r in rows], columns=COLS)
    return (new if ledger.empty else pd.concat([ledger, new], ignore_index=True)), True


def score(ledger: pd.DataFrame, feats: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, int]:
    """Fill in the realised outcome of every prediction whose horizon has passed."""
    if ledger.empty:
        return ledger, 0
    dates = [d.strftime("%Y-%m-%d") for d in feats.index]
    pos = {d: i for i, d in enumerate(dates)}
    n = 0
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    led = ledger.copy()
    for idx in led.index[led["realized"].isna()]:
        i = pos.get(led.at[idx, "as_of"])
        if i is None or i + horizon >= len(dates):
            continue
        y = feats[led.at[idx, "target"]].iloc[i]
        if pd.isna(y):
            continue
        led.at[idx, "realized"] = float(y)
        led.at[idx, "target_date"] = dates[i + horizon]
        led.at[idx, "scored_at"] = stamp
        n += 1
    return led, n


def metrics(df: pd.DataFrame) -> dict:
    """Accuracy of one model on one target over the scored rows in `df`."""
    d = df.dropna(subset=["realized"])
    n = len(d)
    if n == 0:
        return {"n": 0}
    p, y = d["pred"].to_numpy(float), d["realized"].to_numpy(float)
    err = y - p
    nz = (y != 0) & (p != 0)
    out = {"n": n, "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
           "skill": float(1 - np.mean(err ** 2) / np.mean(y ** 2)) if np.mean(y ** 2) > 0 else None,
           "hit": float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))) if nz.any() else None,
           "ic": float(pd.Series(p).rank().corr(pd.Series(y).rank())) if n >= 5 and np.std(p) > 0 else None}
    if "lo80" in d and d["lo80"].notna().all():
        out["cover80"] = float(np.mean((y >= d["lo80"].to_numpy(float)) & (y <= d["hi80"].to_numpy(float))))
    if "p_up" in d:
        pu = d["p_up"].to_numpy(float)
        out["brier"] = float(np.mean((pu - (y > 0)) ** 2))
    return out


def scorecard(df: pd.DataFrame) -> dict:
    """metrics() for every (target, model) in the ledger."""
    out = {}
    for (tgt, model), g in df.groupby(["target", "model"]):
        out.setdefault(tgt, {})[model] = metrics(g)
    return out
