"""The state-vector inputs and prediction targets, one row per trading session.

Every feature on row t uses only data available at session t's close. The targets on row t are
the log returns from close t to close t+HORIZON, so they are unknown (NaN) for the last HORIZON
rows. tests/test_features.py checks that truncating the data never changes earlier rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

TARGETS = ["y_sector", "y_relative"]


def build(prices: dict[str, pd.DataFrame], news_feats: pd.DataFrame | None, cfg: Config) -> pd.DataFrame:
    tickers = [cfg.target, *cfg.constituents, *cfg.related]
    missing = [t for t in tickers if t not in prices]
    if missing:
        raise ValueError(f"no price history for {', '.join(missing)}")
    px = pd.DataFrame({t: prices[t]["adj_close"] for t in tickers}).sort_index()
    vol = prices[cfg.target]["adj_volume"].reindex(px.index)
    # Sessions are the target ETF's trading days from the start date; other series are aligned to them.
    px = px[px.index >= pd.Timestamp(cfg.data_start)]
    px = px[px[cfg.target].notna()].ffill(limit=3)
    lp = np.log(px)
    r = lp.diff() * 100  # daily log returns, %

    f = pd.DataFrame(index=px.index)
    for t in tickers:
        f[f"r_{t}"] = r[t]
    sec, mkt = lp[cfg.target], lp[cfg.benchmark]
    for n in (5, 20, 60):
        f[f"mom{n}"] = (sec - sec.shift(n)) * 100
    for n in (5, 20):
        f[f"rel{n}"] = ((sec - sec.shift(n)) - (mkt - mkt.shift(n))) * 100
    f["rv20"] = r[cfg.target].rolling(20).std()
    f["rv_ratio"] = f["rv20"] / r[cfg.target].rolling(120).std()
    f["drawdown"] = (sec - sec.rolling(252, min_periods=60).max()) * 100
    f["dist_ma200"] = (sec - sec.rolling(200, min_periods=60).mean()) * 100
    cons = r[list(cfg.constituents)]
    f["dispersion"] = cons.std(axis=1)
    f["breadth50"] = (px[list(cfg.constituents)] > px[list(cfg.constituents)].rolling(50).mean()).mean(axis=1)
    f["breadth_up"] = (cons > 0).mean(axis=1)
    f["corr_mkt60"] = r[cfg.target].rolling(60).corr(r[cfg.benchmark])
    lv = np.log(vol.replace(0, np.nan))
    f["volume_z"] = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
    for t in ("TLT", "UUP", "SMH"):
        if t in px:
            f[f"mom20_{t}"] = (lp[t] - lp[t].shift(20)) * 100
    f["dow"] = px.index.dayofweek / 4.0

    if news_feats is not None:
        f = f.join(news_feats.reindex(f.index).fillna(0.0))

    h = cfg.horizon
    f["y_sector"] = (sec.shift(-h) - sec) * 100
    f["y_relative"] = f["y_sector"] - (mkt.shift(-h) - mkt) * 100
    f["close"] = px[cfg.target]
    # Drop the warm-up period where rolling features aren't defined yet.
    first = f.drop(columns=[*TARGETS, "close"]).notna().all(axis=1).idxmax()
    return f.loc[first:]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (*TARGETS, "close")]


class Scaler:
    """Robust standardisation fitted on the training rows only; values are clipped at 5 sd."""

    def fit(self, x: np.ndarray) -> Scaler:
        self.med = np.nanmedian(x, axis=0)
        iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
        self.scale = np.where(iqr > 1e-9, iqr / 1.349, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.nan_to_num((x - self.med) / self.scale), -5, 5).astype(np.float32)

    def state(self) -> dict:
        return {"med": self.med.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> Scaler:
        o = cls()
        o.med, o.scale = np.array(s["med"]), np.array(s["scale"])
        return o
