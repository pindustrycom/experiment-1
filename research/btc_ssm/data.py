"""Hourly Bitcoin panel for the state-space study, built only from information available at each bar's close.

Row t is the Binance hour that opens at t and closes at t+1h. Everything in row t (return, features, side channels) is
known at t+1h, when the forecast for row t+1 is made. Targets are the next row's return, never a column of row t.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

H = 24   # hours per day


def load(folder: str) -> pd.DataFrame:
    b = pd.read_csv(f"{folder}/btc_1h.csv", index_col=0)
    b.index = pd.to_datetime(b.index, format="mixed")
    idx = pd.date_range(b.index[0], b.index[-1], freq="h")
    b = b.reindex(idx)
    b["gap"] = b.close.isna()                       # exchange outages: flat price, zero volume, excluded from scoring
    b["close"] = b.close.ffill()
    for k in ("open", "high", "low"):
        b[k] = b[k].fillna(b.close)
    b[["volume", "quote_volume", "trades"]] = b[["volume", "quote_volume", "trades"]].fillna(0.0)
    d = pd.DataFrame(index=idx)
    d["close"] = b.close
    d["open"], d["high"], d["low"] = b.open, b.high, b.low
    d["y"] = 100 * np.log(b.close)                  # log price in percent
    d["r"] = d.y.diff()                             # hourly log return, percent
    d["gap"] = b.gap | b.gap.shift(1, fill_value=False)
    d["lvol"] = np.log1p(b.quote_volume)
    d["fng"] = _fng(folder, idx)
    d["dvol"] = _dvol(folder, idx)
    return d.iloc[1:]


def _fng(folder: str, idx) -> pd.Series:
    f = f"{folder}/fng.csv"
    if not os.path.exists(f):
        return pd.Series(np.nan, index=idx)
    s = pd.read_csv(f, index_col=0, parse_dates=True)["value"]
    # The day-D value is published shortly after 00:00 UTC on D; use it from 02:00 UTC on D (hour bars opening >= 01:00
    # close >= 02:00) so publication and ingestion delays cannot leak the day's mood into earlier hours.
    s.index = s.index + pd.Timedelta(hours=1)
    return s.reindex(idx, method="ffill")


def _dvol(folder: str, idx) -> pd.Series:
    f = f"{folder}/dvol.csv"
    if not os.path.exists(f):
        return pd.Series(np.nan, index=idx)
    s = pd.read_csv(f, index_col=0, parse_dates=True)["close"]
    # Deribit stamps a candle with its start; its close is final at start+1h, the same moment as our bar's close.
    # Shift one more hour for safety (index refresh lag).
    s.index = s.index + pd.Timedelta(hours=1)
    return s.reindex(idx).ffill(limit=6)


def features(d: pd.DataFrame) -> pd.DataFrame:
    """Causal features of row t (all from rows <= t). Raw scale; standardize with train statistics only."""
    r = d.r.where(~d.gap, 0.0)
    f = pd.DataFrame(index=d.index)
    f["r"] = r
    f["absr"] = r.abs()
    rv24 = (r ** 2).rolling(H, min_periods=H // 2).sum()
    bv24 = (np.pi / 2) * (r.abs() * r.abs().shift()).rolling(H, min_periods=H // 2).sum()
    f["lrv24"] = np.log(rv24 + 1e-4)
    f["lrv168"] = np.log((r ** 2).rolling(7 * H, min_periods=H).sum() / 7 + 1e-4)
    f["jump"] = ((rv24 - bv24).clip(lower=0) / (rv24 + 1e-8)).fillna(0)  # share of 24h variance from jumps (BNS)
    f["r24"] = r.rolling(H).sum()                   # last day's return (momentum / reversal input)
    f["r168"] = r.rolling(7 * H).sum()
    lv = d.lvol.rolling(H).mean()
    f["vol_z"] = (lv - lv.rolling(30 * H, min_periods=7 * H).mean()) / (lv.rolling(30 * H, min_periods=7 * H).std() + 1e-6)
    hr = d.index.hour + 1                           # hour of the NEXT bar (known calendar), drives intraday seasonality
    f["hs"], f["hc"] = np.sin(2 * np.pi * hr / H), np.cos(2 * np.pi * hr / H)
    f["fng"] = d.fng
    f["dvol"] = np.log(d.dvol)
    f["dvol_gap"] = f.dvol - 0.5 * (f.lrv168 + np.log(365.0))   # log implied minus log realized annual vol (both %)
    return f


PRICE = ["r", "absr", "hs", "hc"]
RV = ["lrv24", "lrv168", "jump", "r24", "r168", "vol_z"]
SETS = {
    "price": PRICE,
    "price+rv": PRICE + RV,
    "price+rv+fng": PRICE + RV + ["fng"],
    "price+rv+dvol": PRICE + RV + ["dvol", "dvol_gap"],
    "all": PRICE + RV + ["fng", "dvol", "dvol_gap"],
}


def standardize(f: pd.DataFrame, train_end) -> pd.DataFrame:
    """Z-score with the mean/std of rows before train_end only (no look-ahead in normalization). Missing -> 0."""
    tr = f[f.index < train_end]
    z = (f - tr.mean()) / (tr.std() + 1e-8)
    return z.clip(-8, 8).fillna(0.0)
