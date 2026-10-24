"""Bars for the backtests.
crypto: Binance spot klines, public monthly files on data.binance.vision (no key), cached as CSV.
stocks daily: Yahoo chart API, OHLC scaled by adjclose/close so dividends count (total return).
stocks intraday: SPY/QQQ Tiingo IEX 1-minute bars from the exp7-data release, regular hours only."""
from __future__ import annotations

import glob
import io
import os
import time
import urllib.request
import zipfile

import pandas as pd
import requests

KLINES = "https://data.binance.vision/data/spot/monthly/klines/{s}/{tf}/{s}-{tf}-{m}.zip"


def binance(cache: str, sym: str, tf: str, start: str) -> pd.DataFrame:
    os.makedirs(cache, exist_ok=True)
    last = pd.Timestamp.now("UTC").tz_localize(None).to_period("M") - 1
    frames = []
    for m in pd.period_range(start, last, freq="M").strftime("%Y-%m"):
        f = f"{cache}/{sym}-{tf}-{m}.csv"
        if not os.path.exists(f):
            for attempt in range(4):
                try:
                    raw = urllib.request.urlopen(KLINES.format(s=sym, tf=tf, m=m), timeout=120).read()
                    break
                except Exception:  # retry network errors
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            zipfile.ZipFile(io.BytesIO(raw)).extract(f"{sym}-{tf}-{m}.csv", cache)
        frames.append(pd.read_csv(f, header=None, usecols=range(6), names=["t", "open", "high", "low", "close", "volume"]))
    d = pd.concat(frames)
    d["t"] = pd.to_datetime(d.t.where(d.t > 1e14, d.t * 1000), unit="us")   # ms before 2025, us after
    return d.drop_duplicates("t").set_index("t").sort_index().astype(float)


def yahoo_ohlc(ticker: str, start: str = "2017-01-01") -> pd.DataFrame:
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    for attempt in range(4):
        try:
            r = requests.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
                             params={"period1": t0, "period2": int(time.time()) + 86400, "interval": "1d",
                                     "includeAdjustedClose": "true"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            r.raise_for_status()
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    d = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")}, index=idx, dtype=float)
    adj = pd.Series(res["indicators"]["adjclose"][0]["adjclose"], index=idx, dtype=float)
    f = (adj / d.close).ffill()
    for k in ("open", "high", "low", "close"):
        d[k] = d[k] * f
    d = d[~d.index.duplicated(keep="last")].dropna(subset=["open", "close"])
    today = pd.Timestamp.now("America/New_York").normalize().tz_localize(None)
    return d[d.index < today]          # only finished sessions


def minute_bars(folder: str, minutes: int) -> pd.DataFrame:
    d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f"{folder}/*.csv")))
    d["ts"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert("America/New_York")
    d = d.drop_duplicates("ts").set_index("ts").sort_index().between_time("09:30", "15:59")
    d = d.groupby(pd.Grouper(freq=f"{minutes}min", origin="start_day", offset="30min")).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    day = d.index.normalize()
    dead = d.volume.eq(0).groupby(day).transform("mean") > 0.5
    d = d[~dead]
    d.index = d.index.tz_localize(None)
    return d
