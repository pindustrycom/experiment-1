"""Hourly Binance spot klines (public files on data.binance.vision, no key) for the Kronos test.

Writes <out>/<SYMBOL>_1h.csv with columns t (bar open, UTC), open, high, low, close, volume, amount (quote volume).
A bar stamped t closes at t + 1 hour."""
from __future__ import annotations

import io
import sys
import time
import urllib.request
import zipfile

import pandas as pd

MONTH = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{m}.zip"
DAY = "https://data.binance.vision/data/spot/daily/klines/{s}/1h/{s}-1h-{d}.zip"


def _get(url: str) -> bytes:
    for attempt in range(4):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(url)


def _klines(raw: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    return pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=range(8),
                       names=["t", "open", "high", "low", "close", "volume", "t_close", "amount"])


def klines(symbol: str, start: str = "2019-01") -> pd.DataFrame:
    now = pd.Timestamp.now("UTC").tz_localize(None)
    last_month = now.to_period("M") - 1
    frames = [_klines(_get(MONTH.format(s=symbol, m=m)))
              for m in pd.period_range(start, last_month, freq="M").strftime("%Y-%m")]
    for d in pd.date_range(last_month.to_timestamp(how="end").normalize() + pd.Timedelta(days=1),
                           now.normalize() - pd.Timedelta(days=1)):
        try:
            frames.append(_klines(_get(DAY.format(s=symbol, d=d.strftime("%Y-%m-%d")))))
        except Exception as e:  # noqa: BLE001  (the newest daily file can lag a day)
            print("skip day", d.date(), e)
    d = pd.concat(frames)
    d["t"] = pd.to_datetime(d.t.where(d.t > 1e14, d.t * 1000), unit="us")   # ms before 2025, us after
    return d.drop(columns="t_close").drop_duplicates("t").set_index("t").sort_index().astype(float)


if __name__ == "__main__":
    out = sys.argv[1]
    for sym in sys.argv[2:] or ["BTCUSDT", "ETHUSDT"]:
        d = klines(sym)
        d.to_csv(f"{out}/{sym}_1h.csv")
        print(sym, len(d), d.index[0], d.index[-1])
