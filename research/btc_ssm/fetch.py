"""Collect the inputs for the Bitcoin state-space study and write them as plain CSVs into one folder.

- btc_1h.csv: Binance spot BTCUSDT hourly klines (public files on data.binance.vision, no key), 2018 to yesterday.
  Bar time is the bar's open (UTC); close is known one hour later.
- fng.csv: Crypto Fear & Greed index (alternative.me), one value per UTC day, published shortly after 00:00 UTC.
- dvol.csv: Deribit BTC implied-volatility index (DVOL), hourly, from 2021-03 (public API, no key).
The two side channels are optional: a failure is logged and the study runs price-only."""
from __future__ import annotations

import io
import sys
import time
import urllib.request
import zipfile

import pandas as pd
import requests

MONTH = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{m}.zip"
DAY = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{d}.zip"


def _get(url: str) -> bytes:
    for attempt in range(4):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def _klines(raw: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    return pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=range(9),
                       names=["t", "open", "high", "low", "close", "volume", "t_close", "quote_volume", "trades"])


def binance(start: str = "2018-01") -> pd.DataFrame:
    now = pd.Timestamp.now("UTC").tz_localize(None)
    last_month = now.to_period("M") - 1
    frames = [_klines(_get(MONTH.format(m=m))) for m in pd.period_range(start, last_month, freq="M").strftime("%Y-%m")]
    for d in pd.date_range(last_month.to_timestamp(how="end").normalize() + pd.Timedelta(days=1), now.normalize() - pd.Timedelta(days=1)):
        try:
            frames.append(_klines(_get(DAY.format(d=d.strftime("%Y-%m-%d")))))
        except Exception as e:  # the newest daily file can lag a day
            print("skip day", d.date(), e)
    d = pd.concat(frames)
    d["t"] = pd.to_datetime(d.t.where(d.t > 1e14, d.t * 1000), unit="us")   # ms before 2025, us after
    d = d.drop(columns="t_close").drop_duplicates("t").set_index("t").sort_index()
    return d.astype(float)


def fear_greed() -> pd.DataFrame:
    js = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=60).json()["data"]
    d = pd.DataFrame(js)
    d["date"] = pd.to_datetime(d.timestamp.astype(int), unit="s")
    return d[["date", "value"]].astype({"value": float}).set_index("date").sort_index()


def dvol() -> pd.DataFrame:
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    end = int(time.time() * 1000)
    start = int(pd.Timestamp("2021-03-01").timestamp() * 1000)
    rows = []
    while end > start:
        r = requests.get(url, params={"currency": "BTC", "start_timestamp": start, "end_timestamp": end,
                                      "resolution": "3600"}, timeout=60).json()["result"]
        if not r["data"]:
            break
        rows += r["data"]
        nxt = r.get("continuation")
        if not nxt or nxt >= end:
            break
        end = nxt
        time.sleep(0.2)
    d = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"])
    d["t"] = pd.to_datetime(d.t, unit="ms")
    return d.drop_duplicates("t").set_index("t").sort_index()


if __name__ == "__main__":
    out = sys.argv[1]
    b = binance()
    b.to_csv(f"{out}/btc_1h.csv")
    print("btc_1h", b.index[0], b.index[-1], len(b))
    for name, fn in (("fng", fear_greed), ("dvol", dvol)):
        try:
            x = fn()
            x.to_csv(f"{out}/{name}.csv")
            print(name, x.index[0], x.index[-1], len(x))
        except Exception as e:
            print(name, "failed:", e)
