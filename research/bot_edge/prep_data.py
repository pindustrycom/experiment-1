"""Bars in freqtrade's feather layout (<BASE>_USDT-<tf>.feather, plus futures/ copies with zero funding).

  python prep_data.py releases RELEASE_DIR OUT   # project data: btc-ssm-data (BTC 1h), kronos-results (ETH 1h),
                                                 # exp7-data (SPY/QQQ Tiingo 1-min, regular hours)
  python prep_data.py binance OUT SYM,SYM START  # CI: Binance spot 5m monthly files from data.binance.vision

Stocks are stored as BASE/USDT pairs only so freqtrade can read them; prices are SPY/QQQ in dollars, not dividend
adjusted (the benchmark uses the same bars). Funding for the futures copies is zero (see README).
"""
import glob
import io
import os
import sys
import time
import urllib.request
import zipfile

import pandas as pd

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def save(df, out, base, tf):
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    f = df.copy()
    f.index = idx
    f = f.rename_axis("date").reset_index()
    f.to_feather(f"{out}/{base}_USDT-{tf}.feather")
    os.makedirs(f"{out}/futures", exist_ok=True)
    f.to_feather(f"{out}/futures/{base}_USDT_USDT-{tf}-futures.feather")
    if tf in ("1h", "4h", "1d"):
        f.to_feather(f"{out}/futures/{base}_USDT_USDT-{tf}-mark.feather")
        z = f.copy()
        z[["open", "high", "low", "close", "volume"]] = 0.0
        z.to_feather(f"{out}/futures/{base}_USDT_USDT-{tf}-funding_rate.feather")
    if tf == "1h":
        e = f.set_index("date").resample("8h").agg(AGG).dropna().reset_index()
        e.to_feather(f"{out}/futures/{base}_USDT_USDT-8h-mark.feather")
        e[["open", "high", "low", "close", "volume"]] = 0.0
        e.to_feather(f"{out}/futures/{base}_USDT_USDT-8h-funding_rate.feather")


def crypto_frames(d, out, base, tfs):
    d = d[~d.index.duplicated()].sort_index()
    for tf in tfs:
        rule = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "12h": "12h", "1d": "1D"}[tf]
        save(d.resample(rule, label="left", closed="left").agg(AGG).dropna(), out, base, tf)


def stock_frames(m, out, base):
    """m: 1-minute bars indexed in America/New_York, regular session only."""
    naive = m.copy()
    naive.index = naive.index.tz_localize(None)
    for tf, mins in (("1m", 1), ("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60), ("4h", 240)):
        g = naive if mins == 1 else naive.groupby(pd.Grouper(freq=f"{mins}min", origin="start_day", offset="30min")).agg(AGG).dropna()
        g = g.copy()
        g.index = g.index.tz_localize("America/New_York")
        save(g, out, base, tf)
    g = naive.groupby(naive.index.normalize()).agg(AGG)
    g.index = (g.index + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York")
    save(g, out, base, "1d")
    save(g, out, base, "12h")


def from_releases(rel, out):
    os.makedirs(out, exist_ok=True)
    for base, path in (("BTC", f"{rel}/btc-ssm-data/btc_1h.csv"), ("ETH", f"{rel}/kronos-results/ETHUSDT_1h.csv")):
        d = pd.read_csv(path)
        d["t"] = pd.to_datetime(d.t, format="mixed")
        crypto_frames(d.set_index("t"), out, base, ("1h", "4h", "12h", "1d"))
    for s in ("SPY", "QQQ"):
        m = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f"{rel}/exp7-data/{s}/*.csv")))
        m["ts"] = pd.to_datetime(m["date"], utc=True)
        m = m.drop_duplicates("ts").set_index("ts").sort_index().tz_convert("America/New_York").between_time("09:30", "15:59")
        stock_frames(m, out, s)


def binance(out, syms, start):
    os.makedirs(out, exist_ok=True)
    url = "https://data.binance.vision/data/spot/monthly/klines/{s}/5m/{s}-5m-{m}.zip"
    last = pd.Timestamp.now("UTC").tz_localize(None).to_period("M") - 1
    for sym in syms:
        frames = []
        for m in pd.period_range(start, last, freq="M").strftime("%Y-%m"):
            for attempt in range(4):
                try:
                    raw = urllib.request.urlopen(url.format(s=sym, m=m), timeout=120).read()
                    break
                except urllib.error.HTTPError as e:
                    raw = None
                    if e.code == 404:
                        break
                    time.sleep(2 ** attempt)
                except Exception:
                    raw = None
                    time.sleep(2 ** attempt)
            if raw is None:
                continue
            z = zipfile.ZipFile(io.BytesIO(raw))
            frames.append(pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=range(6),
                                      names=["t", "open", "high", "low", "close", "volume"]))
        d = pd.concat(frames)
        d["t"] = pd.to_datetime(d.t.where(d.t > 1e14, d.t * 1000), unit="us")   # ms before 2025, us after
        base = sym.replace("USDT", "")
        crypto_frames(d.set_index("t"), out, base, ("5m", "15m", "30m", "1h", "4h", "1d"))
        print(sym, len(d), d.t.min(), d.t.max(), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "releases":
        from_releases(sys.argv[2], sys.argv[3])
    else:
        binance(sys.argv[2], sys.argv[3].split(","), sys.argv[4])
