"""Long-history inputs, all free and keyless.
Ken French data library: daily Fama-French 5 factors and momentum (US stocks, long/short, in %, since 1963).
Cboe strategy benchmark indexes: PUT (S&P 500 put-write) and BXM (buy-write), daily since 1986, from Cboe's CDN with
Yahoo as fallback."""
from __future__ import annotations

import io
import os
import sys
import time
import zipfile

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kakushadze151"))
from data import UA, yahoo

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"


def _get(url: str) -> bytes | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print(f"  {url}: {e}")
            time.sleep(2 ** attempt)
    return None


def parse_french(text: str) -> pd.DataFrame:
    """The daily file: a few note lines, a header row, YYYYMMDD rows in percent, then a copyright line."""
    lines = text.splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.strip() and ln.split(",")[0].strip().isdigit()) - 1
    cols = [c.strip() for c in lines[head].split(",")][1:]
    rows = []
    for ln in lines[head + 1:]:
        f = [x.strip() for x in ln.split(",")]
        if not f[0].isdigit() or len(f[0]) != 8:
            break
        rows.append([pd.Timestamp(f[0])] + [float(x) / 100 for x in f[1:]])
    d = pd.DataFrame(rows, columns=["date", *cols]).set_index("date")
    return d


def french(cache: str) -> pd.DataFrame | None:
    out = []
    for name in ("F-F_Research_Data_5_Factors_2x3_daily", "F-F_Momentum_Factor_daily"):
        raw = _cached(cache, name + ".zip", FRENCH.format(name))
        if raw is None:
            return None
        z = zipfile.ZipFile(io.BytesIO(raw))
        out.append(parse_french(z.read(z.namelist()[0]).decode("latin-1")))
    d = out[0].join(out[1], how="inner")
    d.columns = [c.replace("Mom", "UMD") for c in d.columns]
    return d


def cboe(sym: str, cache: str) -> pd.Series | None:
    raw = _cached(cache, f"{sym}.csv", CBOE.format(sym))
    if raw is not None:
        try:
            d = pd.read_csv(io.BytesIO(raw))
            d.columns = [c.strip().upper() for c in d.columns]
            col = sym if sym in d.columns else "CLOSE"
            s = pd.Series(pd.to_numeric(d[col], errors="coerce").to_numpy(), index=pd.to_datetime(d["DATE"]))
            return clean(s.dropna().sort_index(), sym)
        except (KeyError, ValueError, pd.errors.ParserError) as e:
            print(f"  cboe {sym}: {e}")
    y = yahoo(f"^{sym}", cache, start="1985-01-01")
    return None if y is None else y.close


def clean(s: pd.Series, name: str, limit=0.2) -> pd.Series:
    """Drop bad prints: an option-selling index never moves 20% in a day in this window, so a level that jumps
    20%+ and comes back is a data error."""
    s = s[s > 0]
    gaps = s.index.to_series().diff().dt.days
    if (gaps > 7).any():                     # early history is sparse (monthly points): keep the daily part only
        s = s[s.index >= gaps[gaps > 7].index[-1]]
        print(f"  {name}: daily data from {s.index[0].date()}")
    for _ in range(20):
        r = s.pct_change()
        bad = (r.abs() > limit) & (r.shift(-1).abs() > limit)     # a spike that reverses next day
        if not bad.any():
            break
        print(f"  {name}: dropping {int(bad.sum())} bad prints, e.g. {s[bad].head(3).to_dict()}")
        s = s[~bad]
    return s


def _cached(cache: str, name: str, url: str) -> bytes | None:
    os.makedirs(cache, exist_ok=True)
    f = f"{cache}/{name}"
    if os.path.exists(f) and time.time() - os.path.getmtime(f) < 20 * 3600:
        with open(f, "rb") as fh:
            return fh.read()
    raw = _get(url)
    if raw is not None:
        with open(f, "wb") as fh:
            fh.write(raw)
    return raw
