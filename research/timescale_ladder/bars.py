"""Build the five time scales (minute, hour, day, week, month) from SPY/QQQ 1-minute bars.

Input: the exp7-data release (one CSV per ticker and month, UTC timestamps, Tiingo IEX bars 2019-2026).
Regular hours only (09:30-16:00 New York). For every period of every scale: log return r and realized variance rv
(sum of squared 1-minute log returns inside it; day and longer also include the overnight gap), plus the id of its
parent period one scale up, so each scale can hand forecasts down and errors up.

    python bars.py EXP7_DIR TICKER OUT.pkl
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["month", "week", "day", "hour", "minute"]


def load(d: Path, ticker: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sorted((d / ticker).glob("*.csv"))])
    t = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York")
    df = df.assign(ts=t.dt.tz_localize(None)).drop(columns="date").drop_duplicates("ts").sort_values("ts")
    mins = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[(mins >= 570) & (mins < 960)].copy()  # 09:30 <= bar start < 16:00
    df["day"] = df["ts"].dt.normalize()
    n = df.groupby("day")["close"].transform("size")
    u = df.groupby("day")["close"].transform("nunique")
    return df[(n >= 300) & (u >= 50)].reset_index(drop=True)  # drop half days, and holidays the feed fills with flat bars


def build(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lc = np.log(df["close"].to_numpy())
    lo = np.log(df["open"].to_numpy())
    first = (df["day"] != df["day"].shift()).to_numpy()
    r = np.where(first, lc - lo, lc - np.roll(lc, 1))           # intraday 1-minute log return
    gap = np.where(first, lo - np.roll(lc, 1), 0.0)              # overnight gap, booked on the day's first minute
    gap[0] = 0.0
    m = pd.DataFrame({"ts": df["ts"], "day": df["day"], "r": r, "gap": gap})
    m["minute_of_day"] = (m["ts"].dt.hour * 60 + m["ts"].dt.minute - 570).astype(int)
    m["hour"] = m["day"] + pd.to_timedelta(m["minute_of_day"] // 60, unit="h")   # 7 bins, the last is 30 min
    iso = m["day"].dt.isocalendar()
    m["week"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    m["month"] = m["day"].dt.year * 100 + m["day"].dt.month
    out = {}
    mi = m.assign(rv=m["r"] ** 2, slot=m["minute_of_day"])
    out["minute"] = mi[["ts", "r", "rv", "slot", "hour"]].rename(columns={"ts": "key", "hour": "parent"}).reset_index(drop=True)
    for lvl, parent in [("hour", "day"), ("day", "week"), ("week", "month"), ("month", None)]:
        g = m.groupby(lvl, sort=True)
        inc_gap = lvl != "hour"
        f = pd.DataFrame({"r": g["r"].sum() + (g["gap"].sum() if inc_gap else 0),
                          "rv": g["r"].apply(lambda x: (x ** 2).sum()) + (g["gap"].apply(lambda x: (x ** 2).sum()) if inc_gap else 0)})
        if parent:
            f["parent"] = g[parent].first()
        f["slot"] = g["minute_of_day"].first() // 60 if lvl == "hour" else 0
        out[lvl] = f.reset_index().rename(columns={lvl: "key"})
    out["month"]["parent"] = 0
    return out


if __name__ == "__main__":
    B = build(load(Path(sys.argv[1]), sys.argv[2]))
    pd.to_pickle(B, sys.argv[3])
    for k in LEVELS:
        f = B[k]
        print(f"{k:>6}: {len(f):>7} periods, mean r {1e4 * f['r'].mean():7.3f} bp, vol {100 * f['r'].std():.3f}%")
