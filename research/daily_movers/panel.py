"""Point-in-time S&P 500 daily OHLC panel for the daily big-mover study.

Reads github.com/Johnbrick123/sp500-data (release "data": prices.parquet, membership_intervals.parquet,
no_trade_days.csv) and writes wide date x ticker frames (all split+dividend adjusted) to one pickle.

    python panel.py SP500_DIR OUT.pkl
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

START = "2000-01-01"


def build(sp: Path) -> dict:
    iv = pd.read_parquet(sp / "membership_intervals.parquet")
    iv["start"], iv["end"] = pd.to_datetime(iv["start"]), pd.to_datetime(iv["end"])
    iv = iv[iv["end"].isna() | (iv["end"] > START)]
    etfs = ["SPY", "QQQ", "IWM"]
    tickers = sorted(set(iv["ticker"]) | set(etfs))
    px = pd.read_parquet(sp / "prices.parquet", columns=["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"],
                         filters=[("ticker", "in", tickers), ("date", ">=", pd.Timestamp(START))])
    px = px.drop_duplicates(["date", "ticker"], keep="last")
    k = px["adj_close"] / px["close"]  # adjustment factor applied to the day's open/high/low
    for c in ["open", "high", "low"]:
        px[c] = px[c] * k
    wide = {c: px.pivot(index="date", columns="ticker", values=s).sort_index()
            for c, s in [("close", "adj_close"), ("open", "open"), ("high", "high"), ("low", "low"), ("volume", "volume")]}
    dates, cols = wide["close"].index, wide["close"].columns
    member = pd.DataFrame(False, index=dates, columns=cols)
    for t, s, e in iv[["ticker", "start", "end"]].itertuples(index=False):
        if t in member.columns:
            member.loc[(dates >= s) & ((dates < e) if pd.notna(e) else True), t] = True
    nt = pd.read_csv(sp / "no_trade_days.csv", parse_dates=["date"])
    notrade = pd.DataFrame(False, index=dates, columns=cols)
    for d, t in nt[["date", "ticker"]].itertuples(index=False):
        if t in cols and d in dates:
            notrade.loc[d, t] = True
    etf = {t: wide["close"][t] for t in etfs if t in cols}
    etf_open = {t: wide["open"][t] for t in etfs if t in cols}
    for c in wide:
        wide[c] = wide[c].drop(columns=[t for t in etfs if t in cols and t not in set(iv["ticker"])])
    member = member[wide["close"].columns]
    notrade = notrade[wide["close"].columns]
    return {**wide, "member": member, "notrade": notrade, "etf": pd.DataFrame(etf), "etf_open": pd.DataFrame(etf_open)}


if __name__ == "__main__":
    out = build(Path(sys.argv[1]))
    pd.to_pickle(out, sys.argv[2])
    m = out["member"]
    print(f"{m.shape[0]} days x {m.shape[1]} tickers, members/day {m.sum(axis=1).median():.0f}, "
          f"{m.index[0].date()}..{m.index[-1].date()}")
