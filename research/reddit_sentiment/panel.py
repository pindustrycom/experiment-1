"""Point-in-time S&P 500 daily panel, 2010-2026, from github.com/Johnbrick123/sp500-data (release "data").

    python panel.py SP500_DIR OUT.pkl
Wide date x ticker frames: close (adjusted), open (adjusted), volume, member (True while in the index), plus SPY/QQQ.
Tickers that changed symbol keep the dataset's own ids (e.g. FOX-201903); collectors map them back by stripping the suffix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

START = "2010-01-01"


def build(sp: Path) -> dict:
    iv = pd.read_parquet(sp / "membership_intervals.parquet")
    iv["start"], iv["end"] = pd.to_datetime(iv["start"]), pd.to_datetime(iv["end"])
    iv = iv[iv["end"].isna() | (iv["end"] > START)]
    etfs = ["SPY", "QQQ"]
    tickers = sorted(set(iv["ticker"]) | set(etfs))
    px = pd.read_parquet(sp / "prices.parquet", columns=["date", "ticker", "open", "close", "volume", "adj_close"],
                         filters=[("ticker", "in", tickers), ("date", ">=", pd.Timestamp(START))])
    px = px.drop_duplicates(["date", "ticker"], keep="last")
    px["open"] = px["open"] * px["adj_close"] / px["close"]
    px["dvol"] = px["close"] * px["volume"]
    W = {c: px.pivot(index="date", columns="ticker", values=s).sort_index()
         for c, s in [("close", "adj_close"), ("open", "open"), ("dvol", "dvol")]}
    dates = W["close"].index
    member = pd.DataFrame(False, index=dates, columns=W["close"].columns)
    for t, s, e in iv[["ticker", "start", "end"]].itertuples(index=False):
        if t in member.columns:
            member.loc[(dates >= s) & ((dates < e) if pd.notna(e) else True), t] = True
    etf = W["close"][etfs].copy()
    etf_open = W["open"][etfs].copy()
    keep = [c for c in member.columns if c in set(iv["ticker"])]
    return {**{k: v[keep] for k, v in W.items()}, "member": member[keep], "etf": etf, "etf_open": etf_open}


if __name__ == "__main__":
    out = build(Path(sys.argv[1]))
    pd.to_pickle(out, sys.argv[2])
    print(out["close"].shape, out["member"].sum(axis=1).median())
