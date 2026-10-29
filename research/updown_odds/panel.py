"""Point-in-time S&P 500 daily panel for the up/down odds study.

Reads the public dataset github.com/Johnbrick123/sp500-data (membership_intervals.parquet, prices.parquet) and the
project's papers-data cache (sp500_sectors.csv, SPY, ^VIX) and writes wide date x ticker frames to one pickle:
adj (total-return price), open, close, volume, dividends, member (bool, in the index that day), plus sector, SPY and VIX series.

    python panel.py SP500_DIR PAPERS_DIR OUT.pkl
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

START = "1999-01-01"


def build(sp: Path, papers: Path) -> dict:
    iv = pd.read_parquet(sp / "membership_intervals.parquet")
    iv["start"], iv["end"] = pd.to_datetime(iv["start"]), pd.to_datetime(iv["end"])
    iv = iv[iv["end"].isna() | (iv["end"] > START)]
    tickers = sorted(iv["ticker"].unique())
    px = pd.read_parquet(sp / "prices.parquet", columns=["date", "ticker", "open", "close", "volume", "adj_close", "dividends"],
                         filters=[("ticker", "in", tickers), ("date", ">=", pd.Timestamp(START))])
    px = px.drop_duplicates(["date", "ticker"], keep="last")
    wide = {k: px.pivot(index="date", columns="ticker", values=v).sort_index()
            for k, v in [("adj", "adj_close"), ("open", "open"), ("close", "close"), ("volume", "volume"), ("dividends", "dividends")]}
    dates, cols = wide["adj"].index, wide["adj"].columns
    member = pd.DataFrame(False, index=dates, columns=cols)
    for t, s, e in iv[["ticker", "start", "end"]].itertuples(index=False):
        if t in member.columns:
            member.loc[(dates >= s) & ((dates < e) if pd.notna(e) else True), t] = True
    sec = pd.read_csv(papers / "sp500_sectors.csv").set_index("ticker")["sector"]
    spy = pd.read_csv(papers / "prices" / "SPY.csv", parse_dates=["date"]).set_index("date")
    vix = pd.read_csv(papers / "prices" / "^VIX.csv", parse_dates=["date"]).set_index("date")["close"]
    return {**wide, "member": member, "sector": sec.reindex(cols), "spy": spy, "vix": vix}


if __name__ == "__main__":
    out = build(Path(sys.argv[1]), Path(sys.argv[2]))
    pd.to_pickle(out, sys.argv[3])
    m = out["member"]
    print(f"{m.shape[0]} days x {m.shape[1]} tickers, members/day {m.sum(axis=1).median():.0f}, "
          f"{out['adj'].index[0].date()}..{out['adj'].index[-1].date()}")
