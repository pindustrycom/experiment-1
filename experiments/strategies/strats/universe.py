"""Point-in-time S&P 500 membership and prices of the companies that left the index, from the public dataset
github.com/Johnbrick123/sp500-data (membership 1970 to today with [start, end) spans, every member priced on every day it
was a member; removed and delisted companies recovered from Tiingo, CRSP and archives; a dead company's label carries a
-YYYYMM suffix so a reused ticker never points at the wrong firm).

Replays and backtests use members_at(d), the index as it was on d, instead of today's list: today's list alone leaves out
every company that fell out of the index, which flatters any rule that buys losers. Rebuilt equal weight from this data, the
index tracks RSP with a 0.994 daily correlation (2004-2026); today's members alone would have earned about 2.5%/yr more.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)
URL = "https://github.com/Johnbrick123/sp500-data/releases/download/data/"
MAX_AGE_DAYS = 7


def _download(name: str, f: Path, max_age_days: float = MAX_AGE_DAYS) -> bool:
    if f.exists() and time.time() - f.stat().st_mtime < max_age_days * 86400:
        return False
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".part")
    with requests.get(URL + name, stream=True, timeout=120, headers={"User-Agent": "experiment-1 research"}) as r:
        r.raise_for_status()
        with tmp.open("wb") as out:
            for chunk in r.iter_content(1 << 20):
                out.write(chunk)
    if tmp.read_bytes()[:4] != b"PAR1":
        tmp.unlink()
        raise ValueError(f"{name} is not a Parquet file")
    tmp.replace(f)
    return True


def intervals(data: Path, fetch: bool = True) -> pd.DataFrame:
    """ticker, start, end (end empty for today's members); a failed download keeps the cached copy."""
    f = data / "pit" / "membership_intervals.parquet"
    if fetch:
        try:
            _download("membership_intervals.parquet", f)
        except Exception as ex:  # noqa: BLE001 - keep last week's copy
            log.warning("membership intervals: %s", ex)
    if not f.exists():
        return pd.DataFrame(columns=["ticker", "start", "end"])
    iv = pd.read_parquet(f)
    iv["start"], iv["end"] = pd.to_datetime(iv["start"]), pd.to_datetime(iv["end"])
    return iv


class Members:
    """Callable d -> the S&P 500 members on d (tickers in Yahoo spelling, BRK-B). Without the dataset, today's list."""

    def __init__(self, iv: pd.DataFrame, today: list[str] | None = None):
        self.iv = iv
        self.today = sorted(today) if today else sorted(iv.loc[iv["end"].isna(), "ticker"])
        self._cache: dict[pd.Timestamp, list[str]] = {}

    def __call__(self, d: pd.Timestamp) -> list[str]:
        d = pd.Timestamp(d)
        if not len(self.iv):
            return self.today
        if d not in self._cache:
            on = (self.iv["start"] <= d) & (self.iv["end"].isna() | (self.iv["end"] > d))
            self._cache[d] = sorted(self.iv.loc[on, "ticker"])
        return self._cache[d]

    def ever(self, start: str) -> list[str]:
        on = self.iv["end"].isna() | (self.iv["end"] > pd.Timestamp(start))
        return sorted(set(self.iv.loc[on, "ticker"]) | set(self.today))


def former_prices(data: Path, iv: pd.DataFrame, start: str, fetch: bool = True) -> dict:
    """Write data/prices/<label>.csv (adj_close, adj_open, adj_low, adj_high, the format of data.update_prices) for every
    company that was a member after `start` but is not one today, once a week. The 235 MB price file is deleted afterwards
    so the state the workflow carries between runs stays small."""
    cur = set(iv.loc[iv["end"].isna(), "ticker"])
    gone = sorted(set(iv.loc[iv["end"].notna() & (iv["end"] > pd.Timestamp(start)), "ticker"]) - cur)
    d = data / "prices"
    d.mkdir(parents=True, exist_ok=True)
    stamp, pq = data / "pit" / "prices_written.txt", data / "pit" / "prices.parquet"
    missing = [t for t in gone if not (d / f"{t}.csv").exists()]
    out = {"former_members": len(gone), "written": 0}
    fresh = stamp.exists() and time.time() - stamp.stat().st_mtime < MAX_AGE_DAYS * 86400
    if fetch and (missing or not fresh):
        try:
            _download("prices.parquet", pq, max_age_days=0)
        except Exception as ex:  # noqa: BLE001 - replays fall back to the names that have prices
            out["warning"] = f"prices: {type(ex).__name__}: {str(ex)[:200]}"
    if pq.exists() and (missing or not fresh or not fetch):
        p = pd.read_parquet(pq, columns=["date", "ticker", "open", "high", "low", "close", "adj_close"],
                            filters=[("ticker", "in", gone), ("date", ">=", pd.Timestamp(start))])
        for t, g in p.groupby("ticker"):
            g = g.sort_values("date").drop_duplicates("date", keep="last")
            f = g["adj_close"] / g["close"]
            pd.DataFrame({"adj_close": g["adj_close"].to_numpy(), **{f"adj_{k}": (g[k] * f).to_numpy() for k in ("open", "low", "high")}},
                         index=pd.DatetimeIndex(g["date"], name="date")).to_csv(d / f"{t}.csv")
            out["written"] += 1
        if fetch:
            pq.unlink()
            stamp.write_text(pd.Timestamp.now().isoformat() + "\n")
    out["missing"] = len([t for t in gone if not (d / f"{t}.csv").exists()])
    return out
