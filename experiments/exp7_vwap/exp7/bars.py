"""Tiingo IEX 1-minute bars, cached as one CSV per ticker and month so a rerun only fetches what is missing.

IEX is one exchange, so volumes are a slice of the consolidated tape; the session VWAP built from them is an
approximation of the one a broker chart shows. Regular session only (09:30-16:00 New York)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)
URL = "https://api.tiingo.com/iex/{t}/prices"


def months(start: str, end: str) -> list[pd.Period]:
    return list(pd.period_range(start, end, freq="M"))


def fetch_month(s: requests.Session, ticker: str, m: pd.Period, key: str, waits: int = 8) -> pd.DataFrame:
    params = {"startDate": m.start_time.strftime("%Y-%m-%d"), "endDate": m.end_time.strftime("%Y-%m-%d"), "resampleFreq": "1min",
              "columns": "open,high,low,close,volume", "token": key}
    for _ in range(waits):
        r = s.get(URL.format(t=ticker.lower()), params=params, timeout=120)
        if r.status_code == 429 or (r.status_code == 200 and "Error" in r.text[:200] and "limit" in r.text[:300].lower()):
            log.warning("Tiingo rate limit on %s %s; waiting 10 minutes", ticker, m)
            time.sleep(600)
            continue
        r.raise_for_status()
        d = pd.DataFrame(r.json())
        return d if d.empty else d[["date", "open", "high", "low", "close", "volume"]]
    raise RuntimeError(f"Tiingo kept rate-limiting {ticker} {m}")


def collect(data: Path, tickers: list[str], start: str, end: str, key: str, deadline: float, pause: float = 4.0) -> dict:
    """Fetch every missing ticker-month (the current month is always refetched). Stops cleanly at `deadline` (epoch s)."""
    s, got, missing = requests.Session(), 0, []
    this_month = pd.Timestamp.now(tz="America/New_York").to_period("M")
    for t in tickers:
        (data / t).mkdir(parents=True, exist_ok=True)
        for m in months(start, end):
            f = data / t / f"{m}.csv"
            if f.exists() and m < this_month:
                continue
            if time.time() > deadline:
                missing.append(f"{t} {m}")
                continue
            d = fetch_month(s, t, m, key)
            if len(d) >= 10000:
                log.warning("%s %s returned %d rows; the month may be cut short", t, m, len(d))
            d.to_csv(f, index=False)
            got += 1
            time.sleep(pause)
    return {"fetched": got, "missing": missing}


def load(data: Path, ticker: str) -> pd.DataFrame:
    """1-minute regular-session bars indexed by bar start (New York time), holidays with stale zero-volume bars dropped."""
    fs = sorted((data / ticker).glob("*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs if f.stat().st_size > 10], ignore_index=True)
    d["ts"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert("America/New_York")
    d = d.drop(columns="date").drop_duplicates("ts").set_index("ts").sort_index().between_time("09:30", "15:59")
    day = d.index.strftime("%Y-%m-%d")
    dead = d["volume"].eq(0).groupby(day).transform("mean") > 0.5
    return d[~dead.to_numpy()]
