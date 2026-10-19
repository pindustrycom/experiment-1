"""VIX and T-bill history through OpenBB (the official `openbb` packages from PyPI, github.com/OpenBB-finance/OpenBB).

Yahoo's chart endpoint is the only source Experiment 2 has for ^VIX and ^IRX: Tiingo carries no indices. Both feed the
live portfolio (VIX sets the variance-swap strike and the vol features, ^IRX the T-bill rate), so when Yahoo fails they
used to go stale without notice. OpenBB reaches two free, keyless official sources instead:
    ^VIX   Cboe's own daily VIX history                       (provider "cboe")
    ^IRX   Federal Reserve H.15 3-month Treasury rate         (provider "federal_reserve")
H.15's 3-month rate is close to Yahoo's ^IRX (the 13-week bill) but not identical, and both sources can run a session
behind (Cboe leaves out the latest day, H.15 is published the next morning); for ^IRX the gap is a few hundredths of a
percent, which hardly moves a daily T-bill accrual.

Frames come back in papers.py's layout: date, open, close, adj_close, volume. openbb is imported only when called.
"""
from __future__ import annotations

import pandas as pd

SOURCES = {"^VIX": "Cboe via OpenBB", "^IRX": "Federal Reserve H.15 via OpenBB"}
# How far a fallback may sit from the stored series on their common dates before it is refused (median gap):
# VIX in relative terms, the T-bill rate in percentage points.
TOLERANCE = {"^VIX": ("rel", 0.01), "^IRX": ("abs", 0.25)}


def supported(ticker: str) -> bool:
    return ticker in SOURCES


def fetch(ticker: str, start: str) -> pd.DataFrame:
    from openbb import obb

    if ticker == "^VIX":
        df = obb.index.price.historical("VIX", provider="cboe", start_date=start, interval="1d").to_df()
        df = df.reset_index()[["date", "open", "close"]]
    elif ticker == "^IRX":
        df = obb.fixedincome.government.treasury_rates(provider="federal_reserve", start_date=start).to_df()
        df = df.reset_index()[["date", "month_3"]].rename(columns={"month_3": "close"})
        df["close"] = df["close"] * 100        # OpenBB gives rates as decimals; ^IRX is in percent
        df["open"] = None
    else:
        raise ValueError(f"OpenBB has no source set up for {ticker}")
    return frame(df)


def frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["adj_close"] = df["close"]
    df["volume"] = None
    df = df.dropna(subset=["close"]).drop_duplicates("date", keep="last").sort_values("date")
    return df[["date", "open", "close", "adj_close", "volume"]].reset_index(drop=True)


def gap(ticker: str, old: pd.DataFrame, new: pd.DataFrame) -> float | None:
    """Median gap between two series on their common dates, in the ticker's TOLERANCE units (None: no overlap)."""
    m = old[["date", "close"]].merge(new[["date", "close"]], on="date", suffixes=("_o", "_n")).dropna()
    if m.empty:
        return None
    kind, _ = TOLERANCE[ticker]
    d = (m["close_n"] - m["close_o"]).abs()
    return float((d / m["close_o"].abs() if kind == "rel" else d).median())


def agrees(ticker: str, old: pd.DataFrame | None, new: pd.DataFrame) -> bool:
    if old is None or old.empty:
        return True
    g = gap(ticker, old, new)
    return g is not None and g <= TOLERANCE[ticker][1]
