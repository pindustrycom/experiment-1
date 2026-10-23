"""Inputs for the 151 Trading Strategies backtests. All public, no keys.
Yahoo chart API: daily OHLC scaled by adjclose/close so dividends count (stocks, ETFs, BTC, FX spot, VIX indexes).
FRED: OECD 3-month interbank rates and the US 3-month T-bill (monthly, %), for the FX carry strategies."""
from __future__ import annotations

import io
import os
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str, headers=UA, timeout=30, tries=4, **kw) -> requests.Response:
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, **kw)
            r.raise_for_status()
            return r
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def yahoo(ticker: str, cache: str, start: str = "2013-01-01") -> pd.DataFrame | None:
    """Daily total-return OHLC on finished sessions, or None when Yahoo has nothing."""
    os.makedirs(cache, exist_ok=True)
    f = f"{cache}/{ticker.replace('^', '_').replace('=', '_')}.csv"
    if os.path.exists(f) and time.time() - os.path.getmtime(f) < 20 * 3600:
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        return d if len(d) else None
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    try:
        r = _get(f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
                 params={"period1": t0, "period2": int(time.time()) + 86400, "interval": "1d", "includeAdjustedClose": "true"})
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as e:  # delisted / unknown ticker
        print(f"  yahoo {ticker}: {e}")
        return None
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    d = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")}, index=idx, dtype=float)
    adj = res["indicators"].get("adjclose")
    if adj:
        fac = (pd.Series(adj[0]["adjclose"], index=idx, dtype=float) / d.close).ffill()
        for k in ("open", "high", "low", "close"):
            d[k] = d[k] * fac
    d = d[~d.index.duplicated(keep="last")].dropna(subset=["close"])
    today = pd.Timestamp.now("America/New_York").normalize().tz_localize(None)
    d = d[d.index < today]
    d.to_csv(f)
    return d if len(d) else None


def fred(series: str) -> pd.Series | None:
    """FRED sometimes hangs for cloud runners: try both CSV endpoints, with and without a browser user agent."""
    urls = [(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", UA),
            (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}", {}),
            (f"https://fred.stlouisfed.org/series/{series}/downloaddata/{series}.csv", {})]
    for url, hd in urls:
        try:
            r = _get(url, headers=hd, timeout=30, tries=2)
            d = pd.read_csv(io.StringIO(r.text))
            s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
            s.index = pd.to_datetime(d.iloc[:, 0])
            s = s.dropna()
            if len(s):
                return s
        except (requests.RequestException, pd.errors.ParserError, ValueError, IndexError) as e:   # timeouts, HTML pages
            print(f"  fred {series} {url[:60]}: {str(e)[:120]}")
    return None
