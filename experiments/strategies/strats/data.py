"""Data for the strategy pots: daily prices (Yahoo), the S&P 500 member list (Wikipedia), earnings results and insider
filings (Finnhub, free tier). Everything is cached under the data folder, so a source that fails keeps yesterday's copy."""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)
START = "2017-01-01"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FINNHUB = "https://finnhub.io/api/v1"
UA = {"User-Agent": "experiment-1 research (https://github.com/pindustrycom/experiment-1)"}


def parse_yahoo_chart(payload: dict) -> pd.Series:
    """Adjusted closes by New York date from a Yahoo chart response."""
    res = payload["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    dates = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    s = pd.Series(adj, index=dates, dtype=float)
    close = pd.Series(q["close"], index=dates, dtype=float)
    s = s.fillna(close * (s / close).ffill())          # the latest adjusted close is often missing until the next day
    return s[~s.index.duplicated(keep="last")].dropna()


def fetch_yahoo(ticker: str, start: str = START) -> pd.Series:
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    r = requests.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker.replace('.', '-')}",
                     params={"period1": t0, "period2": int(time.time()) + 86400, "interval": "1d", "includeAdjustedClose": "true"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    return parse_yahoo_chart(r.json())


def update_prices(data: Path, tickers: list[str], pause: float = 0.15) -> dict:
    """Refetch every ticker's history; a failed ticker keeps its cached file."""
    d = data / "prices"
    d.mkdir(parents=True, exist_ok=True)
    out = {"fetched": 0, "failed": []}
    for t in tickers:
        try:
            s = fetch_yahoo(t)
            if len(s) > 30:
                s.rename("adj_close").to_csv(d / f"{t}.csv", index_label="date")
                out["fetched"] += 1
            else:
                out["failed"].append(t)
        except Exception:  # noqa: BLE001 - one bad ticker must not stop the run
            out["failed"].append(t)
        time.sleep(pause)
    return out


def load_prices(data: Path, tickers: list[str]) -> pd.DataFrame:
    cols = {}
    for t in tickers:
        f = data / "prices" / f"{t}.csv"
        if f.exists():
            s = pd.read_csv(f, index_col="date", parse_dates=["date"])["adj_close"]
            cols[t] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(cols).sort_index()


def sp500_members(data: Path) -> list[str]:
    """Today's S&P 500 tickers, refreshed at most once a week."""
    f = data / "sp500.csv"
    if not f.exists() or time.time() - f.stat().st_mtime > 7 * 86400:
        try:
            r = requests.get(WIKI, headers=UA, timeout=30)
            r.raise_for_status()
            tab = next(t for t in pd.read_html(io.StringIO(r.text)) if "Symbol" in t.columns)
            if len(tab) > 400:
                tab[["Symbol"]].rename(columns={"Symbol": "ticker"}).to_csv(f, index=False)
        except Exception as ex:  # noqa: BLE001 - keep last week's list
            log.warning("S&P 500 list: %s", ex)
    return sorted(pd.read_csv(f)["ticker"].astype(str)) if f.exists() else []


def _finnhub(path: str, key: str, **params) -> dict | list:
    for attempt in range(3):
        r = requests.get(f"{FINNHUB}/{path}", params={**params, "token": key}, timeout=30)
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def update_earnings(data: Path, key: str, today: pd.Timestamp, days: int = 120) -> int:
    """Earnings results (symbol, date, hour, epsActual, epsEstimate) for the past `days`, merged into earnings.csv."""
    f = data / "earnings.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    got = []
    for k in range(0, days, 30):
        hi, lo = today - pd.Timedelta(days=k), today - pd.Timedelta(days=min(k + 30, days))
        js = _finnhub("calendar/earnings", key, **{"from": lo.strftime("%Y-%m-%d"), "to": hi.strftime("%Y-%m-%d")})
        got += js.get("earningsCalendar") or []
        time.sleep(1.1)
    new = pd.DataFrame(got, columns=["symbol", "date", "hour", "epsActual", "epsEstimate"]) if got else pd.DataFrame()
    allx = pd.concat([old, new], ignore_index=True)
    if allx.empty:
        return 0
    allx = allx.drop_duplicates(["symbol", "date"], keep="last").sort_values(["date", "symbol"])
    allx.to_csv(f, index=False)
    return len(new)


def update_insiders(data: Path, key: str, symbols: list[str], today: pd.Timestamp, days: int = 45, pause: float = 1.05) -> dict:
    """Insider transactions filed in the past `days` for each symbol, merged into insiders.csv."""
    f = data / "insiders.csv"
    cols = ["symbol", "name", "transactionCode", "change", "transactionDate", "filingDate", "transactionPrice"]
    old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=cols)
    rows, failed = [], 0
    since = (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    for s in symbols:
        try:
            js = _finnhub("stock/insider-transactions", key, symbol=s, **{"from": since})
            for x in js.get("data") or []:
                rows.append({c: x.get(c) for c in cols} | {"symbol": s})
        except Exception:  # noqa: BLE001
            failed += 1
        time.sleep(pause)
    allx = pd.concat([old, pd.DataFrame(rows, columns=cols)], ignore_index=True)
    allx = allx.drop_duplicates(["symbol", "name", "transactionDate", "filingDate", "change"], keep="last")
    allx.to_csv(f, index=False)
    return {"rows": len(rows), "failed": failed}
