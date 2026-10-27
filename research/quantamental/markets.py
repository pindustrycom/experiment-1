"""A universe of markets for the edge pipeline: ETFs for US sectors, countries, bonds, commodities, real estate and
bitcoin, daily from Yahoo (cached), turned into the same month x name panel the stock pipeline uses."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("quantamental.markets")
UNIVERSE = {
    "US sectors": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"],
    "countries": ["EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ", "EWH", "EWS", "EWT", "EWY", "EWW", "EWQ", "EWL", "EWP", "EWI", "EZA", "FXI", "EEM", "EFA"],
    "bonds": ["TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "EMB"],
    "commodities": ["GLD", "SLV", "USO", "DBA", "DBC", "UNG"],
    "other": ["VNQ", "BTC-USD"],
}
SIGNALS = {
    "momentum 12-1": ("quant", lambda p: p["mom12_1"]),
    "momentum 3 months": ("quant", lambda p: p["mom3"]),
    "1-month reversal": ("quant", lambda p: -p["rev1"]),
    "5-year reversal (value)": ("quant", lambda p: -p["rev60"]),
    "trend: above 200-day average": ("quant", lambda p: p["trend"]),
    "low volatility": ("quant", lambda p: -p["vol"]),
    "low beta": ("quant", lambda p: -p["beta"]),
}


def fetch(ticker: str, cache: Path) -> pd.Series | None:
    f = cache / f"{ticker}.csv"
    if f.exists() and time.time() - f.stat().st_mtime < 3 * 86400:
        return pd.read_csv(f, index_col=0, parse_dates=True)["adj_close"]
    try:
        r = requests.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
                         params={"period1": 946684800, "period2": int(time.time()), "interval": "1d", "events": "div,splits"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or res["indicators"]["quote"][0]["close"]
        s = pd.Series(adj, index=pd.to_datetime(res["timestamp"], unit="s").normalize(), name="adj_close", dtype=float).dropna()
        s = s[~s.index.duplicated(keep="last")]
        cache.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_csv(f)
        time.sleep(0.3)
        return s
    except Exception as e:  # noqa: BLE001 - a missing market just leaves the universe
        log.warning("%s: %s", ticker, e)
        return None


def panel(cache: Path, start: str = "2004-12-31", end: str = "2026-09-30") -> tuple[pd.DataFrame, pd.Series]:
    px = {t: s for g in UNIVERSE.values() for t in g if (s := fetch(t, cache)) is not None}
    spy = fetch("SPY", cache)
    daily = pd.DataFrame(px).sort_index()
    daily = daily[daily.index.dayofweek < 5]          # bitcoin trades weekends: keep the stock-market calendar
    months = pd.date_range(start, end, freq="ME")
    m = daily.reindex(daily.index.union(months)).ffill(limit=5).reindex(months)
    r = daily.pct_change(fill_method=None)
    s = spy.reindex(daily.index).pct_change(fill_method=None)
    vol = r.rolling(252, min_periods=150).std() * np.sqrt(252)
    beta = r.rolling(252, min_periods=150).cov(s).div(s.rolling(252, min_periods=150).var(), axis=0)
    resid = np.sqrt((vol**2 - beta.pow(2).mul(s.rolling(252, min_periods=150).var() * 252, axis=0)).clip(lower=0))
    ma200 = daily.rolling(200, min_periods=150).mean()
    at = lambda x: x.reindex(months, method="ffill")
    feats = {"ret": m.shift(-1) / m - 1, "mom12_1": m.shift(1) / m.shift(12) - 1, "mom3": m / m.shift(3) - 1, "rev1": m / m.shift(1) - 1,
             "rev60": m / m.shift(60) - 1, "trend": at(daily / ma200 - 1), "vol": at(vol), "beta": at(beta), "ivol": at(resid)}
    long = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1).reset_index()
    long.columns = ["month", "ticker", *feats]
    long = long[long["mom12_1"].notna() & long["vol"].notna()]
    return long, spy
