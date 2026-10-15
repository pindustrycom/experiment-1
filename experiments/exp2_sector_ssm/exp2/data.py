"""Prices from Tiingo and company news from Finnhub, cached in the experiment's state folder.

State layout (all plain files, so a run can be inspected by hand):
    prices/<TICKER>.csv       daily bars, adjusted for splits and dividends
    news/raw/<YYYY-MM>.jsonl  headlines as fetched, one per line, deduplicated by id
    news/fetched.json         which (ticker, day) ranges have been fetched
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("exp2.data")
PRICE_COLS = ["date", "close", "adj_close", "adj_volume", "div_cash", "split_factor"]


class ProviderError(RuntimeError):
    pass


def _get(url: str, params: dict, tries: int = 4, pause: float = 2.0):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                last = ProviderError(f"{url.split('?')[0]}: HTTP {r.status_code}")
                time.sleep(pause * 2 ** i)
                continue
            if r.status_code != 200:
                raise ProviderError(f"{url.split('?')[0]}: HTTP {r.status_code} {r.text[:200]}")
            return r.json()
        except requests.RequestException as e:
            last = ProviderError(f"{url.split('?')[0]}: {type(e).__name__}")
            time.sleep(pause * 2 ** i)
    raise last


# ---------------------------------------------------------------- prices

def parse_tiingo(payload) -> pd.DataFrame:
    if not isinstance(payload, list):
        raise ProviderError(f"tiingo: unexpected payload {str(payload)[:200]}")
    rows = [{"date": r["date"][:10], "close": r["close"], "adj_close": r["adjClose"], "adj_volume": r.get("adjVolume"),
             "div_cash": r.get("divCash", 0.0) or 0.0, "split_factor": r.get("splitFactor", 1.0) or 1.0} for r in payload]
    return pd.DataFrame(rows, columns=PRICE_COLS)


def fetch_tiingo(ticker: str, start: str, key: str) -> pd.DataFrame:
    return parse_tiingo(_get(f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices",
                             {"startDate": start, "token": key, "format": "json"}))


def update_prices(state: Path, tickers: list[str], start: str, key: str | None, fetch=fetch_tiingo) -> dict:
    """Bring each ticker's file up to date. Adjusted closes change for the whole history after a
    dividend or split, so a ticker is re-downloaded in full whenever a new bar carries one or the
    overlap no longer matches; otherwise only recent bars are fetched."""
    out = {"updated": [], "full": [], "failed": {}}
    if not key:
        out["failed"] = {t: "TIINGO_API_KEY not set" for t in tickers}
        return out
    d = state / "prices"
    d.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        f = d / f"{t}.csv"
        try:
            old = pd.read_csv(f, dtype={"date": str}) if f.exists() else None
            full = old is None or old.empty
            if not full:
                since = (pd.Timestamp(old["date"].iloc[-1]) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
                new = fetch(t, since, key)
                ov = old.merge(new, on="date", suffixes=("", "_n"))
                fresh = new[new["date"] > old["date"].iloc[-1]]
                corporate = (fresh["div_cash"].fillna(0) != 0).any() or (fresh["split_factor"].fillna(1) != 1).any()
                drift = ov.empty or (abs(ov["adj_close"] / ov["adj_close_n"] - 1) > 1e-6).any()
                if corporate or drift:
                    full = True
                else:
                    merged = pd.concat([old, fresh], ignore_index=True)
            if full:
                merged = fetch(t, start, key)
                out["full"].append(t)
            if merged.empty:
                raise ProviderError("no bars returned")
            merged = merged.drop_duplicates("date", keep="last").sort_values("date")
            merged[PRICE_COLS].to_csv(f, index=False)
            out["updated"].append(t)
        except Exception as e:  # noqa: BLE001 - one ticker failing must not stop the others
            out["failed"][t] = str(e)[:200]
            log.warning("prices %s failed: %s", t, e)
    return out


def load_prices(state: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        f = state / "prices" / f"{t}.csv"
        if f.exists():
            df = pd.read_csv(f, dtype={"date": str})
            df["date"] = pd.to_datetime(df["date"])
            out[t] = df.set_index("date").sort_index()
    return out


# ---------------------------------------------------------------- news

def parse_finnhub(payload, ticker: str) -> list[dict]:
    if not isinstance(payload, list):
        raise ProviderError(f"finnhub: unexpected payload {str(payload)[:200]}")
    out = []
    for it in payload:
        if not it.get("headline") or not it.get("datetime"):
            continue
        out.append({"id": "fh" + str(it.get("id") or hashlib.sha1((it.get("url") or it["headline"]).encode()).hexdigest()[:16]), "ticker": ticker,
                    "ts": int(it["datetime"]), "headline": it["headline"].strip()[:400],
                    "summary": (it.get("summary") or "").strip()[:600], "source": it.get("source") or "",
                    "url": it.get("url") or "", "related": it.get("related") or ""})
    return out


def fetch_finnhub(ticker: str, start: date, end: date, key: str) -> list[dict]:
    return parse_finnhub(_get("https://finnhub.io/api/v1/company-news",
                              {"symbol": ticker, "from": start.isoformat(), "to": end.isoformat(), "token": key}), ticker)


def _news_files(state: Path) -> Path:
    p = state / "news" / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_news(state: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(_news_files(state).glob("*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    cols = ["id", "ticker", "ts", "headline", "summary", "source", "url", "related"]
    df = pd.DataFrame(rows, columns=cols)
    return df.drop_duplicates("id", keep="first").reset_index(drop=True)


def _append_news(state: Path, items: list[dict], known: set[str]) -> int:
    by_month: dict[str, list[dict]] = {}
    n = 0
    for it in items:
        if it["id"] in known:
            continue
        known.add(it["id"])
        m = datetime.fromtimestamp(it["ts"], UTC).strftime("%Y-%m")
        by_month.setdefault(m, []).append(it)
        n += 1
    for m, rows in by_month.items():
        with open(_news_files(state) / f"{m}.jsonl", "a") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return n


def update_news(state: Path, tickers: list[str], key: str | None, today: date | None = None,
                backfill_days: int = 365, fetch=fetch_finnhub, pause: float = 1.1, max_calls: int = 900) -> dict:
    """Fetch new headlines. The first run backfills `backfill_days` in weekly windows (big names
    have hundreds of stories a week and a wider window gets cut off); later runs fetch the last
    few days again, since stories are sometimes published late."""
    out = {"new": 0, "calls": 0, "failed": {}, "coverage_start": None}
    meta_f = state / "news" / "fetched.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    if not key:
        out["failed"] = {"*": "FINNHUB_API_KEY not set"}
        out["coverage_start"] = meta.get("coverage_start")
        return out
    today = today or datetime.now(UTC).date()
    known = set(load_news(state)["id"])
    meta.setdefault("coverage_start", (today - timedelta(days=backfill_days)).isoformat())
    meta.setdefault("through", {})
    for t in tickers:
        start = date.fromisoformat(meta["through"].get(t, meta["coverage_start"]))
        start = min(start, today - timedelta(days=3))
        d = start
        try:
            while d <= today and out["calls"] < max_calls:
                e = min(d + timedelta(days=6), today)
                out["new"] += _append_news(state, fetch(t, d, e, key), known)
                out["calls"] += 1
                meta["through"][t] = e.isoformat()
                d = e + timedelta(days=1)
                time.sleep(pause)
        except Exception as ex:  # noqa: BLE001
            out["failed"][t] = str(ex)[:200]
            log.warning("news %s failed: %s", t, ex)
        meta_f.parent.mkdir(parents=True, exist_ok=True)
        meta_f.write_text(json.dumps(meta, indent=1))
    out["coverage_start"] = meta["coverage_start"]
    return out


def secret(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None
