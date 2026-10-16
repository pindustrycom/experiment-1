"""Collect NVDA's history: prices, market cap, analyst actions, news headlines and Reddit posts.

Every collector is resumable: finished time windows are recorded in data/progress.json, so a rerun
only fetches what is missing. Nothing is invented: a window that keeps failing is left out and
reported, and the features treat it as unobserved.

Sources (all free):
- Tiingo daily prices: NVDA, QQQ, SPY.
- SEC XBRL: NVDA's shares outstanding as filed (market cap = close x shares known at the time).
- Yahoo Finance: analyst upgrades, downgrades and price-target changes (the brokers' view).
- Google News search: headlines about NVIDIA, by publication date.
- Reddit, via the Arctic Shift archive: posts whose title mentions NVDA or Nvidia (public speculation).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

from . import config

log = logging.getLogger("exp3.collect")
UA_BROWSER = "Mozilla/5.0 (X11; Linux x86_64) exp3-nvda research"


class SourceError(RuntimeError):
    pass


def _get(session: requests.Session, url: str, params: dict | None = None, tries: int = 5, pause: float = 3.0, ok_codes=(200,)):
    last = None
    for i in range(tries):
        try:
            r = session.get(url, params=params or {}, timeout=60)
            if r.status_code in ok_codes:
                return r
            last = SourceError(f"{url}: HTTP {r.status_code} {r.text[:120]}")
            if r.status_code not in (408, 422, 429) and r.status_code < 500:
                break
        except requests.RequestException as e:
            last = SourceError(f"{url}: {type(e).__name__}")
        time.sleep(pause * 2 ** i)
    raise last


class Progress:
    def __init__(self, data: Path):
        self.f = data / "progress.json"
        self.d = json.loads(self.f.read_text()) if self.f.exists() else {}

    def done(self, key: str) -> bool:
        return key in self.d

    def mark(self, key: str, n: int) -> None:
        self.d[key] = n
        self.f.write_text(json.dumps(self.d))


def _append(path: Path, rows: list[dict]) -> None:
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return pd.DataFrame(rows).drop_duplicates("id", keep="last") if rows else pd.DataFrame()


# ---------------------------------------------------------------- prices, shares, analysts

def prices(data: Path, key: str, tickers=(config.TICKER, *config.MARKET), start=config.PRICE_START) -> dict:
    d = data / "prices"
    d.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    out = {}
    for t in tickers:
        r = _get(s, f"https://api.tiingo.com/tiingo/daily/{t.lower()}/prices", {"startDate": start, "token": key}, pause=20)
        rows = [{"date": x["date"][:10], "close": x["close"], "adj_close": x["adjClose"], "adj_volume": x.get("adjVolume"),
                 "split_factor": x.get("splitFactor", 1.0) or 1.0} for x in r.json()]
        if not rows:
            raise SourceError(f"no prices for {t}")
        pd.DataFrame(rows).to_csv(d / f"{t}.csv", index=False)
        out[t] = len(rows)
        time.sleep(2)
    return out


def shares(data: Path, user_agent: str, cik: str = config.CIK) -> int:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent or "exp3-nvda research"
    r = _get(s, f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/dei/EntityCommonStockSharesOutstanding.json")
    rows = [{"end": v["end"], "filed": v["filed"], "shares": v["val"], "form": v.get("form")} for v in r.json()["units"]["shares"]]
    df = pd.DataFrame(rows).sort_values("filed").drop_duplicates(["end", "filed"])
    df.to_csv(data / "shares.csv", index=False)
    return len(df)


def analysts(data: Path, ticker: str = config.TICKER) -> int:
    s = requests.Session()
    s.headers["User-Agent"] = UA_BROWSER
    try:
        s.get("https://fc.yahoo.com", timeout=20)
    except requests.RequestException:
        pass
    crumb = _get(s, "https://query2.finance.yahoo.com/v1/test/getcrumb").text
    r = _get(s, f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}", {"modules": "upgradeDowngradeHistory", "crumb": crumb})
    hist = r.json()["quoteSummary"]["result"][0]["upgradeDowngradeHistory"]["history"]
    rows = [{"ts": pd.Timestamp(h["epochGradeDate"], unit="s", tz="UTC").isoformat(), "firm": h.get("firm"), "action": h.get("action"),
             "from_grade": h.get("fromGrade") or "", "to_grade": h.get("toGrade") or "", "pt_action": h.get("priceTargetAction") or "",
             "pt": h.get("currentPriceTarget"), "pt_prior": h.get("priorPriceTarget")} for h in hist]
    pd.DataFrame(rows).sort_values("ts").to_csv(data / "analysts.csv", index=False)
    return len(rows)


# ---------------------------------------------------------------- Google News

def parse_rss(content: bytes) -> list[dict]:
    out = []
    for it in ET.fromstring(content).findall(".//item"):
        title, pub = it.findtext("title") or "", it.findtext("pubDate")
        if not pub or not title:
            continue
        ts = parsedate_to_datetime(pub)
        src = it.findtext("source") or ""
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        out.append({"id": "gn" + hashlib.sha1((it.findtext("link") or title).encode()).hexdigest()[:16],
                    "ts": pd.Timestamp(ts).tz_convert("UTC").isoformat(), "title": title.strip(), "source": src})
    return out


def news(data: Path, start=config.SENTIMENT_START, end=config.TEST_END, query=config.NEWS_QUERY, pause: float = 1.5,
         fetch=None, cap: int = 95, max_seconds: float = 3 * 3600) -> dict:
    """Weekly windows; a window that hits Google's ~100 result cap is split in half, down to one day."""
    prog, path = Progress(data), data / "news.jsonl"
    s = requests.Session()
    s.headers["User-Agent"] = UA_BROWSER

    def default_fetch(a: str, b: str) -> list[dict]:
        r = _get(s, "https://news.google.com/rss/search", {"q": f"{query} after:{a} before:{b}", "hl": "en-US", "gl": "US", "ceid": "US:en"})
        return parse_rss(r.content)

    fetch = fetch or default_fetch
    stats = {"windows": 0, "items": 0, "failed": []}
    t0 = time.time()

    def window(a: pd.Timestamp, b: pd.Timestamp):
        key = f"gn:{a.date()}:{b.date()}"
        if prog.done(key):
            return
        if time.time() - t0 > max_seconds:
            stats["failed"].append(key + " (time)")
            return
        try:
            # Google's before: is exclusive and after: inclusive, by date
            items = fetch(a.strftime("%Y-%m-%d"), (b + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        except Exception as e:  # noqa: BLE001
            stats["failed"].append(f"{key}: {e}"[:160])
            return
        finally:
            time.sleep(pause)
        if len(items) >= cap and b > a:
            mid = a + (b - a) / 2
            mid = mid.normalize()
            window(a, mid)
            window(mid + pd.Timedelta(days=1), b)
            if all(prog.done(f"gn:{x.date()}:{y.date()}") for x, y in ((a, mid), (mid + pd.Timedelta(days=1), b))):
                prog.mark(key, -1)
            return
        _append(path, items)
        prog.mark(key, len(items))
        stats["windows"] += 1
        stats["items"] += len(items)

    a = pd.Timestamp(start)
    while a <= pd.Timestamp(end):
        b = min(a + pd.Timedelta(days=6), pd.Timestamp(end))
        window(a, b)
        a = b + pd.Timedelta(days=1)
    return stats


# ---------------------------------------------------------------- Reddit

ARCTIC = "https://arctic-shift.photon-reddit.com/api/posts/search"


def reddit(data: Path, start=config.SENTIMENT_START, end=config.TEST_END, subs=config.SUBREDDITS, terms=config.REDDIT_TERMS,
           pause: float = 1.0, fetch=None, max_seconds: float = 3 * 3600) -> dict:
    """Monthly windows per subreddit and term, paged 100 posts at a time."""
    prog, path = Progress(data), data / "reddit.jsonl"
    s = requests.Session()
    s.headers["User-Agent"] = "exp3-nvda research"

    def default_fetch(sub, term, after, before):
        r = _get(s, ARCTIC, {"subreddit": sub, "title": term, "after": after, "before": before, "limit": 100, "sort": "asc"}, tries=6, pause=4)
        return r.json().get("data") or []

    fetch = fetch or default_fetch
    stats = {"windows": 0, "items": 0, "failed": []}
    t0 = time.time()
    months = pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end), freq="MS")
    for sub in subs:
        for term in terms:
            for m in months:
                key = f"rd:{sub}:{term}:{m.strftime('%Y-%m')}"
                if prog.done(key):
                    continue
                if time.time() - t0 > max_seconds:
                    stats["failed"].append(key + " (time)")
                    continue
                after, before = int(m.timestamp()), int((m + pd.offsets.MonthBegin(1)).timestamp())
                got, ok = [], True
                while True:
                    try:
                        page = fetch(sub, term, after, before)
                    except Exception as e:  # noqa: BLE001
                        stats["failed"].append(f"{key}: {e}"[:160])
                        ok = False
                        break
                    finally:
                        time.sleep(pause)
                    got += page
                    if len(page) < 100:
                        break
                    after = int(page[-1]["created_utc"]) + 1
                if not ok:
                    continue
                rows = [{"id": "rd" + p["id"], "ts": pd.Timestamp(int(p["created_utc"]), unit="s", tz="UTC").isoformat(), "sub": sub,
                         "title": (p.get("title") or "").strip(), "text": (p.get("selftext") or "")[:300],
                         "score": p.get("score") or 0, "comments": p.get("num_comments") or 0} for p in got]
                _append(path, rows)
                prog.mark(key, len(rows))
                stats["windows"] += 1
                stats["items"] += len(rows)
    return stats
