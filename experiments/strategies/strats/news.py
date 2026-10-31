"""News lean for the news-tilted straddle pot: Google News headlines from the 3 days up to today's 16:00 New York close,
the company's name and ticker replaced by "the company", read by Claude Haiku for which way a big move is more likely.

The same reading as the study in research/timescale_ladder/news (collect.py), where on 2019-2026 flagged days that moved
8%+ it told rises from falls with AUC ~0.6, holding after the stock's own recent returns. Uses the anthropic SDK. Any failure gives no lean, and that name is bought as a plain straddle.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

log = logging.getLogger(__name__)
MODEL = os.environ.get("NEWS_MODEL", "claude-haiku-5-5")
UA = "Mozilla/5.0 (X11; Linux x86_64) experiment-1 research"
SUFFIX = re.compile(r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|n\.?v|s\.?a|the)\.?$",
                    re.IGNORECASE)
SYSTEM = """You read news headlines about one listed US company, all published before the stock market closed on one day.
The company's name is replaced by "the company". Judge only from these headlines; do not use anything you may know about
later events. Answer with JSON only:
{"sentiment": number from -1 (very negative) to 1 (very positive),
 "lean": number from -1 to 1, how the stock is more likely to move next if it makes a big move (negative = down),
 "earnings_soon": true if the headlines say results are due within about a day, else false,
 "confidence": number from 0 to 1}"""


def company_names(ua: str) -> dict[str, str]:
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers={"User-Agent": ua}, timeout=60)
    r.raise_for_status()
    out = {}
    for v in r.json().values():
        n = v["title"]
        for _ in range(3):
            n = SUFFIX.sub("", n).strip()
        out[v["ticker"].replace(".", "-")] = n.title() if n.isupper() else n
    return out


def mask(title: str, name: str, ticker: str) -> str:
    t = re.sub(r"\$[A-Za-z.]+", "the company", title)
    first = name.split()[0] if name.split() and len(name.split()[0]) > 3 else name
    for w in sorted({name, first, ticker}, key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(w)}\b('s)?", "the company", t, flags=re.IGNORECASE)
    return re.sub(r"the company( company)+", "the company", t)


def headlines(name: str, ticker: str, d: pd.Timestamp) -> list[str]:
    q = f'"{name}"' if name != ticker else f"{ticker} stock"
    a, b = (d - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    r = requests.get("https://news.google.com/rss/search", headers={"User-Agent": UA}, timeout=30,
                     params={"q": f"{q} after:{a} before:{b}", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    r.raise_for_status()
    close = pd.Timestamp(f"{d.date()} 16:00", tz="America/New_York")
    items = []
    for it in ET.fromstring(r.content).findall(".//item"):
        title, pub, src = it.findtext("title") or "", it.findtext("pubDate"), it.findtext("source") or ""
        if not pub or not title:
            continue
        ts = pd.Timestamp(parsedate_to_datetime(pub))
        if ts >= close:
            continue
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        items.append((ts, title.strip()))
    return [mask(t, name, ticker) for _, t in sorted(items, reverse=True)[:25]]


def read(titles: list[str], client) -> dict:
    """Haiku's reading of the headlines (the same call as the study's collect.py, through the SDK)."""
    msg = "<headlines>\n" + "\n".join(f"- {t}" for t in titles) + "\n</headlines>"
    resp = client.messages.create(model=MODEL, max_tokens=300, system=SYSTEM, messages=[{"role": "user", "content": msg}])
    txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    if "{" not in txt:
        raise ValueError(f"no JSON in the answer (stop {resp.stop_reason}): {txt[:200]!r}")
    return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])


def leans(tickers: list[str], d: pd.Timestamp) -> dict[str, dict]:
    """ticker -> {"lean", "sentiment", "n"} for each name; {} without ANTHROPIC_API_KEY."""
    if not os.environ.get("ANTHROPIC_API_KEY") or not tickers:
        return {}
    import anthropic
    client = anthropic.Anthropic(timeout=60, max_retries=3)
    try:
        nm = company_names(os.environ.get("SEC_USER_AGENT") or "experiment-1 research admin@example.com")
    except Exception as ex:  # noqa: BLE001 - fall back to tickers
        log.warning("SEC names: %s", ex)
        nm = {}
    out = {}
    for t in tickers:
        try:
            titles = headlines(nm.get(t, t), t, d)
            js = read(titles, client) if titles else {}
            out[t] = {"lean": _num(js.get("lean")), "sentiment": _num(js.get("sentiment")), "n": len(titles)}
        except Exception as ex:  # noqa: BLE001 - one name without news is bought as a plain straddle
            log.warning("news %s: %s", t, ex)
            out[t] = {"lean": None, "sentiment": None, "n": 0, "error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        time.sleep(1.0)
    return out


def _num(x) -> float | None:
    try:
        return max(-1.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return None
