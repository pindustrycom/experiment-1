"""News before each flagged stock-day (flagged.csv): Google News headlines from the 3 days up to that day's close, scored
by Claude Haiku and by a small word list. Runs on GitHub Actions (news-flags.yml); resumable, caches in OUT_DIR.

Leakage guards: only headlines published before 16:00 New York on the flagged day; the company's name and ticker are
replaced by "the company" and no dates are shown to the model, which is told not to use anything it knows about later
events. A language model may still recognise a famous event from its headlines, so evaluate.py also checks the word
list, which knows nothing, and the most recent months separately.

    python collect.py flagged.csv OUT_DIR [--max-minutes 270]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) experiment-1 research"
MODEL = os.environ.get("NEWS_MODEL", "claude-haiku-5-5")
PRICE = (0.10, 0.50)  # USD per million input/output tokens
SUFFIX = re.compile(r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|n\.?v|s\.?a|the)\.?$", re.IGNORECASE)
POS = {"beat", "beats", "raises", "raised", "upgrade", "upgraded", "surge", "surges", "soar", "soars", "record", "strong", "jump",
       "jumps", "rally", "rallies", "gain", "gains", "outperform", "buy", "bullish", "boost", "boosts", "approval", "approved", "wins",
       "win", "partnership", "tops", "exceeds", "higher", "rises", "rise", "growth", "upbeat"}
NEG = {"miss", "misses", "cut", "cuts", "downgrade", "downgraded", "plunge", "plunges", "drop", "drops", "fall", "falls", "probe",
       "lawsuit", "sued", "weak", "recall", "layoffs", "warns", "warning", "slump", "slumps", "sink", "sinks", "bearish", "sell",
       "lower", "loss", "losses", "fraud", "investigation", "decline", "declines", "tumble", "tumbles", "risk", "concerns", "fears"}
SYSTEM = """You read news headlines about one listed US company, all published before the stock market closed on one day.
The company's name is replaced by "the company". Judge only from these headlines; do not use anything you may know about
later events. Answer with JSON only:
{"sentiment": number from -1 (very negative) to 1 (very positive),
 "lean": number from -1 to 1, how the stock is more likely to move next if it makes a big move (negative = down),
 "earnings_soon": true if the headlines say results are due within about a day, else false,
 "confidence": number from 0 to 1}"""


def names(ua: str) -> dict[str, str]:
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers={"User-Agent": ua}, timeout=60)
    r.raise_for_status()
    out = {}
    for v in r.json().values():
        n = v["title"]
        for _ in range(3):
            n = SUFFIX.sub("", n).strip()
        out[v["ticker"].replace(".", "-")] = n.title() if n.isupper() else n
    return out


def rss(s: requests.Session, q: str, a: str, b: str) -> list[dict]:
    for attempt in range(4):
        r = s.get("https://news.google.com/rss/search", params={"q": f"{q} after:{a} before:{b}", "hl": "en-US", "gl": "US", "ceid": "US:en"},
                  timeout=30)
        if r.status_code in (429, 503):
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        out = []
        for it in ET.fromstring(r.content).findall(".//item"):
            title, pub, src = it.findtext("title") or "", it.findtext("pubDate"), it.findtext("source") or ""
            if not pub or not title:
                continue
            if src and title.endswith(" - " + src):
                title = title[: -len(src) - 3]
            out.append({"ts": parsedate_to_datetime(pub).astimezone(tz=None).isoformat(), "title": title.strip(), "source": src})
        return out
    raise RuntimeError("Google News kept refusing")


def mask(title: str, name: str, ticker: str) -> str:
    t = re.sub(r"\$[A-Za-z.]+", "the company", title)
    for w in sorted({name, name.split()[0] if len(name.split()[0]) > 3 else name, ticker}, key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(w)}\b('s)?", "the company", t, flags=re.IGNORECASE)
    return re.sub(r"the company( company)+", "the company", t)


def words(titles: list[str]) -> float | None:
    p = n = 0
    for t in titles:
        toks = set(re.findall(r"[a-z]+", t.lower()))
        p, n = p + len(toks & POS), n + len(toks & NEG)
    return (p - n) / (p + n) if p + n else None


def haiku(client, titles: list[str]) -> tuple[dict, float]:
    msg = "<headlines>\n" + "\n".join(f"- {t}" for t in titles) + "\n</headlines>"
    resp = client.messages.create(model=MODEL, max_tokens=200, system=SYSTEM, messages=[{"role": "user", "content": msg}])
    txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    js = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
    cost = resp.usage.input_tokens * PRICE[0] / 1e6 + resp.usage.output_tokens * PRICE[1] / 1e6
    return js, cost


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("flagged", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-minutes", type=float, default=270)
    ap.add_argument("--budget-usd", type=float, default=3.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / "scores.jsonl"
    done = {json.loads(x)["key"] for x in f.read_text().splitlines()} if f.exists() else set()
    fl = pd.read_csv(a.flagged)
    nm = names(os.environ.get("SEC_USER_AGENT") or "experiment-1 research admin@example.com")
    import anthropic
    client = anthropic.Anthropic(timeout=60, max_retries=3) if os.environ.get("ANTHROPIC_API_KEY") else None
    s = requests.Session()
    s.headers["User-Agent"] = UA
    t0, spent, n_new, fails = time.time(), 0.0, 0, 0
    with f.open("a") as fh:
        for r in fl.itertuples(index=False):
            key = f"{r.date}|{r.ticker}"
            if key in done:
                continue
            if time.time() - t0 > a.max_minutes * 60 or spent > a.budget_usd:
                break
            d = pd.Timestamp(r.date)
            name = nm.get(r.ticker, r.ticker)
            q = f'"{name}"' if name != r.ticker else f"{r.ticker} stock"
            close = pd.Timestamp(f"{r.date} 16:00", tz="America/New_York")
            try:
                items = rss(s, q, (d - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            except Exception as ex:  # noqa: BLE001 - try again next run
                fails += 1
                print("rss", key, ex)
                time.sleep(5)
                continue
            finally:
                time.sleep(1.0)
            items = sorted([x for x in items if pd.Timestamp(x["ts"]) < close], key=lambda x: x["ts"], reverse=True)[:25]
            titles = [mask(x["title"], name, r.ticker) for x in items]
            row = {"key": key, "date": r.date, "ticker": r.ticker, "query": q, "n": len(titles), "words": words(titles),
                   "hash": hashlib.sha1("\n".join(titles).encode()).hexdigest()[:12], "titles": titles}
            if titles and client is not None:
                try:
                    js, c = haiku(client, titles)
                    spent += c
                    row.update({k: js.get(k) for k in ("sentiment", "lean", "earnings_soon", "confidence")})
                except Exception as ex:  # noqa: BLE001 - a failed score is retried next run
                    print("haiku", key, ex)
                    fails += 1
                    continue
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n_new += 1
    total = len(done) + n_new
    print(json.dumps({"new": n_new, "total": total, "of": len(fl), "failures": fails, "usd": round(spent, 3),
                      "minutes": round((time.time() - t0) / 60, 1)}))


if __name__ == "__main__":
    main()
