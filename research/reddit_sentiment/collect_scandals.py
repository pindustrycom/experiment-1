"""Leadership-scandal events from Google News, 2012-2026, for the "scandal drop then bounce" test.

Runs on GitHub Actions (reddit-sentiment.yml). For every week, a few Google News RSS searches for CEO / executive
scandals (affairs, misconduct, harassment, arrests, ousters under investigation). Claude Haiku then reads each week's
headlines and lists the distinct company events in them as JSON (company, US ticker if listed, person, kind).
Prices are not used here: scandal_study.py joins the events to point-in-time S&P 500 prices.

Also saves FRED's daily WTI oil price (for the oil-vs-tech question).

    python collect_scandals.py OUT --max-minutes 200 --budget-usd 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

MODEL = os.environ.get("NEWS_MODEL", "claude-haiku-5-5")
PRICE = (0.10, 0.50)  # USD per million input/output tokens (same budget guard as news-flags)
QUERIES = [
    'CEO affair OR "inappropriate relationship" OR "personal conduct" OR "consensual relationship"',
    'CEO misconduct OR harassment OR "code of conduct" resigns OR fired OR ousted',
    'CEO arrested OR charged OR indicted OR "under investigation"',
    'CEO "steps down" OR ousted OR "fired" scandal OR probe OR investigation',
    'CEO reinstated OR "returns as CEO" OR "board reverses"',
]
SYSTEM = """You read one week of news headlines about company executives. List every distinct event in which the CEO,
founder, chair or another top executive of a company faced a personal or leadership scandal or an abrupt exit:
affairs or undisclosed relationships, harassment or misconduct, arrests or criminal charges, being fired or forced out,
a probe into the person, or being reinstated after such an exit. Skip ordinary planned retirements and successions.
Answer with JSON only: {"events": [{"company": str, "ticker": US stock ticker or null if not US-listed or private,
"person": str, "kind": one of "affair_relationship", "harassment_misconduct", "criminal_arrest", "ousted_fired",
"probe_investigation", "reinstated", "other", "summary": short phrase}]}"""


def rss(s: requests.Session, q: str, a: str, b: str) -> list[dict]:
    for attempt in range(5):
        try:
            r = s.get("https://news.google.com/rss/search", params={"q": f"{q} after:{a} before:{b}", "hl": "en-US", "gl": "US",
                                                                    "ceid": "US:en"}, timeout=30)
        except requests.RequestException:
            time.sleep(10)
            continue
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
            out.append({"ts": parsedate_to_datetime(pub).isoformat(), "title": title.strip(), "source": src})
        return out
    raise RuntimeError("Google News kept refusing")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-minutes", type=float, default=200)
    ap.add_argument("--budget-usd", type=float, default=4.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO", timeout=60,
                         headers={"User-Agent": "Mozilla/5.0 experiment-1 research"})
        r.raise_for_status()
        (a.out / "wti.csv").write_text(r.text)
    except Exception as e:  # noqa: BLE001
        print("FRED WTI failed:", e)
    f = a.out / "scandal_weeks.jsonl"
    done = {json.loads(x)["week"] for x in f.read_text().splitlines() if x.strip()} if f.exists() else set()
    import anthropic
    client = anthropic.Anthropic(timeout=90, max_retries=3) if os.environ.get("ANTHROPIC_API_KEY") else None
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) experiment-1 research"
    t0, spent = time.time(), 0.0
    weeks = pd.date_range("2012-01-02", "2026-11-23", freq="7D")
    with f.open("a") as fh:
        for w in weeks:
            key = w.strftime("%Y-%m-%d")
            if key in done:
                continue
            if time.time() - t0 > a.max_minutes * 60 or spent > a.budget_usd:
                print(f"stopping at {key}: spent ${spent:.2f}")
                break
            a_, b_ = (w - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), (w + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            heads, seen = [], set()
            try:
                for q in QUERIES:
                    for h in rss(s, q, a_, b_):
                        if h["title"] not in seen:
                            seen.add(h["title"])
                            heads.append(h)
                    time.sleep(1.0)
            except Exception as e:  # noqa: BLE001
                print(key, "news failed:", e)
                continue
            heads.sort(key=lambda h: h["ts"])
            events, err = [], None
            if client and heads:
                msg = "<headlines>\n" + "\n".join(f"- [{h['ts'][:10]}] {h['title']} ({h['source']})" for h in heads[:400]) + "\n</headlines>"
                try:
                    resp = client.messages.create(model=MODEL, max_tokens=2000, system=SYSTEM, messages=[{"role": "user", "content": msg}])
                    txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                    events = json.loads(txt[txt.find("{"): txt.rfind("}") + 1]).get("events", [])
                    spent += resp.usage.input_tokens * PRICE[0] / 1e6 + resp.usage.output_tokens * PRICE[1] / 1e6
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:200]
            fh.write(json.dumps({"week": key, "n_headlines": len(heads), "headlines": heads, "events": events, "err": err}) + "\n")
            fh.flush()
            print(f"{key}: {len(heads)} headlines, {len(events)} events, ${spent:.2f}", flush=True)
    print(f"done, spent ${spent:.2f}")


if __name__ == "__main__":
    main()
