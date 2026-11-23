"""Every post in five stock subreddits, by time, from the Arctic Shift archive (public, no key).

Runs on GitHub Actions (reddit-sentiment.yml); the research sandbox cannot reach Reddit or Arctic Shift.
Arctic Shift's title keyword search times out on r/wallstreetbets ("not supported with very active subreddits"), so
instead this pages through all posts of each subreddit in time order and reddit_study.py finds the tickers itself
(capitalised ticker or $ticker in the title). Kept per post: id, time, subreddit, title, flair. The archived score
and comment count are not kept: they were recorded after the post (look-ahead).

Resumable: OUT/state_<year>.json holds the last timestamp reached per subreddit; posts go to OUT/posts_<year>.jsonl.

    python collect_reddit.py OUT --year 2021 --max-minutes 300
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
SUBS = ["wallstreetbets", "stocks", "investing", "StockMarket", "options"]
FIELDS = "id,created_utc,subreddit,title,link_flair_text"


def get(s: requests.Session, params: dict) -> list[dict]:
    last = ""
    for i in range(8):
        try:
            r = s.get(API, params=params, timeout=120)
            js = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            if r.status_code == 200 and js.get("data") is not None:
                return js["data"]
            last = f"{r.status_code} {str(js.get('error') or r.text)[:120]}"
        except (requests.RequestException, ValueError) as e:
            last = str(e)[:120]
        time.sleep(min(120, 10 * 2 ** i))  # "Timeout. Maybe slow down a bit": back off
    raise RuntimeError(last)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--pause", type=float, default=1.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    start = datetime(a.year, 6 if a.year == 2018 else 1, 1, tzinfo=timezone.utc)
    end = min(datetime(a.year + 1, 1, 1, tzinfo=timezone.utc), datetime(2026, 11, 22, tzinfo=timezone.utc))
    sf = a.out / f"state_{a.year}.json"
    state = json.loads(sf.read_text()) if sf.exists() else {}
    s = requests.Session()
    s.headers["User-Agent"] = "experiment-1 research (paper study of Reddit attention)"
    t0, calls, n = time.time(), 0, 0
    with (a.out / f"posts_{a.year}.jsonl").open("a") as fh:
        for sub in SUBS:
            after = int(state.get(sub, start.timestamp()))
            while after < end.timestamp():
                if time.time() - t0 > a.max_minutes * 60:
                    print(f"time limit: {calls} calls, {n} posts this run; state {state}")
                    return
                try:
                    page = get(s, {"subreddit": sub, "after": after, "before": int(end.timestamp()), "limit": "auto",
                                   "sort": "asc", "fields": FIELDS})
                except RuntimeError as e:
                    print(f"{sub} {datetime.fromtimestamp(after, timezone.utc):%Y-%m-%d}: giving up for now: {e}", flush=True)
                    break
                calls += 1
                time.sleep(a.pause)
                if not page:
                    after = int(end.timestamp())
                else:
                    for p in page:
                        fh.write(json.dumps({"id": p.get("id"), "ts": int(p.get("created_utc") or 0), "sub": sub,
                                             "title": (p.get("title") or "")[:300], "flair": p.get("link_flair_text")}) + "\n")
                    n += len(page)
                    after = int(page[-1]["created_utc"]) + 1  # posts in the same second as the last one may be skipped
                state[sub] = after
                if calls % 20 == 0:
                    fh.flush()
                    sf.write_text(json.dumps(state))
                    print(f"{datetime.now(timezone.utc):%H:%M} {sub} at {datetime.fromtimestamp(after, timezone.utc):%Y-%m-%d %H:%M}, "
                          f"{n} posts, {calls} calls", flush=True)
            state[sub] = max(state.get(sub, 0), after)
            fh.flush()
            sf.write_text(json.dumps(state))
            print(f"{sub} {a.year} done: {n} posts so far, {calls} calls", flush=True)
    print(f"year {a.year} finished: {calls} calls, {n} posts")


if __name__ == "__main__":
    main()
