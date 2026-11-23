"""Reddit posts that name an S&P 500 ticker in their title, from the Arctic Shift archive (public, no key).

Runs on GitHub Actions (reddit-sentiment.yml); the research sandbox cannot reach Reddit or Arctic Shift.
For every ticker in data/tickers.txt (point-in-time S&P 500 members since mid-2018, minus tickers that are English
words or 1-2 letters) and every subreddit, pages through posts whose title contains the ticker as a keyword.
Only id, time, subreddit, title and flair are kept. The archived score and comment count are kept for reference but
never used as signals, because they were recorded after the post (look-ahead).

Resumable: OUT/done.txt lists finished (ticker, subreddit, year) windows; posts go to OUT/posts_<shard>.jsonl.

    python collect_reddit.py data/tickers.txt OUT --shard 0 --shards 4 --max-minutes 300
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
YEARS = list(range(2018, 2027))
FIELDS = "id,created_utc,subreddit,title,link_flair_text,score,num_comments"


def ts(y: int) -> str:
    return f"{y}-01-01" if y > 2018 else "2018-06-01"


def get(s: requests.Session, params: dict) -> list[dict]:
    last = ""
    for i in range(6):
        try:
            r = s.get(API, params=params, timeout=90)
            if r.status_code == 200:
                return r.json().get("data") or []
            last = f"{r.status_code} {r.text[:120]}"
            if r.status_code == 400:
                break
        except requests.RequestException as e:
            last = str(e)[:120]
        time.sleep(5 * (i + 1))
    raise RuntimeError(last)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--pause", type=float, default=0.7)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tick = [t.strip() for t in a.tickers.read_text().split() if t.strip()]
    tick = [t for i, t in enumerate(tick) if i % a.shards == a.shard]
    donef = a.out / f"done_{a.shard}.txt"
    done = set(donef.read_text().split()) if donef.exists() else set()
    s = requests.Session()
    s.headers["User-Agent"] = "experiment-1 research (paper study of Reddit attention)"
    t0, calls, posts, fails = time.time(), 0, 0, []
    with (a.out / f"posts_{a.shard}.jsonl").open("a") as fh, donef.open("a") as dh:
        for t in tick:
            for sub in SUBS:
                for y in YEARS:
                    key = f"{t}|{sub}|{y}"
                    if key in done:
                        continue
                    if time.time() - t0 > a.max_minutes * 60:
                        print(f"time limit: {calls} calls, {posts} posts, {len(fails)} failed windows")
                        return
                    after, before = ts(y), ts(y + 1) if y < 2026 else "2026-11-22"
                    got, ok = 0, True
                    while True:
                        try:
                            page = get(s, {"subreddit": sub, "title": t, "after": after, "before": before, "limit": "auto",
                                           "sort": "asc", "fields": FIELDS})
                        except RuntimeError as e:
                            fails.append(f"{key}: {e}")
                            ok = False
                            break
                        finally:
                            calls += 1
                            time.sleep(a.pause)
                        for p in page:
                            fh.write(json.dumps({"tk": t, "id": p.get("id"), "ts": int(p.get("created_utc") or 0), "sub": sub,
                                                 "title": (p.get("title") or "")[:300], "flair": p.get("link_flair_text"),
                                                 "score": p.get("score"), "nc": p.get("num_comments")}) + "\n")
                        got += len(page)
                        if len(page) < 100:
                            break
                        after = str(int(page[-1]["created_utc"]) + 1)
                    if ok:
                        dh.write(key + "\n")
                        posts += got
                fh.flush()
                dh.flush()
            print(f"{datetime.now(timezone.utc):%H:%M} {t}: total {posts} posts, {calls} calls, {len(fails)} fails", flush=True)
    print(f"finished shard {a.shard}: {calls} calls, {posts} posts, {len(fails)} failed windows")
    for f in fails[:30]:
        print("  ", f)


if __name__ == "__main__":
    main()
