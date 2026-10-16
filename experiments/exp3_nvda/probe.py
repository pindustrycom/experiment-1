"""Probe: Google News RSS depth for NVIDIA, Reddit title search pacing, FNSPID range reads."""
import time
import xml.etree.ElementTree as ET

import requests

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (exp3-nvda research)"


def gnews(q, a, b):
    r = S.get("https://news.google.com/rss/search", params={"q": f"{q} after:{a} before:{b}", "hl": "en-US", "gl": "US", "ceid": "US:en"}, timeout=40)
    if r.status_code != 200:
        return f"HTTP {r.status_code} {r.text[:150]}"
    items = ET.fromstring(r.content).findall(".//item")
    return f"{len(items)} items; {[(i.findtext('pubDate'), i.findtext('title')[:60]) for i in items[:3]]}"


for a, b in (("2018-03-01", "2018-03-08"), ("2020-06-01", "2020-06-08"), ("2023-05-20", "2023-05-27"), ("2025-03-01", "2025-03-08")):
    for q in ("NVIDIA", "NVIDIA stock", "NVDA"):
        try:
            print(f"[gnews {q} {a}] {gnews(q, a, b)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[gnews {q} {a}] ERROR {e}", flush=True)
        time.sleep(2)

A = "https://arctic-shift.photon-reddit.com/api/posts/search"
for sub in ("wallstreetbets", "stocks", "investing", "nvidia"):
    for a, b in (("2018-01-01", "2018-02-01"), ("2020-06-01", "2020-07-01"), ("2023-05-01", "2023-06-01")):
        for i in range(4):
            r = S.get(A, params={"subreddit": sub, "title": "NVDA" if sub != "nvidia" else "stock", "after": a, "before": b, "limit": 100, "sort": "asc"}, timeout=60)
            if r.status_code == 200:
                d = r.json().get("data") or []
                print(f"[reddit {sub} {a}] {len(d)} posts; score/comments={[(p.get('score'), p.get('num_comments')) for p in d[:3]]}", flush=True)
                break
            print(f"[reddit {sub} {a}] try {i}: HTTP {r.status_code}", flush=True)
            time.sleep(5 * (i + 1))
        time.sleep(2)

u = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv"
r = S.get(u, headers={"Range": "bytes=0-1500"}, timeout=60)
print(f"[fnspid head] HTTP {r.status_code}: {r.text[:1500]!r}")
t0 = time.time()
with S.get(u, stream=True, timeout=60) as r:
    n = 0
    for chunk in r.iter_content(1 << 20):
        n += len(chunk)
        if time.time() - t0 > 20:
            break
print(f"[fnspid speed] {n / 1e6 / (time.time() - t0):.0f} MB/s")
