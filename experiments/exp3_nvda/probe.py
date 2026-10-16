"""Which historical sources for NVDA sentiment (2018-2025) can a GitHub runner reach, and how deep do they go?
Prints one line per source; never prints keys."""
import time

import requests

S = requests.Session()
S.headers["User-Agent"] = "exp3-nvda research"


def probe(name, url, params=None, show=lambda j: str(j)[:300], pause=0.0):
    time.sleep(pause)
    try:
        r = S.get(url, params=params or {}, timeout=60)
        try:
            body = show(r.json()) if r.status_code == 200 else r.text[:300].replace("\n", " ")
        except Exception:  # noqa: BLE001
            body = r.text[:300].replace("\n", " ")
        print(f"[{name}] HTTP {r.status_code}: {body}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)


A = "https://arctic-shift.photon-reddit.com/api/posts/search"
n = lambda j: f"{len(j.get('data') or [])} posts; {[(p.get('title') or '')[:50] for p in (j.get('data') or [])[:2]]}"  # noqa: E731
for y in ("2018", "2020", "2022"):
    probe(f"arctic wsb query {y}", A, {"subreddit": "wallstreetbets", "query": "NVDA", "after": f"{y}-03-01", "before": f"{y}-03-08", "limit": 100}, n)
    probe(f"arctic wsb title {y}", A, {"subreddit": "wallstreetbets", "title": "NVDA", "after": f"{y}-03-01", "before": f"{y}-03-08", "limit": 100}, n)
    probe(f"arctic wsb plain {y}", A, {"subreddit": "wallstreetbets", "after": f"{y}-03-01", "before": f"{y}-03-02", "limit": 100}, n)
probe("arctic nvidia title 2019", A, {"subreddit": "nvidia", "title": "stock", "after": "2019-01-01", "before": "2019-03-01", "limit": 100}, n)
probe("arctic stocks title 2019", A, {"subreddit": "stocks", "title": "NVDA", "after": "2019-01-01", "before": "2019-03-01", "limit": 100}, n)

DS = "https://datasets-server.huggingface.co"
probe("hf splits", f"{DS}/splits", {"dataset": "Zihan1004/FNSPID"})
probe("hf size", f"{DS}/size", {"dataset": "Zihan1004/FNSPID"}, lambda j: str(j)[:600])
probe("hf filter NVDA", f"{DS}/filter", {"dataset": "Zihan1004/FNSPID", "config": "default", "split": "train",
                                          "where": "\"Stock_symbol\"='NVDA'", "length": 3},
      lambda j: f"total={j.get('num_rows_total')}; rows={[r['row'] for r in j.get('rows', [])][:3]}"[:800])
probe("hf tree", "https://huggingface.co/api/datasets/Zihan1004/FNSPID/tree/main/Stock_news", None,
      lambda j: str([(f.get('path'), f.get('size')) for f in j])[:400])

G = "https://api.gdeltproject.org/api/v2/doc/doc"
for y in ("2018", "2020", "2023", "2025"):
    probe(f"gdelt artlist {y}", G, {"query": "nvidia", "mode": "artlist", "format": "json", "sourcelang": "english",
                                    "startdatetime": f"{y}0301000000", "enddatetime": f"{y}0308000000", "maxrecords": 250},
          lambda j: f"{len(j.get('articles') or [])} articles; {[a.get('title', '')[:50] for a in (j.get('articles') or [])[:2]]}", pause=7)
probe("gdelt tone 2018", G, {"query": "nvidia", "mode": "timelinetone", "format": "json", "startdatetime": "20180101000000", "enddatetime": "20180115000000"},
      lambda j: str(j)[:300], pause=7)
