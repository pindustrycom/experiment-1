"""Which historical sources for NVDA sentiment (2018-2025) can a GitHub runner reach, and how deep do they go?
Prints one line per source; never prints keys."""
import json
import os
import sys

import requests

UA = os.environ.get("SEC_USER_AGENT") or "exp3-nvda research"
S = requests.Session()
S.headers["User-Agent"] = UA


def probe(name, url, params=None, show=lambda j: str(j)[:300], headers=None):
    try:
        r = S.get(url, params=params or {}, timeout=40, headers=headers or {})
        try:
            body = show(r.json())
        except Exception:  # noqa: BLE001
            body = r.text[:300].replace("\n", " ")
        print(f"[{name}] HTTP {r.status_code}: {body}")
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] ERROR {type(e).__name__}: {str(e)[:200]}")
    sys.stdout.flush()


fh = os.environ.get("FINNHUB_API_KEY", "")
tg = os.environ.get("TIINGO_API_KEY", "")

for sub in ("wallstreetbets", "stocks", "investing", "nvidia"):
    for y in ("2018", "2021", "2025"):
        probe(f"reddit-arctic {sub} {y}", "https://arctic-shift.photon-reddit.com/api/posts/search",
              {"subreddit": sub, "query": "NVDA", "after": f"{y}-03-01", "before": f"{y}-03-08", "limit": 100},
              lambda j: f"{len(j.get('data') or [])} posts; first={[(p.get('title') or '')[:60] for p in (j.get('data') or [])[:2]]}")
probe("reddit-arctic comments wsb 2019", "https://arctic-shift.photon-reddit.com/api/comments/search",
      {"subreddit": "wallstreetbets", "body": "NVDA", "after": "2019-03-01", "before": "2019-03-03", "limit": 100},
      lambda j: f"{len(j.get('data') or [])} comments")
probe("hf FNSPID", "https://huggingface.co/api/datasets/Zihan1004/FNSPID", None,
      lambda j: str([s.get('rfilename') for s in j.get('siblings', [])])[:400])
for y in ("2018", "2022", "2025"):
    probe(f"gdelt doc {y}", "https://api.gdeltproject.org/api/v2/doc/doc",
          {"query": "nvidia", "mode": "artlist", "format": "json", "startdatetime": f"{y}0301000000", "enddatetime": f"{y}0303000000", "maxrecords": 50},
          lambda j: f"{len(j.get('articles') or [])} articles; {[a.get('title','')[:60] for a in (j.get('articles') or [])[:2]]}")
probe("gdelt timeline tone 2018", "https://api.gdeltproject.org/api/v2/doc/doc",
      {"query": "nvidia", "mode": "timelinetone", "format": "json", "startdatetime": "20180101000000", "enddatetime": "20180201000000"},
      lambda j: str(j)[:300])
probe("tiingo news 2018", "https://api.tiingo.com/tiingo/news",
      {"tickers": "nvda", "startDate": "2018-03-01", "endDate": "2018-03-08", "limit": 100, "token": tg},
      lambda j: f"{len(j) if isinstance(j, list) else j}" if not isinstance(j, list) else f"{len(j)} stories; {[a.get('title','')[:60] for a in j[:2]]}")
probe("finnhub news 2018", "https://finnhub.io/api/v1/company-news", {"symbol": "NVDA", "from": "2018-03-01", "to": "2018-03-08", "token": fh},
      lambda j: f"{len(j)} stories" if isinstance(j, list) else str(j)[:200])
probe("finnhub news 2024", "https://finnhub.io/api/v1/company-news", {"symbol": "NVDA", "from": "2024-03-01", "to": "2024-03-08", "token": fh},
      lambda j: f"{len(j)} stories" if isinstance(j, list) else str(j)[:200])
probe("finnhub upgrade-downgrade", "https://finnhub.io/api/v1/stock/upgrade-downgrade", {"symbol": "NVDA", "from": "2018-01-01", "to": "2025-12-31", "token": fh},
      lambda j: f"{len(j)} actions; oldest={j[-1] if j else None}" if isinstance(j, list) else str(j)[:200])
probe("finnhub recommendation", "https://finnhub.io/api/v1/stock/recommendation", {"symbol": "NVDA", "token": fh},
      lambda j: f"{len(j)} months; oldest={j[-1]['period'] if j else None}" if isinstance(j, list) else str(j)[:200])
probe("finnhub social", "https://finnhub.io/api/v1/stock/social-sentiment", {"symbol": "NVDA", "from": "2021-01-01", "to": "2021-02-01", "token": fh},
      lambda j: str(j)[:200])
probe("sec shares outstanding", "https://data.sec.gov/api/xbrl/companyconcept/CIK0001045810/dei/EntityCommonStockSharesOutstanding.json", None,
      lambda j: f"{len(j['units']['shares'])} values; first={j['units']['shares'][0]}")
probe("stocktwits", "https://api.stocktwits.com/api/2/streams/symbol/NVDA.json", {"max": 1000000000},
      lambda j: str(j)[:200])
s2 = requests.Session()
s2.headers["User-Agent"] = "Mozilla/5.0"
try:
    s2.get("https://fc.yahoo.com", timeout=20)
    crumb = s2.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=20).text
    r = s2.get("https://query2.finance.yahoo.com/v10/finance/quoteSummary/NVDA", params={"modules": "upgradeDowngradeHistory", "crumb": crumb}, timeout=30)
    h = r.json()["quoteSummary"]["result"][0]["upgradeDowngradeHistory"]["history"]
    print(f"[yahoo upgrades] {len(h)} actions; oldest={h[-1]}; newest={h[0]}")
except Exception as e:  # noqa: BLE001
    print(f"[yahoo upgrades] ERROR {type(e).__name__}: {str(e)[:200]}")
print(json.dumps({"done": True}))
