"""Data for the Take-Two (TTWO) / GTA VI study. Runs on GitHub Actions (ttwo-gta.yml), where the API keys live; the
sandbox cannot reach these hosts. Paper research only.

    python fetch.py OUT_DIR

Writes:
  prices_<T>.csv     Tiingo daily OHLC + adjusted OHLC (TTWO, EA, SPY, QQQ, XLC) from 1997
  chain_TTWO.json    Cboe delayed option chain (every listed strike/expiry, bid/ask/IV/greeks), as of the last close
  earnings_TTWO.csv  SEC EDGAR 8-K item 2.02 filing dates (earnings releases), to separate earnings days
  usdcad.csv         Bank of Canada daily USD/CAD
  news_TTWO.jsonl    Google News headlines for Take-Two / GTA 6 over the last 10 days, scored by the word list and
                     Claude Haiku with the same prompt as research/timescale_ladder/news (straddle8_news)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "timescale_ladder" / "news"))

TICKERS = ["TTWO", "EA", "SPY", "QQQ", "XLC"]
CIK = "0000946581"  # Take-Two Interactive


def get(url: str, params: dict | None = None, headers: dict | None = None, tries: int = 5):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
            if r.status_code == 200:
                return r
            print(url, r.status_code, r.text[:200])
        except requests.RequestException as e:
            print(url, type(e).__name__)
        time.sleep(5 * 2 ** i)
    raise RuntimeError(f"failed: {url}")


def tiingo(out: Path) -> None:
    key = os.environ["TIINGO_API_KEY"]
    for t in TICKERS:
        js = get(f"https://api.tiingo.com/tiingo/daily/{t.lower()}/prices",
                 {"startDate": "1997-01-01", "token": key, "format": "json"}).json()
        df = pd.DataFrame(js)
        df["date"] = df["date"].str[:10]
        df.to_csv(out / f"prices_{t}.csv", index=False)
        print(t, len(df), df["date"].iloc[0], df["date"].iloc[-1])
        time.sleep(2)


def cboe(out: Path) -> None:
    r = get("https://cdn.cboe.com/api/global/delayed_quotes/options/TTWO.json", headers={"User-Agent": "Mozilla/5.0"})
    (out / "chain_TTWO.json").write_bytes(r.content)
    js = r.json()
    print("chain", js.get("timestamp"), len(js["data"]["options"]), "options")


def edgar(out: Path) -> None:
    ua = {"User-Agent": os.environ.get("SEC_USER_AGENT") or "experiment-1 research"}
    js = get(f"https://data.sec.gov/submissions/CIK{CIK}.json", headers=ua).json()
    frames = [pd.DataFrame(js["filings"]["recent"])]
    for f in js["filings"].get("files", []):
        frames.append(pd.DataFrame(get(f"https://data.sec.gov/submissions/{f['name']}", headers=ua).json()))
        time.sleep(0.5)
    df = pd.concat(frames)
    df = df[(df["form"] == "8-K") & df["items"].astype(str).str.contains("2.02")]
    df[["filingDate", "acceptanceDateTime", "items"]].sort_values("filingDate").to_csv(out / "earnings_TTWO.csv", index=False)
    print("earnings 8-Ks", len(df))


def fx(out: Path) -> None:
    r = get("https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json", {"start_date": "2017-01-01"})
    obs = r.json()["observations"]
    pd.DataFrame([{"date": o["d"], "usdcad": float(o["FXUSDCAD"]["v"])} for o in obs]).to_csv(out / "usdcad.csv", index=False)
    print("fx", len(obs))


def news(out: Path) -> None:
    import collect  # research/timescale_ladder/news/collect.py: same feed, word list and Haiku prompt

    s = requests.Session()
    s.headers["User-Agent"] = collect.UA
    end = pd.Timestamp.now(tz="America/New_York").normalize() + pd.Timedelta(days=1)
    rows, seen = [], set()
    for q in ['"Take-Two"', '"GTA 6"', '"GTA VI"', "TTWO"]:
        for d in range(10):
            a, b = (end - pd.Timedelta(days=d + 1)).strftime("%Y-%m-%d"), (end - pd.Timedelta(days=d)).strftime("%Y-%m-%d")
            for it in collect.rss(s, q, a, b):
                if it["title"] not in seen:
                    seen.add(it["title"])
                    rows.append({**it, "query": q, "day": a})
            time.sleep(1.5)
    df = pd.DataFrame(rows)
    df.to_csv(out / "news_TTWO_raw.csv", index=False)
    print("headlines", len(df))
    client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
    with open(out / "news_TTWO.jsonl", "w") as f:
        for day, g in df.groupby("day"):
            titles = [collect.mask(t, "Take-Two Interactive Software", "TTWO") for t in g["title"]][:60]
            rec = {"day": day, "n": len(g), "words": collect.words(titles)}
            if client:
                try:
                    rec["haiku"], rec["cost"] = collect.haiku(client, titles)
                except Exception as e:  # noqa: BLE001
                    rec["haiku_error"] = str(e)[:200]
            f.write(json.dumps(rec) + "\n")
            print(rec)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for step in (tiingo, cboe, edgar, fx, news):
        try:
            step(out)
        except Exception as e:  # noqa: BLE001  keep the rest of the data
            print("STEP FAILED", step.__name__, repr(e))


if __name__ == "__main__":
    main()
