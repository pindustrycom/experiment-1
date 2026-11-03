"""Fetch one day's Tiingo IEX 1-minute bars for a ticker list into DATA_DIR/<TICKER>/<YYYY-MM>.csv (oracle.py's input)."""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

day, out, tickers = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split(",")
key, s = os.environ["TIINGO_API_KEY"], requests.Session()
for t in tickers:
    r = s.get(f"https://api.tiingo.com/iex/{t.lower()}/prices", timeout=60, params={
        "startDate": day, "endDate": day, "resampleFreq": "1min", "columns": "open,high,low,close,volume", "token": key})
    if r.status_code != 200 or not r.text.startswith("["):
        print(t, r.status_code, r.text[:120]); continue
    d = pd.DataFrame(r.json())
    if d.empty:
        print(t, "no bars"); continue
    (out / t).mkdir(parents=True, exist_ok=True)
    d[["date", "open", "high", "low", "close", "volume"]].to_csv(out / t / f"{day[:7]}.csv", index=False)
    print(t, len(d), "bars")
    time.sleep(1.5)
