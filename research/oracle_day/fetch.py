"""Fetch START..END Tiingo IEX 1-minute bars for a ticker list into DATA_DIR/<TICKER>/bars.csv (oracle.py's input)."""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

start, end, out, tickers = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4].split(",")
key, s = os.environ["TIINGO_API_KEY"], requests.Session()
for t in tickers:
    r = s.get(f"https://api.tiingo.com/iex/{t.lower()}/prices", timeout=60, params={
        "startDate": start, "endDate": end, "resampleFreq": "1min", "columns": "open,high,low,close,volume", "token": key})
    if r.status_code != 200 or not r.text.startswith("["):
        print(t, r.status_code, r.text[:120]); continue
    d = pd.DataFrame(r.json())
    if d.empty:
        print(t, "no bars"); continue
    (out / t).mkdir(parents=True, exist_ok=True)
    d[["date", "open", "high", "low", "close", "volume"]].to_csv(out / t / "bars.csv", index=False)
    print(t, len(d), "bars")
    time.sleep(1.5)
