"""Deribit BTC option-chain snapshots with real bid/ask for the tradeable volatility-trade test.

Tardis.dev publishes the first day of every month free (no key). For each month from 2021-04 this streams that day's
Deribit options_chain file and keeps, per BTC option, its last quote at or before 00:10 UTC, then stops reading (the
file is in time order, so only the first minutes are downloaded). Also saves Deribit's official BTC delivery
(settlement) prices. Output: options_1st.csv.gz and delivery.csv in OUT_DIR."""
from __future__ import annotations

import csv
import gzip
import io
import os
import sys
import time

import pandas as pd
import requests

URL = "https://datasets.tardis.dev/v1/deribit/options_chain/{y}/{m:02d}/01/OPTIONS.csv.gz"
KEEP = ["symbol", "timestamp", "type", "strike_price", "expiration", "open_interest", "bid_price", "bid_amount", "bid_iv",
        "ask_price", "ask_amount", "ask_iv", "mark_price", "mark_iv", "underlying_index", "underlying_price", "delta",
        "gamma", "vega", "theta"]


def snapshot_retry(day: pd.Timestamp) -> pd.DataFrame:
    """Tardis rate-limits anonymous downloads (HTTP 429): back off and retry."""
    for attempt in range(6):
        try:
            return snapshot(day)
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 429 or attempt == 5:
                raise
            time.sleep(60 * (attempt + 1))
    raise RuntimeError("unreachable")


def snapshot(day: pd.Timestamp, until_min: int = 10) -> pd.DataFrame:
    cut = int((day + pd.Timedelta(minutes=until_min)).timestamp() * 1e6)
    stop = cut + 5 * 60 * 1_000_000
    last = {}
    with requests.get(URL.format(y=day.year, m=day.month), stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        text = io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw), encoding="utf-8")
        rd = csv.reader(text)
        head = next(rd)
        ix = {k: head.index(k) for k in KEEP + ["local_timestamp"]}
        for row in rd:
            lt = int(row[ix["local_timestamp"]])
            if lt > stop:
                break
            sym = row[ix["symbol"]]
            if not sym.startswith("BTC-") or int(row[ix["timestamp"]]) > cut:
                continue
            last[sym] = [row[ix[k]] for k in KEEP]
    d = pd.DataFrame(list(last.values()), columns=KEEP)
    d.insert(0, "date", day.date())
    return d


def delivery() -> pd.DataFrame:
    rows, off = [], 0
    while True:
        js = requests.get("https://www.deribit.com/api/v2/public/get_delivery_prices",
                          params={"index_name": "btc_usd", "offset": off, "count": 1000}, timeout=60).json()["result"]
        rows += js["data"]
        off += 1000
        if off >= js["records_total"] or not js["data"]:
            break
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = sys.argv[1]
    prev = sys.argv[2] if len(sys.argv) > 2 else None      # earlier options_1st.csv.gz: months in it are not fetched again
    days = pd.date_range("2021-04-01", pd.Timestamp.now("UTC").tz_localize(None).normalize(), freq="MS")
    frames, have = [], set()
    if prev and os.path.exists(prev):
        old = pd.read_csv(prev)
        frames.append(old)
        have = set(pd.to_datetime(old.date).dt.normalize())
        print("kept", len(have), "months from the earlier file")
    for day in days:
        if day in have:
            continue
        try:
            s = snapshot_retry(day)
            frames.append(s)
            print(day.date(), len(s), "options", flush=True)
            pd.concat(frames).to_csv(f"{out}/options_1st.csv.gz", index=False)   # save as we go (rate limits are slow)
            time.sleep(5)
        except Exception as e:  # noqa: BLE001  (a missing month is logged and skipped)
            print(day.date(), "failed:", e, flush=True)
    pd.concat(frames).to_csv(f"{out}/options_1st.csv.gz", index=False)
    try:
        dv = delivery()
        dv.to_csv(f"{out}/delivery.csv", index=False)
        print("delivery prices", len(dv))
    except Exception as e:  # noqa: BLE001
        print("delivery failed:", e)
