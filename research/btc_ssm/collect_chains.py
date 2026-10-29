"""Daily snapshot of Deribit's BTC and ETH option chains with bid/ask, from Deribit's free public API (one request per
coin per run), saved in the same columns as the Tardis first-of-month files so vrp_variants.entry() reads both.

    python collect_chains.py OUT_DIR [PREVIOUS_FILE ...]

Writes OUT_DIR/chains_<coin>_<YYYY-MM>.csv.gz for the current month, adding this run's snapshot to the previous file
of that month when one is given (a run repeated on the same day replaces that day's rows). deribit-chains.yml runs it
at 00:10 UTC every day, the entry time of the vol trade. Greeks are not in the summary feed: delta comes from Black-76
at the mark IV and the expiry's forward (Deribit quotes options on the forward, so the underlying price is that forward).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
UA = {"User-Agent": "experiment-1 research (https://github.com/pindustrycom/experiment-1)"}
COINS = ("BTC", "ETH")
COLS = ["date", "symbol", "timestamp", "type", "strike_price", "expiration", "open_interest", "bid_price", "ask_price",
        "mark_price", "mark_iv", "underlying_index", "underlying_price", "delta", "volume"]


def fetch(coin: str) -> list[dict]:
    for attempt in range(4):
        try:
            r = requests.get(URL, params={"currency": coin, "kind": "option"}, headers=UA, timeout=60)
            r.raise_for_status()
            return r.json()["result"]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(30 * (attempt + 1))
    return []


def parse(rows: list[dict], now: pd.Timestamp) -> pd.DataFrame:
    """Instrument names look like BTC-27DEC26-100000-C; expiries settle at 08:00 UTC."""
    out = []
    for x in rows:
        parts = x["instrument_name"].split("-")
        if len(parts) != 4:
            continue
        exp = pd.to_datetime(parts[1], format="%d%b%y") + pd.Timedelta(hours=8)
        out.append({"symbol": x["instrument_name"], "timestamp": int(x.get("creation_timestamp", now.value // 10**6)) * 1000,
                    "type": "call" if parts[3] == "C" else "put", "strike_price": float(parts[2]),
                    "expiration": int(exp.value // 1000), "open_interest": x.get("open_interest"),
                    "bid_price": x.get("bid_price"), "ask_price": x.get("ask_price"), "mark_price": x.get("mark_price"),
                    "mark_iv": x.get("mark_iv"), "underlying_index": x.get("underlying_index"),
                    "underlying_price": x.get("underlying_price"), "volume": x.get("volume")})
    d = pd.DataFrame(out)
    if d.empty:
        return pd.DataFrame(columns=COLS)
    d["date"] = now.normalize().strftime("%Y-%m-%d")
    T = (pd.to_datetime(d.expiration, unit="us") - now.tz_localize(None)) / pd.Timedelta(days=365)
    sq = pd.to_numeric(d.mark_iv, errors="coerce") / 100 * np.sqrt(T.clip(lower=1e-6))
    d1 = (np.log(d.underlying_price / d.strike_price) + 0.5 * sq ** 2) / sq
    d["delta"] = np.where(d.type == "call", norm.cdf(d1), norm.cdf(d1) - 1).round(5)
    return d[COLS]


def main(out: str, previous: list[str]) -> None:
    os.makedirs(out, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    for coin in COINS:
        snap = parse(fetch(coin), now)
        name = f"chains_{coin}_{now:%Y-%m}.csv.gz"
        old = next((pd.read_csv(p) for p in previous if os.path.basename(p) == name and os.path.exists(p)), None)
        if old is not None:
            snap = pd.concat([old[old.date != snap.date.iloc[0]] if len(snap) else old, snap])
        snap.to_csv(f"{out}/{name}", index=False)
        print(coin, name, len(snap), "rows,", snap.date.nunique() if len(snap) else 0, "days")
        time.sleep(2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
