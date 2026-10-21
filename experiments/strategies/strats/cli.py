"""python -m strats live --data data --state state/state.json --out site

Fetches prices, the S&P 500 list, earnings and insider filings, new 13Fs and Pelosi PTRs, moves every pot to the latest
close, replays the pots that have a history, and writes the pages. Prints a JSON report."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from . import copy, copy_data, data, live, rules, site


def run(data_dir: Path, state_f: Path, out: Path, fetch: bool = True) -> dict:
    rep: dict = {}
    members = data.sp500_members(data_dir) if fetch else (sorted(pd.read_csv(data_dir / "sp500.csv")["ticker"]) if (data_dir / "sp500.csv").exists() else [])
    key = os.environ.get("FINNHUB_API_KEY", "")
    if fetch and (data_dir / "copy").exists():
        rep["copy_data"] = copy_data.update(data_dir, pd.Timestamp.today().normalize())
    cp = copy.load(data_dir) if (data_dir / "copy").exists() else None
    tickers = sorted(set(rules.TREND_ETFS) | {m.replace(".", "-") for m in members} | (copy.tickers(cp) if copy.has_data(cp) else set()))
    if fetch:
        rep["prices"] = data.update_prices(data_dir, tickers)
        rep["prices"]["failed"] = rep["prices"]["failed"][:30]
    px = data.load_prices(data_dir, tickers)
    spy = px["SPY"].dropna()
    d = spy.index[-1]
    px = px.loc[px.index.isin(spy.index)]
    rep["last_close"] = str(d.date())
    members = [m.replace(".", "-") for m in members]
    if fetch and key:
        for name, fn in (("earnings", lambda: data.update_earnings(data_dir, key, d)),
                         ("insiders", lambda: data.update_insiders(data_dir, key, members, d))):
            try:
                rep[name] = fn()
            except Exception as ex:  # noqa: BLE001 - a pot without fresh events keeps its holdings
                rep[name] = f"failed: {type(ex).__name__}: {str(ex)[:200]}"
    elif fetch:
        rep["finnhub"] = "FINNHUB_API_KEY not set: earnings and insider pots get no new events"
    read = lambda f: pd.read_csv(f) if f.exists() else pd.DataFrame()  # noqa: E731
    earnings, insiders = read(data_dir / "earnings.csv"), read(data_dir / "insiders.csv")
    if not earnings.empty:
        earnings["symbol"] = earnings["symbol"].astype(str).str.replace(".", "-")
    if not insiders.empty:
        insiders["symbol"] = insiders["symbol"].astype(str).str.replace(".", "-")
    state = live.load_state(state_f)
    rep["step"] = live.step(state, px, d, members, earnings, insiders, cp)
    live.save_state(state_f, state)
    rp = None
    try:
        rp = live.replay_summary(live.replay(px, members, cp=cp))
        rep["replay"] = {k: {m: round(v, 4) for m, v in rp[k].items() if m != "years"} for k in rp if k in rules.STRATEGIES or k == "SPY"}
    except Exception as ex:  # noqa: BLE001 - the pages still build
        rep["replay"] = f"failed: {type(ex).__name__}: {str(ex)[:200]}"
    rep["site"] = site.write(out, state, rp, {"members": len(members), "priced": int(px.loc[d].notna().sum())})
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("live")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--state", type=Path, default=Path("state/state.json"))
    p.add_argument("--out", type=Path, default=Path("site"))
    p.add_argument("--no-network", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    print(json.dumps(run(a.data, a.state, a.out, fetch=not a.no_network), indent=1, default=str))
