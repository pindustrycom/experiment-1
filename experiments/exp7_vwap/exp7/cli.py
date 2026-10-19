"""python -m exp7 backtest --data DIR --out DIR [--tickers SPY,QQQ] [--start 2019-01] [--minutes-budget 300]"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from . import backtest, bars


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="exp7")
    p.add_argument("cmd", choices=["backtest"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tickers", default="SPY,QQQ")
    p.add_argument("--start", default="2019-01")
    p.add_argument("--end", default=None)
    p.add_argument("--minutes-budget", type=float, default=300, help="stop fetching after this long and test what is there")
    a = p.parse_args()
    tickers = a.tickers.split(",")
    end = a.end or time.strftime("%Y-%m")
    report = {"tickers": tickers}
    key = os.environ.get("TIINGO_API_KEY")
    if key:
        report["fetch"] = bars.collect(a.data, tickers, a.start, end, key, time.time() + 60 * a.minutes_budget)
    loaded = {t: bars.load(a.data, t) for t in tickers if (a.data / t).exists() and any((a.data / t).glob("*.csv"))}
    report["coverage"] = {t: {"days": int(d.index.normalize().nunique()), "first": str(d.index[0].date()), "last": str(d.index[-1].date())} for t, d in loaded.items()}
    res = backtest.run(loaded)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=1))
    (a.out / "table.md").write_text(backtest.table(res) + "\n")
    json.dump(report, sys.stdout, indent=1)
    print("\n" + backtest.table({k: v for k, v in res.items() if "cost0.3bp" in k or "hold" in k}))
