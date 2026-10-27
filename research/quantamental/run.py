"""Quantamental backtests end to end.

  python run.py DATA OUT [--max-docs N] [--no-text]

DATA holds the papers-data cache (prices/, sp500_current.csv, sp500_changes.csv). SEC company facts, the yearly
fundamentals table and the 10-K word counts are cached in DATA/sec/ so a rerun only fetches what is new.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

import backtest
import sec
import text

log = logging.getLogger("quantamental")

COLS = [("ann", "Per year", "{:+.1%}"), ("vol", "Vol", "{:.1%}"), ("sharpe", "Sharpe", "{:.2f}"), ("max_dd", "Worst drop", "{:.0%}"),
        ("beta", "Beta", "{:.2f}"), ("alpha_ann", "Alpha vs SPY", "{:+.1%}"), ("alpha_t", "t", "{:.1f}"),
        ("vs_spy_ann", "Minus SPY", "{:+.1%}"), ("ff_alpha_ann", "FF6 alpha", "{:+.1%}"), ("ff_alpha_t", "FF6 t", "{:.1f}"),
        ("sharpe_2010-2017", "Sharpe 2010-17", "{:.2f}"), ("sharpe_2018-2026", "Sharpe 2018-26", "{:.2f}"),
        ("dsr_here", "Deflated Sharpe (this study)", "{:.2f}"), ("dsr_project", "Deflated Sharpe (+300 project trials)", "{:.2f}")]


def table(res: dict) -> str:
    head = "| Strategy | " + " | ".join(c[1] for c in COLS) + " |\n|---|" + "---|" * len(COLS) + "\n"
    lines = []
    for k, r in res.items():
        cells = []
        for key, _, fmt in COLS:
            v = r.get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) and not pd.isna(v) else "")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def report(rep: dict) -> str:
    u = rep["universe"]
    s = ["# Quantamental backtests\n",
         (f"Point-in-time S&P 500 members, {u['months']} months, about {u['avg_names']:.0f} names a month with a 10-K on file. "
          "Monthly rebalance, 10 bp per side, 0.5%/yr borrow on shorts. Long-only books are compared with SPY; long-short books "
          "are market-neutral in dollars. FF6 = Fama-French five factors plus momentum.\n"),
         table(rep["results"]),
         "\n## Information coefficient and the fundamental law (IR = IC x sqrt(breadth))\n",
         "| Signal | Mean IC | IC t | Names a month | IR the law predicts if bets were independent |\n|---|---|---|---|---|"]
    for k, v in rep["ic"].items():
        s.append(f"| {k} | {v['ic_mean']:.3f} | {v['ic_t']:.1f} | {v['names']:.0f} | {v['law_ir']:.2f} |")
    if rep.get("filing_reaction"):
        s.append("\n## 10-K tone and the market's reaction (filing day -1 to +3, minus SPY, Fama-MacBeth by month)\n")
        s.append("| Measure | Return per 1 sd more negative words | t | Months |\n|---|---|---|---|")
        for k, v in rep["filing_reaction"].items():
            if isinstance(v, dict):
                s.append(f"| {k} | {v['slope_pct_per_sd']:+.3f}% | {v['t']:.1f} | {v['months']} |")
    s.append("\n## Universe\n\n```\n" + json.dumps(u, indent=1) + "\n```\n")
    return "\n".join(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-docs", type=int, default=20000)
    ap.add_argument("--no-text", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cur = pd.read_csv(a.data / "sp500_current.csv")
    changes = pd.read_csv(a.data / "sp500_changes.csv", parse_dates=["date"])
    member = backtest.membership(cur, changes, pd.date_range(backtest.START, backtest.END, freq="ME"))
    m = sec.ticker_ciks()
    ciks = {t: m[t] for t in member.columns if t in m}
    log.info("universe: %d tickers ever members, %d with a SEC CIK", member.shape[1], len(ciks))
    secdir = a.data / "sec"
    secdir.mkdir(parents=True, exist_ok=True)
    sec.fundamentals(ciks, secdir / "facts", secdir / "fundamentals.csv")
    tc = None
    if not a.no_text:
        tc = secdir / "text_counts.csv"
        text.collect(sorted(set(ciks.values())), tc, max_docs=a.max_docs)
    rep = backtest.run(a.data, secdir / "fundamentals.csv", tc, a.out)
    rep["universe"]["tickers_with_cik"] = len(ciks)
    (a.out / "README.md").write_text(report(rep))
    (a.out / "results.json").write_text(json.dumps(rep, indent=1, default=str))
    print(report(rep))


if __name__ == "__main__":
    main()
