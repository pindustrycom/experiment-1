"""Run the carousel's steps 0-6 as one walk-forward process on companies and on markets, and print the report.

  python edge.py DATA OUT

DATA is the folder run.py fills (papers-data prices + sec/ cache); market prices are cached in DATA/markets/.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

import backtest
import markets
import pipeline
import run as runmod

log = logging.getLogger("quantamental.edge")
WATCH = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "WMT"]
COLS = [("ann", "Per year", "{:+.1%}"), ("sharpe", "Sharpe", "{:.2f}"), ("max_dd", "Worst drop", "{:.0%}"), ("beta", "Beta", "{:.2f}"),
        ("alpha_ann", "Alpha vs SPY", "{:+.1%}"), ("alpha_t", "t", "{:.1f}"), ("ff_alpha_ann", "FF6 alpha", "{:+.1%}"),
        ("ff_alpha_t", "FF6 t", "{:.1f}"), ("dsr_project", "Deflated Sharpe", "{:.2f}")]


def section(title: str, res: dict, watch: list[str]) -> str:
    s = [f"\n# {title}\n", "## Steps 1-2: each idea as a number, its IC and breadth\n",
         "| Signal | Camp | Mean IC | IC t | Names a month | Naive law IR (IC x sqrt(12 x names)) | Realized IR of the IC |", "|---|---|---|---|---|---|---|"]
    for k, r in res["ic"].sort_values("ic_t", ascending=False).iterrows():
        s.append(f"| {k} | {r['camp']} | {r['ic']:+.3f} | {r['ic_t']:.1f} | {r['names']:.0f} | {r['law_ir_naive']:.2f} | {r['ir_of_ic']:.2f} |")
    s += ["\n## Step 3: which signals the process chose, month by month (past IC t > 2 only)\n",
          "| Signal | Share of months used | Direction |", "|---|---|---|"]
    for k, r in res["signal_use"].iterrows():
        s.append(f"| {k} | {r['share_of_months_used']:.0%} | {'as named' if r['sign'] > 0 else 'reversed'} |")
    s += ["\n## Steps 5-6: books built from the combined score, out of sample, after costs\n",
          runmod.table(json.loads(res["results"].to_json(orient="index")), COLS)]
    c = res["cards"]
    if len(c):
        fmt = lambda d: "\n".join(f"| {r.ticker} | {r.score:+.2f} | {r.rank_pct:.0%} | {r.exp_alpha_yr:+.1%} | {r.drivers} |" for r in d.itertuples())
        head = "| Name | Score (z) | Rank | Expected alpha / yr | Main drivers (signal z x weight) |\n|---|---|---|---|---|\n"
        s += [f"\n## Edge cards, formed {c['month'].iloc[0].date()}\n", "Top 10\n", head + fmt(c.head(10)), "\nBottom 10\n", head + fmt(c.tail(10))]
        w = c[c["ticker"].isin(watch)]
        if len(w):
            s += ["\nNames you might ask about\n", head + fmt(w)]
    return "\n".join(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)
    F = backtest.french_factors()
    fund = pd.read_csv(a.data / "sec" / "fundamentals.csv", parse_dates=["end", "avail", "float_date"])
    tc = a.data / "sec" / "text_counts.csv"
    P, spy, _ = backtest.build_panel(a.data, fund, pd.read_csv(tc) if tc.exists() else None)
    stocks = pipeline.run(P, pipeline.STOCK_SIGNALS, spy, "2013-01-31", backtest.COST, backtest.BORROW, F)
    Pm, spy_m = markets.panel(a.data / "markets")
    mk = pipeline.run(Pm, markets.SIGNALS, spy_m, "2010-01-31", 0.0003, 0.003 / 12, F)
    rep = section("Companies: S&P 500 stocks, tested 2013-02 to 2026-09", stocks, WATCH)
    rep += "\n" + section(f"Markets: {Pm['ticker'].nunique()} ETFs (sectors, countries, bonds, commodities, REITs, bitcoin), tested 2010-02 to 2026-09",
                          mk, ["SPY", "TLT", "GLD", "BTC-USD", "XLK", "EEM"])
    (a.out / "edge_report.md").write_text(rep)
    for lab, r in (("stocks", stocks), ("markets", mk)):
        pd.DataFrame(r["series"]).to_csv(a.out / f"edge_{lab}_monthly.csv", float_format="%.6g")
        r["cards"].to_csv(a.out / f"edge_{lab}_cards.csv", index=False, float_format="%.4g")
        r["chosen"].to_csv(a.out / f"edge_{lab}_weights.csv", float_format="%.4g")
    print(rep)


if __name__ == "__main__":
    main()
