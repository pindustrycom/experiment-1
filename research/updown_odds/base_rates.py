"""How the odds of 'up' change with the holding period, for SPY and for single S&P 500 members.

P(SPY up), P(member up) and P(member beats SPY) over 1 day to 5 years, overlapping windows that start on every day a
stock was a member (point-in-time; a stock that leaves keeps its last price).

    python base_rates.py PANEL.pkl OUT_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = {"1 day": 1, "1 week": 5, "1 month": 21, "3 months": 63, "1 year": 252, "3 years": 756, "5 years": 1260}


def main(panel: Path, out: Path) -> None:
    P = pd.read_pickle(panel)
    adjf, mem = P["adj"].ffill(), P["member"] & P["adj"].notna()
    spy = P["spy"]["adj_close"].reindex(adjf.index).ffill()
    rows = []
    for name, h in H.items():
        s = (spy.shift(-h) / spy - 1).loc["2001":].dropna()
        f = (adjf.shift(-h) / adjf - 1).loc["2001":]
        m = mem.loc[f.index] & f.notna()
        st = f.where(m).stack().dropna()
        ex = f.sub(spy.shift(-h) / spy - 1, axis=0).where(m).stack().dropna()
        rows.append(dict(horizon=name, days=h, spy_up=(s > 0).mean(), spy_avg=s.mean(), spy_median=s.median(),
                         stock_up=(st > 0).mean(), stock_beats_spy=(ex > 0).mean(), stock_avg=st.mean(),
                         stock_median=st.median()))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "base_rates.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
