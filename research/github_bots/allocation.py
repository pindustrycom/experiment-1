"""Monthly allocation rules that many GitHub repos implement (QuantConnect Lean examples, backtrader and vectorbt
notebooks), run on daily adjusted closes. A switch is decided on the last close of a month and traded at the next close.

faber_10m: hold SPY while its month-end close is above its 10-month average of month-end closes, else T-bills (BIL).
gem:       Antonacci's global equities momentum: if SPY's 12-month return beats T-bills, hold whichever of SPY and EFA
           (international stocks) did better over 12 months, otherwise hold AGG (bonds).
"""
from __future__ import annotations

import pandas as pd


def _run(px: pd.DataFrame, pick: pd.Series, cost_bp: float) -> pd.Series:
    """pick: month-end date -> ticker held from the next session. Returns daily returns."""
    rets = px.pct_change()
    hold = pick.reindex(px.index).ffill().shift(1)
    r = pd.Series(0.0, index=px.index)
    for t in hold.dropna().unique():
        m = hold == t
        r[m] = rets[t][m]
    switch = hold.ne(hold.shift()) & hold.notna() & hold.shift().notna()
    return (r - switch * 2 * cost_bp / 1e4)[hold.notna()]


def month_ends(px: pd.DataFrame) -> pd.DataFrame:
    return px.groupby(px.index.to_period("M")).tail(1)


def faber_10m(px: pd.DataFrame, cost_bp: float, asset: str = "SPY") -> pd.Series:
    me = month_ends(px)
    pick = pd.Series(["BIL"] * len(me), index=me.index).where(me[asset] <= me[asset].rolling(10).mean(), asset)
    pick[me[asset].rolling(10).mean().isna()] = None
    return _run(px, pick.dropna(), cost_bp)


def gem(px: pd.DataFrame, cost_bp: float) -> pd.Series:
    me = month_ends(px)
    m12 = me / me.shift(12) - 1
    pick = pd.Series("AGG", index=me.index)
    risk_on = m12["SPY"] > m12["BIL"]
    pick[risk_on] = (m12["SPY"] >= m12["EFA"]).map({True: "SPY", False: "EFA"})[risk_on]
    return _run(px, pick[m12["SPY"].notna()], cost_bp)
