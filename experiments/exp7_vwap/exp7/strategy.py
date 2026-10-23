"""The reel's rule: on an N-minute chart, trade when the 9-period EMA of closes crosses the session VWAP (anchored at
09:30). A cross is seen at a bar's close and traded at the next bar's open, so nothing uses a price before it exists.
Everything is flat by the last bar's close, and no new trade starts in the final 15 minutes.

Variants
- flip: always in after the first cross, reversing on every cross (long above VWAP, short below).
- long: flip, long side only (flat below VWAP).
- first: one trade a day, the first cross from 10:00 on, held to the opposite cross or the close.

Daily stop (stop=, a fraction): once the day's result (closed trades plus the open one at the bar's low, or high if short)
reaches -stop, the trade is closed at that level (or the bar's open, if it opened past it) and the rule is flat for the rest
of the day. It starts again the next morning."""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

LATE, OPEN_WAIT = time(15, 45), time(10, 0)


def resample(d1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 1:
        d = d1.copy()
    else:
        d = d1.groupby(pd.Grouper(freq=f"{minutes}min", origin="start_day", offset="30min")).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).dropna()
    d["day"] = d.index.strftime("%Y-%m-%d")
    return d


def indicators(d: pd.DataFrame, span: int = 9) -> pd.DataFrame:
    d = d.copy()
    tp = (d.high + d.low + d.close) / 3
    cum_v = d.volume.groupby(d.day).cumsum().replace(0, np.nan)
    d["vwap"] = ((tp * d.volume).groupby(d.day).cumsum() / cum_v).fillna(tp)
    d["ema"] = d.close.ewm(span=span, adjust=False).mean()
    return d


def _trade(day, side, entry, exit_time, px) -> dict:
    return {"day": day, "side": side, "entry_time": entry[0], "entry": entry[1], "exit_time": exit_time, "exit": px, "ret": side * (px / entry[1] - 1)}


def trades(d: pd.DataFrame, variant: str, stop: float | None = None) -> pd.DataFrame:
    """One row per trade: day, side, entry and exit time and price, gross return (before costs)."""
    out = []
    for day, g in d.groupby("day"):
        side = np.sign((g.ema - g.vwap).to_numpy())
        o, c, t = g.open.to_numpy(), g.close.to_numpy(), g.index
        lo, hi = g.low.to_numpy(), g.high.to_numpy()
        pos, entry, done, booked, halted = 0, None, False, 0.0, False

        for k in range(1, len(g)):
            if stop and pos != 0:
                level = entry[1] * (1 - pos * (stop + booked))      # the price where the day is down `stop`
                if (lo[k] <= level) if pos > 0 else (hi[k] >= level):
                    px = min(o[k], level) if pos > 0 else max(o[k], level)
                    out.append(_trade(day, pos, entry, t[k], px))
                    pos, entry, halted = 0, None, True
            if k == len(g) - 1:
                break
            if halted:
                continue
            if side[k] == 0 or side[k - 1] == 0 or side[k] == side[k - 1] or t[k + 1].time() >= LATE:
                continue
            want = int(side[k])
            if variant == "long" and want < 0:
                want = 0
            if variant == "first":
                if pos == 0 and (done or t[k + 1].time() < OPEN_WAIT):
                    continue
                if pos != 0:
                    want = 0
            if want == pos:
                continue
            if pos != 0:
                out.append(_trade(day, pos, entry, t[k + 1], o[k + 1]))
                booked += out[-1]["ret"]
                done = True
            pos, entry = want, ((t[k + 1], o[k + 1]) if want else None)
        if pos != 0:
            out.append(_trade(day, pos, entry, t[-1], c[-1]))
    return pd.DataFrame(out, columns=["day", "side", "entry_time", "entry", "exit_time", "exit", "ret"])


def daily(tr: pd.DataFrame, days: pd.Index, cost_bp: float) -> pd.Series:
    """Day return with the whole pot in each trade (1x), cost_bp per side. Days without a trade are 0."""
    net = tr.ret - 2 * cost_bp / 1e4
    return net.groupby(tr.day).sum().reindex(days, fill_value=0.0)
