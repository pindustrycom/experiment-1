"""An event-driven backtester for the daily strategy pots, to check the vectorized replay (live.replay).

The replay assumes each book is bought at the same close its signal was computed from, for 1 bp a side, with fractional
shares. Here every session is two events in time order, and each part only sees what has happened so far:

  open    the broker fills the orders queued at the last close at this session's open (slippage and commission per
          COSTS), and the portfolio's cash and shares change
  close   the data feed releases this close; the portfolio marks its shares; the strategy (the same rules.py functions
          as the live pots) sees prices up to this close only and, when its rebalance is due, sends target weights; the
          portfolio turns them into share orders for the next open

so a signal can never trade on the price it was computed from, and a rule never gets tomorrow's prices to peek at. Only
the trading calendar is known ahead (which session ends a week or month), as it is in real life. Unlike the replay, which
earns the target weights every day (a free daily rebalance), the portfolio holds shares that drift until the next
rebalance. Shorts pay a borrow fee per day held. Dividends are in the adjusted prices. A stock without an open (an old
price cache) is filled at that session's close.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import rules

# Cost settings compared on the page: (label, fill at, commission per side, slippage per side on stocks / ETFs,
# short borrow per year, whole shares and starting cash or None for fractional)
SETTINGS = [
    ("Same close, 1 bp (the replay's assumptions)", "close", 0.0001, (0.0, 0.0), 0.0, None),
    ("Next open, 1 bp", "open", 0.0001, (0.0, 0.0), 0.0, None),
    ("Next open, spread + slippage", "open", 0.0, (0.0005, 0.0002), 0.005, None),
    ("Next open, spread + slippage, whole shares, 1,000 CAD", "open", 0.0, (0.0005, 0.0002), 0.005, 715.0),
]
ETFS = set(rules.TREND_ETFS)


@dataclass
class Event:
    kind: str                 # "open" | "close" | "fill"
    date: pd.Timestamp
    data: dict = field(default_factory=dict)


class Feed:
    """Releases one session at a time. history() is everything the strategy may look at."""

    def __init__(self, close: pd.DataFrame, open_: pd.DataFrame | None):
        self.close, self.open = close, open_
        self.i = -1

    def advance(self) -> pd.Timestamp:
        self.i += 1
        return self.close.index[self.i]

    @property
    def now(self) -> pd.Timestamp:
        return self.close.index[self.i]

    def history(self) -> pd.DataFrame:
        return self.close.iloc[: self.i + 1]

    def close_px(self, t: str) -> float:
        return self.close.iat[self.i, self.close.columns.get_loc(t)] if t in self.close.columns else np.nan

    def open_px(self, t: str) -> float:
        if self.open is not None and t in self.open.columns:
            v = self.open.iat[self.i, self.open.columns.get_loc(t)]
            if pd.notna(v) and v > 0:
                return float(v)
        return self.close_px(t)


class Broker:
    def __init__(self, commission: float, slip_stock: float, slip_etf: float):
        self.commission, self.slip = commission, (slip_stock, slip_etf)

    def fill(self, t: str, qty: float, px: float) -> dict:
        s = self.slip[1] if t in ETFS else self.slip[0]
        price = px * (1 + s * np.sign(qty))
        return {"ticker": t, "qty": qty, "price": price, "fee": abs(qty) * price * self.commission,
                "slippage": abs(qty) * px * s}


class Portfolio:
    def __init__(self, cash: float, whole: bool, borrow: float):
        self.cash, self.whole, self.borrow = cash, whole, borrow
        self.shares: dict[str, float] = {}
        self.costs = 0.0
        self.trades = 0

    def value(self, feed: Feed) -> float:
        v = self.cash
        for t, q in self.shares.items():
            p = feed.close_px(t)
            v += q * p if pd.notna(p) else 0.0
        return v

    def orders(self, target: dict[str, float], feed: Feed) -> list[tuple[str, float]]:
        """Share changes that bring the book to target weights of today's value, priced at today's close."""
        eq = self.value(feed)
        out = []
        for t in sorted(set(target) | set(self.shares)):
            p = feed.close_px(t)
            if not (pd.notna(p) and p > 0):
                continue
            want = target.get(t, 0.0) * eq / p
            if self.whole:
                want = float(np.trunc(want))
            q = want - self.shares.get(t, 0.0)
            if abs(q) * p > 1e-9 * max(eq, 1.0):
                out.append((t, q))
        return out

    def on_fill(self, f: dict) -> None:
        t = f["ticker"]
        self.cash -= f["qty"] * f["price"] + f["fee"]
        self.costs += f["fee"] + f["slippage"]
        self.trades += 1
        q = self.shares.get(t, 0.0) + f["qty"]
        if abs(q) < 1e-12:
            self.shares.pop(t, None)
        else:
            self.shares[t] = q

    def carry(self, feed: Feed) -> None:
        """A day's borrow fee on short positions."""
        short = sum(-q * feed.close_px(t) for t, q in self.shares.items() if q < 0 and pd.notna(feed.close_px(t)))
        fee = short * self.borrow / 252
        self.cash -= fee
        self.costs += fee


def strategy(key: str, members, calendar: pd.DatetimeIndex):
    """The live pots' rule as an event handler: (history up to this close) -> target weights, or None when not due."""
    due = rules.month_end(calendar) if key == "trend" else rules.week_end(calendar)

    def on_close(hist: pd.DataFrame, d: pd.Timestamp, first: bool) -> dict[str, float] | None:
        if not (first or due[d]):
            return None
        return rules.trend_weights(hist, d) if key == "trend" else rules.reversal_weights(hist, d, members)
    return on_close


def run(key: str, close: pd.DataFrame, open_: pd.DataFrame | None, members, start: str,
        setting: tuple) -> dict:
    """One strategy through the event loop. Returns daily returns and totals."""
    _, fill_at, commission, (slip_stock, slip_etf), borrow, cash = setting
    feed = Feed(close, open_)
    broker = Broker(commission, slip_stock, slip_etf)
    cash0 = cash or 1.0
    pf = Portfolio(cash0, whole=cash is not None, borrow=borrow)
    on_close = strategy(key, members, close.index)
    begin = int(close.index.searchsorted(pd.Timestamp(start)))
    queue: deque[Event] = deque()
    pending: list[tuple[str, float]] = []
    values, drag, first = [], 0.0, True
    for _ in range(len(close)):
        d = feed.advance()
        if feed.i < begin:
            continue
        c0 = pf.costs
        queue.append(Event("open", d))
        queue.append(Event("close", d))
        while queue:
            ev = queue.popleft()
            if ev.kind == "open":
                for t, q in pending:
                    p = feed.open_px(t)
                    if pd.notna(p) and p > 0:
                        queue.appendleft(Event("fill", d, broker.fill(t, q, p)))
                pending = []
            elif ev.kind == "fill":
                pf.on_fill(ev.data)
            elif ev.kind == "close":
                pf.carry(feed)
                target = on_close(feed.history(), d, first)
                first = False
                if target is not None:
                    pending = pf.orders(target, feed)
                    if fill_at == "close":     # the replay's assumption: filled at the price the signal saw
                        for t, q in pending:
                            pf.on_fill(broker.fill(t, q, feed.close_px(t)))
                        pending = []
                values.append((d, pf.value(feed)))
                drag += (pf.costs - c0) / max(values[-1][1], 1e-9)
    v = pd.Series(dict(values))
    r = v.pct_change().dropna()
    return {"returns": r, "cost_pct_per_year": 100 * drag / max(len(r) / 252, 1e-9), "trades": pf.trades, "end": float(v.iloc[-1] / cash0)}


def compare(close: pd.DataFrame, open_: pd.DataFrame | None, members, start: str, vectorized: pd.DataFrame,
            keys: tuple[str, ...] = ("trend", "reversal")) -> dict:
    """Each strategy under each setting, next to the vectorized replay's returns."""
    from .live import stats
    out: dict = {}
    for key in keys:
        rows = []
        if key in vectorized.columns:
            rows.append({"setting": "Vectorized replay (the pots' backtest)", **_brief(stats(vectorized[key]))})
        daily = {"Vectorized replay": vectorized[key].dropna()} if key in vectorized.columns else {}
        for st in SETTINGS:
            res = run(key, close, open_, members, start, st)
            daily[st[0]] = res["returns"]
            rows.append({"setting": st[0], **_brief(stats(res["returns"])), "trades": res["trades"],
                         "cost_pct_per_year": round(float(res["cost_pct_per_year"]), 2)})
        same = daily[SETTINGS[0][0]].iloc[1:]
        v = vectorized[key].reindex(same.index).dropna() if key in vectorized.columns else pd.Series(dtype=float)
        same = same.reindex(v.index)
        out[key] = {"rows": rows,
                    # how close the event loop with the replay's assumptions comes to the replay (they differ only by drift)
                    "check": {"corr": round(float(np.corrcoef(same, v)[0, 1]), 4), "mean_abs_gap_bp": round(float((same - v).abs().mean() * 1e4), 2)}
                    if len(v) > 20 else None,
                    "equity": {k: _monthly_equity(r) for k, r in daily.items()}}
    return out


def _brief(s: dict) -> dict:
    if not s:
        return {}
    return {"cagr_pct": round(100 * s["cagr"], 1), "sharpe": round(s["sharpe"], 2), "max_dd_pct": round(100 * s["max_dd"], 1)}


def _monthly_equity(r: pd.Series) -> dict[str, float]:
    eq = (1 + r).cumprod()
    m = eq.groupby(eq.index.to_period("M")).last()
    return {str(p): round(float(x), 4) for p, x in m.items()}
