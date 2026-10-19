"""Backtest grid: tickers x bar sizes x variants x costs, with every calendar year reported on its own. The rule's
settings (9 EMA, session VWAP, 3-minute bars) come from the reel, not from fitting on this data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import strategy

BARS, VARIANTS, COSTS = (1, 3, 5, 10), ("flip", "long", "first"), (0.0, 0.3, 1.0)


def stats(r: pd.Series, tr: pd.DataFrame | None = None, cost_bp: float = 0.0) -> dict:
    if len(r) == 0:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    out = {"days": int(len(r)), "annual_pct": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 2), "total_pct": round(100 * (eq.iloc[-1] - 1), 1),
           "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else 0.0,
           "max_drawdown_pct": round(100 * float((eq / eq.cummax() - 1).min()), 1)}
    if tr is not None and len(tr):
        net = tr.ret - 2 * cost_bp / 1e4
        out |= {"trades": int(len(tr)), "trades_per_day": round(len(tr) / len(r), 2), "win_pct": round(100 * float((net > 0).mean()), 1),
                "avg_trade_bp": round(1e4 * float(net.mean()), 2)}
    return out


def run(bars1: dict[str, pd.DataFrame]) -> dict:
    res = {}
    for tk, d1 in bars1.items():
        hold = d1.groupby(d1.index.strftime("%Y-%m-%d")).apply(lambda g: g.close.iloc[-1] / g.open.iloc[0] - 1)
        res[f"{tk} open-to-close hold"] = {"all": stats(hold)} | {y: stats(hold[hold.index.str[:4] == y]) for y in sorted(set(hold.index.str[:4]))}
        for m in BARS:
            d = strategy.indicators(strategy.resample(d1, m))
            days = pd.Index(sorted(d.day.unique()))
            for v in VARIANTS:
                tr = strategy.trades(d, v)
                for c in COSTS:
                    r = strategy.daily(tr, days, c)
                    row = {"all": stats(r, tr, c)}
                    for y in sorted(set(days.str[:4])):
                        row[y] = stats(r[r.index.str[:4] == y], tr[tr.day.str[:4] == y], c)
                    res[f"{tk} {m}m {v} cost{c:g}bp"] = row
    return res


def table(res: dict) -> str:
    years = sorted({y for row in res.values() for y in row if y != "all"})
    lines = ["| Test | " + " | ".join(years) + " | All: annual % | Sharpe | Worst drop % | Trades/day |", "|---" * (len(years) + 5) + "|"]
    for k, row in res.items():
        a = row["all"]
        lines.append(f"| {k} | " + " | ".join(f"{row[y]['annual_pct']:+.1f}" if row.get(y) else "" for y in years)
                     + f" | {a['annual_pct']:+.1f} | {a['sharpe']:.2f} | {a['max_drawdown_pct']:.1f} | {a.get('trades_per_day', '')} |")
    return "\n".join(lines)
