"""Daily stop loss for every strategy pot (Joaquin, 2026-10-23).

Each position has a stop a fixed percentage below the previous close (above it for a short). When a session's low
(high for a short) reaches the stop, the position is sold at the stop price, or at the open if the stock opened past it,
and sits in cash for the rest of the day. At that close the pot forms its book as usual, so the strategy carries on the
next day (the stopped position is bought back if the rule still wants it, at 1 bp a side like any trade).

The weekly reversal pot with the stop (reversal_stop; the plain reversal pot has none, for comparison) is 20 stocks at 5% each, so its stop is also its daily loss cap: the pot cannot lose much more
than REVERSAL_CAP in a session unless stocks open below their stop (an overnight gap fills at the open).

Prices are Yahoo's daily bars, adjusted like the closes. A ticker without that day's low/high (an old cache) is not
checked. Levels and their backtest: research/stop_losses/README.md.
"""
from __future__ import annotations

import pandas as pd

REVERSAL_CAP = 0.05
STOPS = {                    # stop distance from the previous close, per position
    "trend": 0.04,           # ETFs: a 4% day against a position is rare (about 1 in 100 sessions for EEM)
    "pead": 0.08,
    "insider": 0.08,
    "best_ideas": 0.08,
    "conviction": 0.08,
    "pelosi": 0.08,
    "reversal_stop": REVERSAL_CAP,   # the plain "reversal" pot keeps no stop, to compare the two
    "btc_donchian": 0.06,    # BTC from its price at the last US close, checked on 4-hour bars (crypto.py)
}


def held_return(w: dict[str, float], px: pd.DataFrame, bars: dict[str, pd.DataFrame] | None, d0: pd.Timestamp,
                d1: pd.Timestamp, stop: float | None, cost: float = 0.0) -> tuple[float, list[str]]:
    """Return from close d0 to close d1 of holding weights w, with each position stopped out stop below (above, if short)
    the previous close of any session in between, and the stopped positions (each charged cost on the sale).
    A ticker without both closes counts as flat."""
    if not w or d0 not in px.index or d1 not in px.index:
        return 0.0, []
    sessions = px.index[(px.index > d0) & (px.index <= d1)]
    tot, out = 0.0, []
    for t, x in w.items():
        if t not in px.columns or pd.isna(px.at[d0, t]) or pd.isna(px.at[d1, t]) or not px.at[d0, t]:
            continue
        ref, end = px.at[d0, t], px.at[d1, t]
        if stop and bars:
            prev = ref
            for s in sessions:
                c = px.at[s, t]
                hit = _hit(bars, s, t, prev, x, stop)
                if hit is not None:
                    end = hit
                    out.append(t)
                    break
                prev = c if pd.notna(c) else prev
        tot += x * (end / ref - 1) - (cost * abs(x) if t in out else 0.0)
    return float(tot), out


def _hit(bars: dict[str, pd.DataFrame], s: pd.Timestamp, t: str, prev: float, x: float, stop: float) -> float | None:
    """The fill price if the position's stop was reached in session s, else None."""
    lo, hi, op = (bars.get(k) for k in ("low", "high", "open"))
    o = op.at[s, t] if op is not None and t in op.columns and s in op.index else float("nan")
    if x > 0:
        level = prev * (1 - stop)
        low = lo.at[s, t] if lo is not None and t in lo.columns and s in lo.index else float("nan")
        if pd.notna(low) and low <= level:
            return min(o, level) if pd.notna(o) else level
    else:
        level = prev * (1 + stop)
        high = hi.at[s, t] if hi is not None and t in hi.columns and s in hi.index else float("nan")
        if pd.notna(high) and high >= level:
            return max(o, level) if pd.notna(o) else level
    return None


def explain(state: dict, px: pd.DataFrame, d: pd.Timestamp, top: int = 5) -> dict:
    """What moved each pot on the session d: its day return and the positions that added or cost the most."""
    out = {}
    if d not in px.index:
        return out
    i = px.index.get_loc(d)
    if i == 0:
        return out
    d0 = px.index[i - 1]
    ds0 = d0.strftime("%Y-%m-%d")
    out["SPY"] = round(float(px.at[d, "SPY"] / px.at[d0, "SPY"] - 1), 5) if "SPY" in px.columns else None
    for key, rows in (state.get("pots") or {}).items():
        book = next((r["w"] for r in reversed(rows) if r["date"] <= ds0), None)
        day = next((r for r in rows if r["date"] == d.strftime("%Y-%m-%d")), None)
        if book is None or day is None:
            continue
        c = {t: x * (px.at[d, t] / px.at[d0, t] - 1) for t, x in book.items()
             if t in px.columns and pd.notna(px.at[d, t]) and pd.notna(px.at[d0, t]) and px.at[d0, t]}
        s = sorted(c.items(), key=lambda kv: kv[1])
        out[key] = {"ret": round(day["ret"], 5), "positions": len(book),
                    "worst": {t: f"{100 * v:+.2f}pp ({100 * (px.at[d, t] / px.at[d0, t] - 1):+.1f}%)" for t, v in s[:top]},
                    "best": {t: f"{100 * v:+.2f}pp" for t, v in s[::-1][:2]}}
    return out


def backtest(px: pd.DataFrame, bars: dict[str, pd.DataFrame], members: list[str], start: str, levels: dict[str, list[float]],
             cost: float) -> dict:
    """Trend and reversal replayed from start with each stop level (0 = no stop): yearly return, worst day, worst drop, stops,
    return per calendar year and the growth of 1 at each month end (one "YYYY-MM:value" string, to keep the log short)."""
    from . import rules
    dates = px.index[px.index >= start]
    me, we = rules.month_end(px.index), rules.week_end(px.index)
    res = {}
    for key, lv in levels.items():
        form = (lambda d: rules.trend_weights(px, d)) if key == "trend" else (lambda d: rules.reversal_weights(px, d, members))
        due = me if key == "trend" else we
        books, w = {}, form(dates[0])
        for d in dates[1:]:
            books[d] = w
            if due[d]:
                w = form(d)
        for s in lv:
            r, n, prev = [], 0, dates[0]
            for i, d1 in enumerate(dates[1:], 1):
                w = books[d1]
                ret, hit = held_return(w, px, bars, prev, d1, s, cost)
                n += len(hit)
                nxt = books[dates[i + 1]] if i + 1 < len(dates) else w
                r.append(ret - cost * rules.turnover({k: v for k, v in w.items() if k not in hit}, nxt))
                prev = d1
            res[f"{key} stop {s:.0%}"] = summary(pd.Series(r, index=dates[1:]), n)
    if "SPY" in px:
        res["SPY"] = summary(px["SPY"].loc[dates].pct_change().dropna(), 0)
    return res


def summary(r: pd.Series, n: int) -> dict:
    """One backtest line from its daily returns and the number of stops hit."""
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return {"annual_pct": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 1),
            "worst_day_pct": round(100 * r.min(), 1), "worst_drop_pct": round(100 * (eq / eq.cummax() - 1).min(), 1),
            "days_below_minus5pct": int((r < -0.05).sum()), "stops": n,
            "since_2025_pct": round(100 * ((1 + r[r.index >= "2025-01-01"]).prod() - 1), 1),
            "years": {str(y): round(100 * ((1 + g).prod() - 1), 1) for y, g in r.groupby(r.index.year)},
            "month_ends": ",".join(f"{d:%Y-%m}:{v:.4f}" for d, v in eq.groupby(eq.index.to_period("M")).tail(1).items())}


def backtest_btc(btc: pd.DataFrame, levels: list[float]) -> dict:
    """The BTC rule over the cached 4-hour bars with each stop level (0 = no stop)."""
    from . import crypto
    res = {}
    if len(btc) <= crypto.TREND:
        return res
    for s in levels:
        eq = crypto.run_rule(btc, stop=s or None)["equity"]
        day = eq.groupby(eq.index.date).last().pct_change().dropna()
        res[f"btc stop {s:.0%}"] = {"from": str(btc.index[0].date()), "total_pct": round(100 * (eq.iloc[-1] - 1), 1),
                                    "worst_day_pct": round(100 * day.min(), 1), "worst_drop_pct": round(100 * (eq / eq.cummax() - 1).min(), 1)}
    return res
