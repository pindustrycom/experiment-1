"""Daily paper trading of the strategy pots, and a replay of those whose history we have (all but pead and insider).

After each US close: mark every pot's book to the close, then form the next book (rules.py). A pot's day return is its
book's return from the previous close minus COST per unit of turnover when the book changes. The state (each pot's
daily rows with the book held, and the event holdings with sessions left) is a JSON file the workflow keeps between runs.
Every pot starts at the ORIGIN close; one added later is rebuilt from there with only the data known at each close.

The earnings and insider pots need event histories that the free data does not reach back for, so they are tracked
forward only. Trend and reversal are also replayed from 2019 with the same functions; the reversal replay uses today's
S&P 500 members, which flatters it (stocks that fell out of the index are missing). The copy pots are replayed from the
13F and PTR history in data/copy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import copy, crypto, rules

COST = 0.0001          # 1 bp of the traded amount, each side
REPLAY_FROM = "2019-01-01"
EVENT_FRESH_DAYS = 7   # an earnings beat older than this when first seen is not bought
# Every pot forms its first book at this close, the last one before the paper portfolios' start (experiments/hub/portfolio.py
# START, 2026-10-19, when Exp1 began), so all pots are compared over the same days. A pot added later is replayed from here.
ORIGIN = "2026-10-16"


def _last(rows: list[dict]) -> dict | None:
    return rows[-1] if rows else None


def step(state: dict, px: pd.DataFrame, d: pd.Timestamp, members: list[str], earnings: pd.DataFrame, insiders: pd.DataFrame,
         cp: dict | None = None, btc: pd.DataFrame | None = None, origin: str = ORIGIN) -> dict:
    """Move every pot to the close of d. Safe to repeat: a pot already at d is left alone. A pot that does not reach back to
    the origin close (a new pot, or one added after the first) is first rebuilt from there, session by session, with only
    what was known at each close; those rows are marked as replayed."""
    pots = state.setdefault("pots", {k: [] for k in rules.STRATEGIES})
    state.setdefault("events", {"pead": {}, "insider": {}})
    state.setdefault("seen", {"pead": []})
    ds = d.strftime("%Y-%m-%d")
    after = px.index[(px.index >= origin) & (px.index < d)]
    o = after[0] if len(after) else None
    rep = {}
    for key in rules.STRATEGIES:
        rows = pots.get(key) or []
        if o is not None and (not rows or rows[0]["date"] > o.strftime("%Y-%m-%d")):
            if (done := catch_up(state, key, px, after, members, earnings, insiders, cp, btc)):
                rep[f"{key}_replayed"] = done
        rep[key] = step_pot(state, key, px, d, ds, members, earnings, insiders, cp, btc)
    state["seen"]["pead"] = state["seen"]["pead"][-3000:]
    firsts = [r[0]["date"] for r in pots.values() if r]
    if firsts:
        state["start"] = min(firsts)
    return rep


def catch_up(state: dict, key: str, px: pd.DataFrame, sessions: pd.DatetimeIndex, members: list[str], earnings: pd.DataFrame,
             insiders: pd.DataFrame, cp: dict | None, btc: pd.DataFrame | None) -> str:
    """Rebuild one pot from the first of sessions, each step seeing prices only up to its own close."""
    state["pots"][key] = []
    if key in state["events"]:
        state["events"][key] = {}
    if key == "pead":
        state["seen"]["pead"] = []
    for s in sessions:
        step_pot(state, key, px.loc[:s], s, s.strftime("%Y-%m-%d"), members, earnings, insiders, cp, btc)
    rows = state["pots"].get(key) or []
    for r in rows:
        r["replayed"] = True
    return f"replayed {len(rows)} sessions from {rows[0]['date']}" if rows else ""


def step_pot(state: dict, key: str, px: pd.DataFrame, d: pd.Timestamp, ds: str, members: list[str], earnings: pd.DataFrame,
             insiders: pd.DataFrame, cp: dict | None, btc: pd.DataFrame | None) -> str:
    """Move one pot to the close of d."""
    pots, events, seen = state["pots"], state["events"], state["seen"]
    rows = pots.setdefault(key, [])
    last = _last(rows)
    if last and last["date"] >= ds:
        return f"already at {last['date']}"
    if key == crypto.KEY:
        out = step_btc(rows, last, btc, d, ds)
        if not rows:
            pots.pop(key, None)
        return out
    old_w = last["w"] if last else {}
    hold = book_return(old_w, px, pd.Timestamp(last["date"]), d) if last else 0.0
    if key in copy.KEYS:
        if not copy.has_data(cp):
            if not last:
                pots.pop(key, None)
                return "no filings data yet: starts once it is there"
            new_w, reb = old_w, False
        else:
            new_w, reb = copy.WEIGHTS[key](cp, px, d), True
    elif key in ("trend", "reversal"):
        freq, due = ("M", rules.is_last_session_of_month) if key == "trend" else ("W", rules.is_last_session_of_week)
        if due_now(rows, d, freq, due):
            new_w = rules.trend_weights(px, d) if key == "trend" else rules.reversal_weights(px, d, members)
            reb = True
        else:
            new_w, reb = old_w, False
    else:
        reb = True
        h = {t: n - 1 for t, n in events[key].items() if n - 1 > 0} if last else dict(events[key])
        if key == "pead":
            beats = rules.earnings_beats(earnings)
            if not beats.empty:
                bd = pd.to_datetime(beats["date"])
                fresh = beats[(bd < d) & (bd >= d - pd.Timedelta(days=EVENT_FRESH_DAYS))]
                for s, day in zip(fresh["symbol"], fresh["date"]):
                    tag = f"{s}|{day}"
                    if tag not in seen["pead"] and s in px.columns and pd.notna(px.at[d, s]):
                        seen["pead"].append(tag)
                        h.setdefault(s, rules.PEAD_HOLD)
        else:
            for s in rules.insider_clusters(insiders, d):
                if s in px.columns and pd.notna(px.at[d, s]):
                    h.setdefault(s, rules.INSIDER_HOLD)
        events[key] = h
        new_w = rules.event_weights(h)
    ret = hold - COST * rules.turnover(old_w, new_w) if last else -COST * rules.turnover({}, new_w)
    idx = (last["index"] if last else 1.0) * (1 + ret)
    rows.append({"date": ds, "ret": ret, "index": idx, "w": {k: round(v, 6) for k, v in new_w.items()}, "reb": reb})
    return f"{ds}: {ret:+.4%}, {len(new_w)} positions"


def step_btc(rows: list[dict], last: dict | None, btc: pd.DataFrame | None, d: pd.Timestamp, ds: str) -> str:
    """The BTC pot earns the 4-hour rule's equity change between the bars marked at the last and this US close."""
    eq = crypto.run_rule(btc) if btc is not None and len(btc) > crypto.TREND else None
    b = crypto.mark_bar(eq, d) if eq is not None else None
    if b is None:
        if not last:
            return "no BTC bars yet: starts once they are there"
        ret, w, bar = 0.0, last["w"], last.get("bar")
    else:
        prev = pd.Timestamp(last["bar"]) if last and last.get("bar") else None
        ret = float(eq.at[b, "equity"] / eq.at[prev, "equity"] - 1) if prev is not None and prev in eq.index else 0.0
        w, bar = ({"BTC": 1.0} if eq.at[b, "after"] else {}), str(b)
    idx = (last["index"] if last else 1.0) * (1 + ret)
    rows.append({"date": ds, "ret": ret, "index": idx, "w": w, "reb": False, "bar": bar})
    return f"{ds}: {ret:+.4%}, {'BTC' if w else 'cash'} after the bar opening {bar} UTC"


def due_now(rows: list[dict], d: pd.Timestamp, freq: str, due) -> bool:
    """Rebalance on the period's last session, on the first run, and on the first run after a missed rebalance."""
    reb = [r["date"] for r in rows if r.get("reb")]
    if not reb:
        return True
    return due(d) or pd.Timestamp(reb[-1]).to_period(freq) < d.to_period(freq) - 1


def book_return(w: dict, px: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    if d0 not in px.index or d1 not in px.index:
        return 0.0
    return rules.book_return(w, px, d0, d1)


# ---------------------------------------------------------------- replay

def replay(px: pd.DataFrame, members: list[str], start: str = REPLAY_FROM, cp: dict | None = None) -> pd.DataFrame:
    """Daily returns of trend, reversal and (with their data) the copy pots, after costs, and SPY, from start to the last close."""
    dates = px.index[px.index >= start]
    me, we = rules.month_end(px.index), rules.week_end(px.index)
    wt, wr = rules.trend_weights(px, dates[0]), rules.reversal_weights(px, dates[0], members)
    out = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        rt, rr = rules.book_return(wt, px, d0, d1), rules.book_return(wr, px, d0, d1)
        if me[d1]:
            nw = rules.trend_weights(px, d1)
            rt -= COST * rules.turnover(wt, nw)
            wt = nw
        if we[d1]:
            nw = rules.reversal_weights(px, d1, members)
            rr -= COST * rules.turnover(wr, nw)
            wr = nw
        out.append({"date": d1, "trend": rt, "reversal": rr, "SPY": px.at[d1, "SPY"] / px.at[d0, "SPY"] - 1})
    rep = pd.DataFrame(out).set_index("date")
    if copy.has_data(cp):
        rep = rep.join(copy.replay(cp, px, start))
    return rep


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    vol = r.std() * math.sqrt(252)
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol": float(vol), "sharpe": float(r.mean() * 252 / vol) if vol else 0.0,
            "max_dd": float((eq / eq.cummax() - 1).min()), "total": float(eq.iloc[-1] - 1),
            "years": {str(y): float((1 + g).prod() - 1) for y, g in r.groupby(r.index.year)}}


def replay_summary(rep: pd.DataFrame) -> dict:
    keys = [k for k in rep.columns if k != "SPY"]
    out = {k: stats(rep[k]) for k in rep.columns}
    out["from"], out["to"] = str(rep.index[0].date()), str(rep.index[-1].date())
    # how the two move with SPY: correlation of monthly returns
    m = (1 + rep).groupby(rep.index.to_period("M")).prod() - 1
    out["corr_spy"] = {k: float(np.corrcoef(m[k], m["SPY"])[0, 1]) for k in keys}
    out["monthly"] = {str(p): {k: round(float(m.at[p, k]), 5) for k in m.columns} for p in m.index}
    return out


def load_state(f: Path) -> dict:
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {}


def save_state(f: Path, state: dict) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state, indent=1))
