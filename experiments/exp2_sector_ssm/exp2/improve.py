"""Experiment 2's weekly improvement loop: try one small change to the portfolio's settings at a time, on paper, and
keep it only if it does better over the following 12 weeks and Joaquin approves it.

The settings are five knobs on the frozen portfolio of exp2/live.py (the defaults reproduce it exactly):
  leverage     how much the whole portfolio is scaled                     (1.35; capped at 1.7)
  regime_cut   turbulent-regime probability above which SPY is sold      (0.5)
  vol_size     size of the short-volatility trade, times its usual size   (1.0; capped at 1.5, it carries crash risk)
  vol_bar      VIX^2 must exceed this times our variance forecast to sell (1.0; since 2019 VIX^2 has always been at
               least 1.37 times the forecast when checked, so lower bars would change nothing and the grid only goes up)
  stock_size   size of the stock book, times its usual size               (1.0)

Every week, after Friday's close:
  1. If the live portfolio has fallen 15% from its peak since the current settings were adopted, go back to the
     previous settings and pause the loop until Joaquin restarts it.
  2. With no trial running, start one: the next knob in turn moved one step on its grid (alternating up and down).
     From the next session on, the trial is tracked in the shadow: each day's return is recomputed from the
     prediction log as if the trial setting had been live. Nothing real changes.
  3. A trial that has run 12 weeks is judged on its Sharpe ratio against the current settings over the same days.
     If it wins it waits for Joaquin's approval (python -m exp2 live --decision approve); otherwise it is dropped.
     Either way the next trial starts the following week.

Shadow returns use the logged SPY return, stock-book return, regime probability and VIX strike of each day plus the
month's variance forecast from plans.json, so they are exact for the default settings (tested) and a faithful
what-if for the others. Paper tracking only: no orders are placed anywhere.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = {"leverage": 1.35, "regime_cut": 0.5, "vol_size": 1.0, "vol_bar": 1.0, "stock_size": 1.0}
GRID = {"leverage": [1.0, 1.2, 1.35, 1.5, 1.7], "regime_cut": [0.3, 0.4, 0.5, 0.6, 0.7],
        "vol_size": [0.5, 0.75, 1.0, 1.25, 1.5], "vol_bar": [1.0, 1.25, 1.5, 1.75, 2.0], "stock_size": [0.0, 0.5, 1.0, 1.5, 2.0]}
ORDER = ["vol_size", "leverage", "regime_cut", "vol_bar", "stock_size"]
NAMES = {"leverage": "leverage", "regime_cut": "regime cut-off", "vol_size": "volatility trade size",
         "vol_bar": "volatility trade bar", "stock_size": "stock book size"}
TRIAL_WEEKS = 12
MAX_DRAWDOWN = 0.15
WEIGHTS = {"hrp": 0.125, "regime": 0.359, "varswap": 0.517}
VEGA, ROLL_COST = 0.5, 0.5


def settings_of(s: dict | str | None) -> dict:
    """Full settings from a partial dict (or its JSON); unknown keys are dropped, values held inside the grid's range."""
    if isinstance(s, str):
        try:
            s = json.loads(s) if s.strip() else {}
        except ValueError:
            s = {}
    if not isinstance(s, dict):
        s = {}
    out = dict(DEFAULTS)
    for k, v in (s or {}).items():
        if k in GRID:
            try:
                out[k] = min(max(float(v), GRID[k][0]), GRID[k][-1])
            except (TypeError, ValueError):
                continue
    return out


def swap_on(plan: dict, s: dict) -> bool:
    """Is the month's variance swap struck under these settings (VIX^2 above the bar times the forecast, calm month end)."""
    fvar, K, p = plan.get("fvar"), plan.get("K"), plan.get("p_hi_end")
    if (s["vol_bar"], s["regime_cut"]) == (DEFAULTS["vol_bar"], DEFAULTS["regime_cut"]) and "on" in plan:
        return bool(plan["on"])                      # the month's own decision, as made when it was formed
    if fvar is None or K is None or p is None or (isinstance(fvar, float) and math.isnan(fvar)):
        return bool(plan.get("on"))
    return bool(K ** 2 > s["vol_bar"] * fvar and p < s["regime_cut"])


# ---------------------------------------------------------------- shadow returns

def shadow(L: pd.DataFrame, plans: dict, s: dict) -> pd.Series:
    """Each scored session's portfolio return (in %) recomputed from the log as if settings s had been live."""
    x = L[L["status"] == "scored"]
    if x.empty:
        return pd.Series(dtype=float)
    f = lambda c: pd.to_numeric(x[c], errors="coerce").to_numpy()  # noqa: E731
    b = lambda c: x[c].astype(str).str.lower().eq("true").to_numpy()  # noqa: E731
    spy, hrp, p_hi, K, n, first = f("spy_ret") / 100, f("hrp_ret"), f("p_hi"), f("K"), f("n_days"), b("first_day")
    months = pd.to_datetime(x["date"]).dt.to_period("M").astype(str)
    on = np.array([swap_on(plans.get(m, {}), s) for m in months])
    g1 = VEGA / (2 * K * n)
    rv = np.where(on, g1 * K ** 2 - g1 * 252e4 * spy ** 2 - VEGA * ROLL_COST * first, 0.0)
    rr = 100 * spy * (p_hi < s["regime_cut"])
    tot = s["leverage"] * (WEIGHTS["hrp"] * s["stock_size"] * hrp + WEIGHTS["regime"] * rr + WEIGHTS["varswap"] * s["vol_size"] * rv)
    return pd.Series(np.nan_to_num(tot), index=x["date"].astype(str).to_numpy())


def stats(r: pd.Series) -> dict:
    """Return, Sharpe and worst drop of a series of daily % returns."""
    if r.empty:
        return {"days": 0, "total_pct": 0.0, "sharpe": None, "max_dd_pct": 0.0}
    eq = (1 + r / 100).cumprod()
    sd = float(r.std())
    return {"days": int(len(r)), "total_pct": float(100 * (eq.iloc[-1] - 1)),
            "sharpe": float(r.mean() / sd * math.sqrt(252)) if sd > 0 else None,
            "max_dd_pct": float(100 * (1 - eq / eq.cummax()).max())}


# ---------------------------------------------------------------- the weekly step

def load(work: Path) -> dict:
    try:
        st = json.loads((work / "improve.json").read_text())
    except (OSError, ValueError):
        st = {}
    st.setdefault("settings", dict(DEFAULTS))
    st.setdefault("previous", None)
    st.setdefault("adopted", None)
    st.setdefault("turn", 0)
    st.setdefault("trial", None)
    st.setdefault("pending", None)
    st.setdefault("paused", None)
    st.setdefault("history", [])
    st.setdefault("last_review", None)
    st["settings"] = settings_of(st["settings"])
    return st


def save(work: Path, st: dict) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "improve.json").write_text(json.dumps(st, indent=1))


def next_trial(st: dict) -> dict | None:
    """The next knob in turn, one grid step away from its current value (up on even rounds, down on odd ones)."""
    cur = st["settings"]
    for _ in range(2 * len(ORDER)):
        t = st["turn"]
        st["turn"] = t + 1
        k = ORDER[t % len(ORDER)]
        up = (t // len(ORDER)) % 2 == 0
        g = GRID[k]
        i = min(range(len(g)), key=lambda j: abs(g[j] - cur[k]))
        for j in ((i + 1, i - 1) if up else (i - 1, i + 1)):
            if 0 <= j < len(g) and g[j] != cur[k]:
                return {"key": k, "from": cur[k], "to": g[j]}
    return None


def _sessions(L: pd.DataFrame, since: str | None, until: str | None = None) -> pd.DataFrame:
    """Scored live sessions (tracked, not replayed) from `since` to `until`."""
    x = L[(L["status"] == "scored") & L["source"].isin(["live", "catch-up"])]
    d = x["date"].astype(str)
    return x[(d >= (since or "")) & (d <= (until or "9999"))]


def weekly(L: pd.DataFrame, plans: dict, st: dict, today: str, force: bool = False) -> dict:
    """Run the week's review once, after its last close (normally Friday evening; a missed Friday is caught up later).
    `today` is the date of the latest close in the data."""
    d = pd.Timestamp(today)
    week = d.to_period("W-FRI") if d.weekday() >= 4 else d.to_period("W-FRI") - 1     # the last week whose close is in
    if not force and st["last_review"] and pd.Timestamp(st["last_review"]).to_period("W-FRI") >= week:
        return {"review": "not due"}
    st["last_review"] = week.end_time.strftime("%Y-%m-%d") if not force else today
    note = {"date": today}
    # 1. drawdown guard on the real (live) pot since the current settings were adopted
    live = _sessions(L, st["adopted"], today)
    if not live.empty and st["previous"] is not None:
        eq = (1 + pd.to_numeric(live["real_ret"], errors="coerce").fillna(0.0) / 100).cumprod()
        dd = float(1 - eq.iloc[-1] / eq.cummax().iloc[-1])
        if dd >= MAX_DRAWDOWN:
            st["history"].append({**note, "event": "reverted", "detail": f"down {100 * dd:.0f}% from its peak",
                                  "from": st["settings"], "to": st["previous"]})
            st["settings"], st["previous"], st["adopted"] = st["previous"], None, today
            st["trial"], st["pending"], st["paused"] = None, None, f"reverted on {today} after a {100 * dd:.0f}% drop"
            return {"review": "reverted", "drawdown": dd}
    if st["paused"]:
        return {"review": "paused", "why": st["paused"]}
    # 2. judge a trial that has run its course
    out = {"review": "done"}
    t = st["trial"]
    if t:
        days = _sessions(L, t["start"], today)
        weeks = days["date"].astype(str).map(lambda d: pd.Timestamp(d).to_period("W-FRI")).nunique() if not days.empty else 0
        t["weeks_done"] = int(weeks)
        if weeks >= TRIAL_WEEKS:
            trial_s = {**st["settings"], t["key"]: t["to"]}
            a, b = stats(shadow(days, plans, st["settings"])), stats(shadow(days, plans, trial_s))
            win = (b["sharpe"] or -9) > (a["sharpe"] or -9)
            rec = {**note, "event": "trial won" if win else "trial lost", "key": t["key"], "from": t["from"], "to": t["to"],
                   "start": t["start"], "current": a, "trial": b}
            st["history"].append(rec)
            if win:
                st["pending"] = {**rec, "settings": trial_s}
            st["trial"] = None
            out["judged"] = rec
        else:
            out["trial_weeks_done"] = t["weeks_done"]
    # 3. start the next trial (one at a time; none while a winner waits for approval)
    if not st["trial"] and not st["pending"]:
        nt = next_trial(st)
        if nt:
            st["trial"] = {**nt, "start": (pd.Timestamp(today) + pd.offsets.BDay(1)).strftime("%Y-%m-%d"), "weeks_done": 0}
            st["history"].append({**note, "event": "trial started", **nt})
            out["started"] = nt
    return out


def decide(st: dict, decision: str, today: str) -> dict:
    """Joaquin's call on a trial that won: approve makes it the live setting from the next morning; reject drops it.
    'resume' restarts a loop paused by the drawdown guard."""
    if decision == "resume":
        st["paused"] = None
        st["history"].append({"date": today, "event": "resumed"})
        return {"decision": "resumed"}
    p = st["pending"]
    if not p:
        return {"decision": f"nothing to {decision}"}
    if decision == "approve":
        st["previous"], st["settings"], st["adopted"] = st["settings"], settings_of(p["settings"]), today
        st["history"].append({"date": today, "event": "approved", "key": p["key"], "from": p["from"], "to": p["to"]})
    else:
        st["history"].append({"date": today, "event": "rejected", "key": p["key"], "from": p["from"], "to": p["to"]})
    st["pending"] = None
    return {"decision": decision, "key": p["key"], "to": p["to"]}


def progress(L: pd.DataFrame, plans: dict, st: dict) -> dict:
    """The running trial's shadow result so far against the current settings, for the page."""
    t = st.get("trial")
    if not t:
        return {}
    days = _sessions(L, t["start"])
    trial_s = {**st["settings"], t["key"]: t["to"]}
    return {"current": stats(shadow(days, plans, st["settings"])), "trial": stats(shadow(days, plans, trial_s))}
