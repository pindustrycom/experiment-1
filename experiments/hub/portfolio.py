"""Paper portfolios in Canadian dollars, traded every day from the two experiments' signals. No real money.

Three pots, each starting with a fake 1000 CAD on START:
  Exp1 portfolio: equal money in every pair Experiment 1 has a call on, rebalanced at each close.
      A pair whose SI-v2 direction favours one company: long that one, short the other (the "bias" paper trade).
      Otherwise, a pair whose out-of-sample spread z-score is beyond +-2: short the rich leg, long the cheap one
      (the "convergence" paper trade). Pairs with neither are left in cash. Each pair's slice is long and short
      the same amount, so the day's return of a slice is r_long - r_short.
  Exp2 portfolio: follows Experiment 2's combined portfolio; each session earns its scored return (real_ret).
  Combined (the main dashboard): 500 CAD in each of the two.
  Exp7 portfolio (its own pot, not in the combined one): the VWAP + 9 EMA day-trading rule on SPY; each session earns the
      day's result of the rule's trades after 0.3 bp per side (exp7/data/daily.csv, built by experiments/exp7_vwap).
  Strategy pots (each its own 1000 CAD, not in the combined one): trend following, earnings drift, insider buying and
      weekly reversal; each session earns the pot's return after 1 bp per side (strategies/data/daily.csv, built by
      experiments/strategies).
  Top 3 pot (its own 1000 CAD, not in the combined one): the three best strategies so far (TOP3), equal money in each,
      evened out at every week's last close; between Fridays each slice moves with its pot's CAD value.
The pots are invested in US markets, so each day's value is converted to CAD at the Bank of Canada USD/CAD rate.
No costs, borrow fees or interest are included.

Inputs are files the site already has: exp1/data/pairs.json and pair_prices.json (built by backend/app/static_site.py)
exp2/data/predictions.csv (built by exp2/live_site.py) and exp7/data/daily.csv (built by exp7/live.py). Experiment 1's books need a history, so they are kept in
a small state file (portfolio.json) that only the full nightly run updates; Experiment 2 and the FX rates are re-read
every time. Standard library only.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path

START = "2026-10-19"          # first session the portfolios are tracked
CAPITAL = 1000.0              # CAD in each pot
# The concentrated pot: picked on 2020-2025 replays (best return per strategy and best 3-way mix: +28.8%/yr vs SPY +15.0%),
# then checked on Jan-Oct 2026 (+27.4%/yr vs SPY +17.7%). research/regime_walkforward has the study.
TOP3 = ("reversal", "pelosi", "exp2")
Z_ENTRY = 2.0
# Stop losses (Joaquin, 2026-10-23), checked at the close since these pots only have closing prices: a pair slice that
# lost EXP1_STOP or more in a session sits in cash for the next one, and so does the whole Exp2 pot after a session down
# EXP2_STOP or more; both then trade as usual. In the 2019-2026 replay (research/portfolio_2019) they cost Exp1 0.3 points
# a year and added 0.2 to Exp2, with 8 Exp2 stops in 7.8 years.
EXP1_STOP, EXP2_STOP = 0.05, 0.02
FX_URL = "https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json?start_date={start}"


def _json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------- Experiment 1

def exp1_book(pairs: list[dict], prices: list[dict]) -> list[dict]:
    """Today's positions from Experiment 1's latest scores: one long/short leg per pair with a call."""
    px = {p["slug"]: p for p in prices}
    book = []
    for pr in sorted(pairs, key=lambda q: q["slug"]):
        p = px.get(pr["slug"])
        if not p:
            continue
        a, b = pr["company_a"]["ticker"], pr["company_b"]["ticker"]
        d, z = pr.get("direction"), p.get("z_last")
        if d in ("a_advantaged", "b_advantaged"):
            lg, sh = (a, b) if d == "a_advantaged" else (b, a)
            why = f"news and fundamentals favour {lg}"
            kind = "bias"
        elif z is not None and abs(z) >= Z_ENTRY:
            lg, sh = (b, a) if z > 0 else (a, b)
            why = f"spread {z:+.1f} sd from normal, betting it closes"
            kind = "convergence"
        else:
            continue
        book.append({"slug": pr["slug"], "long": lg, "short": sh, "kind": kind, "why": why})
    return book


def _ret(p: dict, ticker: str, d0: str, d1: str) -> float | None:
    side = "a" if p["a"] == ticker else "b"
    v = {r["date"]: r[side] for r in p.get("rows") or []}
    if d0 in v and d1 in v and v[d0]:
        return v[d1] / v[d0] - 1
    return None


def book_return(book: list[dict], prices: list[dict], d0: str, d1: str) -> tuple[float, int]:
    """Return of an equal-money book held from close d0 to close d1, and how many legs could be priced
    (a leg without both prices counts as flat)."""
    if not book:
        return 0.0, 0
    px = {p["slug"]: p for p in prices}
    tot, n = 0.0, 0
    for leg in book:
        if leg.get("stopped"):            # in cash for this session
            continue
        p = px.get(leg["slug"])
        rl = _ret(p, leg["long"], d0, d1) if p else None
        rs = _ret(p, leg["short"], d0, d1) if p else None
        if rl is not None and rs is not None:
            tot += rl - rs
            n += 1
    return tot / len(book), n


def update_exp1(state: dict, site: Path) -> dict:
    """After a full run: mark the last book to today's closes and form the next one. Safe to repeat."""
    d = site / "exp1" / "data"
    pairs, prices = _json(d / "pairs.json", []), _json(d / "pair_prices.json", [])
    if not prices:
        return {"exp1": "no prices"}
    today = max(p["last_date"] for p in prices)
    rows = state.setdefault("exp1", [])
    if rows and rows[-1]["date"] >= today:
        return {"exp1": f"already up to {rows[-1]['date']}"}
    idx, ret, priced, hit = 1.0, 0.0, 0, []
    if rows:
        last = rows[-1]
        ret, priced = book_return(last["book"], prices, last["date"], today)
        if today < START:
            ret = 0.0
        idx = last["index"] * (1 + ret)
        hit = [g["slug"] for g in last["book"] if not g.get("stopped") and today >= START
               and (r := book_return([g], prices, last["date"], today))[1] and r[0] <= -EXP1_STOP]
    book = [{**g, "stopped": True, "why": f"stopped out: lost {100 * EXP1_STOP:.0f}% or more on {today}; back tomorrow"}
            if g["slug"] in hit else g for g in exp1_book(pairs, prices)]
    rows.append({"date": today, "index": idx, "ret": ret, "priced_legs": priced, "book": book})
    return {"exp1": f"{today}: {ret:+.4%}, {len(book)} positions" + (f", stopped: {' '.join(hit)}" if hit else "")}


# ---------------------------------------------------------------- Experiment 2

def exp2_returns(site: Path) -> list[tuple[str, float]]:
    """Scored sessions from START on, as (date, return as a fraction). Replayed history before live tracking is left out."""
    f = site / "exp2" / "data" / "predictions.csv"
    try:
        with f.open() as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return []
    out = []
    for r in rows:
        if r.get("date", "") >= START and r.get("status") == "scored" and r.get("source") in ("live", "catch-up"):
            try:
                out.append((r["date"], float(r["real_ret"]) / 100))
            except (KeyError, ValueError):
                continue
    return sorted(out)


def with_day_stop(rets: list[tuple[str, float]], stop: float) -> list[tuple[str, float]]:
    """A session after one that lost stop or more is spent in cash (return 0)."""
    out, prev = [], 0.0
    for d, r in rets:
        out.append((d, 0.0 if prev <= -stop else r))
        prev = r
    return out


def exp2_position(site: Path) -> str:
    f = site / "exp2" / "data" / "predictions.csv"
    try:
        with f.open() as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return ""
    if not rows:
        return ""
    r = rows[-1]
    on = lambda k: str(r.get(k, "")).lower() == "true"  # noqa: E731
    bits = ["SPY held" if on("regime_in") else "SPY out (cash)",
            f"short volatility at VIX {float(r['K']):.1f}" if on("varswap_on") and r.get("K") else "volatility trade off",
            "S&P 500 stock book"]
    return f"{r['date']}: " + ", ".join(bits)


# ---------------------------------------------------------------- Experiment 7

def exp7_returns(site: Path) -> list[tuple[str, float]]:
    """Sessions from START on, as (date, return as a fraction, after costs)."""
    try:
        with (site / "exp7" / "data" / "daily.csv").open() as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return []
    out = []
    for r in rows:
        try:
            if r["date"] >= START:
                out.append((r["date"], float(r["ret"])))
        except (KeyError, ValueError):
            continue
    return sorted(out)


def exp7_position(site: Path) -> str:
    return (_json(site / "exp7" / "data" / "summary.json", {}) or {}).get("position", "")


# ---------------------------------------------------------------- Strategy pots (experiments/strategies)

STRATS = ["trend", "pead", "insider", "best_ideas", "conviction", "pelosi", "reversal", "reversal_stop", "btc_donchian", "btc_donchian_vol", "straddle8", "straddle8_news"]


def strategy_returns(site: Path) -> dict[str, list[tuple[str, float]]]:
    """Each strategy pot's sessions from START on, as (date, return as a fraction, after costs); strategies/data/daily.csv."""
    try:
        with (site / "strategies" / "data" / "daily.csv").open() as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return {}
    out = {k: [] for k in STRATS}
    for r in rows:
        for k in STRATS:
            try:
                if r["date"] >= START and r.get(k):
                    out[k].append((r["date"], float(r[k])))
            except (KeyError, ValueError):
                continue
    return {k: sorted(v) for k, v in out.items() if v}


def strategy_info(site: Path) -> dict:
    """name, category, what and position of each strategy pot, from strategies/data/summary.json."""
    return (_json(site / "strategies" / "data" / "summary.json", {}) or {}).get("strategies", {})


# ---------------------------------------------------------------- FX and the daily table

def fetch_fx(start: str) -> dict[str, float]:
    req = urllib.request.Request(FX_URL.format(start=start), headers={"User-Agent": "market-experiments"})
    with urllib.request.urlopen(req, timeout=20) as r:
        obs = json.load(r).get("observations") or []
    return {o["d"]: float(o["FXUSDCAD"]["v"]) for o in obs if (o.get("FXUSDCAD") or {}).get("v")}


def _asof(series: dict[str, float], d: str) -> float | None:
    k = [x for x in series if x <= d]
    return series[max(k)] if k else None


def table(state: dict, site: Path) -> dict:
    """Daily CAD values of the three pots, from START on."""
    fx = state.get("fx") or {}
    e1 = [(r["date"], r["index"]) for r in state.get("exp1", []) if r["date"] >= START]
    e2, idx2 = {}, 1.0
    for d, r in with_day_stop(exp2_returns(site), EXP2_STOP):
        idx2 *= 1 + r
        e2[d] = idx2
    e7, idx7 = {}, 1.0
    for d, r in exp7_returns(site):
        idx7 *= 1 + r
        e7[d] = idx7
    es = {}
    for k, rs in strategy_returns(site).items():
        idx, es[k] = 1.0, {}
        for d, r in rs:
            idx *= 1 + r
            es[k][d] = idx
    dates = sorted({d for d, _ in e1} | set(e2) | set(e7) | {d for v in es.values() for d in v})
    base_day = (date.fromisoformat(START) - timedelta(days=1)).isoformat()
    fx0 = _asof(fx, base_day) or (fx[min(fx)] if fx else None)
    e1d = dict(e1)
    pre = [r["index"] for r in state.get("exp1", []) if r["date"] < START]
    i1 = pre[-1] if pre else 1.0
    base1 = i1
    rows = [{"date": base_day, "exp1": CAPITAL, "exp2": CAPITAL, "combined": CAPITAL, "exp7": CAPITAL, **{s: CAPITAL for s in es}, "fx": fx0}]
    i2 = i7 = 1.0
    ix = {s: 1.0 for s in es}
    # a strategy pot that joined later is converted to CAD from its own first close, not from START's rate
    fxs = {s: _asof(fx, min(es[s])) or fx0 for s in es}
    for d in dates:
        i1, i2, i7 = e1d.get(d, i1), e2.get(d, i2), e7.get(d, i7)
        ix = {s: es[s].get(d, ix[s]) for s in es}
        f = _asof(fx, d) or fx0
        k = f / fx0 if f and fx0 else 1.0
        v1, v2 = CAPITAL * i1 / base1 * k, CAPITAL * i2 * k
        rows.append({"date": d, "exp1": v1, "exp2": v2, "combined": (v1 + v2) / 2, "exp7": CAPITAL * i7 * k,
                     **{s: CAPITAL * ix[s] * (f / fxs[s] if f and fxs[s] and d >= min(es[s]) else 1.0) for s in es}, "fx": f})
    last = state["exp1"][-1] if state.get("exp1") else None
    top3 = list(TOP3) if concentrate(rows) else []
    return {"start": START, "rows": rows, "top3": top3, "fx_known": fx0 is not None,
            "exp1_book": last["book"] if last else [], "exp1_book_date": last["date"] if last else None,
            "exp2_position": exp2_position(site), "exp7": bool(e7) or (site / "exp7" / "index.html").exists(), "exp7_position": exp7_position(site),
            "strategies": {k: v for k, v in strategy_info(site).items() if k in es}}


def concentrate(rows: list[dict], keys: tuple[str, ...] = TOP3, name: str = "top3") -> bool:
    """Add a pot that holds equal CAD in each of keys, reset to equal at each week's last row. False if a pot is missing."""
    if not rows or any(k not in rows[0] for k in keys):
        return False
    week = [date.fromisoformat(r["date"]).isocalendar()[:2] for r in rows]
    w = dict.fromkeys(keys, 1 / len(keys))
    v = rows[0][name] = CAPITAL
    for i in range(1, len(rows)):
        g = {k: rows[i][k] / rows[i - 1][k] for k in keys}
        grow = sum(w[k] * g[k] for k in keys)
        v *= grow
        rows[i][name] = v
        w = dict.fromkeys(keys, 1 / len(keys)) if i + 1 == len(rows) or week[i + 1] != week[i] else {k: w[k] * g[k] / grow for k in keys}
    return True


def run(site: Path, state_file: Path, update: bool) -> dict:
    """update=True (the nightly full run) moves Experiment 1's book forward and saves the state."""
    state = _json(state_file, {}) or {}
    rep = {}
    try:
        state["fx"] = {**(state.get("fx") or {}), **fetch_fx((date.fromisoformat(START) - timedelta(days=10)).isoformat())}
    except Exception as ex:  # noqa: BLE001 - keep the last saved rates
        rep["fx_warning"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    if update:
        rep.update(update_exp1(state, site))
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=1))
    t = table(state, site)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "portfolio.json").write_text(json.dumps(t, indent=1))
    rep["latest"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in t["rows"][-1].items()}
    return {"portfolio": rep, "table": t}
