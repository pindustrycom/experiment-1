"""Paper portfolios in Canadian dollars, traded every day from the two experiments' signals. No real money.

Three pots, each starting with a fake 1000 CAD on START:
  Exp1 portfolio: equal money in every pair Experiment 1 has a call on, rebalanced at each close.
      A pair whose SI-v2 direction favours one company: long that one, short the other (the "bias" paper trade).
      Otherwise, a pair whose out-of-sample spread z-score is beyond +-2: short the rich leg, long the cheap one
      (the "convergence" paper trade). Pairs with neither are left in cash. Each pair's slice is long and short
      the same amount, so the day's return of a slice is r_long - r_short.
  Exp2 portfolio: follows Experiment 2's combined portfolio; each session earns its scored return (real_ret).
  Combined (the main dashboard): 500 CAD in each of the two.
The pots are invested in US markets, so each day's value is converted to CAD at the Bank of Canada USD/CAD rate.
No costs, borrow fees or interest are included.

Inputs are files the site already has: exp1/data/pairs.json and pair_prices.json (built by backend/app/static_site.py)
and exp2/data/predictions.csv (built by exp2/live_site.py). Experiment 1's books need a history, so they are kept in
a small state file (portfolio.json) that only the full nightly run updates; Experiment 2 and the FX rates are re-read
every time. Standard library only.

Each night the real closes are also used to grade the day: for Exp1, how many pair bets lost money; for Exp2, whether
the morning call got the direction of the day right (and whether the result landed inside its 68% band). The share
that failed is shown on the pages next to the money.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path

START = "2026-10-19"          # first session the portfolios are tracked
CAPITAL = 1000.0              # CAD in each pot
Z_ENTRY = 2.0
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


def leg_returns(book: list[dict], prices: list[dict], d0: str, d1: str) -> list[float | None]:
    """Each leg's return (long minus short) from close d0 to close d1; None when a price is missing."""
    px = {p["slug"]: p for p in prices}
    out = []
    for leg in book:
        p = px.get(leg["slug"])
        rl = _ret(p, leg["long"], d0, d1) if p else None
        rs = _ret(p, leg["short"], d0, d1) if p else None
        out.append(rl - rs if rl is not None and rs is not None else None)
    return out


def book_return(book: list[dict], prices: list[dict], d0: str, d1: str) -> tuple[float, int]:
    """Return of an equal-money book held from close d0 to close d1, and how many legs could be priced
    (a leg without both prices counts as flat)."""
    if not book:
        return 0.0, 0
    legs = [r for r in leg_returns(book, prices, d0, d1) if r is not None]
    return sum(legs) / len(book), len(legs)


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
    idx, ret, priced, legs = 1.0, 0.0, 0, []
    if rows:
        last = rows[-1]
        ret, priced = book_return(last["book"], prices, last["date"], today)
        if today < START:
            ret = 0.0
        else:   # grade each bet of the book just closed against the real closes
            legs = [{"slug": b["slug"], "long": b["long"], "short": b["short"], "ret": r}
                    for b, r in zip(last["book"], leg_returns(last["book"], prices, last["date"], today))]
        idx = last["index"] * (1 + ret)
    rows.append({"date": today, "index": idx, "ret": ret, "priced_legs": priced, "legs": legs,
                 "book": exp1_book(pairs, prices)})
    return {"exp1": f"{today}: {ret:+.4%}, {len(rows[-1]['book'])} positions"}


# ---------------------------------------------------------------- Experiment 2

def _exp2_rows(site: Path) -> list[dict]:
    f = site / "exp2" / "data" / "predictions.csv"
    try:
        with f.open() as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def exp2_scored(site: Path) -> list[dict]:
    """Scored sessions from START on: the morning call next to the real result. Replayed history is left out."""
    out = []
    for r in _exp2_rows(site):
        if r.get("date", "") >= START and r.get("status") == "scored" and r.get("source") in ("live", "catch-up"):
            real = _num(r.get("real_ret"))
            if real is None:
                continue
            exp = _num(r.get("exp_ret"))
            right = str(r.get("up_right", "")).lower()
            in68 = str(r.get("in68", "")).lower()
            out.append({"date": r["date"], "real_ret": real, "exp_ret": exp,
                        "up_right": right == "true" if right in ("true", "false") else None,
                        "in68": in68 == "true" if in68 in ("true", "false") else None})
    return sorted(out, key=lambda x: x["date"])


def exp2_returns(site: Path) -> list[tuple[str, float]]:
    """Scored sessions from START on, as (date, return as a fraction)."""
    return [(r["date"], r["real_ret"] / 100) for r in exp2_scored(site)]


def exp2_position(site: Path) -> str:
    rows = _exp2_rows(site)
    if not rows:
        return ""
    r = rows[-1]
    on = lambda k: str(r.get(k, "")).lower() == "true"  # noqa: E731
    bits = ["SPY held" if on("regime_in") else "SPY out (cash)",
            f"short volatility at VIX {float(r['K']):.1f}" if on("varswap_on") and r.get("K") else "volatility trade off",
            "S&P 500 stock book"]
    return f"{r['date']}: " + ", ".join(bits)


# ---------------------------------------------------------------- FX and the daily table

def fetch_fx(start: str) -> dict[str, float]:
    req = urllib.request.Request(FX_URL.format(start=start), headers={"User-Agent": "market-experiments"})
    with urllib.request.urlopen(req, timeout=20) as r:
        obs = json.load(r).get("observations") or []
    return {o["d"]: float(o["FXUSDCAD"]["v"]) for o in obs if (o.get("FXUSDCAD") or {}).get("v")}


def _asof(series: dict[str, float], d: str) -> float | None:
    k = [x for x in series if x <= d]
    return series[max(k)] if k else None


def _rate(bad: int, n: int) -> float | None:
    return bad / n if n else None


def grades(state: dict, site: Path) -> dict:
    """The end-of-day report card: each day's bets against the real closes, and the share that failed."""
    days = {}
    lost = bets = 0
    for r in state.get("exp1", []):
        legs = [x for x in r.get("legs") or [] if x.get("ret") is not None]
        if r["date"] < START or not r.get("legs"):
            continue
        bad = sum(x["ret"] < 0 for x in legs)
        lost, bets = lost + bad, bets + len(legs)
        days.setdefault(r["date"], {})["exp1"] = {"ret": r["ret"], "lost": bad, "bets": len(legs),
                                                  "legs": [{k: x[k] for k in ("long", "short", "ret")} for x in legs]}
    wrong = graded = out68 = banded = 0
    for r in exp2_scored(site):
        if r["up_right"] is not None:
            wrong, graded = wrong + (not r["up_right"]), graded + 1
        if r["in68"] is not None:
            out68, banded = out68 + (not r["in68"]), banded + 1
        days.setdefault(r["date"], {})["exp2"] = r
    return {"days": [{"date": d, **v} for d, v in sorted(days.items(), reverse=True)],
            "exp1": {"lost": lost, "bets": bets, "fail_rate": _rate(lost, bets)},
            "exp2": {"wrong": wrong, "days": graded, "fail_rate": _rate(wrong, graded),
                     "outside_band": out68, "band_days": banded, "band_miss_rate": _rate(out68, banded)}}


def table(state: dict, site: Path) -> dict:
    """Daily CAD values of the three pots, from START on."""
    fx = state.get("fx") or {}
    e1 = [(r["date"], r["index"]) for r in state.get("exp1", []) if r["date"] >= START]
    e2, idx2 = {}, 1.0
    for d, r in exp2_returns(site):
        idx2 *= 1 + r
        e2[d] = idx2
    dates = sorted({d for d, _ in e1} | set(e2))
    base_day = (date.fromisoformat(START) - timedelta(days=1)).isoformat()
    fx0 = _asof(fx, base_day) or (fx[min(fx)] if fx else None)
    e1d = dict(e1)
    pre = [r["index"] for r in state.get("exp1", []) if r["date"] < START]
    i1 = pre[-1] if pre else 1.0
    base1 = i1
    rows = [{"date": base_day, "exp1": CAPITAL, "exp2": CAPITAL, "combined": CAPITAL, "fx": fx0}]
    i2 = 1.0
    for d in dates:
        i1, i2 = e1d.get(d, i1), e2.get(d, i2)
        f = _asof(fx, d) or fx0
        k = f / fx0 if f and fx0 else 1.0
        v1, v2 = CAPITAL * i1 / base1 * k, CAPITAL * i2 * k
        rows.append({"date": d, "exp1": v1, "exp2": v2, "combined": (v1 + v2) / 2, "fx": f})
    last = state["exp1"][-1] if state.get("exp1") else None
    return {"start": START, "rows": rows, "fx_known": fx0 is not None,
            "exp1_book": last["book"] if last else [], "exp1_book_date": last["date"] if last else None,
            "exp2_position": exp2_position(site), "grades": grades(state, site)}


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
