"""The three copy pots (long-term), built on public filings and tested in research/copy_traders:
  best_ideas  each of 23 famous funds' biggest stock in its latest 13F, bought after the filing; funds equal weight.
  conviction  stocks held at 7.5%+ of a long-term fund's book (not Renaissance or Bridgewater), ranked by how many funds
              hold them that way, top 50 equal weight; rebuilt 47 days after each quarter end with the 13Fs filed by then.
  pelosi      every stock Nancy Pelosi discloses buying (shares or calls, copied with shares), bought after the filing and
              held until she discloses a sale or for 252 sessions; equal weight, cash when nothing is held.
Like rules.py: pure functions from what was public at a close to the target weights held until the next close.
The data (13F top 25 per fund, CUSIP -> ticker, Pelosi's PTRs) lives in data/copy, seeded from the trader-skill-data
release and kept current by copy_data.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import rules

FUNDS = {"SPY", "IVV", "VOO", "QQQ", "IWM", "GLD", "EEM", "EFA", "VTI"}   # index ETFs are nobody's stock pick
NOT_LONG_TERM = {"0001037389", "0001350694"}                              # Renaissance, Bridgewater
STALE_DAYS = 200        # a fund with no 13F for this long has stopped filing: its last book is dropped
CONV_MIN_W, CONV_MAX, CONV_LAG_DAYS = 0.075, 50, 47
PELOSI_HOLD = 252
KEYS = ("best_ideas", "conviction", "pelosi")


def load(data: Path) -> dict:
    """Filers (cik, name, quarters), CUSIP map, and Pelosi's PTR rows; empty when the data has not been seeded."""
    d = data / "copy"
    filers = [json.loads(p.read_text()) for p in sorted((d / "13f").glob("*.json"))]
    cusips = json.loads((d / "cusips.json").read_text()) if (d / "cusips.json").exists() else {}
    recs = []
    for p in sorted((d / "ptr").glob("*.json")):
        for doc in json.loads(p.read_text())["docs"].values():
            if doc.get("last") == "Pelosi":
                recs += [{"ticker": r.get("ticker"), "type": r.get("type"), "date": r.get("date"), "filed": doc["filed"]}
                         for r in doc["rows"]]
    return {"filers": filers, "cusips": cusips, "pelosi": pelosi_frame(recs)}


def pelosi_frame(recs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(recs, columns=["ticker", "type", "date", "filed"])
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.replace("/", "-").str.replace(".", "-")
    df["date"], df["filed"] = pd.to_datetime(df["date"], errors="coerce"), pd.to_datetime(df["filed"], errors="coerce")
    return df[(df["ticker"] != "") & df["date"].notna() & df["filed"].notna()].reset_index(drop=True)


def ticker(cusips: dict, cusip: str) -> str:
    return (cusips.get(cusip) or "").replace(".", "-").replace("/", "-")


def tickers(cp: dict, since: str = "2018-01-01") -> set[str]:
    """Every ticker the three pots could hold from `since`, for the price fetch."""
    out = set(cp["pelosi"]["ticker"]) if len(cp.get("pelosi", [])) else set()
    for f in cp.get("filers", []):
        for q in f["quarters"]:
            if q["filed"] >= since:
                out |= {t for h in q["top"][:15] if (t := ticker(cp["cusips"], h["cusip"]))}
    return out - FUNDS - {""}


def _priced(px: pd.DataFrame, d: pd.Timestamp, t: str) -> bool:
    return t in px.columns and pd.notna(px.at[d, t])


def best_ideas_weights(cp: dict, px: pd.DataFrame, d: pd.Timestamp) -> dict[str, float]:
    ds, old = str(d.date()), str((d - pd.Timedelta(days=STALE_DAYS)).date())
    picks: dict[str, float] = {}
    n = 0
    for f in cp["filers"]:
        known = [q for q in f["quarters"] if q["filed"] < ds]          # filed before d: bought at d's close at the earliest
        if not known:
            continue
        q = max(known, key=lambda q: (q["filed"], q["period"]))
        if q["filed"] < old:
            continue
        names = [t for h in q["top"] if (t := ticker(cp["cusips"], h["cusip"])) and t not in FUNDS and _priced(px, d, t)]
        if names:
            picks[names[0]] = picks.get(names[0], 0.0) + 1.0
            n += 1
    return {t: x / n for t, x in picks.items()} if n else {}


def conviction_book(cp: dict, px: pd.DataFrame, per: pd.Timestamp) -> list[str]:
    """The names for the quarter ending `per`, decided at the first session on or after per + CONV_LAG_DAYS."""
    cache = cp.setdefault("_conv", {})
    if per in cache:
        return cache[per]
    reb = per + pd.Timedelta(days=CONV_LAG_DAYS)
    i = px.index.searchsorted(reb)
    if i >= len(px.index):
        return []
    day = px.index[i]
    score: dict[str, list[float]] = {}
    for f in cp["filers"]:
        if f["cik"] in NOT_LONG_TERM:
            continue
        qs = [q for q in f["quarters"] if pd.Timestamp(q["filed"]) <= reb and pd.Timestamp(q["period"]) <= per]
        if not qs:
            continue
        q = max(qs, key=lambda q: q["period"])
        if pd.Timestamp(q["period"]) < per - pd.Timedelta(days=100):
            continue                                                       # fund stopped filing
        tot = q.get("stock_total") or q.get("total")
        ws: dict[str, float] = {}
        for h in q["top"]:
            t = ticker(cp["cusips"], h["cusip"])
            if t and tot:
                ws[t] = ws.get(t, 0.0) + h["value"] / tot
        for t, x in ws.items():
            if x >= CONV_MIN_W and t not in FUNDS:
                score.setdefault(t, []).append(x)
    ranked = sorted(score, key=lambda t: (-len(score[t]), -sum(score[t])))
    cache[per] = [t for t in ranked if _priced(px, day, t)][:CONV_MAX]
    return cache[per]


def conviction_weights(cp: dict, px: pd.DataFrame, d: pd.Timestamp) -> dict[str, float]:
    pers = sorted({pd.Timestamp(q["period"]) for f in cp["filers"] for q in f["quarters"]})
    due = [p for p in pers if px.index.searchsorted(p + pd.Timedelta(days=CONV_LAG_DAYS)) < len(px.index)
           and px.index[px.index.searchsorted(p + pd.Timedelta(days=CONV_LAG_DAYS))] <= d]
    if not due:
        return {}
    names = [t for t in conviction_book(cp, px, due[-1]) if _priced(px, d, t)]
    return {t: 1.0 / len(names) for t in names} if names else {}


def pelosi_spans(cp: dict, px: pd.DataFrame) -> list[tuple[str, int, int]]:
    """(ticker, first session held, first session not held) for each disclosed buy; positions in px.index."""
    key = (len(px.index), px.index[-1])
    if cp.get("_pelosi_key") == key:
        return cp["_pelosi"]
    cal = px.index
    p = cp["pelosi"]
    p = p[p["ticker"].isin(px.columns)].sort_values("filed")
    buys = p[p["type"] == "P"].drop_duplicates(["ticker", "date"])        # shares and calls on one day = one signal
    sells = p[p["type"].astype(str).str.startswith("S")]
    spans = []
    for b in buys.itertuples():
        i0 = int(cal.searchsorted(b.filed, side="right"))
        if i0 >= len(cal):
            continue
        i1 = i0 + PELOSI_HOLD
        later = sells[(sells["ticker"] == b.ticker) & (sells["date"] > b.date)]
        if len(later):
            i1 = min(i1, int(cal.searchsorted(later["filed"].iloc[0], side="right")))
        spans.append((b.ticker, i0, i1))
    cp["_pelosi_key"], cp["_pelosi"] = key, spans
    return spans


def pelosi_weights(cp: dict, px: pd.DataFrame, d: pd.Timestamp) -> dict[str, float]:
    i = int(px.index.get_loc(d))
    held = sorted({t for t, i0, i1 in pelosi_spans(cp, px) if i0 <= i < i1 and _priced(px, d, t)})
    return {t: 1.0 / len(held) for t in held} if held else {}


WEIGHTS = {"best_ideas": best_ideas_weights, "conviction": conviction_weights, "pelosi": pelosi_weights}


def has_data(cp: dict | None) -> bool:
    return bool(cp and cp.get("filers"))


def replay(cp: dict, px: pd.DataFrame, start: str) -> pd.DataFrame:
    """Daily returns of the three pots after costs (cost: live.COST per unit of turnover), same functions as the live run."""
    from .live import COST
    dates = px.index[px.index >= start]
    w = {k: WEIGHTS[k](cp, px, dates[0]) for k in KEYS}
    out = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        row = {"date": d1}
        for k in KEYS:
            r = rules.book_return(w[k], px, d0, d1)
            nw = WEIGHTS[k](cp, px, d1)
            row[k] = r - COST * rules.turnover(w[k], nw)
            w[k] = nw
        out.append(row)
    return pd.DataFrame(out).set_index("date")
