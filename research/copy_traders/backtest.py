"""Backtests of the copy-trading ideas from the "Copying known traders" doc, on the trader-skill cache
(release trader-skill-data, cache.tar.gz: 13F top-25 holdings of 23 famous filers since 2013, House PTRs since 2019,
Yahoo adjusted closes). Everything enters at the first close after the information was public and pays 5 bp a side.

  python backtest.py CACHE_DIR OUT_DIR

1. best ideas:   each filer's biggest position (top 1 and top 3) bought after its 13F is filed, held to its next filing,
                 equal weight across filers.
2. conviction:   stocks held at 7.5%+ of a long-term filer's book, ranked by how many filers hold them that way;
                 rebalanced 47 days after each quarter end with the filings known by then.
4. Pelosi pot:   every stock Pelosi discloses buying (shares or calls) bought after the filing, held until a disclosed
                 sale or 252 sessions, equal weight, cash when nothing is held.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COST_BP = 5
FROM = "2019-01-01"
# Quant and macro books: not the "long-term fundamental" funds the conviction papers use.
NOT_LONG_TERM = {"0001037389", "0001350694"}   # Renaissance, Bridgewater
FUNDS = {"SPY", "IVV", "VOO", "QQQ", "IWM", "GLD", "EEM", "EFA", "VTI"}   # index ETFs are not anyone's stock pick


def load_prices(cache: Path) -> pd.DataFrame:
    cols = {p.stem: pd.read_csv(p, index_col=0, parse_dates=True)["adj"] for p in sorted((cache / "prices").glob("*.csv"))}
    px = pd.DataFrame(cols).sort_index()
    px = px[px.index >= "2013-01-01"]
    return px.loc[px["SPY"].notna()]


def stats(r: pd.Series, bench: pd.Series) -> dict:
    r = r.dropna()
    b = bench.reindex(r.index).fillna(0.0)
    years = len(r) / 252
    eq, beq = (1 + r).cumprod(), (1 + b).cumprod()
    beta = float(np.cov(r, b)[0, 1] / b.var())
    resid = r - beta * b
    return {"start": str(r.index[0].date()), "end": str(r.index[-1].date()),
            "cagr": float(eq.iloc[-1] ** (1 / years) - 1), "spy_cagr": float(beq.iloc[-1] ** (1 / years) - 1),
            "sharpe": float(r.mean() / r.std() * math.sqrt(252)), "spy_sharpe": float(b.mean() / b.std() * math.sqrt(252)),
            "max_dd": float((eq / eq.cummax() - 1).min()), "spy_max_dd": float((beq / beq.cummax() - 1).min()),
            "beta": beta, "alpha": float(resid.mean() * 252),
            "alpha_t": float(resid.mean() / (resid.std() / math.sqrt(len(resid)))),
            "years": {str(y): float((1 + g).prod() - 1) for y, g in r.groupby(r.index.year)},
            "spy_years": {str(y): float((1 + g).prod() - 1) for y, g in b.groupby(b.index.year)}}


def next_day(cal: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    i = cal.searchsorted(d, side="right")
    return cal[i] if i < len(cal) else None


def weights_returns(px: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    """Daily returns of holding target weights w (set at each close, earned the next day), cost on weight changes."""
    rets = px[w.columns].pct_change().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    r = (held * rets).sum(axis=1)
    turnover = w.fillna(0.0).diff().abs().sum(axis=1)
    return r - turnover.shift(1).fillna(0.0) * COST_BP / 1e4


def ticker(cusips: dict, cusip: str) -> str:
    return (cusips.get(cusip) or "").replace(".", "-").replace("/", "-")


def load_13f(cache: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((cache / "13f").glob("*.json"))]


# ---------------------------------------------------------------- 1. best ideas

def best_ideas(filers: list[dict], cusips: dict, px: pd.DataFrame, top: int) -> tuple[pd.Series, dict]:
    """Each filer's `top` biggest priced positions, from the close after its filing to its next one; filers equal weight."""
    cal = px.index
    books: dict[str, list[tuple[pd.Timestamp, list[str]]]] = {}
    for f in filers:
        for q in f["quarters"]:
            day = next_day(cal, pd.Timestamp(q["filed"]))
            if day is None:
                continue
            names = [t for t in (ticker(cusips, h["cusip"]) for h in q["top"])
                     if t in px.columns and t not in FUNDS and pd.notna(px.at[day, t])]
            names = list(dict.fromkeys(names))[:top]
            if names:
                books.setdefault(f["cik"], []).append((day, names))
    cols = sorted({t for b in books.values() for _, ns in b for t in ns})
    w = pd.DataFrame(0.0, index=cal, columns=cols)
    live = pd.Series(0.0, index=cal)
    for b in books.values():
        b.sort()
        for k, (d0, ns) in enumerate(b):
            d1 = b[k + 1][0] if k + 1 < len(b) else cal[-1] + pd.Timedelta(days=1)
            sl = (cal >= d0) & (cal < d1)
            w.loc[sl, ns] += 1 / len(ns)
            live[sl] += 1
    w = w.div(live.replace(0, np.nan), axis=0).fillna(0.0)
    picks = {}
    for b in books.values():
        for d0, ns in b:
            picks.setdefault(str(d0.date())[:7], set()).update(ns)
    return weights_returns(px, w), {"names_now": sorted(c for c in cols if w[c].iloc[-1] > 0)}


# ---------------------------------------------------------------- 2. conviction + consensus

def conviction(filers: list[dict], cusips: dict, px: pd.DataFrame, min_w: float = 0.075, max_n: int = 50,
               min_funds: int = 1) -> tuple[pd.Series, dict]:
    cal = px.index
    filings = []   # (filed, period, cik, {ticker: weight})
    for f in filers:
        if f["cik"] in NOT_LONG_TERM:
            continue
        for q in f["quarters"]:
            tot = q.get("stock_total") or q.get("total")
            if not tot:
                continue
            ws: dict[str, float] = {}
            for h in q["top"]:
                t = ticker(cusips, h["cusip"])
                if t:
                    ws[t] = ws.get(t, 0.0) + h["value"] / tot
            filings.append((pd.Timestamp(q["filed"]), pd.Timestamp(q["period"]), f["cik"], ws))
    periods = sorted({p for _, p, _, _ in filings})
    w = pd.DataFrame(0.0, index=cal, columns=sorted(px.columns))
    hist = []
    for k, per in enumerate(periods):
        reb = per + pd.Timedelta(days=47)
        day = next_day(cal, reb - pd.Timedelta(days=1))
        if day is None:
            continue
        latest: dict[str, dict] = {}
        for filed, p, cik, ws in filings:          # each fund's most recent filing known on the rebalance day
            if filed <= reb and p <= per and (cik not in latest or p > latest[cik][0]):
                latest[cik] = (p, ws)
        score: dict[str, list] = {}
        for p, ws in latest.values():
            if p < per - pd.Timedelta(days=100):
                continue                            # fund stopped filing
            for t, x in ws.items():
                if x >= min_w and t not in FUNDS:
                    score.setdefault(t, []).append(x)
        ranked = sorted(score, key=lambda t: (-len(score[t]), -sum(score[t])))
        names = [t for t in ranked if len(score[t]) >= min_funds and t in px.columns and pd.notna(px.at[day, t])][:max_n]
        nxt = next_day(cal, periods[k + 1] + pd.Timedelta(days=46)) if k + 1 < len(periods) else None
        sl = (cal >= day) & (cal < nxt) if nxt is not None else cal >= day
        if names:
            w.loc[sl, names] = 1 / len(names)
        hist.append((str(day.date()), len(names)))
    w = w.loc[:, (w != 0).any()]
    return weights_returns(px, w), {"books": hist[-6:], "names_now": sorted(c for c in w.columns if w[c].iloc[-1] > 0)}


# ---------------------------------------------------------------- 4. Pelosi pot

def ptr_rows(cache: Path) -> pd.DataFrame:
    recs = []
    for p in sorted((cache / "ptr").glob("*.json")):
        for doc_id, d in json.loads(p.read_text())["docs"].items():
            for r in d["rows"]:
                recs.append({"doc_id": doc_id, "member": d["name"], "last": d["last"], "filed": d["filed"], **r})
    df = pd.DataFrame(recs)
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].fillna("").str.upper().str.replace("/", "-").str.replace(".", "-")
    return df


def pelosi(rows: pd.DataFrame, px: pd.DataFrame, hold: int = 252, when: str = "filed") -> tuple[pd.Series, dict]:
    cal = px.index
    pos = {d: i for i, d in enumerate(cal)}
    p = rows[(rows["last"] == "Pelosi") & rows["ticker"].isin(px.columns)].sort_values(when)
    buys = p[p["type"] == "P"]
    sells = p[p["type"].astype(str).str.startswith("S")]
    w = pd.DataFrame(0.0, index=cal, columns=sorted(set(buys["ticker"])))
    trades, seen = [], set()
    for b in buys.itertuples():
        if (b.ticker, b.date) in seen:     # shares and calls bought the same day are one signal
            continue
        seen.add((b.ticker, b.date))
        d0 = next_day(cal, getattr(b, when))
        if d0 is None:
            continue
        i0 = pos[d0]
        later = sells[(sells["ticker"] == b.ticker) & (sells["date"] > b.date)]
        i1 = i0 + hold                     # sold at the close of day i1 (past the end = still held)
        if len(later):
            ds = next_day(cal, getattr(later.iloc[0], when))
            if ds is not None:
                i1 = min(i1, pos[ds])
        w.iloc[i0:i1, w.columns.get_loc(b.ticker)] = 1.0
        stop = cal[min(i1, len(cal) - 1)]
        trades.append({"ticker": b.ticker, "asset_type": b.asset_type, "trade": str(b.date.date()), "entry": str(d0.date()),
                       "exit": str(stop.date()), "ret": float(px.at[stop, b.ticker] / px.at[d0, b.ticker] - 1),
                       "spy": float(px.at[stop, "SPY"] / px.at[d0, "SPY"] - 1)})
    n = (w > 0).sum(axis=1)
    w = w.div(n.replace(0, np.nan), axis=0).fillna(0.0)
    r = weights_returns(px, w)
    first = n[n > 0].index[0]
    return r[r.index > first], {"buys": len(trades), "invested_share": float((n[n.index > first] > 0).mean()),
                                "trades": trades, "names_now": sorted(c for c in w.columns if w[c].iloc[-1] > 0)}


def main(cache: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    px = load_prices(cache)
    spy = px["SPY"].pct_change()
    cusips = json.loads((cache / "cusips.json").read_text())
    filers = load_13f(cache)
    res, eq = {}, {}

    def add(name: str, r: pd.Series, extra: dict) -> None:
        for span, start in (("2019", FROM), ("2013", "2013-09-01")):
            rr = r[r.index >= max(pd.Timestamp(start), r.index[0])]
            res.setdefault(name, {})[span] = stats(rr, spy)
        res[name]["info"] = {k: v for k, v in extra.items() if k != "trades"}
        if "trades" in extra:
            pd.DataFrame(extra["trades"]).to_csv(out / f"{name}_trades.csv", index=False)
        eq[name] = (1 + r).cumprod()

    for top in (1, 3):
        add(f"best_ideas_top{top}", *best_ideas(filers, cusips, px, top))
    add("conviction_7.5pct", *conviction(filers, cusips, px))
    add("conviction_2plus_funds", *conviction(filers, cusips, px, min_w=0.05, min_funds=2))
    rows = ptr_rows(cache)
    for hold in (126, 252):
        add(f"pelosi_hold{hold}", *pelosi(rows, px, hold))
    add("pelosi_hold252_tradedate", *pelosi(rows, px, 252, when="date"))
    eq["SPY"] = (1 + spy.fillna(0.0)).cumprod()
    pd.DataFrame(eq).to_csv(out / "equity.csv", float_format="%.5f")
    (out / "results.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        s = v["2019"]
        print(f"{k:28s} 2019+: {s['cagr']:+.1%} vs SPY {s['spy_cagr']:+.1%}  Sharpe {s['sharpe']:.2f}/{s['spy_sharpe']:.2f}  "
              f"DD {s['max_dd']:.0%}/{s['spy_max_dd']:.0%}  beta {s['beta']:.2f} alpha {s['alpha']:+.1%} (t {s['alpha_t']:.1f})"
              f" | 2013+: {v['2013']['cagr']:+.1%} vs {v['2013']['spy_cagr']:+.1%} alpha {v['2013']['alpha']:+.1%} (t {v['2013']['alpha_t']:.1f})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
