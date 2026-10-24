"""Backtest the two signals in Jake Schincariol's trader-skill pack against our paper setup.

trader-skill (github.com/Jakeschincariol/trader-skill, MIT) is 11 Claude skills. Nine of them are plumbing (budget sizing,
risk limits, order tickets, tax lots, alerts, rebalancing, screening, earnings facts, news). Two produce a trading signal
we can test on history:

  copycat  copy a famous investor's 13F top holdings (trader-copycat/edgar13f.py reads SEC EDGAR)
  pelosi   copy the stock buys members of the House disclose in their PTRs (trader-pelosi/ptr.py reads the House Clerk)

This script uses the pack's own readers (pinned commit, loaded from --skill DIR) to collect the history, then asks what a
copier could actually have earned: you only see a 13F when it is filed (up to 45 days after the quarter) and a PTR when it
is disclosed (up to 45 days after the trade). Each test is also run "with hindsight" (entering on the quarter end or the
trade date) to show how much of the edge the delay costs.

    python run.py collect13f --skill DIR --cache cache
    python run.py collectptr --skill DIR --cache cache --years 2019-2026
    python run.py prices --cache cache
    python run.py backtest --cache cache --out results

Prices are Yahoo daily adjusted closes (Tiingo's free tier allows 500 symbols a month and Exp1/Exp2 need them). Yahoo only
has stocks that still trade, so names that were delisted drop out: that biases both tests upward, and coverage is reported.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("trader_skill")

START = "2013-06-30"          # 13F information tables are XML from Q2 2013
COST_BP = 5.0                 # per side, on turnover
HORIZONS = (5, 21, 63, 126, 252)
HOLD_DAYS = 126               # Congress portfolio: each disclosed buy is held about six months
FILER_TOP = (5, 10)


def load_skill(skill: Path, folder: str, name: str):
    spec = importlib.util.spec_from_file_location("ts_" + name, skill / folder / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 13F collection

def all_13f(e, fetcher, cik: str) -> tuple[str, list[dict]]:
    """Every 13F filing of a filer, including the older pages the submissions API splits off."""
    sub = fetcher.get_json(e.SUBMISSIONS % e.cik10(cik))
    filings = e.list_filings(sub)
    for extra in sub.get("filings", {}).get("files", []):
        page = fetcher.get_json("https://data.sec.gov/submissions/" + extra["name"])
        filings += e.list_filings({"filings": {"recent": page}})
    seen, out = set(), []
    for f in sorted(filings, key=lambda f: (f["filed"], f["accession"])):
        if f["accession"] not in seen:
            seen.add(f["accession"])
            out.append(f)
    return sub.get("name") or cik, out


def quarter_top(e, fetcher, cik: str, f: dict, keep: int = 25) -> dict:
    base = e.archive_url(cik, f["accession"])
    table = e.info_table_name(fetcher.get_json(base + "index.json"))
    rows = e.parse_info_table(fetcher.get(base + table), multiplier=e.value_multiplier(f["filed"]))
    positions, total = e.aggregate(rows)
    stocks = [p for p in positions if e.is_stock(p)]
    stock_total = sum(p["value"] for p in stocks)
    top = [{"cusip": p["cusip"], "issuer": p["issuer"], "title": p["title"], "value": p["value"]} for p in stocks[:keep]]
    return {"period": f["period"], "filed": f["filed"], "accession": f["accession"], "total": total, "stock_total": stock_total,
            "n_positions": len(stocks), "top": top}


def collect13f(args) -> None:
    e = load_skill(args.skill, "trader-copycat", "edgar13f")
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        sys.exit("SEC_USER_AGENT is not set")
    fetcher = e.Fetcher(ua=ua, pause=0.15)
    filers = e.load_filers()
    out_dir = args.cache / "13f"
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in filers:
        cik = row["cik"]
        path = out_dir / f"{cik}.json"
        old = json.loads(path.read_text()) if path.exists() else {"quarters": []}
        done = {q["accession"] for q in old["quarters"]}
        try:
            name, filings = all_13f(e, fetcher, cik)
        except e.EdgarError as exc:
            log.warning("%s: %s", cik, exc)
            continue
        quarters = list(old["quarters"])
        for f in filings:
            if f["form"] != "13F-HR" or f["period"] < START or f["accession"] in done:
                continue
            try:
                quarters.append(quarter_top(e, fetcher, cik, f))
            except Exception as exc:  # noqa: BLE001 - one bad filing must not stop the rest
                log.warning("%s %s: %s", name, f["accession"], exc)
        # Two originals for one period (a refiling): keep the first one filed, it is what a copier saw first.
        by_period = {}
        for q in sorted(quarters, key=lambda q: (q["filed"], q["accession"])):
            by_period.setdefault(q["period"], q)
        quarters = sorted(by_period.values(), key=lambda q: q["filed"])
        path.write_text(json.dumps({"cik": cik, "name": name, "person": row.get("person"), "quarters": quarters}))
        log.info("%s: %d quarters", name, len(quarters))
    map_cusips(e, fetcher, args.cache)


def map_cusips(e, fetcher, cache: Path) -> None:
    """CUSIP -> ticker with OpenFIGI first (it knows old CUSIPs), SEC's ticker list by name second."""
    path = cache / "cusips.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    issuers = {}
    for p in sorted((cache / "13f").glob("*.json")):
        for q in json.loads(p.read_text())["quarters"]:
            for t in q["top"][:10]:
                issuers.setdefault(t["cusip"], t)
    todo = [c for c in issuers if c not in known]
    log.info("mapping %d new CUSIPs (%d known)", len(todo), len(known))
    figi = {}
    for i in range(0, len(todo), 10):
        for _ in range(3):
            try:
                figi.update(e.figi_lookup(fetcher, todo[i:i + 10]))
                break
            except e.EdgarError as exc:        # 25 requests a minute without a key: back off and retry the chunk
                log.warning("openfigi: %s; waiting a minute", exc)
                time.sleep(60)
        time.sleep(2.5)
    index = e.ticker_index(fetcher)
    for c in todo:
        p = dict(issuers[c], put_call="", share_type="SH", figi="")
        e.resolve_tickers([p], index, {}, figi)
        known[c] = p.get("ticker")
    path.write_text(json.dumps(known, indent=0, sort_keys=True))


# ---------------------------------------------------------------- House PTR collection

def collectptr(args) -> None:
    ptr = load_skill(args.skill, "trader-pelosi", "ptr")
    if not shutil.which("pdftotext"):
        sys.exit("pdftotext (poppler-utils) is needed")
    out_dir = args.cache / "ptr"
    out_dir.mkdir(parents=True, exist_ok=True)
    y0, y1 = (int(x) for x in args.years.split("-"))
    deadline = time.time() + args.minutes * 60
    for year in range(y0, y1 + 1):
        path = out_dir / f"{year}.json"
        old = json.loads(path.read_text()) if path.exists() else {"docs": {}}
        docs = old["docs"]
        index = ptr.select(ptr.load_index(year), kind="P")
        log.info("%d: %d PTRs in the index, %d already read", year, len(index), len(docs))
        with tempfile.TemporaryDirectory() as tmp:
            for r in index:
                if r["doc_id"] in docs:
                    continue
                if time.time() > deadline:
                    log.warning("time budget used; the next run carries on")
                    break
                rec = {"name": r["name"], "last": r["last"], "first": r["first"], "filed": r["filed"], "rows": []}
                if not r["doc_id"].startswith("2"):
                    rec["skip"] = "scanned"      # 8xxxxxxx = a scanned paper form: no text to read
                    docs[r["doc_id"]] = rec
                    continue
                try:
                    _, report = ptr._fetch_and_read(r["doc_id"], r["year"], tmp, filed=r["filed"])
                    rec["rows"] = [{k: x.get(k) for k in ("owner", "asset", "ticker", "asset_type", "type", "date", "amount_low", "amount_high")}
                                   for x in (report or {}).get("rows", [])]
                    os.remove(os.path.join(tmp, r["doc_id"] + ".pdf"))
                except Exception as exc:  # noqa: BLE001
                    rec["skip"] = f"error: {exc}"[:200]
                docs[r["doc_id"]] = rec
                time.sleep(0.2)
        path.write_text(json.dumps({"year": year, "docs": docs}))
        if time.time() > deadline:
            break


def ptr_rows(cache: Path) -> pd.DataFrame:
    recs = []
    for p in sorted((cache / "ptr").glob("*.json")):
        for doc_id, d in json.loads(p.read_text())["docs"].items():
            for r in d["rows"]:
                recs.append({"doc_id": doc_id, "member": d["name"], "last": d["last"], "filed": d["filed"], **r})
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].fillna("").str.upper().str.replace("/", "-").str.replace(".", "-")
    # A stock and its options bought the same day in one filing (Pelosi does this) count as one signal.
    return df.drop_duplicates(["doc_id", "ticker", "type", "date"]).reset_index(drop=True)


def ptr_coverage(cache: Path) -> dict:
    n = scanned = err = rows = 0
    for p in sorted((cache / "ptr").glob("*.json")):
        for d in json.loads(p.read_text())["docs"].values():
            n += 1
            skip = d.get("skip") or ""
            scanned += skip == "scanned"
            err += skip.startswith("error")
            rows += len(d["rows"])
    return {"ptr_filings": n, "scanned_paper": scanned, "errors": err, "transactions": rows}


# ---------------------------------------------------------------- prices

def fetch_yahoo(ticker: str, start: str = "2012-12-01") -> pd.Series | None:
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    for attempt in range(4):
        try:
            r = requests.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
                             params={"period1": t0, "period2": int(time.time()), "interval": "1d", "includeAdjustedClose": "true"},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            time.sleep(5)
            continue
        res = (r.json().get("chart", {}).get("result") or [None])[0]
        if not res or not res.get("timestamp"):
            return None
        adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
        if not adj:
            return None
        idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
        s = pd.Series(adj, index=idx, dtype=float).dropna()
        return s[~s.index.duplicated(keep="last")]
    return None


def wanted_tickers(cache: Path) -> set[str]:
    out = {"SPY"}
    cus = cache / "cusips.json"
    if cus.exists():
        out |= {t.replace(".", "-").replace("/", "-") for t in json.loads(cus.read_text()).values() if t}
    df = ptr_rows(cache) if (cache / "ptr").exists() else pd.DataFrame()
    if not df.empty:
        out |= set(df.loc[df.ticker.str.fullmatch(r"[A-Z][A-Z0-9\-]{0,7}") & df.type.isin(["P", "S", "S (partial)"]), "ticker"])
    return out


def prices(args) -> None:
    d = args.cache / "prices"
    d.mkdir(parents=True, exist_ok=True)
    missing_path = args.cache / "no_prices.json"
    missing = set(json.loads(missing_path.read_text())) if missing_path.exists() else set()
    fresh = time.time() - 20 * 3600
    todo = sorted(t for t in wanted_tickers(args.cache)
                  if t not in missing and not ((d / f"{t}.csv").exists() and (d / f"{t}.csv").stat().st_mtime > fresh))
    log.info("fetching %d tickers from Yahoo", len(todo))
    deadline = time.time() + args.minutes * 60
    for i, t in enumerate(todo):
        if time.time() > deadline:
            log.warning("time budget used at %d/%d", i, len(todo))
            break
        s = fetch_yahoo(t)
        if s is None or len(s) < 5:
            missing.add(t)
        else:
            s.rename("adj").to_csv(d / f"{t}.csv", index_label="date")
        time.sleep(0.25)
    missing_path.write_text(json.dumps(sorted(missing)))


def load_prices(cache: Path) -> pd.DataFrame:
    cols = {p.stem: pd.read_csv(p, index_col=0, parse_dates=True)["adj"] for p in sorted((cache / "prices").glob("*.csv"))}
    px = pd.DataFrame(cols).sort_index()
    px = px[px.index >= "2013-01-01"]
    return px.loc[px["SPY"].notna()]


# ---------------------------------------------------------------- stats

def stats(r: pd.Series, bench: pd.Series) -> dict:
    r = r.dropna()
    b = bench.reindex(r.index).fillna(0.0)
    if len(r) < 20:
        return {"days": int(len(r))}
    years = len(r) / 252
    eq = (1 + r).cumprod()
    beq = (1 + b).cumprod()
    cagr = eq.iloc[-1] ** (1 / years) - 1
    bcagr = beq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * math.sqrt(252)
    beta = float(np.cov(r, b)[0, 1] / b.var()) if b.var() > 0 else float("nan")
    resid = r - beta * b
    alpha = resid.mean() * 252
    alpha_t = resid.mean() / (resid.std() / math.sqrt(len(resid))) if resid.std() > 0 else float("nan")
    return {"start": str(r.index[0].date()), "end": str(r.index[-1].date()), "days": int(len(r)),
            "cagr": round(float(cagr), 4), "spy_cagr": round(float(bcagr), 4), "excess_cagr": round(float(cagr - bcagr), 4),
            "vol": round(float(vol), 4), "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() > 0 else None,
            "spy_sharpe": round(float(b.mean() / b.std() * math.sqrt(252)), 2) if b.std() > 0 else None,
            "max_dd": round(float((eq / eq.cummax() - 1).min()), 4), "beta": round(beta, 2),
            "alpha": round(float(alpha), 4), "alpha_t": round(float(alpha_t), 2)}


def next_day(cal: pd.DatetimeIndex, d: pd.Timestamp, strictly_after: bool = True) -> pd.Timestamp | None:
    i = cal.searchsorted(d, side="right" if strictly_after else "left")
    return cal[i] if i < len(cal) else None


# ---------------------------------------------------------------- 13F backtest

def hold_series(px: pd.DataFrame, events: list[tuple[pd.Timestamp, dict[str, float]]], cost_bp: float = COST_BP) -> pd.Series:
    """Daily returns of buying the weights at each event's close and holding (weights drift) until the next event."""
    cal = px.index
    rets = pd.Series(0.0, index=cal)
    started = None
    prev_w: dict[str, float] = {}
    for k, (d0, w) in enumerate(events):
        d1 = events[k + 1][0] if k + 1 < len(events) else cal[-1]
        if d1 <= d0 or not w:
            continue
        seg = px.loc[d0:d1, list(w)].ffill()
        rel = seg / seg.iloc[0]
        val = (rel * pd.Series(w)).sum(axis=1)
        r = val.pct_change().iloc[1:]
        turnover = sum(abs(w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in set(w) | set(prev_w))
        if len(r):
            r.iloc[0] -= turnover * cost_bp / 1e4
        rets.loc[r.index] = r.values
        last = rel.iloc[-1] * pd.Series(w)
        prev_w = (last / last.sum()).to_dict()
        started = started or d0
    return rets[rets.index > started] if started is not None else rets.iloc[:0]


def filer_events(quarters: list[dict], cusips: dict, px: pd.DataFrame, top: int, when: str) -> tuple[list, list[float]]:
    cal = px.index
    events, coverage = [], []
    for q in quarters:
        names = q["top"][:top]
        if not names:
            continue
        d = pd.Timestamp(q["filed"] if when == "filed" else q["period"])
        day = next_day(cal, d, strictly_after=(when == "filed"))
        if day is None:
            continue
        w, tot = {}, sum(t["value"] for t in names)
        for t in names:
            tk = (cusips.get(t["cusip"]) or "").replace(".", "-").replace("/", "-")
            if tk in px.columns and pd.notna(px.at[day, tk]):
                w[tk] = w.get(tk, 0.0) + t["value"]
        got = sum(w.values())
        coverage.append(got / tot if tot else 0.0)
        if got > 0:
            events.append((day, {k: v / got for k, v in w.items()}))
    return events, coverage


def backtest13f(cache: Path, px: pd.DataFrame, out: Path) -> dict:
    cusips = json.loads((cache / "cusips.json").read_text())
    spy = px["SPY"].pct_change()
    res, daily = {}, {}
    for p in sorted((cache / "13f").glob("*.json")):
        f = json.loads(p.read_text())
        qs = f["quarters"]
        if len(qs) < 4:
            continue
        entry = {"name": f["name"], "person": f.get("person"), "quarters": len(qs), "first_period": qs[0]["period"],
                 "median_lag_days": int(np.median([(pd.Timestamp(q["filed"]) - pd.Timestamp(q["period"])).days for q in qs]))}
        for top in FILER_TOP:
            for when in ("filed", "period"):
                ev, cov = filer_events(qs, cusips, px, top, when)
                r = hold_series(px, ev)
                key = f"top{top}_{when}"
                entry[key] = stats(r, spy)
                entry[key]["weight_priced"] = round(float(np.mean(cov)), 3) if cov else None
                daily[f"{f['name']}|{key}"] = r
        res[f["cik"]] = entry
    df = pd.DataFrame(daily)
    comp = {}
    for key in [f"top{t}_{w}" for t in FILER_TOP for w in ("filed", "period")]:
        cols = [c for c in df.columns if c.endswith("|" + key)]
        r = df[cols].mean(axis=1).dropna()   # equal weight across the filers active that day
        comp[key] = stats(r, spy)
        df["ALL|" + key] = r
    out.mkdir(parents=True, exist_ok=True)
    eq = (1 + df.fillna(0.0)).cumprod()
    eq["SPY"] = (1 + spy.reindex(eq.index).fillna(0.0)).cumprod()
    eq.round(5).to_csv(out / "copycat_equity.csv", index_label="date")
    return {"filers": res, "all_filers_equal_weight": comp}


# ---------------------------------------------------------------- Congress backtest

def event_study(rows: pd.DataFrame, px: pd.DataFrame, when: str) -> pd.DataFrame:
    """Excess return over SPY after each transaction, from the close of the entry day."""
    cal = px.index
    pos = {d: i for i, d in enumerate(cal)}
    out = []
    for r in rows.itertuples():
        if r.ticker not in px.columns:
            continue
        d = r.filed if when == "filed" else r.date
        if pd.isna(d):
            continue
        day = next_day(cal, d, strictly_after=(when == "filed"))
        if day is None or pd.isna(px.at[day, r.ticker]):
            continue
        i = pos[day]
        rec = {"idx": r.Index, "entry": day}
        for h in HORIZONS:
            if i + h < len(cal):
                a, b = px[r.ticker].iloc[i], px[r.ticker].iloc[i + h]
                s0, s1 = px["SPY"].iloc[i], px["SPY"].iloc[i + h]
                if pd.notna(b):
                    rec[f"x{h}"] = b / a - s1 / s0
        out.append(rec)
    return pd.DataFrame(out)


def summarize_events(ev: pd.DataFrame) -> dict:
    out = {"n": int(len(ev))}
    for h in HORIZONS:
        x = ev.get(f"x{h}", pd.Series(dtype=float)).dropna()
        if len(x) < 10:
            continue
        # Many members buy the same stock in the same weeks, so t is an upper bound (events overlap).
        out[f"{h}d"] = {"n": int(len(x)), "mean": round(float(x.mean()), 4), "median": round(float(x.median()), 4),
                        "hit": round(float((x > 0).mean()), 3), "t": round(float(x.mean() / (x.std() / math.sqrt(len(x)))), 2)}
    return out


def congress_portfolio(rows: pd.DataFrame, ev: pd.DataFrame, px: pd.DataFrame, hold: int = HOLD_DAYS) -> pd.Series:
    """Equal weight in every stock with a disclosed buy in the last `hold` trading days (rebalanced daily, cost on entry/exit)."""
    cal = px.index
    rets = px.pct_change()
    held = pd.DataFrame(0.0, index=cal, columns=sorted(set(rows.loc[ev.idx, "ticker"])) if len(ev) else [])
    pos = {d: i for i, d in enumerate(cal)}
    for e in ev.itertuples():
        t = rows.at[e.idx, "ticker"]
        i = pos[e.entry]
        held.iloc[i + 1:i + 1 + hold, held.columns.get_loc(t)] = 1.0     # earn from the day after the entry close
    if held.empty:
        return pd.Series(dtype=float)
    held = (held > 0).astype(float)
    n = held.sum(axis=1)
    w = held.div(n.replace(0, np.nan), axis=0)
    r = (w * rets[held.columns].fillna(0.0)).sum(axis=1)
    turnover = w.fillna(0.0).diff().abs().sum(axis=1)
    r = r - turnover * COST_BP / 1e4
    active = n > 0
    first = active.idxmax() if active.any() else None
    return r[r.index >= first] if first is not None else r.iloc[:0]


def backtest_congress(cache: Path, px: pd.DataFrame, out: Path) -> dict:
    rows = ptr_rows(cache)
    spy = px["SPY"].pct_change()
    res = {"coverage": ptr_coverage(cache)}
    if rows.empty:
        return res
    stocks = rows[rows.asset_type.isin(["ST", "OP"]) | rows.asset_type.isna()]
    stocks = stocks[stocks.ticker.str.fullmatch(r"[A-Z][A-Z0-9\-]{0,7}")]
    buys = stocks[stocks.type == "P"]
    sells = stocks[stocks.type.isin(["S", "S (partial)"])]
    res["coverage"].update({"stock_buys": int(len(buys)), "stock_sells": int(len(sells)),
                            "buys_priced": int(buys.ticker.isin(px.columns).sum()),
                            "median_disclosure_lag_days": int((buys.filed - buys.date).dt.days.median())})
    groups = {
        "all_buys": buys,
        "big_buys_50k_plus": buys[buys.amount_low.fillna(0) >= 50001],
        "pelosi_buys": buys[buys["last"].str.lower() == "pelosi"],
        "all_sells": sells,
    }
    curves = {}
    for g, df in groups.items():
        res[g] = {}
        for when in ("filed", "date"):
            ev = event_study(df, px, when)
            res[g][f"events_{when}"] = summarize_events(ev) if len(ev) else {"n": 0}
            if g != "all_sells" and len(ev):
                r = congress_portfolio(df, ev, px)
                res[g][f"portfolio_{when}"] = stats(r, spy)
                curves[f"{g}|{when}"] = r
    # By year, for the copier's version: is the edge (if any) stable or one lucky stretch?
    ev = event_study(buys, px, "filed")
    if len(ev):
        ev["year"] = pd.to_datetime(ev.entry).dt.year
        res["all_buys"]["by_year_63d"] = {int(y): {"n": int(g["x63"].notna().sum()), "mean": round(float(g["x63"].mean()), 4)}
                                          for y, g in ev.groupby("year") if g["x63"].notna().sum() >= 10}
    out.mkdir(parents=True, exist_ok=True)
    eq = (1 + pd.DataFrame(curves).fillna(0.0)).cumprod()
    eq["SPY"] = (1 + spy.reindex(eq.index).fillna(0.0)).cumprod()
    eq.round(5).to_csv(out / "congress_equity.csv", index_label="date")
    return res


# ---------------------------------------------------------------- report

def pct(x) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.1f}%"


def report(res: dict) -> str:
    lines = ["# trader-skill signals, backtested", ""]
    c = res.get("copycat")
    if c:
        lines += ["## Copycat (13F top holdings, bought the day after filing, held to the next filing)", "",
                  "| filer | quarters | lag d | top10 CAGR | SPY | excess | Sharpe | alpha (t) | hindsight excess | priced |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for f in sorted(c["filers"].values(), key=lambda f: -(f.get("top10_filed", {}).get("excess_cagr") or -9)):
            a, h = f.get("top10_filed", {}), f.get("top10_period", {})
            if "cagr" not in a:
                continue
            lines.append(f"| {f['name']} | {f['quarters']} | {f['median_lag_days']} | {pct(a['cagr'])} | {pct(a['spy_cagr'])} | "
                         f"{pct(a['excess_cagr'])} | {a['sharpe']} | {pct(a['alpha'])} ({a['alpha_t']}) | {pct(h.get('excess_cagr'))} | "
                         f"{a.get('weight_priced')} |")
        lines += ["", "All filers, equal weight:", ""]
        for k, s in c["all_filers_equal_weight"].items():
            if "cagr" in s:
                lines.append(f"- {k}: CAGR {pct(s['cagr'])} vs SPY {pct(s['spy_cagr'])}, Sharpe {s['sharpe']} vs {s['spy_sharpe']}, "
                             f"alpha {pct(s['alpha'])} (t {s['alpha_t']}), beta {s['beta']}, max DD {pct(s['max_dd'])}")
        lines.append("")
    g = res.get("congress")
    if g:
        lines += ["## Congress (House PTRs)", "", f"Coverage: {json.dumps(g['coverage'])}", "",
                  "Excess return over SPY after the entry close (filed = copier enters after disclosure; date = hindsight, on the trade date):", "",
                  "| group | entry | n | 21d | 63d | 126d | 252d | 63d t | 63d hit |", "|---|---|---|---|---|---|---|---|---|"]
        for name in ("all_buys", "big_buys_50k_plus", "pelosi_buys", "all_sells"):
            for when in ("filed", "date"):
                e = g.get(name, {}).get(f"events_{when}", {})
                if not e.get("n"):
                    continue
                m = {h: e.get(f"{h}d", {}).get("mean") for h in HORIZONS}
                t63 = e.get("63d", {})
                lines.append(f"| {name} | {when} | {e['n']} | {pct(m[21])} | {pct(m[63])} | {pct(m[126])} | {pct(m[252])} | "
                             f"{t63.get('t', '')} | {t63.get('hit', '')} |")
        lines += ["", f"Portfolios (equal weight in each disclosed buy for {HOLD_DAYS} trading days, {COST_BP:.0f} bp a side):", ""]
        for name in ("all_buys", "big_buys_50k_plus", "pelosi_buys"):
            for when in ("filed", "date"):
                s = g.get(name, {}).get(f"portfolio_{when}")
                if s and "cagr" in s:
                    lines.append(f"- {name} ({when}): CAGR {pct(s['cagr'])} vs SPY {pct(s['spy_cagr'])}, Sharpe {s['sharpe']} vs {s['spy_sharpe']}, "
                                 f"alpha {pct(s['alpha'])} (t {s['alpha_t']}), beta {s['beta']}, max DD {pct(s['max_dd'])}, {s['start']} to {s['end']}")
        if "by_year_63d" in g.get("all_buys", {}):
            lines += ["", "All buys, 63-day excess by year (copier entry): " +
                      ", ".join(f"{y} {pct(v['mean'])} (n {v['n']})" for y, v in g["all_buys"]["by_year_63d"].items())]
    return "\n".join(lines) + "\n"


def backtest(args) -> None:
    px = load_prices(args.cache)
    log.info("prices: %d tickers, %s to %s", px.shape[1], px.index[0].date(), px.index[-1].date())
    res = {"generated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "skill_commit": args.skill_commit,
           "price_tickers": int(px.shape[1]), "cost_bp_per_side": COST_BP}
    if (args.cache / "13f").exists():
        res["copycat"] = backtest13f(args.cache, px, args.out)
    if (args.cache / "ptr").exists():
        res["congress"] = backtest_congress(args.cache, px, args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    text = report(res)
    (args.out / "summary.md").write_text(text)
    print(text)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect13f", "collectptr", "prices", "backtest"):
        p = sub.add_parser(name)
        p.add_argument("--cache", type=Path, default=Path("cache"))
        p.add_argument("--skill", type=Path, default=Path(os.environ.get("TRADER_SKILL", "trader-skill")))
        p.add_argument("--minutes", type=float, default=120)
        p.add_argument("--years", default="2019-2026")
        p.add_argument("--out", type=Path, default=Path("results"))
        p.add_argument("--skill-commit", default=os.environ.get("TRADER_SKILL_COMMIT", ""))
    args = ap.parse_args(argv)
    {"collect13f": collect13f, "collectptr": collectptr, "prices": prices, "backtest": backtest}[args.cmd](args)


if __name__ == "__main__":
    main()
