"""Activist 13D backtest: buy a stock at the first close after a known activist files an original Schedule 13D on it
(a 5%+ stake with intent to push for change), hold 6 or 12 months, 5 bp a side. Baseline: every original 13D.

  python activist13d.py CACHE_DIR OUT_DIR [--from 2019]

Reads SEC EDGAR's quarterly master index (every filing, free; needs SEC_USER_AGENT) and maps subject companies to tickers
with SEC's current ticker list, so companies that were later delisted or acquired drop out (the result looks better than
it was for losers, and worse for takeover winners). Prices are Yahoo daily adjusted closes, cached in CACHE_DIR/prices.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from backtest import next_day, stats, weights_returns

EDGAR = "https://www.sec.gov/Archives/edgar/full-index"
FORMS = {"SC 13D", "SCHEDULE 13D"}      # originals only; SCHEDULE 13D is the form name since the 2025 EDGAR change
ACTIVISTS = {
    "Icahn": r"ICAHN", "Elliott": r"ELLIOTT (INVESTMENT|ASSOCIATES|INTERNATIONAL|MANAGEMENT)", "Starboard": r"STARBOARD VALUE",
    "Trian": r"TRIAN (FUND|PARTNERS)", "Jana": r"JANA PARTNERS", "Pershing Square": r"PERSHING SQUARE",
    "Third Point": r"THIRD POINT", "ValueAct": r"VALUEACT", "Corvex": r"CORVEX", "Sachem Head": r"SACHEM HEAD",
    "Land & Buildings": r"LAND (&|AND) BUILDINGS", "Engaged": r"ENGAGED CAPITAL", "Ancora": r"ANCORA",
    "Legion": r"LEGION PARTNERS", "Mantle Ridge": r"MANTLE RIDGE", "Cevian": r"CEVIAN", "Engine": r"ENGINE (CAPITAL|NO)",
    "Irenic": r"IRENIC", "Blackwells": r"BLACKWELLS", "Barington": r"BARINGTON", "Sarissa": r"SARISSA",
    "Pale Fire": r"PALE FIRE", "Voce": r"VOCE CAPITAL", "Politan": r"POLITAN", "Inclusive": r"INCLUSIVE CAPITAL",
}
ACT_RE = re.compile("|".join(f"(?P<g{i}>{p})" for i, p in enumerate(ACTIVISTS.values())))
NAMES = list(ACTIVISTS)


def sec_get(url: str) -> bytes:
    ua = os.environ.get("SEC_USER_AGENT") or "experiment-1 research admin@example.com"
    for k in range(4):
        r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=60)
        if r.status_code == 200:
            return r.content
        time.sleep(2 ** k)
    r.raise_for_status()
    return b""


def parse_master(text: str) -> pd.DataFrame:
    """Rows of a master.idx (CIK|Company Name|Form Type|Date Filed|Filename) whose form is an original 13D."""
    recs = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) == 5 and parts[2].strip() in FORMS:
            recs.append({"cik": int(parts[0]), "name": parts[1].strip().upper(), "form": parts[2].strip(),
                         "filed": parts[3], "file": parts[4].strip().rsplit("/", 1)[-1]})   # the accession, shared by all rows
    return pd.DataFrame(recs, columns=["cik", "name", "form", "filed", "file"])


def filings(cache: Path, start: int) -> pd.DataFrame:
    (cache / "edgar").mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today()
    frames = []
    for y in range(start, today.year + 1):
        for q in range(1, 5):
            if pd.Timestamp(y, 3 * q - 2, 1) > today:
                break
            f = cache / "edgar" / f"{y}Q{q}.csv"
            current = y == today.year and q == (today.month - 1) // 3 + 1
            if not f.exists() or current:
                raw = sec_get(f"{EDGAR}/{y}/QTR{q}/master.idx")
                parse_master(raw.decode("latin-1")).to_csv(f, index=False)
                time.sleep(0.2)
            frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)


def pair_up(rows: pd.DataFrame, tickers: dict[int, str]) -> pd.DataFrame:
    """One event per filing: the subject is the listed company in it that is not the filer.
    A filing is listed once under the subject and once under each reporting person, with the same accession."""
    out = []
    for file, g in rows.groupby("file"):
        acts = [m for m in (ACT_RE.search(n) for n in g["name"]) if m]
        act = NAMES[int(acts[0].lastgroup[1:])] if acts else None
        subj = g[~g["name"].map(lambda n: bool(ACT_RE.search(n))) & g["cik"].isin(tickers)]
        if len(subj) != 1 or len(g) < 2:
            continue                                          # unlisted subject, or can't tell who is who
        s = subj.iloc[0]
        out.append({"filed": s["filed"], "ticker": tickers[s["cik"]], "subject": s["name"], "activist": act, "file": file})
    df = pd.DataFrame(out, columns=["filed", "ticker", "subject", "activist", "file"])
    df["filed"] = pd.to_datetime(df["filed"])
    # the first original 13D on a company within a year is the event; later ones are the same campaign or a new group member
    df = df.sort_values("filed")
    keep, last = [], {}
    for i, r in df.iterrows():
        key = (r["ticker"], r["activist"] or "*")
        if key not in last or (r["filed"] - last[key]).days > 365:
            keep.append(i)
        last[key] = r["filed"]
    return df.loc[keep].reset_index(drop=True)


def sec_tickers() -> dict[int, str]:
    j = json.loads(sec_get("https://www.sec.gov/files/company_tickers.json"))
    out: dict[int, str] = {}
    for v in j.values():
        out.setdefault(int(v["cik_str"]), v["ticker"].replace(".", "-").replace("/", "-"))
    return out


def fetch_yahoo(t: str, start: str) -> pd.Series | None:
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    try:
        r = requests.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{t}",
                         params={"period1": t0, "period2": int(time.time()) + 86400, "interval": "1d", "includeAdjustedClose": "true"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        res = r.json()["chart"]["result"][0]
        adj = res["indicators"]["adjclose"][0]["adjclose"]
        idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
        s = pd.Series(adj, index=idx, dtype=float)
        return s[~s.index.duplicated(keep="last")].dropna()
    except Exception:  # noqa: BLE001 - delisted or unknown symbol
        return None


def prices(cache: Path, tickers: set[str], start: str) -> pd.DataFrame:
    d = cache / "prices"
    d.mkdir(parents=True, exist_ok=True)
    cols, missing = {}, []
    for t in sorted(tickers | {"SPY"}):
        f = d / f"{t}.csv"
        if f.exists() and pd.Timestamp.fromtimestamp(f.stat().st_mtime) > pd.Timestamp.now() - pd.Timedelta(days=1):
            cols[t] = pd.read_csv(f, index_col=0, parse_dates=True)["adj"]
            continue
        s = fetch_yahoo(t, start)
        time.sleep(0.15)
        if s is None or s.empty:
            missing.append(t)
            continue
        s.rename("adj").to_csv(f, index_label="date")
        cols[t] = s
    px = pd.DataFrame(cols).sort_index()
    px = px.loc[px["SPY"].notna()]
    print(f"prices: {len(cols)} tickers, {len(missing)} not on Yahoo")
    return px


def event_study(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    cal = px.index
    pos = {d: i for i, d in enumerate(cal)}
    recs = []
    for e in ev.itertuples():
        if e.ticker not in px.columns:
            continue
        d0 = next_day(cal, e.filed)
        if d0 is None or pd.isna(px.at[d0, e.ticker]):
            continue
        i0 = pos[d0]
        rec = {"ticker": e.ticker, "activist": e.activist, "entry": d0}
        # filing-day jump that a copier can't get: close before the filing to the entry close
        if i0 >= 2 and pd.notna(px[e.ticker].iloc[i0 - 2]):
            rec["jump"] = px[e.ticker].iloc[i0] / px[e.ticker].iloc[i0 - 2] - 1 - (px["SPY"].iloc[i0] / px["SPY"].iloc[i0 - 2] - 1)
        for h in (21, 63, 126, 252):
            if i0 + h < len(cal):
                p1 = px[e.ticker].iloc[:i0 + h + 1].dropna().iloc[-1]   # delisted mid-way: last price
                rec[f"x{h}"] = p1 / px.at[d0, e.ticker] - 1 - (px["SPY"].iloc[i0 + h] / px["SPY"].iloc[i0] - 1)
        recs.append(rec)
    return pd.DataFrame(recs)


def portfolio(es: pd.DataFrame, px: pd.DataFrame, hold: int) -> pd.Series:
    cal = px.index
    pos = {d: i for i, d in enumerate(cal)}
    w = pd.DataFrame(0.0, index=cal, columns=sorted(set(es["ticker"])))
    for e in es.itertuples():
        i0 = pos[e.entry]
        w.iloc[i0:i0 + hold, w.columns.get_loc(e.ticker)] = 1.0
    n = (w > 0).sum(axis=1)
    w = w.div(n.replace(0, np.nan), axis=0).fillna(0.0)
    r = weights_returns(px.ffill(limit=5), w)
    first = n[n > 0].index[0]
    return r[r.index > first]


def summarize(es: pd.DataFrame) -> dict:
    out = {"n": len(es)}
    if "jump" in es:
        out["jump_missed"] = float(es["jump"].mean())
    for h in (21, 63, 126, 252):
        x = es.get(f"x{h}", pd.Series(dtype=float)).dropna()
        if len(x) > 5:
            out[f"{h}d"] = {"mean": float(x.mean()), "median": float(x.median()), "t": float(x.mean() / x.std() * np.sqrt(len(x))),
                           "hit": float((x > 0).mean()), "n": len(x)}
    return out


def main(cache: Path, out: Path, start: int = 2019) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = filings(cache, start)
    tk = sec_tickers()
    ev = pair_up(rows, tk)
    print(f"13D originals: {rows['file'].nunique()} filings, {len(ev)} events on listed companies, "
          f"{ev['activist'].notna().sum()} by the {len(ACTIVISTS)} activists")
    ev.to_csv(out / "events_13d.csv", index=False)
    if ev.empty:
        raise SystemExit("no 13D could be matched to a listed subject: the index format may have changed")
    act = ev[ev["activist"].notna()]
    px = prices(cache, set(ev["ticker"]), f"{start - 1}-06-01")
    spy = px["SPY"].pct_change()
    res = {}
    for name, sub in (("activists", act), ("all_13d", ev)):
        es = event_study(sub, px)
        es.to_csv(out / f"study_{name}.csv", index=False)
        res[name] = {"events": summarize(es)}
        if es.empty:
            raise SystemExit(f"{name}: no event has a Yahoo price")
        for hold in (126, 252):
            r = portfolio(es, px, hold)
            res[name][f"hold{hold}"] = stats(r, spy)
    res["by_activist"] = {a: summarize(g) for a, g in event_study(act, px).groupby("activist")}
    (out / "results_13d.json").write_text(json.dumps(res, indent=1, default=str))
    for name in ("activists", "all_13d"):
        e = res[name]["events"]
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v)
                                for k, v in e.items()}))
        for hold in (126, 252):
            s = res[name][f"hold{hold}"]
            print(f"  hold {hold}: {s['start']}..{s['end']} CAGR {s['cagr']:+.1%} vs SPY {s['spy_cagr']:+.1%}, Sharpe {s['sharpe']:.2f}/{s['spy_sharpe']:.2f}, "
                  f"DD {s['max_dd']:.0%}, beta {s['beta']:.2f}, alpha {s['alpha']:+.1%} (t {s['alpha_t']:.1f})")
    for a, s in sorted(res["by_activist"].items(), key=lambda kv: -kv[1]["n"]):
        x = s.get("252d", {})
        print(f"  {a:18s} n={s['n']:3d} 252d excess {x.get('mean', float('nan')):+.1%} (t {x.get('t', float('nan')):.1f})")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), Path(a[1]), int(a[3]) if len(a) > 3 and a[2] == "--from" else 2019)
