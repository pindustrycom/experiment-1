"""Does a stock drop on a CEO / leadership scandal get undone more than an ordinary drop of the same size?

    python scandal_study.py PANEL.pkl SCANDAL_WEEKS.jsonl OUT_DIR

Events: scandal_weeks.jsonl (collect_scandals.py: Google News headlines read by Claude Haiku), kept when the ticker was
an S&P 500 member on the event day. Event day t0 = the trading day, from the first headline of that event's week on,
with the most negative market-adjusted return within 3 trading days (the day the news hit the price).
Returns are market-adjusted with a 252-day beta to SPY (estimated on days -260..-10), cumulated from the close of t0
(you sell into the drop or buy it at the close). Baseline: every S&P 500 member-day 2012+ with a market-adjusted drop
in the same size bucket. Entry at the close of t0 or t0+1, 5 bp per side.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

H = [1, 5, 20, 60]
COST = 0.0005


def market_adjusted(P: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = P["close"].pct_change(fill_method=None)
    m = P["etf"]["SPY"].pct_change()
    cov = r.mul(m, axis=0).rolling(252, min_periods=150).mean() - r.rolling(252, min_periods=150).mean().mul(m.rolling(252, min_periods=150).mean(), axis=0)
    beta = (cov.div(m.rolling(252, min_periods=150).var(), axis=0)).shift(10).clip(0, 3).fillna(1.0)
    ar = r - beta.mul(m, axis=0)
    return r, ar


def fwd_car(ar: pd.DataFrame, h: int, skip: int = 0) -> pd.DataFrame:
    """Sum of abnormal returns over days t+1+skip .. t+h+skip, aligned on t."""
    c = ar.fillna(0).cumsum()
    return c.shift(-(h + skip)) - c.shift(-skip)


def bucket(x: float) -> str:
    for lo, name in [(-0.15, "<-15%"), (-0.10, "-15..-10%"), (-0.06, "-10..-6%"), (-0.03, "-6..-3%")]:
        if x <= lo:
            return name
    return ">-3%"


def load_events(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        w = json.loads(line)
        for e in w.get("events") or []:
            t = (e.get("ticker") or "").upper().strip().replace(".", "-")
            t = re.sub(r"^(NYSE|NASDAQ|NASDAQGS|NYSEARCA)\s*:\s*", "", t)
            if not t or not re.fullmatch(r"[A-Z\-]{1,6}", t):
                continue
            rows.append({"week": pd.Timestamp(w["week"]), "ticker": t, "company": e.get("company"), "person": e.get("person"),
                         "kind": e.get("kind"), "summary": e.get("summary")})
    return pd.DataFrame(rows)


def main(panel: str, weeks: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    P = pd.read_pickle(panel)
    r, ar = market_adjusted(P)
    mem = P["member"]
    base = {}
    ids = {re.sub(r"-\d{6}$", "", c): c for c in r.columns}  # strip the dataset's renamed-ticker suffix

    car = {h: fwd_car(ar, h) for h in H}
    car1 = {h: fwd_car(ar, h, skip=1) for h in H}
    A = ar.where(mem).loc["2012-01-01":]
    st = A.stack().rename("ar0").reset_index()
    st.columns = ["date", "ticker", "ar0"]
    st = st[st["ar0"] <= -0.03]
    for h in H:
        st[f"car{h}"] = car[h].stack().reindex(pd.MultiIndex.from_frame(st[["date", "ticker"]])).to_numpy()
        st[f"car{h}_t1"] = car1[h].stack().reindex(pd.MultiIndex.from_frame(st[["date", "ticker"]])).to_numpy()
    st["bucket"] = st["ar0"].map(bucket)
    base_tab = st.groupby("bucket")[[f"car{h}" for h in H] + [f"car{h}_t1" for h in H]].mean().mul(100).round(2)
    base_tab["n"] = st.groupby("bucket").size()
    base["baseline_car_%_by_drop"] = base_tab.to_dict(orient="index")
    st.to_csv(out / "drops_baseline.csv.gz", index=False, compression="gzip")

    ev = load_events(Path(weeks)) if Path(weeks).exists() else pd.DataFrame()
    res = {"baseline": base}
    if len(ev):
        ev = ev.drop_duplicates(["ticker", "week", "kind"])
        rows = []
        dates = ar.index
        for e in ev.itertuples(index=False):
            col = ids.get(e.ticker)
            if col is None:
                continue
            i0 = dates.searchsorted(e.week - pd.Timedelta(days=1))
            win = dates[i0: i0 + 9]  # the event week plus two days (headlines are dated within the week)
            if len(win) == 0 or not mem[col].reindex(win).any():
                continue
            x = ar[col].reindex(win)
            if x.isna().all():
                continue
            t0 = x.idxmin()
            rec = {"ticker": e.ticker, "company": e.company, "person": e.person, "kind": e.kind, "summary": e.summary,
                   "week": e.week.date(), "t0": t0.date(), "ar0": x.min()}
            for h in H:
                rec[f"car{h}"] = car[h].at[t0, col]
                rec[f"car{h}_t1"] = car1[h].at[t0, col]
            rows.append(rec)
        E = pd.DataFrame(rows).drop_duplicates(["ticker", "t0"])
        E["bucket"] = E["ar0"].map(bucket)
        E.to_csv(out / "scandal_events.csv", index=False)
        # excess bounce vs same-size ordinary drops
        bmean = st.groupby("bucket")[[f"car{h}" for h in H] + [f"car{h}_t1" for h in H]].mean()
        for h in H:
            for s in ("", "_t1"):
                E[f"x{h}{s}"] = E[f"car{h}{s}"] - E["bucket"].map(bmean[f"car{h}{s}"])
        tests = {}
        groups = {"all scandal events": E, "drop <= -3% on the news": E[E["ar0"] <= -0.03],
                  "drop <= -6%": E[E["ar0"] <= -0.06]}
        for k in sorted(E["kind"].dropna().unique()):
            groups[f"kind={k} (drop<=-3%)"] = E[(E["kind"] == k) & (E["ar0"] <= -0.03)]
        for g, d in groups.items():
            tt = {"n": int(len(d)), "mean_drop_%": round(100 * d["ar0"].mean(), 2) if len(d) else None}
            for h in H:
                for s in ("", "_t1"):
                    v = d[f"car{h}{s}"].dropna()
                    xv = d[f"x{h}{s}"].dropna()
                    if len(v) >= 5:
                        tt[f"car{h}{s}_%"] = round(100 * v.mean(), 2)
                        tt[f"car{h}{s}_t"] = round(float(stats.ttest_1samp(v, 0).statistic), 2)
                        tt[f"vs_baseline{h}{s}_%"] = round(100 * xv.mean(), 2)
                        tt[f"vs_baseline{h}{s}_t"] = round(float(stats.ttest_1samp(xv, 0).statistic), 2)
            tests[g] = tt
        # buy the scandal drop at the close of t0+1, hold 20 days, net of 2 x 5 bp: mean per trade and hit rate
        d = E[E["ar0"] <= -0.03].dropna(subset=["car20_t1"])
        tests["trade: buy close t0+1, hold 20d, net"] = {"n": int(len(d)), "mean_%": round(100 * (d["car20_t1"].mean() - 2 * COST), 2),
                                                       "hit_%": round(100 * (d["car20_t1"] > 2 * COST).mean(), 1)}
        res["scandal"] = tests
        res["events_found"] = {"raw_events": int(len(ev)), "matched_sp500": int(len(E))}
    (out / "scandal.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
