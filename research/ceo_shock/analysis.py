"""Key-person shock bounce study (paper research only, nothing is traded).

Question (Joaquin, 2026-11-23): when a large cap drops on a shock headline about its CEO or another key person
(scandal, firing, legal trouble, sudden exit, health), is the drop overdone, so that buying it pays?

Data: point-in-time S&P 500 daily prices and membership from github.com/Johnbrick123/sp500-data (release "data":
prices.parquet, membership_intervals.parquet), split+dividend adjusted. Event list: events.csv (hand-collected,
each date checked against the price reaction). NVDA Nov 2026 events are the live case, kept out of every test.

Method (Sharma ch.3-4 conventions: as-was universe, beta-neutral, S&P 500 cost ~5 bp a side):
  * abnormal return AR = r - beta * r_SPY, beta from a 3-year window ending 21 days before day 0 (min 250 days)
  * day 0 = first trading day the news could move the price; entry at day 0 close
  * post-event CAR over 1, 5, 10, 20, 60 days after the entry close
  * baseline: every PIT S&P 500 member-day 1996-2026 with AR0 <= -3% (any cause), same CAR measure; event CARs are
    compared with the baseline CAR predicted for their own AR0 size (binned)
  * trade rule: buy at day 0 close when AR0 <= -3% on a key-person headline, hold 20 days, 10 bp round trip, vs SPY
  * p-values: t-test (and sign test), Holm across all horizons and both comparisons

    python analysis.py SP500_DIR OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

H = [1, 5, 10, 20, 60]
THRESH = -0.03
COST_RT = 0.0010
HOLD = 20
START = "1995-01-01"


def load(sp: Path):
    iv = pd.read_parquet(sp / "membership_intervals.parquet")
    iv["start"], iv["end"] = pd.to_datetime(iv["start"]), pd.to_datetime(iv["end"])
    iv = iv[iv["end"].isna() | (iv["end"] > START)]
    px = pd.read_parquet(sp / "prices.parquet", columns=["date", "ticker", "adj_close"],
                         filters=[("date", ">=", pd.Timestamp(START))])
    px = px.drop_duplicates(["date", "ticker"], keep="last")
    close = px.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    spy = close["SPY"]
    close = close[[t for t in close.columns if t in set(iv["ticker"])]]
    dates = close.index
    member = pd.DataFrame(False, index=dates, columns=close.columns)
    for t, s, e in iv[["ticker", "start", "end"]].itertuples(index=False):
        if t in member.columns:
            member.loc[(dates >= s) & ((dates < e) if pd.notna(e) else True), t] = True
    return close, spy, member


def betas(r: pd.DataFrame, m: pd.Series) -> pd.DataFrame:
    """Rolling 3-year beta, lagged 21 days so day 0 never feeds its own beta."""
    cov = r.rolling(756, min_periods=250).cov(m)
    var = m.rolling(756, min_periods=250).var()
    b = cov.div(var, axis=0).shift(21)
    return b.clip(0.2, 3.0).fillna(1.0)


def car_table(ar: pd.DataFrame, rows: list[tuple[pd.Timestamp, str]]) -> pd.DataFrame:
    idx = {d: i for i, d in enumerate(ar.index)}
    a = ar.to_numpy()
    cols = {c: j for j, c in enumerate(ar.columns)}
    out = []
    for d, t in rows:
        i, j = idx[d], cols[t]
        rec = {"date": d, "ticker": t, "ar0": a[i, j]}
        for h in H:
            seg = a[i + 1:i + 1 + h, j]
            rec[f"car{h}"] = np.nansum(seg) if len(seg) == h and np.isfinite(seg).sum() >= 0.8 * h else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def resolve(ticker: str, d: pd.Timestamp, close: pd.DataFrame) -> str | None:
    for t in [c for c in close.columns if c == ticker or c.startswith(ticker + "-")]:
        s = close[t].dropna()
        if len(s) and s.index[0] < d - pd.Timedelta(days=30) and s.index[-1] >= d:
            return t
    return None


def holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: p[k])
    n, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (n - i) * p[k]))
        out[k] = run
    return out


def main(sp: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    close, spy, member = load(sp)
    r = close.pct_change(fill_method=None)
    m = spy.pct_change(fill_method=None).reindex(r.index)
    b = betas(r, m)
    ar = r - b.mul(m, axis=0)
    ar = ar.where(ar.abs() < 0.9)  # bad prints

    ev = pd.read_csv(Path(__file__).with_name("events.csv"), parse_dates=["date"])
    ev["sym"] = [resolve(t, d, close) for t, d in zip(ev["ticker"], ev["date"])]
    missing = ev[ev["sym"].isna()][["ticker", "date", "event"]]
    ev = ev.dropna(subset=["sym"]).copy()
    ev["date"] = [ar.index[ar.index.searchsorted(d)] for d in ev["date"]]
    ct = car_table(ar, list(zip(ev["date"], ev["sym"])))
    ev = pd.concat([ev.reset_index(drop=True), ct.drop(columns=["date", "ticker"])], axis=1)
    raw = close.pct_change(fill_method=None)
    # raw holding-period returns for the trade rule
    ix = {d: i for i, d in enumerate(close.index)}
    tr = []
    for _, e in ev.iterrows():
        i = ix[e["date"]]
        s = close[e["sym"]].iloc[i:i + HOLD + 1]
        q = spy.reindex(close.index).iloc[i:i + HOLD + 1]
        if s.notna().sum() < HOLD + 1:
            tr.append((np.nan, np.nan))
            continue
        tr.append((s.iloc[-1] / s.iloc[0] - 1 - COST_RT, q.iloc[-1] / q.iloc[0] - 1))
    ev["trade_ret"], ev["spy_ret"] = zip(*tr)
    ev["trade_excess"] = ev["trade_ret"] - ev["spy_ret"]
    ev.to_csv(out / "events_car.csv", index=False, float_format="%.5f")

    # baseline: every member-day with AR0 <= -3%, 1996+
    A = ar.where(member)
    A = A.loc["1996-01-01":]
    st = A.stack()
    st = st[st <= THRESH]
    base = car_table(ar, list(st.index))
    base = base.dropna(subset=["car20"])
    bins = [-1, -0.20, -0.12, -0.08, -0.06, -0.045, -0.03]
    base["bin"] = pd.cut(base["ar0"], bins)
    bmean = base.groupby("bin", observed=True)[[f"car{h}" for h in H]].mean()
    bcount = base.groupby("bin", observed=True).size()
    base["crisis_year"] = base["date"].dt.year.isin([2000, 2001, 2002, 2008, 2009, 2020])
    summ = base.groupby(["bin", "crisis_year"], observed=True)[[f"car{h}" for h in H]].agg(["count", "mean", "median"])
    summ.to_csv(out / "baseline_summary.csv", float_format="%.5f")

    test = ev[(ev["category"] != "live") & (ev["ar0"] <= THRESH)].copy()
    test["bin"] = pd.cut(test["ar0"], bins)
    res = {"n_events_listed": int(len(ev) + len(missing)), "missing_data": missing.astype(str).to_dict("records"),
           "n_events_with_data": int(len(ev)), "n_test": int(len(test)), "threshold_ar0": THRESH,
           "baseline_n": int(len(base)), "baseline_bins": {str(k): int(v) for k, v in bcount.items()},
           "horizons": {}}
    pvals = {}
    for h in H:
        c = test[f"car{h}"].dropna()
        exp = test.loc[c.index, "bin"].map(bmean[f"car{h}"]).astype(float)
        d = c - exp
        t1, p1 = stats.ttest_1samp(c, 0.0)
        t2, p2 = stats.ttest_1samp(d, 0.0)
        ps = stats.binomtest(int((c > 0).sum()), len(c), 0.5).pvalue
        pvals[f"car{h}_vs0"], pvals[f"car{h}_vs_baseline"] = p1, p2
        res["horizons"][h] = {"n": int(len(c)), "mean_car": c.mean(), "median_car": c.median(),
                              "hit_rate": float((c > 0).mean()), "t": t1, "p": p1, "sign_p": ps,
                              "baseline_expected": exp.mean(), "excess_vs_baseline": d.mean(), "t_vs_baseline": t2,
                              "p_vs_baseline": p2,
                              "baseline_all_drops_mean": float(base[f"car{h}"].mean())}
    hp = holm(pvals)
    for h in H:
        res["horizons"][h]["holm_p"] = hp[f"car{h}_vs0"]
        res["horizons"][h]["holm_p_vs_baseline"] = hp[f"car{h}_vs_baseline"]

    # by category, 20 day
    res["by_category_car20"] = {k: {"n": int(len(g)), "mean": g["car20"].mean(), "median": g["car20"].median()}
                                for k, g in test.groupby("category")}
    # ordinary baseline by bin
    res["baseline_by_bin"] = {str(k): {f"car{h}": float(v) for h, v in zip(H, row)} for k, row in bmean.iterrows()}

    # trade rule
    tt = test.dropna(subset=["trade_excess"]).sort_values("date")
    tx = tt["trade_excess"]
    t3, p3 = stats.ttest_1samp(tx, 0.0)
    # calendar pot: 500 CAD-equivalent in SPY; during a signal window, 100% in the stock instead
    s_ret = spy.pct_change(fill_method=None).loc["2001-01-01":].fillna(0)
    pot_ret = s_ret.copy()
    for _, e in tt.iterrows():
        i = close.index.get_loc(e["date"])
        win = close.index[i + 1:i + 1 + HOLD]
        stock = raw[e["sym"]].reindex(win).fillna(0)
        pot_ret.loc[win] = stock.values
        pot_ret.loc[win[0]] -= COST_RT
    eq = (1 + pot_ret).cumprod() * 500
    eqs = (1 + s_ret).cumprod() * 500
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    res["trade_rule"] = {"n_trades": int(len(tx)), "mean_net_excess_vs_spy": tx.mean(), "median": tx.median(),
                         "hit_rate": float((tx > 0).mean()), "t": t3, "p": p3,
                         "best": tt.loc[tx.idxmax(), ["sym", "date"]].astype(str).tolist() + [float(tx.max())],
                         "worst": tt.loc[tx.idxmin(), ["sym", "date"]].astype(str).tolist() + [float(tx.min())],
                         "pot_2001_end": float(eq.iloc[-1]), "spy_2001_end": float(eqs.iloc[-1]),
                         "pot_cagr": float((eq.iloc[-1] / 500) ** (1 / yrs) - 1),
                         "spy_cagr": float((eqs.iloc[-1] / 500) ** (1 / yrs) - 1),
                         "days_in_stocks_pct": float(len(tt) * HOLD / len(s_ret))}
    pd.DataFrame({"pot": eq, "spy": eqs}).to_csv(out / "pot_vs_spy.csv", float_format="%.2f")

    # live NVDA case, for reference only
    live = ev[ev["category"] == "live"][["sym", "date", "ar0", "car1", "car5"]]
    res["live_nvda"] = live.astype(str).to_dict("records")
    res["data_end"] = str(close.index[-1].date())
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    cols = ["sym", "date", "category", "ar0", "car1", "car5", "car20", "car60", "trade_excess", "event"]
    print(ev[cols].to_string(float_format=lambda x: f"{x * 100:+.1f}"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("baseline_by_bin",)}, indent=1, default=float))
    print(pd.DataFrame(res["baseline_by_bin"]).T.mul(100).round(2), bcount)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
