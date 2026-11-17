"""Is "go long Nasdaq on Monday, open to close" an edge?  Reproduces every check from the video and extends it.

Inputs
  DAILY.pkl  : rows of (date, ticker, open, close, ...) for QQQ and SPY, from github.com/Johnbrick123/sp500-data
               (release "data", prices.parquet; QQQ from 1999-03, SPY from 1995-01). Same-day open->close needs no adjustment.
  EXP7_DIR   : the exp7-data release (Tiingo IEX 1-minute bars, one CSV per ticker and month, 2019-01..2026-10).
Output
  OUT_DIR/results.json plus a few CSVs.

    python monday.py DAILY.pkl EXP7_DIR OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
NQ_PER_QQQ = 41.0      # NDX / QQQ price ratio (about 40.5-41.5 in 2020-26); NQ contract = $20 x NDX
NQ_COST = 15.0         # $ per NQ round trip: ~$5 commission+fees, 1 tick ($5) slippage each side
ETF_COST = 0.0002      # 2 bp round trip on QQQ/SPY (spread + slippage at the open and close auctions)
N_SHUF = 2000


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


def load_daily(path: str, ticker: str) -> pd.DataFrame:
    px = pd.read_pickle(path)
    d = px[px.ticker == ticker].drop_duplicates("date", keep="last").set_index("date").sort_index()
    d.index = pd.to_datetime(d.index)
    d = d[(d.open > 0) & (d.close > 0)]
    out = pd.DataFrame({"open": d.open, "close": d.close})
    out["oc"] = out.close / out.open - 1                       # open -> close, the trade
    out["cc"] = out.close.pct_change()                         # close -> close
    out["on"] = out.open / out.close.shift() - 1               # previous close -> open (overnight / weekend gap)
    out["dow"] = out.index.dayofweek
    wk = out.index.to_period("W-SUN")
    out["first_of_week"] = ~pd.Series(wk, index=out.index).duplicated().values
    out["prev_oc"], out["prev_cc"] = out.oc.shift(), out.cc.shift()
    out["prev_dow"] = out.dow.shift()
    return out.iloc[1:]


def weekday_table(d: pd.DataFrame, cost: float = 0.0) -> list[dict]:
    rows = []
    for i, n in enumerate(DAYS):
        x = d.oc[d.dow == i].values - cost
        rows.append({"day": n, "n": int(len(x)), "mean_bp": round(1e4 * x.mean(), 2), "t": round(tstat(x), 2),
                     "hit": round(float((x > 0).mean()), 3), "total_pct": round(100 * (np.prod(1 + x) - 1), 1)})
    allx = d.oc.values - cost
    rows.append({"day": "All", "n": int(len(allx)), "mean_bp": round(1e4 * allx.mean(), 2), "t": round(tstat(allx), 2),
                 "hit": round(float((allx > 0).mean()), 3), "total_pct": round(100 * (np.prod(1 + allx) - 1), 1)})
    return rows


def shuffle_test(d: pd.DataFrame, day: int = 0) -> dict:
    """Video's test: shuffle which days are called Monday (keep the count), how often does t beat the real one.
    Also the honest version: how often does the BEST of the 5 shuffled weekdays beat it (he picked Monday after looking)."""
    oc, lab = d.oc.values, d.dow.values
    t0 = tstat(oc[lab == day])
    single, best, diff = 0, 0, 0
    m0 = oc[lab == day].mean() - oc[lab != day].mean()
    for _ in range(N_SHUF):
        p = rng.permutation(lab)
        ts = [tstat(oc[p == k]) for k in range(5)]
        single += ts[day] >= t0
        best += max(ts) >= t0
        diff += (oc[p == day].mean() - oc[p != day].mean()) >= m0
    return {"t_obs": round(t0, 2), "p_single_day": single / N_SHUF, "p_best_of_5_days": best / N_SHUF,
            "p_monday_minus_other_days": diff / N_SHUF, "n_shuffles": N_SHUF}


def dollars_nq(d: pd.DataFrame, mask) -> pd.Series:
    """$ per NQ contract per trade: points = NDX level x return, NDX ~ 41 x QQQ open."""
    return 20 * NQ_PER_QQQ * d.open[mask] * d.oc[mask]


def block(d: pd.DataFrame, mask, label: str, nq: bool) -> dict:
    x = d.oc[mask]
    r = {"label": label, "n": int(len(x)), "mean_bp": round(1e4 * x.mean(), 2), "t": round(tstat(x), 2),
         "hit": round(float((x > 0).mean()), 3)}
    if nq:
        usd = dollars_nq(d, mask)
        r.update(usd_gross=round(usd.sum()), usd_net=round(usd.sum() - NQ_COST * len(usd)), usd_mean=round(usd.mean()))
    return r


def drop_top(d: pd.DataFrame, mask, k: int, nq: bool) -> dict:
    x = d.oc[mask]
    keep = x.drop(x.nlargest(k).index)
    r = {"k": k, "t": round(tstat(keep), 2), "mean_bp": round(1e4 * keep.mean(), 2)}
    if nq:
        usd = dollars_nq(d, mask)
        r["usd_gross"] = round(usd.drop(usd.nlargest(k).index).sum())
    return r


def intraday_split(exp7: Path, ticker: str, start: str) -> pd.DataFrame:
    """Per day: return in 4 windows (09:30-10:30, 10:30-12:30, 12:30-14:30, 14:30-16:00 New York) from 1-minute bars."""
    frames = []
    for f in sorted((exp7 / ticker).glob("*.csv")):
        b = pd.read_csv(f, usecols=["date", "open", "close"])
        frames.append(b)
    b = pd.concat(frames)
    ts = pd.to_datetime(b.date, utc=True).dt.tz_convert("America/New_York")
    b.index = pd.DatetimeIndex(ts)
    b = b.sort_index()
    b = b[~b.index.duplicated()]
    b = b.between_time("09:30", "15:59")
    b = b[b.index >= pd.Timestamp(start, tz="America/New_York")]
    hm = b.index.hour * 60 + b.index.minute
    cuts = [570, 630, 750, 870, 960]
    names = ["09:30-10:30", "10:30-12:30", "12:30-14:30", "14:30-16:00"]
    day = b.index.normalize().tz_localize(None)
    out = {}
    first_open = b.open.groupby(day).first()
    for i, n in enumerate(names):
        m = (hm >= cuts[i]) & (hm < cuts[i + 1])
        last = b.close[m].groupby(day[m]).last()
        out[n] = last
    o = pd.DataFrame(out)
    o.insert(0, "open", first_open)
    o = o.dropna()
    lv = o[["open"] + names].values
    rets = pd.DataFrame(lv[:, 1:] / lv[:, :-1] - 1, index=o.index, columns=names)
    rets["open_px"] = o["open"]
    rets["dow"] = rets.index.dayofweek
    return rets


def main(daily: str, exp7: str, out: str) -> None:
    out_p = Path(out); out_p.mkdir(parents=True, exist_ok=True)
    res: dict = {"assumptions": {"nq_per_qqq": NQ_PER_QQQ, "nq_cost_usd_rt": NQ_COST, "etf_cost_rt": ETF_COST,
                                 "video_window": "2020-01-01 .. 2026-11-13", "shuffles": N_SHUF}}
    q, s = load_daily(daily, "QQQ"), load_daily(daily, "SPY")
    res["data"] = {"QQQ": [str(q.index[0].date()), str(q.index[-1].date()), len(q)],
                   "SPY": [str(s.index[0].date()), str(s.index[-1].date()), len(s)]}
    v = q.loc["2020-01-01":]
    mon = v.dow == 0

    # 1. the video's headline: one NQ contract, Monday open->close since 2020
    res["headline"] = block(v, mon, "QQQ Mon 2020+ (NQ $)", True)
    res["weekday_2020"] = {"QQQ": weekday_table(v), "SPY": weekday_table(s.loc["2020-01-01":])}
    # Mondays only vs every other day (difference test, Welch)
    a, b_ = v.oc[mon], v.oc[~mon]
    res["mon_minus_rest_2020"] = {"diff_bp": round(1e4 * (a.mean() - b_.mean()), 2),
                                  "t": round((a.mean() - b_.mean()) / np.sqrt(a.var() / len(a) + b_.var() / len(b_)), 2)}
    # 2. shuffle
    res["shuffle_2020"] = shuffle_test(v)
    # 3. halves
    res["halves"] = [block(v, mon & (v.index < "2023-01-01"), "2020-2022", True),
                     block(v, mon & (v.index >= "2023-01-01"), "2023-now", True)]
    res["by_year"] = [block(v, mon & (v.index.year == y), str(y), True) for y in sorted(set(v.index.year))]
    # 4. where in the day (1-minute bars)
    sp = intraday_split(Path(exp7), "QQQ", "2020-01-01")
    sp = sp[sp.index.isin(q.index)]   # IEX feed prints a few bars on exchange holidays; keep real trading days only
    res["intraday_days"] = {"n_days": int(len(sp)), "corr_with_daily_oc": round(float(
        (1 + sp.iloc[:, :4]).prod(axis=1).sub(1).corr(q.oc.reindex(sp.index))), 3)}
    split = {}
    for i, n in enumerate(DAYS):
        x = sp[sp.dow == i]
        usd = {c: round(float((20 * NQ_PER_QQQ * x.open_px * x[c]).sum())) for c in sp.columns[:4]}
        tt = {c: round(tstat(x[c]), 2) for c in sp.columns[:4]}
        split[n] = {"n": int(len(x)), "usd_by_window": usd, "t_by_window": tt}
    res["intraday_split_2020"] = split
    # 5. red Friday / green Friday
    after = {}
    for defn, col in [("Friday open->close", "prev_oc"), ("Friday close->close", "prev_cc")]:
        m = mon & (v.prev_dow == 4)
        red, green = v[m & (v[col] < 0)].oc, v[m & (v[col] >= 0)].oc
        after[defn] = {"red": block(v, m & (v[col] < 0), "after red", True), "green": block(v, m & (v[col] >= 0), "after green", True),
                       "diff_t": round((red.mean() - green.mean()) / np.sqrt(red.var() / len(red) + green.var() / len(green)), 2)}
    # same split for every other day: is it a Monday thing or plain 1-day reversal?
    gen = {}
    for i, n in enumerate(DAYS):
        m = (v.dow == i)
        red, green = v[m & (v.prev_oc < 0)].oc, v[m & (v.prev_oc >= 0)].oc
        gen[n] = {"after_red_bp": round(1e4 * red.mean(), 1), "after_green_bp": round(1e4 * green.mean(), 1),
                  "diff_t": round((red.mean() - green.mean()) / np.sqrt(red.var() / len(red) + green.var() / len(green)), 2)}
    after["every_weekday_after_red_vs_green_prev_day_oc"] = gen
    res["after_friday"] = after
    # 6. break it: drop the best Mondays
    res["drop_top"] = [drop_top(v, mon, k, True) for k in (0, 5, 10, 20, 30)]
    res["drop_worst"] = {"k": 10, "t": round(tstat(v.oc[mon].drop(v.oc[mon].nsmallest(10).index)), 2)}

    # 7. longer history, both ETFs
    hist = {}
    for name, d in [("QQQ", q), ("SPY", s)]:
        pre = d.loc[:"2019-12-31"]
        hist[name] = {"pre2020": weekday_table(pre), "pre2020_shuffle": shuffle_test(pre),
                      "full": weekday_table(d), "full_shuffle": shuffle_test(d)}
        rows = []
        for y in range(d.index.year.min(), 2027):
            w = d[(d.index.year >= y - 2) & (d.index.year <= y)]
            if w.index.year.min() > y - 2:
                continue
            rows.append({"window": f"{y-2}-{y}", **{n: round(tstat(w.oc[w.dow == i]), 2) for i, n in enumerate(DAYS)}})
        hist[name]["rolling_3y_t"] = rows
        # start-year sensitivity: Monday t from each start year to now
        hist[name]["start_year_t"] = {str(y): round(tstat(d.oc[(d.dow == 0) & (d.index.year >= y)]), 2)
                                      for y in range(d.index.year.min() + 1, 2025)}
        # Monday premium over other days, by decade-ish blocks
        blocks = {}
        for lo, hi in [(1995, 2004), (2005, 2014), (2015, 2019), (2020, 2026)]:
            w = d[(d.index.year >= lo) & (d.index.year <= hi)]
            if len(w) < 200:
                continue
            a_, b2 = w.oc[w.dow == 0], w.oc[w.dow != 0]
            blocks[f"{lo}-{hi}"] = {"mon_bp": round(1e4 * a_.mean(), 1), "rest_bp": round(1e4 * b2.mean(), 1),
                                    "mon_t": round(tstat(a_), 2),
                                    "diff_t": round((a_.mean() - b2.mean()) / np.sqrt(a_.var() / len(a_) + b2.var() / len(b2)), 2)}
        hist[name]["blocks"] = blocks
    res["history"] = hist

    # 8. as a strategy for a small pot: costs, vs holding
    strat = {}
    for name, d in [("QQQ", q), ("SPY", s)]:
        for lo in ["2020-01-01", "2023-01-01", d.index[0].strftime("%Y-%m-%d")]:
            w = d.loc[lo:]
            yrs = (w.index[-1] - w.index[0]).days / 365.25
            series = {
                "Monday open->close, 2bp": np.where(w.dow == 0, w.oc - ETF_COST, 0.0),
                "Monday open->close, gross": np.where(w.dow == 0, w.oc, 0.0),
                "First day of week open->close, 2bp": np.where(w.first_of_week, w.oc - ETF_COST, 0.0),
                "Every day open->close, 2bp": w.oc - ETF_COST,
                "Buy and hold (close->close)": w.cc.fillna(0).values,
                "Hold Fri close -> Mon close (weekend+Monday)": np.where(w.dow == 0, w.cc.fillna(0), 0.0),
            }
            rows = []
            for k, r in series.items():
                r = np.asarray(r, float)
                eq = np.cumprod(1 + r)
                dd = (eq / np.maximum.accumulate(eq) - 1).min()
                cagr = eq[-1] ** (1 / yrs) - 1
                sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
                rows.append({"rule": k, "cagr_pct": round(100 * cagr, 2), "sharpe": round(float(sh), 2),
                             "max_dd_pct": round(100 * dd, 1), "in_market_pct": round(100 * float((r != 0).mean()), 0),
                             "500_to": round(500 * eq[-1], 0)})
            strat[f"{name} {lo[:4]}-now"] = rows
    res["strategy"] = strat
    # alpha of the 2bp Monday rule over holding (daily, Newey-West 5 lags)
    import statsmodels.api as sm
    alpha = {}
    for lo in ["2020-01-01", "2023-01-01", "1999-03-11"]:
        w = q.loc[lo:]
        r = pd.Series(np.where(w.dow == 0, w.oc - ETF_COST, 0.0), index=w.index)
        for bn, bd in [("SPY", s), ("QQQ", q)]:
            X = sm.add_constant(bd.cc.reindex(r.index))
            m = sm.OLS(r, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            alpha[f"QQQ Monday {lo[:4]}-now vs {bn}"] = {"alpha_pct_yr": round(252 * 100 * float(m.params.iloc[0]), 2),
                                                        "t": round(float(m.tvalues.iloc[0]), 2), "beta": round(float(m.params.iloc[1]), 3)}
    res["alpha"] = alpha
    # NQ $ net of costs, video window
    usd = dollars_nq(v, mon)
    res["nq_net"] = {"gross": round(usd.sum()), "costs": round(NQ_COST * len(usd)), "net": round(usd.sum() - NQ_COST * len(usd))}
    # multiple testing: 5 weekdays x 2 markets the author could have picked; Bonferroni bar
    res["multiple_testing"] = {"tests_minimum": 10, "bonferroni_bar_t_two_sided_5pct": 2.81,
                               "note": "5 weekdays x {QQQ,SPY}, before counting the 'something I found this week' search"}
    (out_p / "results.json").write_text(json.dumps(res, indent=1))
    sp.to_csv(out_p / "intraday_windows_qqq.csv")
    v.assign(nq_usd=20 * NQ_PER_QQQ * v.open * v.oc)[["open", "close", "oc", "dow", "nq_usd"]].to_csv(out_p / "qqq_daily_2020.csv")
    print(json.dumps({k: res[k] for k in ["headline", "shuffle_2020", "halves", "drop_top", "nq_net"]}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
