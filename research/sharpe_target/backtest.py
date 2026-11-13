"""Backtest of the quarterly 25-stock books against SPY, the equal-weight universe and 200 random 25-stock books.

Book: at each quarter end buy the 25 members with the highest score, equal weight, hold untouched to the next
quarter end; 5 bp a side on turnover (Sharma ch. 9 for S&P 500 names).
Periods: walk-forward 2005-01 .. 2025-06 ("tuning" in the video's words); sealed holdout 2025-07 .. 2026-09
(15 months, scored once, like the video's locked 15 months).
Args: out_dir
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from stats import alpha_nw, perf  # noqa: E402

OUT = Path(sys.argv[1])
R = pd.read_parquet(OUT / "daily_returns.parquet")
panel = pd.read_parquet(OUT / "panel.parquet")
preds = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("preds_*.parquet"))])
cal = R.index
tick = list(R.columns)
col = {t: k for k, t in enumerate(tick)}
Rv = np.nan_to_num(R.to_numpy(np.float64))
spy = pd.read_csv(Path(sys.argv[2]) / "prices" / "SPY.csv", parse_dates=["date"]).set_index("date")["adj_close"]
spy_r = spy.reindex(cal).pct_change().fillna(0)
irx = pd.read_csv(Path(sys.argv[2]) / "prices" / "^IRX.csv", parse_dates=["date"]).set_index("date")["close"]
rf = (irx.reindex(cal).ffill().shift(1).bfill() / 100 / 252)

N, COST = 25, 0.0005
REB = sorted(d for d in preds["date"].unique() if d <= pd.Timestamp("2026-06-30"))
START, HOLD0, END = pd.Timestamp("2004-12-31"), pd.Timestamp("2025-06-30"), pd.Timestamp("2026-09-30")
qend = sorted(panel["date"].unique())
nxt = {d: qend[k + 1] for k, d in enumerate(qend[:-1])}
universe = {d: g["ticker"].tolist() for d, g in panel.groupby("date")}


def book_returns(picks: dict) -> pd.Series:
    """picks: quarter end -> list of tickers (equal weight). Daily returns after costs."""
    out = pd.Series(0.0, index=cal)
    w_old = {}
    for d in REB:
        names = picks[d]
        i, j = cal.get_loc(d), cal.get_loc(nxt[d])
        idx = [col[t] for t in names]
        w0 = np.full(len(idx), 1 / len(idx))
        turn = sum(abs(w0[k] - w_old.get(t, 0)) for k, t in enumerate(names)) + \
            sum(v for t, v in w_old.items() if t not in names)
        growth = np.cumprod(1 + Rv[i + 1:j + 1][:, idx], axis=0)
        val = growth @ w0
        daily = np.diff(np.r_[1.0, val]) / np.r_[1.0, val[:-1]]
        daily[0] -= COST * turn
        out.iloc[i + 1:j + 1] = daily
        wend = w0 * growth[-1]
        w_old = dict(zip(names, wend / wend.sum()))
    return out


def top(score: pd.DataFrame, asc: bool = False) -> dict:
    return {d: g.sort_values("score", ascending=asc)["ticker"].head(N).tolist()
            for d, g in score.groupby("date") if d in set(REB)}


books = {}
for (m, t), g in preds.groupby(["model", "target"]):
    books[f"{m.upper()} predicts {'return' if t == 'ret' else 'Sharpe'}"] = book_returns(top(g))
pr = panel[panel["date"].isin(REB)]
rules = {
    "Rule: 25 calmest (lowest 1y vol)": ("vol252", True),
    "Rule: 25 best past-year Sharpe": ("sharpe252", False),
    "Rule: 25 strongest 12-1 momentum": ("mom12_1", False),
    "Rule: 25 wildest (highest 1y vol)": ("vol252", False),
}
for name, (c, asc) in rules.items():
    books[name] = book_returns(top(pr.rename(columns={c: "score"})[["date", "ticker", "score"]], asc))
ew = pd.Series(0.0, index=cal)
for d in REB:   # equal-weight universe, rebalanced quarterly (no costs: a benchmark)
    i, j = cal.get_loc(d), cal.get_loc(nxt[d])
    idx = [col[t] for t in universe[d]]
    val = np.cumprod(1 + Rv[i + 1:j + 1][:, idx], axis=0).mean(1)
    ew.iloc[i + 1:j + 1] = np.diff(np.r_[1.0, val]) / np.r_[1.0, val[:-1]]
books["Equal-weight S&P 500 members"] = ew
books["SPY"] = spy_r

rng = np.random.default_rng(7)
rand = []
for k in range(200):
    rand.append(book_returns({d: list(rng.choice(universe[d], N, replace=False)) for d in REB}))
rand = pd.concat(rand, axis=1)

periods = {"walk-forward 2005-01..2025-06": (START, HOLD0), "holdout 2025-07..2026-09": (HOLD0, END),
           "whole 2005-01..2026-09": (START, END)}
res = {}
for pname, (a, b) in periods.items():
    sl = (cal > a) & (cal <= b)
    rp = [perf(rand[c][sl], rf[sl]) for c in rand]
    rw = np.array([x["end"] for x in rp])
    rs = np.array([x["sharpe"] for x in rp])
    res[pname] = {"random_200": {"end_median": float(np.median(rw)), "end_p5": float(np.percentile(rw, 5)),
                                 "end_p95": float(np.percentile(rw, 95)), "sharpe_median": float(np.median(rs)),
                                 "sharpe_p5": float(np.percentile(rs, 5)), "sharpe_p95": float(np.percentile(rs, 95))}}
    for name, r in books.items():
        p = perf(r[sl], rf[sl])
        if name != "SPY":
            p.update(alpha_nw(r[sl], spy_r[sl], rf[sl]))
        p["pct_of_random_end"] = float((rw < p["end"]).mean() * 100)
        p["pct_of_random_sharpe"] = float((rs < p["sharpe"]).mean() * 100)
        res[pname][name] = p
    print(f"\n== {pname}  (5000 start; random 200: median {5000 * np.median(rw):,.0f}, "
          f"5-95% {5000 * np.percentile(rw, 5):,.0f}-{5000 * np.percentile(rw, 95):,.0f}; "
          f"Sharpe median {np.median(rs):.2f})")
    print(f"{'book':40s} {'$5000->':>8s} {'CAGR':>6s} {'Sharpe':>6s} {'MaxDD':>6s} {'beta':>5s} {'alpha':>6s} "
          f"{'t':>5s} {'%rand$':>6s} {'%randS':>6s}")
    for name in books:
        p = res[pname][name]
        print(f"{name:40s} {5000 * p['end']:8,.0f} {100 * p['cagr']:6.1f} {p['sharpe']:6.2f} {100 * p['max_dd']:6.1f} "
              f"{p.get('beta', 1):5.2f} {100 * p.get('alpha', 0):6.1f} {p.get('alpha_t', 0):5.1f} "
              f"{p['pct_of_random_end']:6.0f} {p['pct_of_random_sharpe']:6.0f}")

# how well each model ranks stocks (rank IC with next-quarter return and Sharpe), per period
ic = {}
pp = preds.merge(panel[["date", "ticker", "y_ret", "y_sharpe"]], on=["date", "ticker"])
for (m, t), g in pp.groupby(["model", "target"]):
    by = g.groupby("date").apply(lambda x: pd.Series({"ic_ret": x["score"].corr(x["y_ret"], method="spearman"),
                                                      "ic_sharpe": x["score"].corr(x["y_sharpe"], method="spearman")}),
                                 include_groups=False)
    for pname, (a, b) in periods.items():
        s = by[(by.index >= a) & (by.index < b)]
        ic[f"{m}/{t}|{pname}"] = {"ic_ret": s["ic_ret"].mean(), "ic_ret_t": s["ic_ret"].mean() / s["ic_ret"].std() * math.sqrt(len(s)),
                                  "ic_sharpe": s["ic_sharpe"].mean(), "n_q": len(s)}
print("\nrank IC (score vs next-quarter outcome, mean per quarter)")
for k, v in ic.items():
    print(f"{k:55s} IC ret {v['ic_ret']:+.3f} (t {v['ic_ret_t']:+.1f})  IC sharpe {v['ic_sharpe']:+.3f}  n {v['n_q']}")
res["rank_ic"] = ic
pd.DataFrame(books).loc[START:END].to_csv(OUT / "books_daily.csv")
rand.loc[START:END].to_parquet(OUT / "random_daily.parquet")
(OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
