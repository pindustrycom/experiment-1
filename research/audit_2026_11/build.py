"""Candidate built from the audit: "core-plus". The spec below was fixed before any of it was run.

What the audit supports: (1) only market beta pays reliably, (2) volatility is forecastable while direction is not,
(3) the timed short-volatility trade is the one edge, at about half its paper size once it uses real VIX futures,
(4) the BTC breakout does not beat holding BTC on return but cuts its drawdown and is uncorrelated with stocks.

Spec (fixed in advance):
  A  Core: SPY held at  min(1.5, 15% / vol)  of the pot, vol = SPY's realised volatility over the last 22 sessions.
     The size decided at one close is traded at the next close, and only when it differs from the current size by
     more than 0.10. 2 bp per unit traded; cash earns the T-bill rate; borrowing costs T-bill + 1%.
  B  Vol overlay: Experiment 2's timed short 2nd-month VIX future (monthly, from 2006), at half the research size
     (0.25% of the pot per vol point).
  C  BTC sleeve: 5% of the pot in the live vol-sized BTC breakout rule (25 bp a side, 6% stop), financed at T-bill + 1%.
Reported: A alone daily 1998-2026; A+B monthly 2006-2026; A+B+C monthly 2019-2026; all against SPY, with the
market-model alpha and its Newey-West t. A sensitivity grid around A shows whether the fixed choice was lucky.

Args: papers_data_dir vol_products_monthly.csv btc_1h.csv repo_strategies_dir out_dir"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import TD, alpha, beta_matched, load_px, perf, rf_daily

D, VOLCSV, BTC, REPO, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5])
P = D / "prices"
sys.path.insert(0, REPO)


def vol_managed(px: pd.Series, rf: pd.Series, target=0.15, look=22, cap=1.5, band=0.10, cost=0.0002, spread=0.01,
                lag=1) -> pd.DataFrame:
    r = px.pct_change()
    vol = np.sqrt((r ** 2).rolling(look).mean() * TD)
    want = (target / vol).clip(upper=cap).shift(lag)          # decided at close t-lag, in place from close t
    w = np.zeros(len(r))
    cur, out = np.nan, np.zeros(len(r))
    rv, wv, rfv = r.to_numpy(), want.to_numpy(), rf.to_numpy()
    for i in range(1, len(r)):
        # weight held over day i is the one set at close i-1
        if np.isfinite(cur):
            fin = rfv[i] + (spread / TD if cur > 1 else 0.0)
            out[i] = cur * rv[i] + (1 - cur) * fin
        nxt = wv[i]
        if np.isfinite(nxt) and (not np.isfinite(cur) or abs(nxt - cur) > band):
            out[i] -= cost * abs(nxt - (cur if np.isfinite(cur) else 0.0))
            cur = float(nxt)
        w[i] = cur if np.isfinite(cur) else 0.0
    return pd.DataFrame({"ret": out, "w": w}, index=px.index)


def table(X: pd.DataFrame, mkt: str, rf: pd.Series, per: int, title: str) -> dict:
    res = {}
    print(f"\n{title}\n{'':26} {'cagr':>6} {'vol':>6} {'sharpe':>6} {'maxdd':>7} {'worst':>6} {'beta':>5} {'alpha':>6} {'t':>5}")
    for k in X:
        p, a = perf(X[k], rf, per), alpha(X[k], X[mkt], rf, per)
        res[k] = {**{x: float(v) for x, v in p.items()}, **{x: float(v) for x, v in a.items()}}
        print(f"{k:26} {100*p['cagr']:6.1f} {100*p['vol']:6.1f} {p['sharpe']:6.2f} {100*p['max_dd']:7.1f} {100*p['worst']:6.1f} "
              f"{a['beta']:5.2f} {100*a['alpha']:6.1f} {a['alpha_t']:5.1f}")
    return res


res = {}
spy = load_px(P, "SPY").loc["1997-11-01":]
rf = rf_daily(P, spy.index)
A = vol_managed(spy, rf)
daily = pd.DataFrame({"SPY": spy.pct_change(), "core_A": A["ret"]}).loc["1998-01-05":]
avg_w = A["w"].loc["1998-01-05":].mean()
daily[f"SPY_const_{avg_w:.2f}x"] = beta_matched(daily["SPY"], rf.reindex(daily.index), avg_w)
print(f"core A: average size {avg_w:.2f}, share of days above 1x {(A['w'].loc['1998-01-05':] > 1).mean():.0%}, "
      f"trades a year {(A['w'].diff().abs() > 0).loc['1998-01-05':].sum() / (len(daily) / TD):.0f}")
for name, (a, b) in {"1998-2026": ("1998", "2026"), "1998-2008": ("1998", "2008"), "2009-2018": ("2009", "2018"),
                     "2019-2026": ("2019", "2026")}.items():
    res[f"daily|{name}"] = table(daily.loc[a:b], "SPY", rf, TD, f"A alone, daily, {name}")
yr = (1 + daily).groupby(daily.index.year).prod() - 1
print((100 * yr).round(1).to_string())
res["years_A"] = {str(y): {k: float(v) for k, v in row.items()} for y, row in yr.iterrows()}

# same rule on QQQ (from 1999) as an out-of-sample check
qqq = load_px(P, "QQQ")
rq = rf_daily(P, qqq.index)
Q = vol_managed(qqq, rq, target=0.15 * 1.3)      # QQQ's long-run vol is about 1.3x SPY's: same average size
dq = pd.DataFrame({"QQQ": qqq.pct_change(), "core_A_on_QQQ": Q["ret"]}).iloc[30:]
print("QQQ average size", round(Q["w"].iloc[30:].mean(), 2))
res["qqq"] = table(dq, "QQQ", rq, TD, "A on QQQ, daily, 1999-2026")

# sensitivity grid: was the fixed choice lucky?
grid = []
for tg, lk, cp in itertools.product([0.12, 0.15, 0.18], [10, 22, 63], [1.0, 1.5, 2.0]):
    g = vol_managed(spy, rf, target=tg, look=lk, cap=cp)["ret"].loc["1998-01-05":]
    p, a = perf(g, rf), alpha(g, daily["SPY"], rf)
    grid.append({"target": tg, "look": lk, "cap": cp, "cagr": p["cagr"], "sharpe": p["sharpe"], "max_dd": p["max_dd"],
                 "alpha": a["alpha"], "alpha_t": a["alpha_t"], "beta": a["beta"]})
G = pd.DataFrame(grid)
G.to_csv(OUT / "core_grid.csv", index=False, float_format="%.4f")
print("\ngrid of 27: alpha t min %.1f median %.1f max %.1f; Sharpe min %.2f median %.2f max %.2f (SPY %.2f); share with alpha t > 2: %.0f%%"
      % (G.alpha_t.min(), G.alpha_t.median(), G.alpha_t.max(), G.sharpe.min(), G.sharpe.median(), G.sharpe.max(),
         res["daily|1998-2026"]["SPY"]["sharpe"], 100 * (G.alpha_t > 2).mean()))
res["grid"] = {"alpha_t": [float(G.alpha_t.min()), float(G.alpha_t.median()), float(G.alpha_t.max())],
               "sharpe": [float(G.sharpe.min()), float(G.sharpe.median()), float(G.sharpe.max())]}

# ---- monthly: add the vol overlay (B) and the BTC sleeve (C)
mon = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
rfm = (1 + rf.reindex(daily.index)).groupby(daily.index.to_period("M")).prod() - 1
V = pd.read_csv(VOLCSV, index_col=0)
V.index = pd.PeriodIndex(V.index, freq="M")
vx = (V["vxfut_timed"] / 100).dropna()
vx_always = (V["vxfut_always"] / 100).dropna()
m = mon.join(vx.rename("B_full")).dropna()
m["A+B"] = m["core_A"] + 0.5 * m["B_full"]
m["SPY+B"] = m["SPY"] + 0.5 * m["B_full"]
m["B alone (half size, on cash)"] = rfm.reindex(m.index) + 0.5 * m["B_full"]
m = m.drop(columns=["B_full"])
res["monthly_2006"] = table(m, "SPY", rfm, 12, f"A + B, monthly, {m.index[0]} to {m.index[-1]}")
for name, (a, b) in {"2006-2018": ("2006", "2018"), "2019-2026": ("2019", "2026")}.items():
    res[f"monthly|{name}"] = table(m.loc[a:b], "SPY", rfm, 12, f"A + B, monthly, {name}")

from strats import crypto, stops  # noqa: E402
b1 = pd.read_csv(BTC, index_col="t")
b1.index = pd.to_datetime(b1.index, format="mixed")
b4 = b1.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
btc = {}
for key, size in (("btc_breakout", False), ("btc_breakout_vol_sized", True)):
    eq = crypto.run_rule(b4, stop=stops.STOPS[crypto.KEY], size=size)["equity"]
    btc[key] = eq.groupby(eq.index.to_period("M")).last().pct_change()
btc["btc_hold"] = b4["close"].groupby(b4.index.to_period("M")).last().pct_change()
Bm = pd.DataFrame(btc).loc["2019-01":"2026-09"]
res["btc"] = table(Bm.assign(SPY=mon["SPY"]), "SPY", rfm, 12, "BTC rules, monthly, 2019-01 to 2026-09 (25 bp a side, 6% stop)")
d = Bm["btc_breakout_vol_sized"] - Bm["btc_hold"]
print("breakout (vol-sized) minus hold: mean %.1f%%/month, t %.2f" % (100 * d.mean(), d.mean() / d.std() * np.sqrt(len(d))))

f = m.join(Bm["btc_breakout_vol_sized"].rename("C")).dropna()
f["A+B+C (core-plus)"] = f["A+B"] + 0.05 * (f["C"] - rfm.reindex(f.index) - 0.01 / 12)
f["SPY 95% + BTC hold 5%"] = 0.95 * f["SPY"] + 0.05 * Bm["btc_hold"].reindex(f.index)
f = f.drop(columns=["C"])
res["monthly_2019_full"] = table(f, "SPY", rfm, 12, f"A + B + C, monthly, {f.index[0]} to {f.index[-1]}")
f.to_csv(OUT / "core_plus_monthly.csv", float_format="%.5f")
m.to_csv(OUT / "core_plus_monthly_2006.csv", float_format="%.5f")
daily.to_csv(OUT / "core_daily.csv", float_format="%.6f")
yr2 = (1 + f[["SPY", "core_A", "A+B", "A+B+C (core-plus)"]]).groupby(f.index.year).prod() - 1
print((100 * yr2).round(1).to_string())
res["years_full"] = {str(y): {k: float(v) for k, v in row.items()} for y, row in yr2.iterrows()}
json.dump(res, open(OUT / "build.json", "w"), indent=1)
