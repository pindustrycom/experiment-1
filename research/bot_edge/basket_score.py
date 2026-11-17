"""Score a multi-pair freqtrade run (NostalgiaForInfinity on a 12-coin basket, up to 6 trades open).

  python basket_score.py DATADIR RESULT_ZIP OUT.csv

Each trade's weight is its stake over the account balance when it opened; the portfolio return of a bar is the sum of
weight x pair return over open trades, minus fees. Random baseline: each pair's weight series is shifted in time by
its own random offset (at least 20 days), 2000 draws; p = share of draws that do at least as well. Benchmark:
equal-weight basket of the same coins, rebalanced daily.
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as ek  # noqa: E402
from ft_score import load  # noqa: E402

datadir, zpath, out = sys.argv[1:4]
res = load(zpath)
strat, s = next(iter(res.items()))
tf = s["timeframe"]
start = pd.Timestamp(s["backtest_start"], tz="UTC")
pairs = sorted({t["pair"].split("/")[0] for t in s["trades"]} | set(p.split("/")[0] for p in s.get("pairlist", [])))
closes = pd.DataFrame({b: pd.read_feather(f"{datadir}/{b}_USDT-{tf}.feather").set_index("date").close for b in pairs}).sort_index()
closes = closes[closes.index >= start].ffill()
R = closes.pct_change().fillna(0.0)
W = pd.DataFrame(0.0, index=R.index, columns=pairs)
fees = pd.Series(0.0, index=R.index)
bal = s["starting_balance"]
for t in sorted(s["trades"], key=lambda t: t["open_date"]):
    b = t["pair"].split("/")[0]
    i0 = W.index.searchsorted(pd.Timestamp(t["open_date"]))
    i1 = min(W.index.searchsorted(pd.Timestamp(t["close_date"])), len(W) - 1)
    w = t["stake_amount"] / max(bal, 1e-9)
    W.iloc[i0 + 1:i1 + 1, W.columns.get_loc(b)] += w * (-1 if t["is_short"] else 1)
    fees.iloc[i0] += w * t["fee_open"]
    fees.iloc[i1] += w * t["fee_close"]
    bal += t["profit_abs"]
gross = (W * R).sum(axis=1)
net = gross - fees
actual = float(gross.sum())
rng = np.random.default_rng(0)
n = len(R)
ms = int(20 * 288 if tf == "5m" else 20 * 24)
Wv, Rv = W.to_numpy(), R.to_numpy()
draws = np.empty(2000)
for k in range(2000):
    tot = 0.0
    for j in range(Wv.shape[1]):
        sh = int(rng.integers(ms, n - ms))
        tot += float(np.roll(Wv[:, j], sh) @ Rv[:, j])
    draws[k] = tot
p = (1 + np.sum(draws >= actual)) / (1 + len(draws))
z = (actual - draws.mean()) / draws.std()
dn, dg = ek.daily(net), ek.daily(gross)
hold = ek.daily(R.mean(axis=1))
pf, ph = ek.perf(dn, 365), ek.perf(hold, 365)
X = np.column_stack([np.ones(len(dn)), hold.reindex(dn.index).fillna(0).to_numpy()])
b, tt = ek.nw_t(dn.to_numpy(), X)
row = {"source": "NostalgiaForInfinity (basket)", "strategy": strat, "asset": f"basket of {len(pairs)}", "tf": tf, "native_tf": tf,
       "trades": len(s["trades"]), **pf, "gross_cagr": ek.perf(dg, 365)["cagr"], "hold_cagr": ph["cagr"], "hold_sharpe": ph["sharpe"],
       "hold_dd": ph["max_dd"], "p_random": float(p), "z_random": float(z), "alpha_ann": float(b[0] * 365), "alpha_t": float(tt[0]),
       "beta": float(b[1]), "exposure": float((W.abs().sum(axis=1) > 0).mean()), "ft_profit_total": s["profit_total"],
       "win_rate": float(np.mean([t["profit_ratio"] > 0 for t in s["trades"]])), "years": len(dn) / 365}
half = pd.Timestamp("2025-01-01", tz="UTC")
for name, m in (("h1", R.index < half), ("h2", R.index >= half)):
    a = float(gross[m].sum())
    dd = []
    for k in range(500):
        tot = 0.0
        for j in range(Wv.shape[1]):
            tot += float(np.roll(Wv[m][:, j], int(rng.integers(ms, m.sum() - ms))) @ Rv[m][:, j])
        dd.append(tot)
    dd = np.array(dd)
    row[f"z_random_{name}"] = float((a - dd.mean()) / dd.std())
    row[f"p_random_{name}"] = float((1 + np.sum(dd >= a)) / 501)
row["note"] = "halves split at 2025-01-01 (sample starts 2023)"
pd.DataFrame([row]).to_csv(out, index=False)
print(row)
