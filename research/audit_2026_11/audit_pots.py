"""Audit table for every pot with a daily replay (2019-01 to 2026-10): return, risk, beta, alpha, and what plain SPY
held at the same beta would have made. Args: returns_pit.csv prices_dir out_json"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import alpha, beta_matched, perf, rf_daily

R = pd.read_csv(sys.argv[1], parse_dates=["date"]).set_index("date")
P = Path(sys.argv[2])
rf = rf_daily(P, R.index)
spy = R["SPY"]

# Top 3 as the site builds it: equal thirds of reversal, pelosi, exp2, reset at each week's last close
w = R.index.to_period("W")
last = np.r_[w[1:] != w[:-1], True]
legs = R[["reversal", "pelosi", "exp2"]].dropna()
val, out = np.ones(3) / 3, []
for d, row in legs.iterrows():
    val = val * (1 + row.to_numpy())
    tot = val.sum()
    out.append(tot)
    if last[R.index.get_loc(d)]:
        val = np.ones(3) * tot / 3
R["top3"] = pd.Series(out, index=legs.index).pct_change().fillna(out[0] - 1)

res = {}
print(f"{'pot':12} {'cagr':>7} {'vol':>6} {'sharpe':>6} {'maxdd':>7} {'beta':>5} {'alpha':>6} {'t':>5} | {'SPY@beta':>8} {'its dd':>7} | "
      f"{'t 19-22':>7} {'t 23-26':>7}")
for k in ["SPY", "exp1", "exp2", "exp7", "trend", "reversal", "pelosi", "best_ideas", "conviction", "top3"]:
    r = R[k].dropna()
    p = perf(r, rf)
    a = alpha(r, spy, rf)
    bm = perf(beta_matched(spy.reindex(r.index), rf.reindex(r.index), a["beta"]), rf)
    h1 = alpha(r[:"2022-12-31"], spy, rf)
    h2 = alpha(r["2023-01-01":], spy, rf)
    res[k] = {**{x: float(v) for x, v in p.items()}, **{x: float(v) for x, v in a.items()},
              "spy_at_beta_cagr": float(bm["cagr"]), "spy_at_beta_dd": float(bm["max_dd"]),
              "alpha_t_2019_22": float(h1["alpha_t"]), "alpha_t_2023_26": float(h2["alpha_t"]),
              "from": str(r.index[0].date()), "to": str(r.index[-1].date())}
    print(f"{k:12} {100*p['cagr']:7.1f} {100*p['vol']:6.1f} {p['sharpe']:6.2f} {100*p['max_dd']:7.1f} {a['beta']:5.2f} "
          f"{100*a['alpha']:6.1f} {a['alpha_t']:5.1f} | {100*bm['cagr']:8.1f} {100*bm['max_dd']:7.1f} | {h1['alpha_t']:7.1f} {h2['alpha_t']:7.1f}")
json.dump(res, open(sys.argv[3], "w"), indent=1)
