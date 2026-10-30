"""Parametrize each strategy's daily returns as a random process and see what the parameters predict.

For daily account returns with mean m and standard deviation s, log wealth grows by about g = m - s^2/2 a day
(the volatility drag). The Kelly fraction f* = m / s^2 is the share of the account (or leverage) that maximises
long-run growth; f* <= 0 means the best bet is nothing. A lognormal with these parameters gives the odds of
any outcome, such as doubling $20 in a year, which is how "luck" can be priced. A Student-t fit gives the tails.

    python params.py OUT_DIR   (OUT_DIR from study.py, with daily_returns.pkl)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

out = Path(sys.argv[1])
R = pd.read_pickle(out / "daily_returns.pkl")
R = R[R.index >= "2000-01-03"].drop(columns=["Hindsight: tomorrow's #1 gainer (impossible)"])
rows = []
for name, r in R.items():
    r = r[r.index >= (r.ne(0).idxmax())]
    m, s = r.mean(), r.std()
    g = m - s * s / 2
    lg = np.log1p(r).mean()
    nu = stats.t.fit(r[r != 0])[0] if (r != 0).sum() > 100 else np.nan
    nu = nu if nu < 30 else np.nan  # capped (target/stop) returns are not t-shaped
    # One year (252 days) under a lognormal with these parameters
    mu_y, sd_y = 252 * lg, np.sqrt(252) * np.log1p(r).std()
    rows.append({"strategy": name, "mean_bp/day": round(1e4 * m, 1), "vol_%/day": round(100 * s, 2),
                 "drag_bp/day": round(1e4 * s * s / 2, 1), "growth_bp/day": round(1e4 * g, 1),
                 "actual_log_growth_bp/day": round(1e4 * lg, 1), "kelly_f": round(m / (s * s), 2),
                 "tail_nu": round(nu, 1),
                 "P(profit in a year)_%": round(100 * stats.norm.sf(0, mu_y, sd_y), 1),
                 "P(double in a year)_%": round(100 * stats.norm.sf(np.log(2), mu_y, sd_y), 2),
                 "P(10x in a year)_%": round(100 * stats.norm.sf(np.log(10), mu_y, sd_y), 4)})
T = pd.DataFrame(rows)
T.to_csv(out / "params.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 20)
print(T.to_string(index=False))

# The 8%-a-day dream as parameters: 100 trades of +8% need log growth ln(1.08) = 770 bp per trade.
print("\nNeeded for 2,200x in 100 trades: growth", round(1e4 * np.log(1.08)), "bp/trade")
# Kelly on QQQ: how much leverage the parameters say is optimal, and what 3x does
q = pd.read_pickle(out / "daily_returns.pkl")["QQQ hold"]
q = q[q.index >= "2000-01-03"]
m, s = q.mean(), q.std()
for L in (1, 2, 3, round(m / s**2, 1)):
    print(f"QQQ x{L}: model growth {252 * (L * m - (L * s) ** 2 / 2) * 100:.1f}%/yr (log, before fees)")
