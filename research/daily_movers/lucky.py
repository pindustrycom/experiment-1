"""How many no-skill day traders look like winners by luck?

100,000 simulated traders each put all their money in one random S&P 500 member every day for a year
(2025-10-28 to 2026-10-28, real next-day returns, 5 bp a side). No skill at all. Count how many multiply $20.

    python lucky.py PANEL.pkl
"""
import sys

import numpy as np
import pandas as pd

P = pd.read_pickle(sys.argv[1])
C = P["close"].ffill(limit=5)
nxt = C.pct_change(fill_method=None).shift(-1)
ok = P["member"] & nxt.notna()
d = nxt.index[(nxt.index >= "2025-10-28") & (nxt.index < "2026-10-28")]
rng = np.random.default_rng(1)
N, cost = 100_000, 0.0005
logw = np.zeros(N)
for day in d:
    r = nxt.loc[day][ok.loc[day]].to_numpy()
    pick = r[rng.integers(0, len(r), N)]
    logw += np.log1p(pick) + np.log((1 - cost) / (1 + cost))
end = 20 * np.exp(logw)
print(f"{len(d)} days, {N} no-skill traders starting with $20")
for x in (1, 2, 3, 5, 10):
    print(f"  ended above ${20 * x:>4}: {100 * (end > 20 * x).mean():5.2f}%  ({(end > 20 * x).sum()} traders)")
print(f"  median ${np.median(end):.2f}, best ${end.max():.2f}, worst ${end.min():.2f}")
