"""The video's two extra checks applied to our live pots:
1. Harvey & Liu (2015) Sharpe haircut for the number of strategies tried in this project (about 300, from the edge tests).
2. A cloud of 200 random books built with the live weekly reversal pot's rules (20 point-in-time S&P 500 members,
   rebuilt every week, 1 bp a side like the live pot) to see where the real rule lands.
Inputs: the audit's survivorship-fixed pot replays (research/audit_2026_11/inputs/returns_pit.csv), the panel outputs.
Args: out_dir papers_data_dir returns_pit.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from stats import alpha_nw, haircut_sharpe, perf  # noqa: E402

OUT, D, POTS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
pots = pd.read_csv(POTS, parse_dates=["date"]).set_index("date")
irx = pd.read_csv(D / "prices" / "^IRX.csv", parse_dates=["date"]).set_index("date")["close"]
rf = irx.reindex(pots.index).ffill().shift(1).bfill() / 100 / 252
TRIALS = (1, 15, 50, 300)
res = {"pots": {}}
print(f"{'pot (2019-01..2026-10)':22s} {'Sharpe':>6s} {'t':>5s} " + " ".join(f"{'HC M=' + str(m):>9s}" for m in TRIALS)
      + "  excess-over-SPY Sharpe -> haircut at M=300")
for c in pots.columns:
    r = pots[c].dropna()
    p = perf(r, rf)
    hc = {m: haircut_sharpe(p["sharpe"], p["years"], m) for m in TRIALS}
    # the question that matters: does it beat SPY? use the Sharpe of (pot - SPY at the pot's beta)
    row = {"sharpe": p["sharpe"], "years": p["years"], "haircut": hc}
    if c != "SPY":
        a = alpha_nw(r, pots["SPY"], rf)
        ex = (r - rf) - a["beta"] * (pots["SPY"] - rf)
        ir = ex.mean() / ex.std() * np.sqrt(252)
        row.update(a)
        row["info_ratio"] = ir
        row["ir_haircut_300"] = haircut_sharpe(ir, p["years"], 300)
    res["pots"][c] = row
    tail = (f"  IR {row['info_ratio']:+.2f} -> {row['ir_haircut_300']['sr_haircut']:+.2f}" if c != "SPY" else "")
    print(f"{c:22s} {p['sharpe']:6.2f} {hc[1]['t']:5.1f} " + " ".join(f"{hc[m]['sr_haircut']:9.2f}" for m in TRIALS) + tail)

# random cloud for the weekly reversal rule
R = pd.read_parquet(OUT / "daily_returns.parquet")
M = pd.read_parquet(OUT / "members.parquet").to_numpy()
cal = R.index
Rv = np.nan_to_num(R.to_numpy(np.float64))
wk = cal.to_period("W")
wend = np.r_[wk[1:] != wk[:-1], False]
s0, s1 = cal.searchsorted(pots.index[0]), cal.searchsorted(pots.index[-1])
C = (1 + R.fillna(0)).cumprod().to_numpy()


def weekly_book(pick, cost=0.0001):
    out = np.zeros(len(cal))
    w = None
    for i in range(s0 - 7, s1 + 1):
        if w is not None:
            g = w * (1 + Rv[i])
            out[i] = g.sum() / w.sum() - 1
            w = g / g.sum()
        if wend[i]:
            nw = np.zeros(len(cal.values) and Rv.shape[1])
            nw[pick(i)] = 1 / 20
            if w is not None:
                out[i] -= cost * np.abs(nw - w).sum()
            w = nw
    return pd.Series(out, index=cal).iloc[s0:s1 + 1]


def reversal(i):
    ok = M[i] & (C[i - 5] > 0)
    r5 = np.where(ok, C[i] / C[i - 5] - 1, np.inf)
    return np.argsort(r5)[:20]


rng = np.random.default_rng(11)
rev = weekly_book(reversal)
rf2 = rf.reindex(rev.index).ffill().fillna(0)
cloud = [perf(weekly_book(lambda i: rng.choice(np.where(M[i])[0], 20, replace=False)), rf2) for _ in range(200)]
pr = perf(rev, rf2)
ends = np.array([x["end"] for x in cloud])
shs = np.array([x["sharpe"] for x in cloud])
spy = perf(pots["SPY"], rf)
res["reversal_vs_random"] = {"reversal": pr, "random_end_median": float(np.median(ends)),
                             "random_end_p5_p95": [float(np.percentile(ends, 5)), float(np.percentile(ends, 95))],
                             "random_sharpe_median": float(np.median(shs)),
                             "pct_end": float((ends < pr["end"]).mean() * 100),
                             "pct_sharpe": float((shs < pr["sharpe"]).mean() * 100), "spy": spy}
print(f"\nweekly reversal rebuilt here: x{pr['end']:.2f}, Sharpe {pr['sharpe']:.2f}; random 20 weekly: median "
      f"x{np.median(ends):.2f} (5-95% {np.percentile(ends, 5):.2f}-{np.percentile(ends, 95):.2f}), Sharpe median "
      f"{np.median(shs):.2f}; reversal beats {res['reversal_vs_random']['pct_end']:.0f}% of random on money, "
      f"{res['reversal_vs_random']['pct_sharpe']:.0f}% on Sharpe; SPY x{spy['end']:.2f} Sharpe {spy['sharpe']:.2f}")
(OUT / "haircut.json").write_text(json.dumps(res, indent=1, default=float))
