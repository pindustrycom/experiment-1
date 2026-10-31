"""Does news before a flagged day tell an 8% rise from an 8% fall? Reads flagged.csv and collect.py's scores.jsonl.

For each signal (Haiku sentiment, Haiku lean, word-list score): AUC for up vs down among flagged days that moved 8%+,
correlation with the next return on all flagged days, and a tilted-straddle check: if the lean were used to buy only
the call (lean > 0) or only the put (lean < 0), how often would the right side have been picked on the big days.
Also split by period (2019-2023, 2024+) since the language model may know older events.

    python evaluate.py flagged.csv scores.jsonl [OUT.json]
"""
import json
import sys

import numpy as np
import pandas as pd


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


fl = pd.read_csv(sys.argv[1])
with open(sys.argv[2]) as fh:
    sc = pd.DataFrame([json.loads(x) for x in fh])
D = fl.merge(sc.drop(columns=["titles"]), on=["date", "ticker"], how="inner")
D["up"] = (D["next_ret"] > 0).astype(int)
out = {"stock_days": len(D), "with_headlines": int((D["n"] > 0).sum()), "big_days": int(D["big"].sum()),
       "median_headlines": float(D["n"].median())}
for per, m in [("all", D["date"] >= "2019"), ("2019-2023", D["date"] < "2024"), ("2024+", D["date"] >= "2024")]:
    X = D[m]
    big = X[X["big"] == 1]
    res = {"n": len(X), "big": len(big)}
    for sig in ["sentiment", "lean", "words"]:
        if sig not in X or X[sig].notna().sum() < 30:
            continue
        b = big.dropna(subset=[sig])
        a = X.dropna(subset=[sig])
        c = np.corrcoef(a[sig], a["next_ret"])[0, 1] if len(a) > 30 else np.nan
        right = np.mean(np.sign(b[sig]) == np.sign(b["next_ret"])) if len(b) else np.nan
        res[sig] = {"auc_up_vs_down_on_big_days": round(auc(b["up"], b[sig]), 3), "n_big": len(b),
                    "corr_next_ret_all": round(float(c), 3), "t_corr": round(float(c * np.sqrt(len(a) - 2) / np.sqrt(1 - c * c)), 2),
                    "right_side_on_big_days_%": round(100 * right, 1),
                    "nonzero_%": round(100 * (b[sig] != 0).mean(), 1) if len(b) else None}
    if "earnings_soon" in X:
        e = X["earnings_soon"].fillna(False).astype(bool)
        res["P(8%)_if_earnings_soon_%"] = round(100 * X.loc[e, "big"].mean(), 1) if e.any() else None
        res["P(8%)_otherwise_%"] = round(100 * X.loc[~e, "big"].mean(), 1)
        res["earnings_soon_share_%"] = round(100 * e.mean(), 1)
    out[per] = res
print(json.dumps(out, indent=1))
if len(sys.argv) > 3:
    with open(sys.argv[3], "w") as fh:
        fh.write(json.dumps(out, indent=1))
