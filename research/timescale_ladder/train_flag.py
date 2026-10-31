"""Fit the straddle pot's flag model (experiments/strategies/strats/flag_model.json) on the point-in-time S&P 500.

Same features as the live pot (strats.straddle.features), target |next-day return| >= 8%. Reports the out-of-sample
check (train 2000-2015, test 2016-2026), then refits on 2000-2026 for the live model.

    python train_flag.py PANEL.pkl MODEL.json   (PANEL from research/daily_movers/panel.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "strategies"))
from strats import straddle  # noqa: E402

P = pd.read_pickle(sys.argv[1])
C = P["close"].ffill(limit=5)
F = straddle.features(C, P["open"], P["high"], P["low"], P["volume"], P["etf"]["SPY"].ffill())
nxt = np.log(C / C.shift(1)).shift(-1)
ok = P["member"] & ~P["notrade"] & ~P["notrade"].shift(-1, fill_value=False) & (P["close"] > straddle.MIN_PRICE) & nxt.notna()
D = pd.DataFrame({k: v.where(ok).stack() for k, v in F.items()})
D["y"] = nxt.where(ok).stack()
D = D.dropna().reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
D["big"] = (D["y"].abs() >= np.log(1.08)).astype(int)


def fit(X: pd.DataFrame, y: pd.Series) -> dict:
    Z = straddle.design(X)
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    lr = LogisticRegression(max_iter=2000, C=1.0).fit((Z - mu) / sd, y)
    return {"features": straddle.FEATURES, "mean": mu.tolist(), "std": sd.tolist(), "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0])}


tr, te = D["date"] < "2016-01-01", D["date"] >= "2016-01-01"
m = fit(D[tr], D.loc[tr, "big"])
T = D[te].assign(p=straddle.score(m, D[te]).to_numpy())
T["rank"] = T.groupby("date")["p"].rank(ascending=False, method="first")
top = T[T["rank"] <= straddle.TOP_N]
check = {"test_auc": round(roc_auc_score(T["big"], T["p"]), 3), "base_rate_%": round(100 * T["big"].mean(), 2),
         f"top{straddle.TOP_N}_P(8%)_%": round(100 * top["big"].mean(), 1),
         "top1_P(8%)_%": round(100 * T.loc[T["rank"] == 1, "big"].mean(), 1),
         f"top{straddle.TOP_N}_mean_abs_move_%": round(100 * np.expm1(top["y"].abs()).mean(), 2),
         "test_rows": int(te.sum())}
print(json.dumps(check, indent=1))
final = fit(D, D["big"])
final["check_2016_2026"] = check
final["fitted"] = f"{D['date'].min().date()}..{D['date'].max().date()}, {len(D)} stock-days"
Path(sys.argv[2]).write_text(json.dumps(final, indent=1))
