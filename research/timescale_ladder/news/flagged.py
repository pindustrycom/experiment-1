"""The stock-days to look up news for: each session 2019-2026, the TOP point-in-time S&P 500 members the live flag model
(experiments/strategies/strats/flag_model.json) rates most likely to move 8%+ the next day, with that next return.

    python flagged.py PANEL.pkl flagged.csv   (PANEL from research/daily_movers/panel.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments" / "strategies"))
from strats import straddle

TOP, START = 2, "2019-01-01"
P = pd.read_pickle(sys.argv[1])
C = P["close"].ffill(limit=5)
F = straddle.features(C, P["open"], P["high"], P["low"], P["volume"], P["etf"]["SPY"].ffill())
nxt = np.log(C / C.shift(1)).shift(-1)
ok = P["member"] & ~P["notrade"] & ~P["notrade"].shift(-1, fill_value=False) & (P["close"] > straddle.MIN_PRICE) & nxt.notna()
ok &= pd.Series(P["member"].index >= START, index=P["member"].index).to_numpy()[:, None]
D = pd.DataFrame({k: v.where(ok).stack() for k, v in F.items()})
D["next_ret"] = nxt.where(ok).stack()
D = D.dropna().reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})
D["p8"] = straddle.score(json.loads(straddle.FLAG_MODEL.read_text()), D).to_numpy()
D["rank"] = D.groupby("date")["p8"].rank(ascending=False, method="first")
out = D[D["rank"] <= TOP][["date", "ticker", "rank", "p8", "next_ret", "r_d" if "r_d" in D else "lrv_d"]].copy()
out["big"] = (out["next_ret"].abs() >= np.log(1.08)).astype(int)
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
out.round(6).to_csv(sys.argv[2], index=False)
print(len(out), "stock-days,", out["ticker"].nunique(), "tickers, P(8%)", round(100 * out["big"].mean(), 1), "%")
