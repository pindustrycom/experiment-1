"""Daily USD returns (after each pot's own costs) of every strategy with a history, 2019 on, into returns.csv."""
import shutil, sys, pathlib, json
import pandas as pd, numpy as np
S = pathlib.Path(sys.argv[1])
sys.path[:0] = ["strategies", "exp7_vwap"]
from strats import data, rules, live, copy
from exp7 import bars as e7bars, live as e7live
D = S / "data"
# copy pots need data/copy with 13f, ptr, cusips and prices of their names
cp_dir = D / "copy"
if not cp_dir.exists():
    shutil.copytree(S / "dl/ts/13f", cp_dir / "13f"); shutil.copytree(S / "dl/ts/ptr", cp_dir / "ptr"); shutil.copy(S / "dl/ts/cusips.json", cp_dir)
cp = copy.load(D)
names = copy.tickers(cp)
for t in names:   # prices of copy names not in the S&P cache, from the trader-skill cache (closes only)
    f, src = D / "prices" / f"{t}.csv", S / "dl/ts/prices" / f"{t}.csv"
    if not f.exists() and src.exists():
        x = pd.read_csv(src); x.columns = ["date", "adj_close"]; x.to_csv(f, index=False)
members = sorted(pd.read_csv(D / "sp500.csv").ticker.str.replace(".", "-"))
px = data.load_prices(D, sorted(set(rules.TREND_ETFS) | set(members) | names))
spy = px["SPY"].dropna(); px = px.loc[spy.index]
rep = live.replay(px, members, cp=cp)
print("replay", rep.columns.tolist(), rep.index[0].date(), rep.index[-1].date())
# Exp7: SPY 3-min VWAP/EMA rule on the 1-minute cache
d1 = e7bars.load(S / "dl/exp7", "SPY")
e7, _ = e7live.replay(d1, "2019-01-02")
e7 = e7.set_index(pd.to_datetime(e7["date"]))["ret"].rename("exp7")
# Exp1 and Exp2: the 2019 replays (portfolio_2019); Exp2 from CAD back to USD
p = pathlib.Path("/mnt/project-files/research/portfolio_2019")
e1 = pd.read_csv(p / "exp1_daily_usd.csv", index_col="date", parse_dates=True)["ret"].rename("exp1")
c = pd.read_csv(p / "daily_cad.csv", index_col="date", parse_dates=True)
v = c["exp2"] / c["fx"]
e2 = v.pct_change().rename("exp2"); e2.iloc[0] = v.iloc[0] / (1000 / 1.3642) - 1
out = rep.join([e1, e2, e7], how="left")
out.to_csv(S / "alloc/returns.csv")
print(out.describe().T[["count", "mean", "std"]])
print({k: str(out[k].dropna().index[-1].date()) for k in out})
