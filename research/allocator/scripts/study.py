import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, "strategies")
from strats import allocator as A, live, regimes
S = pathlib.Path(sys.argv[1])
R = pd.read_csv(S / "alloc/returns.csv", index_col=0, parse_dates=True)
R = R.loc[:"2026-10-15"]       # Exp1/Exp2 replays end here
spy_full = pd.read_csv(S / "dl/papers/prices/SPY.csv", index_col="date", parse_dates=True)["adj_close"]
lab = A.point_in_time_regimes(spy_full)
pots = ["exp1", "exp2", "exp7", "trend", "reversal", "best_ideas", "conviction", "pelosi"]
START = "2020-01-02"
res = {}
def add(name, r):
    r = r[r.index >= START].dropna(); s = live.stats(r)
    res[name] = {"cagr": round(100*s["cagr"],1), "vol": round(100*s["vol"],1), "sharpe": round(s["sharpe"],2), "max_dd": round(100*s["max_dd"],1),
                 "end_1000": round(1000*(1+s["total"])), "years": {y: round(100*v,1) for y, v in s["years"].items()}}
    return r
for k in pots + ["SPY"]: add(k, R[k])
add("Equal weight, all 8 pots", R[pots].mean(axis=1))
a = A.run(R[pots + ["SPY"]], lab, START); add("Allocator (regimes)", a["returns"])
b = A.run(R[pots + ["SPY"]], lab, START, use_regime=False); add("Allocator, no regimes", b["returns"])
c = A.run(R[pots + ["SPY"]], lab, START, oracle=True); add("Allocator, hindsight (look-ahead)", c["returns"])
d = A.run(R[pots], lab, START); add("Allocator (regimes), no SPY option", d["returns"])
df = pd.DataFrame(res).T[["cagr","vol","sharpe","max_dd","end_1000"]].sort_values("sharpe", ascending=False)
print(df.to_string())
print(pd.DataFrame({k: v["years"] for k, v in res.items()}).T.loc[df.index].to_string())
bk = pd.DataFrame([{**{"date": x["date"], "regime": x["regime"]}, **x["w"]} for x in a["books"]]).fillna(0)
print("rebalances", len(bk)); print(bk.drop(columns=["date"]).groupby("regime").mean().round(2).to_string())
print(bk.tail(3).to_string())
print("regime share since 2020", lab[lab.index>=START].value_counts(normalize=True).round(2).to_dict(), "today", lab.index[-1].date(), lab.iloc[-1])
json.dump({"stats": res, "books": a["books"]}, open(S/"alloc/result.json","w"), default=str)
pd.DataFrame({"allocator": a["returns"], "no_regimes": b["returns"], "equal_weight": R[pots].mean(axis=1), "SPY": R["SPY"]}).loc[START:].to_csv(S/"alloc/equity_daily.csv")
