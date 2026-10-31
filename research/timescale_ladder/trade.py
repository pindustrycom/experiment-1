"""Does any of it make money? Test years 2023-2026, SPY, 0.5 bp a side per unit of position change.

1. Direction: hold +1 (long) or -1 (short) SPY each period by the sign of the ladder's or the GRU's return
   forecast, and long/flat (no shorting, as in a $20 cash account). Compared with simply holding.
2. Volatility: a day-scale volatility-targeted SPY position (10%/yr target, at most 2x) sized by each variance
   forecast; better forecasts should give a steadier ride (higher Sharpe, smaller drawdown) for the same target.

    python trade.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ladder import LEVELS, TEST

COST = 0.5e-4
PER_YEAR = {"month": 12, "week": 52, "day": 252, "hour": 252 * 7, "minute": 252 * 390}


def perf(pos: np.ndarray, r: np.ndarray, per_year: int) -> dict:
    pnl = pos * r - COST * np.abs(np.diff(np.r_[0, pos]))
    eq = np.exp(np.cumsum(np.log1p(pnl)))
    yrs = len(r) / per_year
    return {"cagr_%": 100 * (eq[-1] ** (1 / yrs) - 1), "sharpe": pnl.mean() / pnl.std() * np.sqrt(per_year) if pnl.std() > 0 else 0,
            "max_dd_%": 100 * (eq / np.maximum.accumulate(eq) - 1).min(), "trades/yr": np.abs(np.diff(np.r_[0, pos])).sum() / yrs}


def main(out: Path) -> None:
    L = pd.read_pickle(out / "ladder.pkl")
    G = pd.read_pickle(out / "gru_preds.pkl")
    S, M = L["S"], L["models"]
    rows = []
    for lvl in LEVELS:
        f = S[lvl]
        te = (f["start"] >= TEST).to_numpy()
        r = np.expm1(f["r"].to_numpy()[te])
        sig = {"ladder": M["HAR + full ladder"][lvl]["mu_hat"][te], "GRU": np.nan_to_num(G[lvl]["gru_z"].to_numpy()[te])}
        rows.append({"scale": lvl, "rule": "hold SPY", **perf(np.ones(te.sum()), r, PER_YEAR[lvl])})
        for k, s in sig.items():
            rows.append({"scale": lvl, "rule": f"{k} long/short", **perf(np.sign(s), r, PER_YEAR[lvl])})
            rows.append({"scale": lvl, "rule": f"{k} long/flat", **perf((s > 0).astype(float), r, PER_YEAR[lvl])})
    T = pd.DataFrame(rows)
    # Day-scale volatility targeting
    f = S["day"]
    te = (f["start"] >= TEST).to_numpy()
    r = np.expm1(f["r"].to_numpy()[te])
    vt = []
    for name in ["last period (random walk)", "HAR (own 1/5/22 periods)", "own scale only", "HAR + full ladder"]:
        v = np.exp(M[name]["day"]["h_hat_full"][te])
        vt.append({"forecast": name, **perf(np.minimum(0.10 / np.sqrt(252 * v), 2.0), r, 252)})
    v = np.exp(G["day"]["gru_h"].to_numpy()[te])
    vt.append({"forecast": "HAR + ladder + GRU", **perf(np.minimum(0.10 / np.sqrt(252 * v), 2.0), r, 252)})
    V = pd.DataFrame(vt)
    pd.set_option("display.width", 200)
    print(T.round(2).to_string(index=False))
    print(V.round(2).to_string(index=False))
    T.to_csv(out / "trade_direction.csv", index=False)
    V.to_csv(out / "trade_voltarget.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
