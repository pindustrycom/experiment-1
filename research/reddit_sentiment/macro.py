"""Joaquin's side questions, as quick tests: do oil shocks (e.g. the Iran war) and memory-chip prices move tech stocks,
and do they do it fast (same day, priced in) or slowly (next week, something to trade)?

    python macro.py PANEL.pkl WTI.csv OUT_DIR
Oil: FRED WTI spot (DCOILWTICO). Memory: no free DRAM spot series, so memory makers' shares (MU, WDC, STX) stand in.
Weekly (Friday) returns 2010-2026; Newey-West t-stats (4 lags). Tech = QQQ, and QQQ minus SPY for tech vs the market.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def nw(y: pd.Series, X: pd.DataFrame) -> dict:
    d = pd.concat([y, X], axis=1).dropna()
    m = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return {"n": int(m.nobs), **{f"b_{k}": round(float(m.params[k]), 3) for k in X.columns},
            **{f"t_{k}": round(float(m.tvalues[k]), 2) for k in X.columns}, "r2_%": round(100 * m.rsquared, 2)}


def main(panel: str, wti: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    P = pd.read_pickle(panel)
    px = P["etf"][["SPY", "QQQ"]].join(P["close"][[c for c in ["NVDA", "MU", "WDC", "STX", "AAPL", "MSFT"] if c in P["close"]]])
    res = {}
    if Path(wti).exists():
        w = pd.read_csv(wti)
        w.columns = ["date", "wti"]
        w["date"] = pd.to_datetime(w["date"])
        w["wti"] = pd.to_numeric(w["wti"], errors="coerce")
        w = w.set_index("date")["wti"]
        w = w[w > 0]  # April 2020's negative print breaks log returns
        px = px.join(w, how="left")
    else:  # FRED unreachable: oil producers' shares (S&P 500 energy names) stand in for the oil price
        en = [c for c in ["XOM", "CVX", "COP", "OXY", "EOG", "DVN", "APA", "HES", "MRO"] if c in P["close"]]
        px["wti"] = np.exp(np.log(P["close"][en].reindex(px.index)).diff().mean(axis=1).fillna(0).cumsum())
        res["oil_proxy"] = "equal-weight energy producers " + ",".join(en)
    W = np.log(px.resample("W-FRI").last()).diff()
    D = np.log(px).diff()
    W["tech_vs_mkt"] = W["QQQ"] - W["SPY"]
    D["tech_vs_mkt"] = D["QQQ"] - D["SPY"]
    if "wti" in W.columns:
        res["oil_same_week"] = {k: nw(W[k], W[["wti"]]) for k in ["QQQ", "tech_vs_mkt", "NVDA"]}
        lag = W[["wti"]].shift(1).rename(columns={"wti": "wti_last_week"})
        res["oil_predicts_next_week"] = {k: nw(W[k], lag) for k in ["QQQ", "tech_vs_mkt", "NVDA"]}
        big = D["wti"].abs() > 0.05
        nxt5 = np.log(px["QQQ"]).diff(5).shift(-5)
        res["oil_shock_days"] = {
            "n_days_|wti|>5%": int(big.sum()),
            "QQQ_same_day_when_oil_up_%": round(100 * D.loc[big & (D["wti"] > 0), "QQQ"].mean(), 2),
            "QQQ_same_day_when_oil_down_%": round(100 * D.loc[big & (D["wti"] < 0), "QQQ"].mean(), 2),
            "QQQ_next_5d_after_oil_up_%": round(100 * nxt5[big & (D["wti"] > 0)].mean(), 2),
            "QQQ_next_5d_after_oil_down_%": round(100 * nxt5[big & (D["wti"] < 0)].mean(), 2),
            "QQQ_next_5d_all_days_%": round(100 * nxt5.mean(), 2)}
        y26 = W.loc["2026-01-01":]
        res["oil_2026_same_week"] = {k: nw(y26[k], y26[["wti"]]) for k in ["QQQ", "tech_vs_mkt"]}
    W["memory"] = W[["MU", "WDC", "STX"]].mean(axis=1)
    res["memory_same_week"] = {k: nw(W[k], W[["memory"]]) for k in ["QQQ", "NVDA"]}
    lagm = W[["memory"]].shift(1).rename(columns={"memory": "memory_last_week"})
    res["memory_predicts_next_week"] = {k: nw(W[k], lagm) for k in ["QQQ", "NVDA", "tech_vs_mkt"]}
    res["memory_predicts_next_week_2023plus"] = {k: nw(W.loc["2023":, k], lagm.loc["2023":]) for k in ["QQQ", "NVDA"]}
    (out / "macro.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
