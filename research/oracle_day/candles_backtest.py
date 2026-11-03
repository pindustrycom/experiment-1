"""Trade every candlestick / indicator signal from candles.py on $20, SPY and QQQ 1-minute bars, 2019 to now.

Rule per signal: when it fires at a bar's close, buy at that close and sell 5 minutes later (a new signal while holding
extends the hold; no overnight). Wealth compounds; costs 0, 1 and 2 bp per side. Also the all-signals logistic model,
walk-forward (refit each January on all earlier years), trading its top 1% of minutes. Benchmarks: buy and hold, and
the perfect-hindsight oracle (5-minute bars, 1 bp). Paper research only.

python candles_backtest.py DATA_DIR   (DATA_DIR/<TICKER>/<YYYY-MM>.csv, the exp7-data release)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from candles import H, day_features, load, logistic
from oracle import oracle

START, COSTS = 20.0, [0, 1, 2]


def run(c: np.ndarray, day: np.ndarray, sig: np.ndarray, cost_bp: float) -> tuple[float, int, np.ndarray]:
    """Hold for H bars after each signal; returns final multiple, number of trades and the 1-min in-market mask."""
    n = len(c)
    hold = np.zeros(n, bool)                           # hold[i]: in the market from bar i to i+1
    for k in range(H):                                 # bars i..i+H-1 after a signal at i
        hold[k:] |= sig[: n - k]
    same = day[1:] == day[:-1]
    hold[:-1] &= same                                  # never across the night
    r = np.r_[c[1:] / c[:-1], 1.0]
    entries = hold & ~np.r_[False, hold[:-1]]
    k = (1 - cost_bp / 1e4) ** 2
    m = np.prod(np.where(hold, r, 1.0)) * k ** entries.sum()
    return float(m), int(entries.sum()), hold


def main(data: str) -> None:
    root, out = Path(data), {"start_usd": START, "tickers": {}}
    for t in ["SPY", "QQQ"]:
        d = load(sorted((root / t).glob("*.csv")))
        f = pd.concat([day_features(g).assign(day=day) for day, g in d.groupby("day", sort=True)])
        sigs = [s for s in f.columns if s not in ("fwd", "day")]
        c, day = d["close"].to_numpy(), d["day"].to_numpy()
        years = d.index.year.to_numpy()
        rec = {"days": int(d["day"].nunique()), "first": d["day"].iloc[0], "last": d["day"].iloc[-1]}
        rec["buy_and_hold_usd"] = round(START * c[-1] / c[0], 2)
        intraday = np.prod(np.where(np.r_[day[1:] == day[:-1], False], np.r_[c[1:] / c[:-1], 1.0], 1.0))
        rec["hold_market_hours_only_usd"] = round(START * intraday, 2)
        lg = sum(np.log10(oracle(g["close"].to_numpy()[::5], 1)[0]) for _, g in d.groupby("day"))
        rec["oracle_5min_1bp_usd"] = f"20 x 10^{lg:.0f}"
        rows = {}
        for s in sigs:
            x = f[s].fillna(False).to_numpy(bool)
            res = {}
            for cb in COSTS:
                m, n, hold = run(c, day, x, cb)
                res[f"{cb}bp_usd"] = round(START * m, 2)
            res["trades"] = n
            res["time_in_market_%"] = round(100 * hold.mean(), 1)
            res["by_year_1bp_x"] = {int(y): round(run(c[years == y], day[years == y], x[years == y], 1)[0], 3) for y in np.unique(years)}
            rows[s] = res
        # walk-forward logistic on all signals, top 1% of minutes each year (threshold from the training years)
        X = f[sigs].fillna(False).to_numpy(float)
        fwd = f["fwd"].to_numpy()
        spike = np.zeros(len(f), bool)
        ok = ~np.isnan(fwd)
        spike[ok] = fwd[ok] >= np.quantile(fwd[ok], 0.95)
        pick = np.zeros(len(f), bool)
        for y in np.unique(years)[1:]:
            tr, te = (years < y) & ok, years == y
            w = logistic(X[tr], spike[tr].astype(float), it=25)
            p_tr = 1 / (1 + np.exp(-np.c_[np.ones(tr.sum()), X[tr]] @ w))
            p_te = 1 / (1 + np.exp(-np.c_[np.ones(te.sum()), X[te]] @ w))
            pick[te] = p_te >= np.quantile(p_tr, 0.99)
        res = {}
        for cb in COSTS:
            m, n, hold = run(c, day, pick, cb)
            res[f"{cb}bp_usd"] = round(START * m, 2)
        res["trades"], res["time_in_market_%"] = n, round(100 * hold.mean(), 1)
        res["by_year_1bp_x"] = {int(y): round(run(c[years == y], day[years == y], pick[years == y], 1)[0], 3) for y in np.unique(years)}
        rows["ALL_SIGNALS_model_top1pct (from 2020)"] = res
        rec["strategies"] = dict(sorted(rows.items(), key=lambda kv: -kv[1]["1bp_usd"]))
        out["tickers"][t] = rec
        print(t, {k: v for k, v in rec.items() if k != "strategies"})
        print(pd.DataFrame(rec["strategies"]).T.drop(columns="by_year_1bp_x").to_string())
    Path("candles_backtest.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
