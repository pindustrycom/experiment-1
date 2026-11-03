"""Perfect-hindsight ("oracle") gains for one regular session: own a stock only while it rises, sell before every fall.

Start with $20. On every bar the oracle holds the stock if (and only if) the next price is higher, so the day's gain is the
product of all the up-moves. With a cost per side, a DP picks the best in/out path (tiny wiggles stop being worth a round
trip). The "switch" version may hold any one ticker (or cash) each bar. Paper research only; nobody can trade this way.

python oracle.py DATA_DIR DAY   (DATA_DIR/<TICKER>/*.csv in Tiingo IEX 1-minute format: date,open,high,low,close,volume)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

START = 20.0
COSTS_BP = [0, 1, 5]
FREQS = ["1min", "5min", "15min", "30min", "60min"]


def load(f: Path, day: str) -> pd.Series:
    d = pd.read_csv(f)
    d["ts"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert("America/New_York")
    d = d.set_index("ts").sort_index()
    d = d[d.index.strftime("%Y-%m-%d") == day].between_time("09:30", "15:59")
    if len(d) < 300:
        return pd.Series(dtype=float)
    # price path: the day's first open, then every bar's close
    first = pd.Series([d["open"].iloc[0]], index=[d.index[0] - pd.Timedelta(minutes=1)])
    return pd.concat([first, d["close"]])


def path(p: pd.Series, freq: str) -> np.ndarray:
    if freq == "1min":
        return p.to_numpy()
    r = p.iloc[1:].resample(freq, origin="start_day", offset="9h30min", label="right", closed="right").last().dropna()
    return np.r_[p.iloc[0], r.to_numpy()]


def oracle(prices: np.ndarray, cost_bp: float) -> tuple[float, int]:
    """Best final wealth multiple and number of round trips with `cost_bp` per side (DP over cash / holding)."""
    k = 1 - cost_bp / 1e4
    cash, hold, trips_c, trips_h = 1.0, k, 0, 1      # hold = wealth if holding (bought at bar 0)
    for a, b in zip(prices[:-1], prices[1:]):
        hold *= b / a
        sell = hold * k
        new_cash, new_tc = (sell, trips_h) if sell > cash else (cash, trips_c)
        buy = cash * k
        new_hold, new_th = (buy, trips_c + 1) if buy > hold else (hold, trips_h)
        cash, trips_c, hold, trips_h = new_cash, new_tc, new_hold, new_th
    return max(cash, hold * k), trips_c if cash >= hold * k else trips_h


def switch_oracle(paths: dict[str, np.ndarray], cost_bp: float) -> float:
    """Hold any one ticker or cash each bar; switching costs cost_bp per side (sell + buy)."""
    names = list(paths)
    P = np.vstack([paths[n] for n in names])            # tickers x bars
    k = 1 - cost_bp / 1e4
    cash, hold = 1.0, np.full(len(names), k)
    for t in range(1, P.shape[1]):
        hold = hold * P[:, t] / P[:, t - 1]
        best_exit = hold.max() * k
        cash = max(cash, best_exit)
        hold = np.maximum(hold, cash * k)               # buy from cash (incl. just-sold) at this bar
    return max(cash, hold.max() * k)


def main(data: str, day: str) -> None:
    series = {f.parent.name: load(f, day) for f in sorted(Path(data).glob(f"*/{day[:7]}.csv"))}
    series = {t: s for t, s in series.items() if len(s)}
    rows, out = [], {"day": day, "start_usd": START, "tickers": {}, "switch": {}}
    for t, s in series.items():
        hold = s.iloc[-1] / s.iloc[0]
        rec = {"buy_and_hold_usd": round(START * hold, 2)}
        for fq in FREQS:
            for c in COSTS_BP:
                m, n = oracle(path(s, fq), c)
                rec[f"{fq}_{c}bp_usd"] = round(START * m, 2)
                rec[f"{fq}_{c}bp_trades"] = n
        out["tickers"][t] = rec
        rows.append({"ticker": t, "hold": rec["buy_and_hold_usd"], **{f"{fq} {c}bp": rec[f"{fq}_{c}bp_usd"] for fq in FREQS for c in COSTS_BP}})
    tab = pd.DataFrame(rows).set_index("ticker").sort_values("1min 0bp", ascending=False)
    if len(series) > 1:
        idx = sorted(set.intersection(*[set(s.index) for s in series.values()]))
        al = {t: s.reindex(idx) for t, s in series.items()}
        for fq in FREQS:
            ps = {t: path(s, fq) for t, s in al.items()}
            for c in COSTS_BP:
                out["switch"][f"{fq}_{c}bp_usd"] = round(START * switch_oracle(ps, c), 2)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(tab.to_string())
    print(json.dumps(out["switch"], indent=1))
    Path("oracle_result.json").write_text(json.dumps(out, indent=1))
    tab.to_csv("oracle_table.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
