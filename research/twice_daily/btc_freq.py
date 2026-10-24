"""BTC Donchian 4h (strats/crypto.py rule, no stop): how much is lost if the trades are only placed when a scheduled
run happens, instead of at every 4-hour close? Signals always come from finished 4-hour bars; a run at time t looks
at the rule's state after the last finished bar and, if it differs from the position, trades at the open of the hour
that starts at t (1-hour Binance BTCUSDT bars). Usage: python btc_freq.py KLINE_CACHE"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "github_bots"))
import data  # noqa: E402

CHANNEL, TREND = 20, 200
h = data.binance(sys.argv[1], "BTCUSDT", "1h", "2018-06")
g = h.groupby(h.index.floor("4h"))
b4 = pd.DataFrame({"high": g.high.max(), "low": g.low.min(), "close": g.close.last()})[g.size() == 4]
hi = b4.high.rolling(CHANNEL).max().shift()
lo = b4.low.rolling(CHANNEL).min().shift()
sma = b4.close.rolling(TREND).mean()
want, w = [], False
for c, a, b, s in zip(b4.close, hi, lo, sma):
    if not w and c > a and c > s:
        w = True
    elif w and c < b:
        w = False
    want.append(w)
# rule state known from each 4h bar's close time; carried to every hour
state = pd.Series(want, index=b4.index + pd.Timedelta(hours=4)).reindex(h.index, method="ffill").fillna(False).astype(bool)

ny = h.index.tz_localize("UTC").tz_convert("America/New_York")
wk = ny.weekday < 5
utc_h, utc_m = h.index.hour, h.index.minute
RUNS = {
    "every 4h (what the pot shows)": h.index.hour % 4 == 0,
    "every 4h, weekdays only": (h.index.hour % 4 == 0) & wk,
    "twice a day weekdays (12:00 + 18:00 NY)": wk & ((ny.hour == 12) | (ny.hour == 18)),
    "twice a day every day (06:00 + 18:00 NY)": (ny.hour == 6) | (ny.hour == 18),
    "once a day every day (18:00 NY)": ny.hour == 18,
    "once a day weekdays (22:00 UTC = live cron 21:40)": wk & (utc_h == 22),
}
o = h.open.to_numpy()
ret_hold = np.r_[o[1:] / o[:-1] - 1, 0.0]          # open of hour i to open of hour i+1
START, END = "2019-01-01", "2026-10-01"
sel = (h.index >= START) & (h.index < END)


def run(mask, cost):
    pos, e, n, eq = False, 1.0, 0, np.empty(len(o))
    st = state.to_numpy()
    for i in range(len(o)):
        if mask[i] and st[i] != pos:
            pos = st[i]; e *= 1 - cost; n += 1
        if pos:
            e *= 1 + ret_hold[i]
        eq[i] = e
    return pd.Series(eq, index=h.index)[sel], n


def stats(eq):
    eq = eq / eq.iloc[0]
    d = eq.resample("1D").last().pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (f"{eq.iloc[-1] ** (1 / yrs) - 1:+7.1%}/yr  Sharpe {d.mean() / d.std() * np.sqrt(365):.2f}  "
            f"maxDD {(eq / eq.cummax() - 1).min():6.1%}")


print("BTC hold:", stats(pd.Series(o, index=h.index)[sel]))
for cost in (0.0025, 0.0010):
    print(f"\n--- cost {cost * 1e4:.0f} bp a side, {START} to {END}")
    for name, m in RUNS.items():
        eq, n = run(np.asarray(m), cost)
        yr = " ".join(f"{y}:{(eq[eq.index.year == y].iloc[-1] / eq[eq.index.year == y].iloc[0] - 1):+.0%}"
                      for y in range(2019, 2027))
        print(f"{name:52s} {stats(eq)}  trades {n:4d}  | {yr}")
