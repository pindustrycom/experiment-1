"""Robustness check for the one rule that beat holding on both coins: Jesse's Donchian (buy a close above the previous
20-bar high while above the 200-bar SMA, sell a close below the previous 20-bar low) on 4-hour crypto bars.
Varies the channel, the trend filter, the cost and the coin (coins it was not picked on), and prints hold times.

  python robust.py CRYPTO_CACHE"""
from __future__ import annotations

import sys

import pandas as pd

import data
import ind
from engine import Rules, daily, simulate, stats

FROM = pd.Timestamp("2019-01-01")


def donchian(d, n=20, filt=200):
    hi, lo = d.high.rolling(n).max().shift(), d.low.rolling(n).min().shift()
    entry = d.close > hi
    if filt:
        entry &= d.close > ind.sma(d.close, filt)
    return Rules(entry, d.close < lo)


def line(sym, d, label, rules, cost, bench, hold):
    r, t = simulate(d, rules, 240, cost)
    rd = daily(r)
    s = stats(rd[rd.index >= FROM], bench, 365, t, d.index, 240)
    print(f"{sym:8} {label:22} cost {cost:4.0f}bp  cagr {s['cagr']:+.1%} sharpe {s['sharpe']:.2f} dd {s['max_dd']:.0%} "
          f"alpha {s['alpha_ann']:+.1%} t {s['alpha_t']:.1f} trades {s['trades']:.0f} hold {s.get('avg_hold_days', 0):.1f}d "
          f"in-market {s.get('time_in_market', 0):.0%} 19-22 {s['cagr_2019_22']:+.1%} 23-26 {s['cagr_2023_26']:+.1%} | "
          f"hold {hold['cagr']:+.1%} sh {hold['sharpe']:.2f} dd {hold['max_dd']:.0%}", flush=True)


def main(cache):
    for sym in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "LINKUSDT", "SOLUSDT"):
        start = "2020-08" if sym == "SOLUSDT" else "2018-01"
        try:
            d = data.binance(cache, sym, "4h", start)
        except Exception as e:  # noqa: BLE001 - a missing coin must not stop the others
            print(sym, "no data", e)
            continue
        dd = d.close.resample("1D").last().pct_change()
        bench = dd[dd.index >= FROM].dropna()
        hold = stats(bench, bench, 365)
        line(sym, d, "jesse default 20/200", donchian(d), 10, bench, hold)
        if sym in ("BTCUSDT", "ETHUSDT"):
            line(sym, d, "jesse default 20/200", donchian(d), 25, bench, hold)
            for n in (10, 40, 55):
                line(sym, d, f"channel {n}/200", donchian(d, n), 10, bench, hold)
            for f in (100, 400, 0):
                line(sym, d, f"channel 20/{f or 'none'}", donchian(d, 20, f), 10, bench, hold)


if __name__ == "__main__":
    main(sys.argv[1])
