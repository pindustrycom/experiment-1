"""The strategies, ported rule for rule from public GitHub repos (default or hyperopted parameters as committed).

freqtrade: https://github.com/freqtrade/freqtrade-strategies (user_data/strategies, read 2026-10-21)
Jesse:     https://github.com/jesse-ai/example-strategies (read 2026-10-21)
quant-trading: https://github.com/je-suis-tm/quant-trading (Dual Thrust, read 2026-10-21)
Classic allocation rules found in many repos (QuantConnect Lean examples, backtrader/vectorbt notebooks): golden cross,
Faber's 10-month moving average, Antonacci's global equities momentum (GEM). Those are in allocation.py.

Each builder takes OHLCV bars and returns engine.Rules. Nothing from those repos is imported or run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ind
from engine import Rules

FT = "https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/"
JS = "https://github.com/jesse-ai/example-strategies/tree/master/"


# ---- freqtrade (crypto bots) -------------------------------------------------------------------------------------------

def ft_strategy001(d):
    e20, e50, e100 = ind.ema(d.close, 20), ind.ema(d.close, 50), ind.ema(d.close, 100)
    ha_o, ha_c = ind.heikinashi(d)
    entry = ind.crossed_above(e20, e50) & (ha_c > e20) & (ha_o < ha_c)
    exit_ = ind.crossed_above(e50, e100) & (ha_c < e20) & (ha_o > ha_c)
    return Rules(entry, exit_, roi={60: 0.01, 30: 0.03, 20: 0.04, 0: 0.05}, stoploss=-0.10, exit_profit_only=True)


def ft_strategy005(d):
    m, _ = ind.macd(d.close)
    r = ind.rsi(d.close, 14)
    fr = np.tanh(0.1 * (r - 50))                       # (e^2x - 1)/(e^2x + 1)
    frn = 50 * (fr + 1)
    k, dd = ind.stochf(d)
    sma40 = ind.sma(d.close, 40)
    entry = ((d.close > 0.000002) & (d.volume > d.volume.rolling(150).mean() * 4) & (d.close < sma40) & (dd > k)
             & (r > 26) & (dd > 1) & (frn < 5))
    exit_ = ind.crossed_above(r, 74) & (m < 0) & (ind.minus_di(d) > 4)
    return Rules(entry, exit_, roi={1440: 0.01, 80: 0.02, 40: 0.03, 20: 0.04, 0: 0.05}, stoploss=-0.10,
                 exit_profit_only=True)


def _binh(d, bbdelta, closedelta, tail):
    m = d.close.rolling(40).mean()
    lower = m - 2 * d.close.rolling(40).std()
    bbd = (m - lower).abs()
    cd = (d.close - d.close.shift()).abs()
    tl = (d.close - d.low).abs()
    return (lower.shift().gt(0) & bbd.gt(d.close * bbdelta) & cd.gt(d.close * closedelta) & tl.lt(bbd * tail)
            & d.close.lt(lower.shift()) & d.close.le(d.close.shift()))


def _cluc(d):
    lo, mid, _ = ind.bbands(ind.typical(d), 20, 2)
    entry = (d.close < ind.ema(d.close, 50)) & (d.close < 0.985 * lo) & (d.volume < d.volume.rolling(30).mean().shift(1) * 20)
    return entry, d.close > mid


def ft_binhv45(d):
    return Rules(_binh(d, 0.007, 0.017, 0.25), None, roi={0: 0.0125}, stoploss=-0.05)


def ft_clucmay72018(d):
    entry, exit_ = _cluc(d)
    return Rules(entry, exit_, roi={0: 0.01}, stoploss=-0.05)


def ft_combined_binh_cluc(d):
    entry, exit_ = _cluc(d)
    return Rules(entry | _binh(d, 0.008, 0.0175, 0.25), exit_, roi={0: 0.05}, stoploss=-0.05, exit_profit_only=True)


def ft_bbandrsi(d):
    r = ind.rsi(d.close, 14)
    lo, _, _ = ind.bbands(ind.typical(d), 20, 2)
    return Rules((r < 30) & (d.close < lo), r > 70, roi={0: 0.1}, stoploss=-0.25)


def ft_supertrend(d):
    def up(p, m):
        return ind.supertrend_up(d, p, m) == 1

    def dn(p, m):
        return ind.supertrend_up(d, p, m) == 0

    entry = up(8, 4) & up(9, 7) & up(8, 1) & (d.volume > 0)
    exit_ = dn(16, 1) & dn(18, 3) & dn(18, 6) & (d.volume > 0)
    return Rules(entry, exit_, roi={0: 0.087, 372: 0.058, 861: 0.029, 2221: 0}, stoploss=-0.265, trailing=(0.05, 0.144, False))


def ft_hlhb(d):
    hl2 = (d.close + d.open) / 2
    r = ind.rsi(hl2, 10)
    e5, e10 = ind.ema(d.close, 5), ind.ema(d.close, 10)
    a = ind.adx(d)
    entry = ind.crossed_above(r, 50) & ind.crossed_above(e5, e10) & (a > 25) & (d.volume > 0)
    exit_ = ind.crossed_below(r, 50) & ind.crossed_below(e5, e10) & (a > 25) & (d.volume > 0)
    return Rules(entry, exit_, roi={0: 0.6225, 703: 0.2187, 2849: 0.0363, 5520: 0}, stoploss=-0.3211,
                 trailing=(0.0117, 0.0186, True))


def ft_swinghightosky(d):
    entry = (ind.cci(d, 72) < -175) & (ind.rsi(d.close, 36) < 90)
    exit_ = (ind.cci(d, 66) > -106) & (ind.rsi(d.close, 45) > 88)
    return Rules(entry, exit_, roi={0: 0.27058, 33: 0.0853, 64: 0.04093, 244: 0}, stoploss=-0.34338)


# ---- Jesse example strategies ----------------------------------------------------------------------------------------

def js_sma_crossover(d, short=True):
    f, s = ind.sma(d.close, 50), ind.sma(d.close, 200)
    up, dn = f > s, f < s
    return Rules(up, dn, dn if short else None, up if short else None)


def js_donchian(d):
    hi = d.high.rolling(20).max().shift()          # ta.donchian(candles[:-1]): channel of the previous 20 bars
    lo = d.low.rolling(20).min().shift()
    return Rules((d.close > hi) & (d.close > ind.sma(d.close, 200)), d.close < lo)


def js_rsi2(d, short=True):
    f, s, r = ind.sma(d.close, 5), ind.sma(d.close, 200), ind.rsi(d.close, 2)
    el, es = (d.close > s) & (r <= 10), (d.close < s) & (r >= 90)
    return Rules(el, d.close > f, es if short else None, d.close < f if short else None)


def js_ifr2(d):
    r = ind.rsi(d.close, 2)
    sa, sb = ind.ichimoku_cloud(d, 20, 30, 120, 60)
    tm = ind.ht_trendmode(d.close)
    entry = (r < 10) & (d.close > sa) & (d.close > sb) & (tm == 1)
    exit_ = (d.close > d.high.shift(1)) & (d.close > d.high.shift(2))
    return Rules(entry, exit_)


def js_macd_ema(d):
    m, s = ind.macd(d.close, 12, 26, 9)
    e = ind.ema(d.close, 100)
    return Rules((d.close > e) & (m > s), (m < s) & (d.close < e))


def js_simple_bollinger(d):
    hl2 = (d.high + d.low) / 2
    _, mid, up = ind.bbands(hl2, 20, 2)
    sa, sb = ind.ichimoku_cloud(d)
    return Rules((d.close > up) & (d.close > sa) & (d.close > sb), d.close < mid)


def js_turtle(d):
    """Turtle System 1 (Jesse's TurtleRules defaults): 20-bar breakout in, 10-bar breakout out, stop 2N, skip the
    breakout after a winner. Simplified: full stake instead of 1%-risk units, no pyramiding."""
    hi20, lo20 = d.high.rolling(20).max().shift(), d.low.rolling(20).min().shift()
    hi10, lo10 = d.high.rolling(10).max().shift(), d.low.rolling(10).min().shift()
    n = ind.atr(d, 20)
    return Rules(d.high >= hi20, d.low <= lo10, (d.low <= lo20) & ~(d.high >= hi20), d.high >= hi10,
                 stop_dist=2 * n, skip_after_win=True)


# ---- je-suis-tm/quant-trading: Dual Thrust (intraday) -----------------------------------------------------------------

def qt_dual_thrust(d, k=0.5, n=5):
    """Range = max(HH - LC, HC - LL) over the last n days; long above open + k*range, short below open - k*range,
    reverse on the opposite trigger, flat at the close. Signals on 5-minute bars."""
    day = d.index.normalize()
    g = d.groupby(day)
    dd = pd.DataFrame({"o": g.open.first(), "h": g.high.max(), "l": g.low.min(), "c": g.close.last()})
    rng = pd.concat([dd.h.rolling(n).max() - dd.c.rolling(n).min(), dd.c.rolling(n).max() - dd.l.rolling(n).min()],
                    axis=1).max(axis=1).shift()
    up = pd.Series((dd.o + k * rng).reindex(day).to_numpy(), index=d.index)
    dn = pd.Series((dd.o - k * rng).reindex(day).to_numpy(), index=d.index)
    last = pd.Series(day, index=d.index) != pd.Series(day, index=d.index).shift(-1)
    return Rules(d.close > up, d.close < dn, d.close < dn, d.close > up, flat_at=last)


# name -> (builder, source, horizon, markets {market: [timeframes]})
CATALOG = {
    "ft_strategy001": (ft_strategy001, FT + "Strategy001.py", "short", {"crypto": ["5m"]}),
    "ft_strategy005": (ft_strategy005, FT + "Strategy005.py", "short", {"crypto": ["5m"]}),
    "ft_binhv45": (ft_binhv45, FT + "berlinguyinca/BinHV45.py", "short", {"crypto": ["1m"]}),
    "ft_clucmay72018": (ft_clucmay72018, FT + "berlinguyinca/ClucMay72018.py", "short", {"crypto": ["5m"]}),
    "ft_combined_binh_cluc": (ft_combined_binh_cluc, FT + "berlinguyinca/CombinedBinHAndCluc.py", "short", {"crypto": ["5m"]}),
    "ft_bbandrsi": (ft_bbandrsi, FT + "berlinguyinca/BbandRsi.py", "short", {"crypto": ["1h"], "stock": ["1h"]}),
    "ft_supertrend": (ft_supertrend, FT + "Supertrend.py", "short", {"crypto": ["1h"], "stock": ["1h"]}),
    "ft_hlhb": (ft_hlhb, FT + "hlhb.py", "short", {"crypto": ["4h"]}),
    "ft_swinghightosky": (ft_swinghightosky, FT + "SwingHighToSky.py", "short", {"crypto": ["15m"]}),
    "js_sma_crossover_ls": (lambda d: js_sma_crossover(d, True), JS + "SMACrossover", "long", {"crypto": ["4h", "1d"], "stock": ["1d"]}),
    "js_sma_crossover_long": (lambda d: js_sma_crossover(d, False), JS + "SMACrossover", "long", {"crypto": ["1d"], "stock": ["1d"]}),
    "js_donchian": (js_donchian, JS + "Donchian", "long", {"crypto": ["4h", "1d"], "stock": ["1d"]}),
    "js_turtle": (js_turtle, JS + "TurtleRules", "long", {"crypto": ["1d"], "stock": ["1d"]}),
    "js_macd_ema": (js_macd_ema, JS + "MACD_EMA", "short", {"crypto": ["1h"], "stock": ["1h", "1d"]}),
    "js_simple_bollinger": (js_simple_bollinger, JS + "SimpleBollinger", "short", {"crypto": ["1h"], "stock": ["1h", "1d"]}),
    "js_rsi2_ls": (lambda d: js_rsi2(d, True), JS + "RSI2", "short", {"crypto": ["1d"], "stock": ["1d"]}),
    "js_rsi2_long": (lambda d: js_rsi2(d, False), JS + "RSI2", "short", {"crypto": ["1d", "4h"], "stock": ["1d", "1h"]}),
    "js_ifr2": (js_ifr2, JS + "IFR2", "short", {"crypto": ["4h", "1d"], "stock": ["1d"]}),
    "qt_dual_thrust": (qt_dual_thrust, "https://github.com/je-suis-tm/quant-trading/blob/master/Dual%20Thrust%20backtest.py",
                       "short", {"stock": ["5m"]}),
}
