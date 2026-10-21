"""Indicators as the source repos compute them. TA-Lib for what freqtrade and Jesse take from TA-Lib; qtpylib-style
Bollinger bands, Heikin-Ashi and crosses; freqtrade-technical supertrend; Jesse's Donchian and Ichimoku cloud."""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sma(x, n):
    return pd.Series(talib.SMA(_f(x), n), index=getattr(x, "index", None))


def ema(x, n):
    return pd.Series(talib.EMA(_f(x), n), index=getattr(x, "index", None))


def rsi(x, n):
    return pd.Series(talib.RSI(_f(x), n), index=getattr(x, "index", None))


def atr(d, n):
    return pd.Series(talib.ATR(_f(d.high), _f(d.low), _f(d.close), n), index=d.index)


def adx(d, n=14):
    return pd.Series(talib.ADX(_f(d.high), _f(d.low), _f(d.close), n), index=d.index)


def minus_di(d, n=14):
    return pd.Series(talib.MINUS_DI(_f(d.high), _f(d.low), _f(d.close), n), index=d.index)


def cci(d, n):
    return pd.Series(talib.CCI(_f(d.high), _f(d.low), _f(d.close), n), index=d.index)


def macd(x, fast=12, slow=26, sig=9):
    m, s, _ = talib.MACD(_f(x), fast, slow, sig)
    i = getattr(x, "index", None)
    return pd.Series(m, index=i), pd.Series(s, index=i)


def stochf(d):
    k, dd = talib.STOCHF(_f(d.high), _f(d.low), _f(d.close))       # TA-Lib defaults 5, 3, as freqtrade calls it
    return pd.Series(k, index=d.index), pd.Series(dd, index=d.index)


def sar(d):
    return pd.Series(talib.SAR(_f(d.high), _f(d.low)), index=d.index)


def ht_trendmode(x):
    return pd.Series(talib.HT_TRENDMODE(_f(x)), index=getattr(x, "index", None))


def bbands(x, n=20, k=2.0):
    """qtpylib.bollinger_bands: rolling mean +- k population std."""
    m = x.rolling(n).mean()
    s = x.rolling(n).std(ddof=0)
    return m - k * s, m, m + k * s


def typical(d):
    return (d.high + d.low + d.close) / 3


def heikinashi(d):
    ha_close = (d.open + d.high + d.low + d.close) / 4
    ha_open = np.empty(len(d))
    o, c = _f(d.open), _f(ha_close)
    ha_open[0] = (o[0] + _f(d.close)[0]) / 2
    for i in range(1, len(d)):
        ha_open[i] = (ha_open[i - 1] + c[i - 1]) / 2
    return pd.Series(ha_open, index=d.index), ha_close


def crossed_above(a, b):
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > b) & (a.shift() <= b.shift())


def crossed_below(a, b):
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a < b) & (a.shift() >= b.shift())


def supertrend_up(d, period, mult):
    """freqtrade-technical supertrend direction: True = 'up' (close above the line), NaN before warm-up."""
    a = _f(talib.ATR(_f(d.high), _f(d.low), _f(d.close), period))
    hl2 = (_f(d.high) + _f(d.low)) / 2
    c = _f(d.close)
    bu, bl = hl2 + mult * a, hl2 - mult * a
    n = len(c)
    fu, fl, st = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    out = np.full(n, np.nan)
    for i in range(period, n):
        fu[i] = bu[i] if (np.isnan(fu[i - 1]) or bu[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = bl[i] if (np.isnan(fl[i - 1]) or bl[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if np.isnan(st[i - 1]) or st[i - 1] == fu[i - 1]:
            st[i] = fu[i] if c[i] <= fu[i] else fl[i]
        else:
            st[i] = fl[i] if c[i] >= fl[i] else fu[i]
        out[i] = 1.0 if c[i] > st[i] else 0.0
    return pd.Series(out, index=d.index)


def donchian(d, n=20):
    """Highest high and lowest low of the last n bars, including the current one."""
    return d.high.rolling(n).max(), d.low.rolling(n).min()


def ichimoku_cloud(d, conv=9, base=26, lag=52, disp=26):
    """Jesse's ichimoku_cloud: the cloud (span A, span B) as plotted under the current candle, i.e. computed disp bars ago."""
    def mid(n):
        return (d.high.rolling(n).max() + d.low.rolling(n).min()) / 2

    span_a = ((mid(conv) + mid(base)) / 2).shift(disp)
    span_b = mid(lag).shift(disp)
    return span_a, span_b
