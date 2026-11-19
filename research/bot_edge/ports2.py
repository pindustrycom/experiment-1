"""Round 2 hand ports: more popular GitHub bots, default settings as shipped. Each function returns either a target
position series decided at each bar's close (filled at the next open by ports.signal_trades) or a trade list.

Sources (genuine, non-fork repos, read 2026-11-18; commit in brackets):
  askmike/gekko (10k, archived) [ff5698f]        strategies/{MACD,PPO,RSI,StochRSI,CCI,DEMA,TMA,TSI,UO}.js
                                                 + config/strategies/*.toml defaults, candleSize 60 (sample-config.js)
  DeviaVir/zenbot (8k, archived) [52872fb]       extensions/strategies/{trend_ema,macd,bollinger,sar,trust_distrust}
  vnpy/vnpy (46k) + vnpy/vnpy_ctastrategy [c6e231c]  strategies/{double_ma,atr_rsi,boll_channel,king_keltner,turtle_signal}
  whittlem/pycryptobot (2k) [1fa9aae]            models/Strategy.py default signals (EMA12/26 cross + MACD + golden cross)
  Haehnchen/crypto-trading-bot (3.5k) [cb95633]  src/strategy/strategies/{cci,macd,awesome_oscillator_cross_zero,parabolicsar}.ts
  ccxt/binance-trade-bot (8.8k) [55f8cbb]        auto_trader.py ratio jumping, scout_multiplier 5
  joshyattridge/smart-money-concepts (2.2k)      the library itself (PyPI smartmoneyconcepts 0.0.27): BOS / CHoCH
Spot bots are long/flat ("short" advice = sell to cash). Bots that short (vnpy CTA, crypto-trading-bot, SMC) are
long/short and charged perpetual fees on crypto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ports import atr, ema, rma, sma


def rsi(c, n):
    d = c.diff()
    up, dn = rma(d.clip(lower=0), n), rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _arr(x):
    return np.asarray(x, float)


# ---------------- gekko ----------------
def gekko_persist(long_cond, short_cond, persistence=1):
    """gekko's trend/persistence state machine: advice once a regime has lasted `persistence` candles.
    long advice -> hold (1), short advice -> cash (0)."""
    lc, sc = np.asarray(long_cond, bool), np.asarray(short_cond, bool)
    pos = np.zeros(len(lc))
    cur, direction, dur, adviced = 0, None, 0, False
    p = max(persistence, 1)
    for i in range(len(lc)):
        if lc[i] or sc[i]:
            d = "L" if lc[i] else "S"
            if d != direction:
                direction, dur, adviced = d, 0, False
            dur += 1
            if dur >= p and not adviced:
                adviced = True
                cur = 1 if d == "L" else 0
        pos[i] = cur
    return pos


def gk_macd(d, short=10, long=21, signal=9, up=0.025, down=-0.025, persistence=1):
    diff = ema(d.close, short) - ema(d.close, long)
    res = _arr(diff - ema(diff, signal))
    return gekko_persist(res > up, res < down, persistence)


def gk_ppo(d, short=12, long=26, signal=9, up=0.025, down=-0.025, persistence=2):
    L = ema(d.close, long)
    ppo = 100 * (ema(d.close, short) - L) / L
    h = _arr(ppo - ema(ppo, signal))
    return gekko_persist(h > up, h < down, persistence)


def gk_rsi(d, n=14, low=30, high=70, persistence=1):
    r = _arr(rsi(d.close, n))
    return gekko_persist(r < low, r > high, persistence)


def gk_stochrsi(d, n=3, low=20, high=80, persistence=3):
    r = rsi(d.close, n)
    lo, hi = r.rolling(n).min(), r.rolling(n).max()
    s = _arr(100 * (r - lo) / (hi - lo))
    return gekko_persist(s < low, s > high, persistence)


def gk_cci(d, history=90, constant=0.015, up=100, down=-100):
    tp = (d.high + d.low + d.close) / 3
    avg = tp.rolling(history).mean()
    md = tp.rolling(history).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    c = _arr((tp - avg) / (constant * md))
    return gekko_persist(c <= down, c >= up, 0)


def gk_dema(d, weight=21, up=0.025, down=-0.025):
    e1 = ema(d.close, weight)
    dema = 2 * e1 - ema(e1, weight)
    diff = _arr(sma(d.close, weight) - dema)      # as written in DEMA.js: SMA above DEMA -> long
    return gekko_persist(diff > up, diff < down, 0)


def gk_tma(d, short=7, medium=25, long=99):
    s, m, l_ = (_arr(sma(d.close, n)) for n in (short, medium, long))
    lc = (s > m) & (m > l_)
    sc = ((s < m) & (m > l_)) | ((s > m) & (m < l_))
    pos = np.zeros(len(s))
    cur = 0
    for i in range(len(s)):
        if lc[i]:
            cur = 1
        elif sc[i]:
            cur = 0
        pos[i] = cur
    return pos


def gk_tsi(d, short=13, long=25, low=-25, high=25, persistence=1):
    m = d.close.diff()
    t = _arr(100 * ema(ema(m, long), short) / ema(ema(m.abs(), long), short))
    return gekko_persist(t < low, t > high, persistence)


def gk_uo(d, low=30, high=70, persistence=1):
    pc = d.close.shift()
    bp = d.close - pd.concat([d.low, pc], axis=1).min(axis=1)
    tr = pd.concat([d.high, pc], axis=1).max(axis=1) - pd.concat([d.low, pc], axis=1).min(axis=1)
    a = [bp.rolling(n).sum() / tr.rolling(n).sum() for n in (7, 14, 28)]
    uo = _arr(100 * (4 * a[0] + 2 * a[1] + a[2]) / 7)
    return gekko_persist(uo < low, uo > high, persistence)


# ---------------- zenbot ----------------
def zb_trend_ema(d, trend_ema=26, oversold_rsi_periods=14, oversold_rsi=10, min_periods=52):
    e = ema(d.close, trend_ema)
    rate = e.pct_change() * 100
    sd = _arr(rate.rolling(trend_ema // 2).std(ddof=0))
    rate = _arr(rate)
    rs = _arr(rsi(d.close, oversold_rsi_periods))
    pos = np.zeros(len(rate))
    cur, trend, acted, oversold, cancel_down = 0, None, False, False, False
    for i in range(min_periods, len(rate)):
        if rs[i] <= oversold_rsi and not oversold and not cancel_down:
            oversold = True
        if oversold:
            oversold, trend, cancel_down = False, "oversold", True
            cur = 1
        elif not np.isnan(sd[i]):
            if rate[i] > sd[i]:
                if trend != "up":
                    acted = False
                trend, cancel_down = "up", False
                if not acted:
                    cur, acted = 1, True
            elif not cancel_down and rate[i] < -sd[i]:
                if trend != "down":
                    acted = False
                trend = "down"
                if not acted:
                    cur, acted = 0, True
        pos[i] = cur
    return pos


def zb_macd(d, short=12, long=26, signal=9, ob_periods=25, ob_rsi=70):
    m = ema(d.close, short) - ema(d.close, long)
    h = _arr(m - ema(m, signal))
    r = _arr(rsi(d.close, ob_periods))
    pos = np.zeros(len(h))
    cur = 0
    for i in range(52, len(h)):
        if r[i] >= ob_rsi:
            cur = 0
        elif h[i] > 0 and h[i - 1] <= 0:
            cur = 1
        elif h[i] < 0 and h[i - 1] >= 0:
            cur = 0
        pos[i] = cur
    return pos


def zb_bollinger(d, n=20, k=2.0):
    m, s = d.close.rolling(n).mean(), d.close.rolling(n).std(ddof=0)
    up, lo, c = _arr(m + k * s), _arr(m - k * s), _arr(d.close)
    pos = np.zeros(len(c))
    cur = 0
    for i in range(len(c)):
        if c[i] > up[i]:
            cur = 0
        elif c[i] < lo[i]:
            cur = 1
        pos[i] = cur
    return pos


def zb_sar(d, af0=0.015, af_max=0.3, min_periods=52):
    h, lo, c = _arr(d.high), _arr(d.low), _arr(d.close)
    n = len(c)
    pos = np.zeros(n)
    trend, sar, ep, af, cur = None, None, None, af0, 0
    for i in range(min_periods, n):
        if trend is None:       # lookback[0] = previous period, lookback[-1] = oldest kept (min_periods back)
            if h[i] > h[i - min_periods]:
                trend, sar, ep = "up", min(lo[i - 2], lo[i - 1]), max(h[i - min_periods:i + 1])
            else:
                trend, sar, ep = "down", max(h[i - 2], h[i - 1]), min(lo[i - min_periods:i + 1])
            af = af0
            pos[i] = cur
            continue
        if trend == "up":
            sar = min(lo[i - 2], lo[i - 1], sar + af * (ep - sar))
        else:
            sar = max(h[i - 2], h[i - 1], sar - af * (sar - ep))
        if trend == "down":
            if h[i] >= sar and c[i] > c[i - 1]:
                trend, cur, ep, af = "up", 1, lo[i], af0
                sar = min(lo[i - 1], lo[i], sar + af * (ep - sar))
            elif lo[i] < ep:
                ep = lo[i]
                af = af + af0 if af < af_max else af
        else:
            if lo[i] <= sar and c[i] < c[i - 1]:
                trend, cur, ep, af = "down", 0, h[i], af0
                sar = max(h[i - 1], h[i], sar - af * (sar - ep))
            elif h[i] > ep:
                ep = h[i]
                af = af + af0 if af < af_max else af
        pos[i] = cur
    return pos


def zb_trust_distrust(d, sell_threshold=2, sell_min=1, buy_threshold=2):
    h = _arr(d.high)
    n = len(h)
    pos = np.zeros(n)
    start = highest = lowest = h[0]
    last, cur = None, 0
    for i in range(n):
        highest = max(highest, h[i])
        lowest = min(lowest, h[i])
        if i >= 52:
            if last != "sell" and h[i] > start * (1 + sell_min / 100) and h[i] < highest * (1 - sell_threshold / 100):
                last, cur = "sell", 0
                start = highest = lowest = h[i]
            elif last != "buy" and h[i] < start and h[i] > lowest * (1 + buy_threshold / 100):
                last, cur = "buy", 1
                start = highest = lowest = h[i]
        pos[i] = cur
    return pos


# ---------------- vnpy CTA (long/short, stop orders on the next bar) ----------------
def _stop_fill(side, stop, o, h, lo):
    """A buy stop at `stop` fills if the bar trades through it (at the open if it gaps past); sell stop likewise."""
    if side > 0 and h >= stop:
        return max(o, stop)
    if side < 0 and lo <= stop:
        return min(o, stop)
    return None


def vn_double_ma(d, fast=10, slow=20):
    f, s = _arr(sma(d.close, fast)), _arr(sma(d.close, slow))
    pos = np.zeros(len(f))
    cur = 0
    for i in range(1, len(f)):
        if f[i] > s[i] and f[i - 1] < s[i - 1]:
            cur = 1
        elif f[i] < s[i] and f[i - 1] > s[i - 1]:
            cur = -1
        pos[i] = cur
    return pos


def vn_atr_rsi(d, atr_length=22, atr_ma_length=10, rsi_length=5, rsi_entry=16, trailing_percent=0.8):
    o, h, lo, c = (_arr(d[k]) for k in ("open", "high", "low", "close"))
    a = atr(d, atr_length)
    a_ma = _arr(a.rolling(atr_ma_length).mean())
    a = _arr(a)
    r = _arr(rsi(d.close, rsi_length))
    n = len(c)
    trades, pos, i0, px0, hi_, lo_ = [], 0, None, None, 0.0, 0.0
    for i in range(100, n - 1):
        j = i + 1
        if pos == 0:
            if a[i] > a_ma[i] and (r[i] > 50 + rsi_entry or r[i] < 50 - rsi_entry):
                pos = 1 if r[i] > 50 + rsi_entry else -1
                i0, px0 = j, o[j]                    # limit at close +/- 5 ticks: fills at the next open
                hi_, lo_ = h[j], lo[j]
        else:
            if pos > 0:
                hi_ = max(hi_, h[i])
                px = _stop_fill(-1, hi_ * (1 - trailing_percent / 100), o[j], h[j], lo[j])
            else:
                lo_ = min(lo_, lo[i])
                px = _stop_fill(1, lo_ * (1 + trailing_percent / 100), o[j], h[j], lo[j])
            if px is not None:
                trades.append((i0, j, pos, px0, px))
                pos = 0
    if pos:
        trades.append((i0, n - 1, pos, px0, c[-1]))
    return trades


def vn_boll_channel(d, boll_window=18, boll_dev=3.4, cci_window=10, atr_window=30, sl_multiplier=5.2):
    o, h, lo, c = (_arr(d[k]) for k in ("open", "high", "low", "close"))
    m, s = d.close.rolling(boll_window).mean(), d.close.rolling(boll_window).std(ddof=0)
    up, dn = _arr(m + boll_dev * s), _arr(m - boll_dev * s)
    tp = (d.high + d.low + d.close) / 3
    cci = _arr((tp - tp.rolling(cci_window).mean()) /
               (0.015 * tp.rolling(cci_window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)))
    a = _arr(atr(d, atr_window))
    n = len(c)
    trades, pos, i0, px0, hi_, lo_ = [], 0, None, None, 0.0, 0.0
    for i in range(100, n - 1):
        j = i + 1
        if pos == 0:
            side = 1 if cci[i] > 0 else (-1 if cci[i] < 0 else 0)
            if side:
                px = _stop_fill(side, up[i] if side > 0 else dn[i], o[j], h[j], lo[j])
                if px is not None:
                    pos, i0, px0, hi_, lo_ = side, j, px, h[j], lo[j]
        else:
            if pos > 0:
                hi_ = max(hi_, h[i])
                px = _stop_fill(-1, hi_ - a[i] * sl_multiplier, o[j], h[j], lo[j])
            else:
                lo_ = min(lo_, lo[i])
                px = _stop_fill(1, lo_ + a[i] * sl_multiplier, o[j], h[j], lo[j])
            if px is not None and j > i0:
                trades.append((i0, j, pos, px0, px))
                pos = 0
    if pos:
        trades.append((i0, n - 1, pos, px0, c[-1]))
    return trades


def vn_king_keltner(d, kk_length=11, kk_dev=1.6, trailing_percent=0.8):
    o, h, lo, c = (_arr(d[k]) for k in ("open", "high", "low", "close"))
    mid = sma(d.close, kk_length)
    a = atr(d, kk_length)
    up, dn = _arr(mid + kk_dev * a), _arr(mid - kk_dev * a)
    n = len(c)
    trades, pos, i0, px0, hi_, lo_ = [], 0, None, None, 0.0, 0.0
    for i in range(100, n - 1):
        j = i + 1
        if pos == 0:                     # OCO stop orders at both bands
            pl = _stop_fill(1, up[i], o[j], h[j], lo[j])
            ps = _stop_fill(-1, dn[i], o[j], h[j], lo[j])
            if pl is not None and ps is not None:       # both touched in one bar: take the one nearer the open
                pl, ps = (pl, None) if abs(o[j] - up[i]) <= abs(o[j] - dn[i]) else (None, ps)
            if pl is not None or ps is not None:
                pos = 1 if pl is not None else -1
                i0, px0, hi_, lo_ = j, (pl if pl is not None else ps), h[j], lo[j]
        else:
            if pos > 0:
                hi_ = max(hi_, h[i])
                px = _stop_fill(-1, hi_ * (1 - trailing_percent / 100), o[j], h[j], lo[j])
            else:
                lo_ = min(lo_, lo[i])
                px = _stop_fill(1, lo_ * (1 + trailing_percent / 100), o[j], h[j], lo[j])
            if px is not None:
                trades.append((i0, j, pos, px0, px))
                pos = 0
    if pos:
        trades.append((i0, n - 1, pos, px0, c[-1]))
    return trades


def vn_turtle(d, entry_window=20, exit_window=10, atr_window=20):
    """TurtleSignalStrategy with one unit (the 4-unit pyramid is left out: one position, full stake)."""
    o, h, lo, c = (_arr(d[k]) for k in ("open", "high", "low", "close"))
    eu, ed = _arr(d.high.rolling(entry_window).max()), _arr(d.low.rolling(entry_window).min())
    xu, xd = _arr(d.high.rolling(exit_window).max()), _arr(d.low.rolling(exit_window).min())
    a = _arr(atr(d, atr_window))
    n = len(c)
    trades, pos, i0, px0, stop = [], 0, None, None, None
    for i in range(100, n - 1):
        j = i + 1
        if pos == 0:
            pl = _stop_fill(1, eu[i], o[j], h[j], lo[j])
            ps = _stop_fill(-1, ed[i], o[j], h[j], lo[j])
            if pl is not None and ps is not None:
                pl, ps = (pl, None) if abs(o[j] - eu[i]) <= abs(o[j] - ed[i]) else (None, ps)
            if pl is not None or ps is not None:
                pos = 1 if pl is not None else -1
                i0, px0 = j, (pl if pl is not None else ps)
                stop = px0 - 2 * a[i] if pos > 0 else px0 + 2 * a[i]
        else:
            lvl = max(stop, xd[i]) if pos > 0 else min(stop, xu[i])
            px = _stop_fill(-pos, lvl, o[j], h[j], lo[j])
            if px is not None:
                trades.append((i0, j, pos, px0, px))
                pos = 0
    if pos:
        trades.append((i0, n - 1, pos, px0, c[-1]))
    return trades


# ---------------- pycryptobot ----------------
def pcb_default(d):
    c = d.close
    e12, e26 = ema(c, 12), ema(c, 26)
    gt, lt = e12 > e26, e12 < e26
    gtco = _arr(gt & (gt != gt.shift()))
    ltco = _arr(lt & (lt != lt.shift()))
    macd = e12 - e26
    sig = ema(macd, 9)
    mgt, mlt = _arr(macd > sig), _arr(macd < sig)
    golden = _arr(sma(c, 50) > sma(c, 200))
    pos = np.zeros(len(c))
    cur = 0
    for i in range(200, len(c)):
        if cur == 0 and gtco[i] and mgt[i] and golden[i]:
            cur = 1
        elif cur == 1 and ltco[i] and mlt[i]:
            cur = 0
        pos[i] = cur
    return pos


# ---------------- Haehnchen/crypto-trading-bot (long/short) ----------------
def _ctb(isl, go_long, go_short, close_long, close_short):
    pos = np.zeros(len(isl))
    cur = 0
    for i in range(len(isl)):
        if cur == 1 and close_long[i]:
            cur = 0
        elif cur == -1 and close_short[i]:
            cur = 0
        elif cur == 0:          # the bot only opens from flat (goLong while long is a no-op)
            if isl[i] and go_long[i]:
                cur = 1
            elif not isl[i] and go_short[i]:
                cur = -1
        pos[i] = cur
    return pos


def ctb_macd(d):
    m = ema(d.close, 12) - ema(d.close, 26)
    hst = m - ema(m, 9)
    cur, prev = _arr(hst), _arr(hst.shift())
    # HMA(9) = WMA(2*WMA(close, 4) - WMA(close, 9), 3)
    w = lambda x, n: x.rolling(n).apply(lambda v: np.dot(v, np.arange(1, n + 1)) / (n * (n + 1) / 2), raw=True)  # noqa: E731
    hma = w(2 * w(d.close, 4) - w(d.close, 9), 3)
    isl = _arr(hma >= sma(d.close, 200))
    up, dn = (prev < 0) & (cur > 0), (prev > 0) & (cur < 0)
    return _ctb(isl, up, dn, (prev > 0) & (cur < 0), (prev < 0) & (cur > 0))


def ctb_ao(d):
    mid = (d.high + d.low) / 2
    ao = sma(mid, 5) - sma(mid, 34)
    cur, prev = _arr(ao), _arr(ao.shift())
    isl = _arr(d.close >= sma(d.close, 200))
    return _ctb(isl, (prev < 0) & (cur > 0), (prev > 0) & (cur < 0), (prev > 0) & (cur < 0), (prev < 0) & (cur > 0))


def ctb_cci(d, n=14):
    tp = (d.high + d.low + d.close) / 3
    cci = (tp - tp.rolling(n).mean()) / (0.015 * tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True))
    v = _arr(cci)
    isl = _arr((d.close >= sma(d.close, 200)) | (d.close >= ema(d.close, 200)))
    N = len(v)
    gl, gs, cl, cs = (np.zeros(N, bool) for _ in range(4))
    for i in range(4, N):
        last, before = v[i], v[i - 1]
        cl[i] = before > 100 and last < 100
        cs[i] = before < -100 and last > -100
        if before <= -100 and last >= -100:      # min of the run below -100 in the last 4 values must reach -200
            mn = last
            for k in range(i - 1, i - 4, -1):
                if v[k] >= -100:
                    break
                mn = min(mn, v[k])
            gl[i] = mn <= -200
        if before >= 100 and last <= 100:
            mx = last
            for k in range(i - 1, i - 4, -1):
                if v[k] <= 100:
                    break
                mx = max(mx, v[k])
            gs[i] = mx >= 200
    return _ctb(isl, gl, gs, cl, cs)


def psar(d, step=0.02, mx=0.2):
    h, lo = _arr(d.high), _arr(d.low)
    n = len(h)
    out = np.full(n, np.nan)
    up, af, ep, s = True, step, h[0], lo[0]
    for i in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, lo[i - 1], lo[i - 2] if i > 1 else lo[i - 1])
            if lo[i] < s:
                up, s, ep, af = False, ep, lo[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, mx)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                up, s, ep, af = True, ep, h[i], step
            elif lo[i] < ep:
                ep, af = lo[i], min(af + step, mx)
        out[i] = s
    return out


def ctb_psar(d):
    hist = _arr(d.close) - psar(d)
    prev = np.r_[np.nan, hist[:-1]]
    isl = hist > 0
    return _ctb(isl, (prev < 0) & (hist > 0), (prev > 0) & (hist < 0), (prev > 0) & (hist < 0), (prev < 0) & (hist > 0))


# ---------------- smart-money-concepts (library code) ----------------
def smc_structure(d, swing_length=50):
    """Long after a bullish BOS/CHoCH, short after a bearish one, acting only once the break is known: the later of
    the breaking candle and the candle that confirms the last swing (swing_length bars after it)."""
    from smartmoneyconcepts import smc
    ohlc = d[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    sw = smc.swing_highs_lows(ohlc, swing_length=swing_length)
    bc = smc.bos_choch(ohlc, sw, close_break=True)
    n = len(ohlc)
    ev = np.zeros(n)
    for i in range(n):
        dirn = bc.BOS.iat[i] if not np.isnan(bc.BOS.iat[i]) else bc.CHOCH.iat[i]
        if np.isnan(dirn) or np.isnan(bc.BrokenIndex.iat[i]):
            continue
        t = int(max(bc.BrokenIndex.iat[i], i + swing_length))
        if t < n:
            ev[t] = dirn
    pos = np.zeros(n)
    cur = 0
    for i in range(n):
        if ev[i]:
            cur = ev[i]
        pos[i] = cur
    return pos


# ---------------- ccxt/binance-trade-bot (coin rotation) ----------------
def btb_rotation(px: pd.DataFrame, fee=0.001, scout_multiplier=5.0, start_coin=None):
    """px: closes, one column per coin, aligned. Holds one coin; every bar it checks every other coin's ratio
    against the stored ratio and jumps to the best positive one, then resets the stored ratios into the new coin
    (update_trade_threshold). Returns a 0/1 weight frame decided at each close (held from the next bar)."""
    coins = list(px.columns)
    P = px.to_numpy(float)
    n, k = P.shape
    ratio = P[0][:, None] / P[0][None, :]          # ratio[a, b] = price_a / price_b at initialisation
    cur = coins.index(start_coin) if start_coin else 0
    tf = fee + fee - fee * fee
    W = np.zeros((n, k))
    for t in range(n):
        p = P[t]
        r = p[cur] / p
        score = (r - tf * scout_multiplier * r) - ratio[cur]
        score[cur] = -np.inf
        b = int(np.argmax(score))
        if score[b] > 0:
            cur = b
            ratio[:, cur] = p / p[cur]             # pairs whose to_coin is the new coin
        W[t, cur] = 1.0
    return pd.DataFrame(W, px.index, coins)
