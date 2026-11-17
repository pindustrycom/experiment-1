"""Hand ports of bots whose own engines cannot run on our bars offline (hummingbot, Superalgos, huseinzol05 agents,
quant-trading). Each returns a list of trades (i_entry, i_exit, side, px_in, px_out) for evalkit.

Sources (genuine repos, read 2026-11-17):
  hummingbot/hummingbot            controllers/directional_trading/{bollinger_v1,macd_bb_v1,supertrend_v1}.py
                                   + strategy_v2 directional_trading_controller_base.py (stop 3%, take profit 2%,
                                     time limit 45 min, cooldown 5 min, 3m candles)
  Superalgos/Algorithmic-Trading-Plugins  Trading-Systems/{Sparta-WHB-BTC-USDT, BB-Top-Bounce-BTC-USDT}.json
  huseinzol05/Stock-Prediction-Models     agent/{1.turtle,2.moving-average,3.signal-rolling,6.evolution-strategy}-agent.ipynb
  je-suis-tm/quant-trading         Awesome Oscillator, Parabolic SAR, Heikin-Ashi, RSI pattern, Bollinger W pattern
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------- helpers ----------------
def sma(x, n):
    return x.rolling(n).mean()


def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def rma(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def atr(d, n):
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def bbp(close, n=100, k=2.0):
    m = close.rolling(n).mean()
    s = close.rolling(n).std(ddof=0)
    lo, up = m - k * s, m + k * s
    return (close - lo) / (up - lo)


def supertrend_dir(d, n=20, mult=4.0):
    """pandas_ta supertrend: returns (line, direction +1/-1)."""
    hl2 = (d.high + d.low) / 2
    a = atr(d, n).to_numpy()
    up = (hl2 + mult * a).to_numpy()
    dn = (hl2 - mult * a).to_numpy()
    c = d.close.to_numpy()
    n_ = len(c)
    dirn = np.ones(n_)
    fu, fl = up.copy(), dn.copy()
    line = np.full(n_, np.nan)
    for i in range(1, n_):
        if c[i] > fu[i - 1]:
            dirn[i] = 1
        elif c[i] < fl[i - 1]:
            dirn[i] = -1
        else:
            dirn[i] = dirn[i - 1]
            if dirn[i] > 0 and fl[i] < fl[i - 1]:
                fl[i] = fl[i - 1]
            if dirn[i] < 0 and fu[i] > fu[i - 1]:
                fu[i] = fu[i - 1]
        line[i] = fl[i] if dirn[i] > 0 else fu[i]
    return pd.Series(line, d.index), pd.Series(dirn, d.index)


def triple_barrier(d, signal, sl=0.03, tp=0.02, time_bars=15, cooldown_bars=1, allow_short=True, flat_eod=None):
    """hummingbot PositionExecutor on bars: a signal at a bar's close opens at the next open; then each bar checks the
    stop, then the take profit on its low/high (stop wins inside one bar), then the time limit at the close.
    One position at a time. flat_eod: boolean array, close at that bar's close (stocks: no overnight)."""
    o, h, lo, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    s = np.asarray(signal, float)
    n = len(c)
    trades = []
    i = 0
    next_ok = 0
    while i < n - 1:
        sig = s[i]
        if sig == 0 or (sig < 0 and not allow_short) or i < next_ok or np.isnan(sig):
            i += 1
            continue
        if flat_eod is not None and flat_eod[i]:
            i += 1
            continue
        side = 1 if sig > 0 else -1
        j0 = i + 1
        p0 = o[j0]
        stop = p0 * (1 - side * sl) if sl else None
        targ = p0 * (1 + side * tp) if tp else None
        exit_i, exit_px = None, None
        for j in range(j0, n):
            if stop is not None and ((side > 0 and lo[j] <= stop) or (side < 0 and h[j] >= stop)):
                exit_px = (min(o[j], stop) if side > 0 else max(o[j], stop)) if j > j0 else stop
                exit_i = j
                break
            if targ is not None and ((side > 0 and h[j] >= targ) or (side < 0 and lo[j] <= targ)):
                exit_px = (max(o[j], targ) if side > 0 else min(o[j], targ)) if j > j0 else targ
                exit_i = j
                break
            if time_bars and j - j0 + 1 >= time_bars:
                exit_i, exit_px = j, c[j]
                break
            if flat_eod is not None and flat_eod[j]:
                exit_i, exit_px = j, c[j]
                break
        if exit_i is None:
            exit_i, exit_px = n - 1, c[-1]
        trades.append((j0, exit_i, side, p0, exit_px))
        i = exit_i
        next_ok = exit_i + cooldown_bars
    return trades


def signal_trades(d, pos):
    """A target position series decided at each close (1 long, -1 short, 0 flat) -> trades filled at the next open."""
    p = np.nan_to_num(np.asarray(pos, float))
    o, c = d.open.to_numpy(float), d.close.to_numpy(float)
    trades = []
    cur, i0 = 0, None
    for i in range(len(p) - 1):
        want = p[i]
        if want != cur:
            if cur != 0:
                trades.append((i0, i + 1, cur, o[i0], o[i + 1]))
            cur = want
            i0 = i + 1 if want != 0 else None
    if cur != 0:
        trades.append((i0, len(p) - 1, cur, o[i0], c[-1]))
    return trades


# ---------------- hummingbot directional controllers ----------------
def hb_bollinger_v1(d, length=100, std=2.0, lo=0.0, hi=1.0):
    b = bbp(d.close, length, std)
    return np.where(b < lo, 1, np.where(b > hi, -1, 0))


def hb_macd_bb_v1(d, length=100, std=2.0, lo=0.0, hi=1.0, fast=21, slow=42, sig=9):
    b = bbp(d.close, length, std)
    macd = ema(d.close, fast) - ema(d.close, slow)
    h = macd - ema(macd, sig)
    return np.where((b < lo) & (h > 0) & (macd < 0), 1, np.where((b > hi) & (h < 0) & (macd > 0), -1, 0))


def hb_supertrend_v1(d, length=20, mult=4.0, thr=0.01):
    line, dirn = supertrend_dir(d, length, mult)
    dist = (d.close - line).abs() / d.close
    return np.where((dirn == 1) & (dist < thr), 1, np.where((dirn == -1) & (dist < thr), -1, 0))


# ---------------- Superalgos community trading systems (approximate) ----------------
def _sa_frame(d, rule):
    if rule is None:
        return d
    return d.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def _sa_bb(f, n=20, k=2.0):
    ma = f.close.rolling(n).mean()
    dev = k * f.close.rolling(n).std(ddof=0)          # Superalgos 'deviation' = band half-width
    slope = ma.pct_change().abs()
    direction = np.sign(ma.diff())
    pb = ((f.close - (ma - dev)) / (2 * dev) * 100).rolling(n).mean()  # %B moving average
    return ma, dev, slope, direction, pb


def _to_hourly(series, f, hourly_index):
    # value known at the close of each higher bar, usable from the next hour on
    s = series.copy()
    s.index = s.index + (f.index[1] - f.index[0]) - pd.Timedelta("1h")
    return s.reindex(hourly_index, method="ffill")


def sa_whb(d):
    """Weak Hands Buster (BTC/USDT, short-side system on spot: sells BTC into breakdowns, buys back lower).
    Price thresholds written in dollars for 2020 BTC (~$10k) are converted to % of price: deviation 2-10%,
    candle range < 7.5%, 8h deviation < 15.75%. Slope 'Extreme' approximated as MA change > 0.4%/bar."""
    ma, dev, slope, dirn, _ = _sa_bb(d)
    f2 = _sa_frame(d, "2h")
    _, _, _, dir2, _ = _sa_bb(f2)
    dir2h = _to_hourly(dir2, f2, d.index)
    pbs = []
    for rule in ("2h", "3h", "4h", "6h", "8h"):
        fr = _sa_frame(d, rule)
        pbs.append(_to_hourly(_sa_bb(fr)[4], fr, d.index))
    not_oversold = np.logical_or.reduce([p >= 30 for p in pbs])
    f8 = _sa_frame(d, "8h")
    _, dev8, _, _, _ = _sa_bb(f8)
    dev8h = _to_hourly(dev8 / f8.close, f8, d.index)
    c = d.close
    three_below = (c < ma) & (c.shift() < ma.shift()) & (c.shift(2) < ma.shift(2))
    sig = (three_below & (c < ma - dev) & (dir2h < 0) & (dev / c > 0.02) & ((d.high - d.low) / c < 0.075)
           & (dev / c < 0.10) & not_oversold & (slope < 0.004) & (dev8h < 0.1575))
    o, h, lo, cc = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    mav = ma.to_numpy()
    trades, i, n = [], 0, len(cc)
    sg = sig.to_numpy()
    while i < n - 1:
        if not sg[i]:
            i += 1
            continue
        j0 = i + 1
        p0 = o[j0]
        stop = p0 * 1.03
        phase = 0
        ex = None
        for j in range(j0, n):
            tp = mav[j - 1] * 0.6 if not np.isnan(mav[j - 1]) else 0
            if h[j] >= stop:
                ex = (j, max(o[j], stop) if j > j0 else stop)
                break
            if lo[j] <= tp:
                ex = (j, tp)
                break
            if phase == 0 and cc[j] < p0 * 0.95:
                phase = 1
            if phase == 1 and not np.isnan(mav[j]):
                stop = min(stop, mav[j] * 1.0525)
        if ex is None:
            ex = (n - 1, cc[-1])
        trades.append((j0, ex[0], -1, p0, ex[1]))
        i = ex[0] + 1
    return trades


def sa_bbtb(d):
    """BB Top Bounce (BTC/USDT, short-side): bounce off the top band on 1h inside a 3h down channel with
    medium/steep/extreme slope and a gentle 24h slope; 8h deviation between 2% and 15% of price."""
    ma, dev, _, _, _ = _sa_bb(d)
    f3, f24, f8 = _sa_frame(d, "3h"), _sa_frame(d, "24h"), _sa_frame(d, "8h")
    _, _, sl3, dir3, _ = _sa_bb(f3)
    _, _, sl24, _, _ = _sa_bb(f24)
    _, dev8, _, _, _ = _sa_bb(f8)
    dir3h, sl3h = _to_hourly(dir3, f3, d.index), _to_hourly(sl3, f3, d.index)
    sl24h = _to_hourly(sl24, f24, d.index)
    dev8h = _to_hourly(dev8 / f8.close, f8, d.index)
    c = d.close
    top = ma + dev
    sig = ((c.shift() > top.shift()) & (c < top) & (c < d.open) & (dir3h < 0) & (sl3h > 0.001)
           & (sl24h > 0.0002) & (sl24h < 0.001) & (dev8h > 0.02) & (dev8h < 0.15))
    o, h, lo, cc = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    mav = ma.to_numpy()
    e7 = ema(c, 7).to_numpy()
    trades, i, n = [], 0, len(cc)
    sg = sig.to_numpy()
    while i < n - 1:
        if not sg[i]:
            i += 1
            continue
        j0 = i + 1
        p0 = o[j0]
        stop = p0 * 1.03
        phase = 0
        ex = None
        for j in range(j0, n):
            tp = p0 * 0.95
            if h[j] >= stop:
                ex = (j, max(o[j], stop) if j > j0 else stop)
                break
            if lo[j] <= tp:
                ex = (j, min(o[j], tp) if j > j0 else tp)
                break
            if phase == 0 and cc[j] < p0 * 0.97:
                phase = 1
            if phase == 1 and not np.isnan(mav[j]):
                stop = min(stop, mav[j] * 1.02)
            if e7[j] > p0 and j - j0 > 2 and phase == 1:   # managed exit back above entry
                ex = (j, cc[j])
                break
        if ex is None:
            ex = (n - 1, cc[-1])
        trades.append((j0, ex[0], -1, p0, ex[1]))
        i = ex[0] + 1
    return trades


# ---------------- huseinzol05 agents ----------------
def hz_turtle(d, window):
    """turtle-agent: buy when the close breaks BELOW the prior rolling min, sell when it breaks ABOVE the prior max
    (that is what the notebook's signal code does). Window = 10% of the series in the notebook; we use a fixed
    window in bars equal to 10% of one year."""
    c = d.close
    rmax, rmin = c.shift(1).rolling(window).max(), c.shift(1).rolling(window).min()
    sig = np.where(rmin > c, 1, np.where(rmax < c, -1, 0))
    pos = pd.Series(sig, d.index).replace(0, np.nan).ffill().fillna(0).clip(lower=0)
    return pos


def hz_moving_average(d, short, long):
    c = d.close
    s = c.rolling(short, min_periods=1).mean()
    lg = c.rolling(long, min_periods=1).mean()
    return pd.Series(np.where(s > lg, 1, 0), d.index)


def hz_signal_rolling(d, delay=5):
    """signal-rolling-agent: after `delay` contrary ticks, buy on a down tick when flat, sell on an up tick when long."""
    c = d.close.to_numpy(float)
    state, cnt = 1, 0
    pos = np.zeros(len(c))
    pos[0] = 1
    for i in range(1, len(c)):
        if c[i] < c[i - 1] and state == 0:
            if cnt < delay:
                cnt += 1
            else:
                state, cnt = 1, 0
        if c[i] > c[i - 1] and state == 1:
            if cnt < delay:
                cnt += 1
            else:
                state, cnt = 0, 0
        pos[i] = state
    return pd.Series(pos, d.index)


class _ES:
    """Deep evolution strategy from 6.evolution-strategy-agent.ipynb (population 15, sigma 0.1, lr 0.03,
    one hidden layer of 500, window 30, actions hold/buy/sell). Inputs are returns, not raw price diffs, so one
    model can see BTC and SPY on the same scale."""

    def __init__(self, window=30, layer=500, seed=0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.w = [rng.standard_normal((window, layer)), rng.standard_normal((layer, 3)), rng.standard_normal((1, layer))]
        self.window = window

    @staticmethod
    def policy(w, X):
        return np.argmax((X @ w[0] + w[2]) @ w[1], axis=1)

    def positions(self, w, X):
        act = self.policy(w, X)
        pos = np.zeros(len(act))
        cur = 0
        for i, a in enumerate(act):
            if a == 1:
                cur = 1
            elif a == 2:
                cur = 0
            pos[i] = cur
        return pos

    def reward(self, w, X, r_next, fee):
        p = self.positions(w, X)
        turn = np.abs(np.diff(np.concatenate([[0], p])))
        return float(np.sum(p * r_next - turn * fee)) * 100

    def fit(self, X, r_next, fee, iters=500, pop=15, sigma=0.1, lr=0.03):
        for _ in range(iters):
            noise = [[self.rng.standard_normal(x.shape) for x in self.w] for _ in range(pop)]
            R = np.array([self.reward([w + sigma * e for w, e in zip(self.w, ns)], X, r_next, fee) for ns in noise])
            R = (R - R.mean()) / (R.std() + 1e-7)
            for k in range(3):
                A = np.array([ns[k] for ns in noise])
                self.w[k] = self.w[k] + lr / (pop * sigma) * np.tensordot(R, A, axes=1)


def hz_es_agent(d, train_end, fee, seed=0, window=30, iters=300):
    """Train on bars before train_end, return the position series over all bars (only out-of-sample is scored)."""
    r = d.close.pct_change().fillna(0.0).to_numpy()
    X = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.zeros(window - 1), r]), window)  # returns up to t
    X = X * 100                                               # returns in %
    r_next = np.concatenate([r[1:], [0.0]])                   # decision at close t earns bar t+1
    tr = d.index < train_end
    es = _ES(window, 500, seed)
    es.fit(X[tr], r_next[tr], fee, iters=iters)
    return pd.Series(es.positions(es.w, X), d.index)          # decided at close t, filled next open


# ---------------- je-suis-tm/quant-trading (not covered in PR #22) ----------------
def qt_awesome(d):
    mp = (d.high + d.low) / 2
    ao = sma(mp, 5) - sma(mp, 34)
    # long when AO crosses above zero or saucer; flat when it crosses below (long-only version)
    return pd.Series(np.where(ao > 0, 1, 0), d.index)


def qt_parabolic_sar(d, step=0.02, mx=0.2):
    h, lo = d.high.to_numpy(), d.low.to_numpy()
    n = len(h)
    sar = np.zeros(n)
    up = True
    af, ep = step, h[0]
    sar[0] = lo[0]
    trend = np.ones(n)
    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if up:
            sar[i] = min(sar[i], lo[i - 1], lo[i - 2] if i > 1 else lo[i - 1])
            if lo[i] < sar[i]:
                up, sar[i], ep, af = False, ep, lo[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, mx)
        else:
            sar[i] = max(sar[i], h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar[i]:
                up, sar[i], ep, af = True, ep, h[i], step
            elif lo[i] < ep:
                ep, af = lo[i], min(af + step, mx)
        trend[i] = 1 if up else 0
    return pd.Series(trend, d.index)


def qt_heikin_ashi(d):
    ha_c = (d.open + d.high + d.low + d.close) / 4
    ha_o = ha_c.copy()
    v = ha_o.to_numpy().copy()
    o, c = d.open.to_numpy(), ha_c.to_numpy()
    v[0] = (o[0] + d.close.iloc[0]) / 2
    for i in range(1, len(v)):
        v[i] = (v[i - 1] + c[i - 1]) / 2
    ha_o = pd.Series(v, d.index)
    ha_h = pd.concat([d.high, ha_o, ha_c], axis=1).max(axis=1)
    ha_l = pd.concat([d.low, ha_o, ha_c], axis=1).min(axis=1)
    body = (ha_c - ha_o).abs()
    # repo rules: long after a large bullish HA candle with no lower shadow following a bearish one; exit on 2
    # consecutive bearish candles with no upper shadow
    long_ = (ha_c > ha_o) & (ha_o.shift() > ha_c.shift()) & (ha_l == ha_o) & (body > body.shift())
    exit_ = (ha_o > ha_c) & (ha_h == ha_o) & (ha_o.shift() > ha_c.shift()) & (ha_h.shift() == ha_o.shift())
    s = pd.Series(np.where(long_, 1.0, np.where(exit_, 0.0, np.nan)), d.index).ffill().fillna(0)
    return s


def qt_rsi_pattern(d, n=14):
    delta = d.close.diff()
    up, dn = rma(delta.clip(lower=0), n), rma(-delta.clip(upper=0), n)
    r = 100 - 100 / (1 + up / dn)
    s = pd.Series(np.where(r < 30, 1.0, np.where(r > 70, 0.0, np.nan)), d.index).ffill().fillna(0)
    return s


def qt_bollinger_w(d, n=20, k=2.0):
    """Simplified 'W bottom': two touches of the lower band within 75 bars with the second low above the first,
    then a close above the middle band -> long until the close crosses the upper band."""
    m = d.close.rolling(n).mean()
    s = d.close.rolling(n).std()
    lo, up = m - k * s, m + k * s
    touch = d.low <= lo
    c = d.close.to_numpy()
    lows = d.low.to_numpy()
    tch = touch.to_numpy()
    mv, uv = m.to_numpy(), up.to_numpy()
    pos = np.zeros(len(c))
    last_touch_low, last_touch_i, second = None, -10**9, False
    cur = 0
    for i in range(len(c)):
        if cur == 0:
            if tch[i]:
                if last_touch_low is not None and i - last_touch_i <= 75 and lows[i] > last_touch_low and i - last_touch_i > 5:
                    second = True
                last_touch_low, last_touch_i = lows[i], i
            if second and c[i] > mv[i]:
                cur, second = 1, False
        elif c[i] > uv[i]:
            cur = 0
        pos[i] = cur
    return pd.Series(pos, d.index)
