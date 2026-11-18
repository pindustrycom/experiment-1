"""Round 3 ports: gekko community pack, grid and DCA bots, pairs, EWMAC trend following, supertrend bot, PatternPy,
Alpaca scalper, conor19w futures bot, fmz classics (R-Breaker, Dual Thrust), pskrunner14 DQN.

Every function takes bars (open/high/low/close/volume, sorted DatetimeIndex) and returns either
  - a target position series decided at each close (filled at the next open by ports.signal_trades), "pos"
  - a trade list (i_entry, i_exit, side, px_in, px_out), "trades"
  - a fractional weight held over each bar (decided at the previous close), "weight"
Defaults are each repo's own; where a repo has none, the assumption is stated in the docstring.
"""
import numpy as np
import pandas as pd

from ports import atr, ema, sma
from ports2 import _arr, _stop_fill, rsi


def _persist(long_c, short_c):
    """gekko-style advice: 'long' / 'short' (= sell on spot) and hold the last advice."""
    s = pd.Series(np.where(long_c, 1.0, np.where(short_c, 0.0, np.nan)))
    return s.ffill().fillna(0.0).to_numpy().copy()


def _wilder_adx(d, n):
    h, lo, c = d.high, d.low, d.close
    up, dn = h.diff(), -lo.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), d.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), d.index)
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    a = 1.0 / n
    atr_ = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr_
    ndi = 100 * ndm.ewm(alpha=a, adjust=False).mean() / atr_
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


# ---------------- gekko community strategies (xFFFFF/Gekko-Strategies) ----------------
def gk_rsi_bull_bear(d, sma_long=1000, sma_short=50, bull_rsi=10, bull_hi=80, bull_lo=60, bear_rsi=15, bear_hi=50,
                     bear_lo=20):
    c = d.close
    bull = (sma(c, sma_short) >= sma(c, sma_long)).to_numpy()
    rb, rr = _arr(rsi(c, bull_rsi)), _arr(rsi(c, bear_rsi))
    lc = np.where(bull, rb < bull_lo, rr < bear_lo)
    sc = np.where(bull, rb > bull_hi, rr > bear_hi)
    p = _persist(lc, sc)
    p[: sma_long] = 0
    return p


def gk_rsi_bull_bear_adx(d, sma_long=1000, sma_short=50, bull_rsi=10, bull_hi=80, bull_lo=60, bear_rsi=15, bear_hi=50,
                         bear_lo=20, bull_mod_hi=5, bull_mod_lo=-5, bear_mod_hi=15, bear_mod_lo=-5, adx_n=3,
                         adx_hi=70, adx_lo=50):
    c = d.close
    bull = (sma(c, sma_short) >= sma(c, sma_long)).to_numpy()
    rb, rr = _arr(rsi(c, bull_rsi)), _arr(rsi(c, bear_rsi))
    ad = _arr(_wilder_adx(d, adx_n))
    hi = np.where(bull, bull_hi + np.where(ad > adx_hi, bull_mod_hi, 0), bear_hi + np.where(ad > adx_hi, bear_mod_hi, 0))
    lo = np.where(bull, bull_lo + np.where((ad <= adx_hi) & (ad < adx_lo), bull_mod_lo, 0),
                  bear_lo + np.where((ad <= adx_hi) & (ad < adx_lo), bear_mod_lo, 0))
    r = np.where(bull, rb, rr)
    p = _persist(r < lo, r > hi)
    p[: sma_long] = 0
    return p


def gk_neo(d, sma_long=150, sma_short=40, bull_rsi=10, bull_hi=80, bull_lo=50, idle_rsi=12, idle_hi=65, idle_lo=39,
           bear_rsi=15, bear_hi=50, bear_lo=25, roc=6, roc_lvl=0):
    c = d.close
    bull = (sma(c, sma_short) >= sma(c, sma_long)).to_numpy()
    rc = _arr(100 * (c / c.shift(roc) - 1))
    rb, ri, rr = _arr(rsi(c, bull_rsi)), _arr(rsi(c, idle_rsi)), _arr(rsi(c, bear_rsi))
    idle = bull & (rc <= roc_lvl)
    r = np.where(~bull, rr, np.where(idle, ri, rb))
    hi = np.where(~bull, bear_hi, np.where(idle, idle_hi, bull_hi))
    lo = np.where(~bull, bear_lo, np.where(idle, idle_lo, bull_lo))
    p = _persist(r < lo, r > hi)
    p[: sma_long] = 0
    return p


def gk_bbrsi(d, rsi_n=14, low=40, high=40, persistence=9, bb_n=20, k=2.0):
    c = d.close
    mid = sma(c, bb_n)
    sd = c.rolling(bb_n).std(ddof=0)
    up, dn = (mid + k * sd).to_numpy(), (mid - k * sd).to_numpy()
    m, x, r = mid.to_numpy(), c.to_numpy(), _arr(rsi(c, rsi_n))
    zone = np.where(x >= up, 3, np.where(x >= m, 2, np.where(x > dn, 1, 0)))
    dur = np.zeros(len(x), int)
    for i in range(1, len(x)):
        dur[i] = dur[i - 1] + 1 if zone[i] == zone[i - 1] else 1
    lc = (x <= dn) & (r <= low) & (dur >= persistence)
    sc = (x >= m) & (r >= high)
    p = _persist(lc, sc)
    p[: bb_n + rsi_n] = 0
    return p


def _tema(x, n):
    e1 = ema(x, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return 3 * e1 - 3 * e2 + e3


def gk_tema(d, short=10, long=80, sma_long=200):
    c = d.close
    fast, slow, slowest = sma(c, short), _tema(c, long), sma(c, sma_long)
    safety = (slow < slowest).to_numpy()
    lc = ~safety & (fast > slow).to_numpy()
    sc = safety | (fast < slow).to_numpy()
    p = _persist(lc, sc)
    p[: 3 * long] = 0
    return p


# ---------------- grid and DCA bots ----------------
def grid_bot(d, width=0.10, levels=80, recenter="MS"):
    """Spot grid (51bitquant binance_grid_trader / opentrader grid): `levels` equal steps over +-width around the
    price at the start of each month; the bot holds one step of coin for every level above the price (buys as it falls,
    sells as it rises), all coin below the grid floor, none above the top. Re-centred monthly (assumed: the repos
    leave the range to the user). Bars only see closes, so swings inside a bar are missed (understates grid gains)."""
    c = d.close
    key = c.index.to_period("M") if recenter == "MS" else c.index.normalize()
    centre = c.groupby(key).transform("first").to_numpy()
    lo, hi = centre * (1 - width), centre * (1 + width)
    step = (hi - lo) / levels
    filled = np.floor((hi - c.to_numpy()) / step)
    w = np.clip(filled, 0, levels) / levels
    return pd.Series(w, d.index).shift().fillna(0.0)


def dca_bot(d, safety_orders=5, step=0.02, take_profit=0.02, step_scale=1.0, volume_scale=1.0):
    """DCA bot (opentrader 'dca' / 3Commas style), defaults assumed (opentrader requires the user to set them):
    base order + 5 safety orders every -2% from the entry, equal sizes; sell everything when the close is 2% above the
    average entry; start again on the next bar. Weight = units held / units the full ladder can hold."""
    c = d.close.to_numpy(float)
    sizes = np.array([volume_scale ** k for k in range(safety_orders + 1)])
    cap = sizes.sum()
    n = len(c)
    w = np.zeros(n)
    units, cost, k, entry = 0.0, 0.0, 0, None
    for i in range(n):
        if entry is None:
            entry, units, cost, k = c[i], sizes[0], sizes[0] * c[i], 1
        else:
            drop = step * sum(step_scale ** j for j in range(k))
            if k <= safety_orders and c[i] <= entry * (1 - drop):
                units += sizes[k]
                cost += sizes[k] * c[i]
                k += 1
            elif c[i] >= cost / units * (1 + take_profit):
                entry, units, cost, k = None, 0.0, 0.0, 0
        w[i] = units / cap
    return pd.Series(w, d.index).shift().fillna(0.0)


# ---------------- pairs / mean reversion (positions on the spread A - B) ----------------
def kq_ratio_z(a, b, fast=5, slow=60, entry=1.0, exit_=0.75):
    """KidQuant Pairs-Trading-With-Python: z = (MA5 - MA60) / std60 of the price ratio A/B. z < -1: long A short B,
    z > 1: short A long B, close when |z| < 0.75."""
    ratio = a / b
    z = ((ratio.rolling(fast).mean() - ratio.rolling(slow).mean()) / ratio.rolling(slow).std()).to_numpy()
    p = np.zeros(len(z))
    cur = 0.0
    for i, v in enumerate(z):
        if np.isnan(v):
            p[i] = 0
            continue
        if cur == 0:
            cur = 1.0 if v < -entry else (-1.0 if v > entry else 0.0)
        elif abs(v) < exit_:
            cur = 0.0
        p[i] = cur
    return p


def chan_kalman(a, b, delta=1e-4, ve=1e-3):
    """Ernie Chan, Algorithmic Trading ex. 3.3 (zazhang/ep-chan-book-algo-trading): Kalman-filter hedge ratio of A on
    B (log prices here, so the spread is a return spread); long the spread when the forecast error e < -sqrt(Q), exit
    when e > -sqrt(Q); short when e > sqrt(Q), exit when e < sqrt(Q). Returns (position, beta)."""
    y, x = np.log(a.to_numpy(float)), np.log(b.to_numpy(float))
    n = len(y)
    beta = np.zeros(2)
    P = np.zeros((2, 2))
    R = None
    vw = delta / (1 - delta) * np.eye(2)
    pos, betas = np.zeros(n), np.zeros(n)
    cur = 0.0
    for t in range(n):
        F = np.array([x[t], 1.0])
        R = P + vw if t > 0 else np.zeros((2, 2))
        yhat = F @ beta
        Q = F @ R @ F + ve
        e = y[t] - yhat
        K = R @ F / Q
        beta = beta + K * e
        P = R - np.outer(K, F) @ R
        betas[t] = beta[0]
        if t < 60:
            continue
        sq = np.sqrt(Q)
        if cur == 0:
            cur = 1.0 if e < -sq else (-1.0 if e > sq else 0.0)
        elif cur > 0 and e > -sq:
            cur = 0.0
        elif cur < 0 and e < sq:
            cur = 0.0
        pos[t] = cur
    return pos, betas


# ---------------- trend following ----------------
EWMAC_SCALARS = {(8, 32): 5.3, (16, 64): 3.75, (32, 128): 2.65, (64, 256): 1.91}


def ewmac(d, pairs=((8, 32), (16, 64), (32, 128), (64, 256)), vol_target=0.125, per_year=365, cap=20.0, fdm=1.1,
          max_lev=2.0):
    """chrism2671/PyTrendFollow (Carver's EWMAC): forecast = (EMA_f - EMA_s) / (25-day EW std of price changes) x
    scalar, capped at +-20, averaged with the forecast diversification multiplier; position = forecast / 10 x
    vol target / instrument vol (36-day EW std of returns), capped at 2x. Daily bars. Weight held over the next bar."""
    c = d.close
    dp_std = c.diff().ewm(span=25, min_periods=25).std()
    fc = []
    for f, s in pairs:
        raw = (c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()) / dp_std
        fc.append((raw * EWMAC_SCALARS.get((f, s), 1.0)).clip(-cap, cap))
    F = (pd.concat(fc, axis=1).mean(axis=1) * fdm).clip(-cap, cap)
    vol = c.pct_change().ewm(span=36, min_periods=36).std() * np.sqrt(per_year)
    w = (F / 10 * vol_target / vol).clip(-max_lev, max_lev)
    return w.shift().fillna(0.0)


def htm_supertrend(d, period=7, mult=3.0):
    """hackingthemarkets/supertrend-crypto-bot: ATR = rolling mean of true range (7), bands hl2 +- 3 ATR, buy when the
    trend flips up, sell when it flips down (long/flat). Native 1m."""
    h, lo, c = (_arr(d[k]) for k in ("high", "low", "close"))
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    a = _arr(tr.rolling(period).mean())
    hl2 = (h + lo) / 2
    ub, lb = hl2 + mult * a, hl2 - mult * a
    up = np.ones(len(c), bool)
    for i in range(1, len(c)):
        if c[i] > ub[i - 1]:
            up[i] = True
        elif c[i] < lb[i - 1]:
            up[i] = False
        else:
            up[i] = up[i - 1]
            if up[i] and lb[i] < lb[i - 1]:
                lb[i] = lb[i - 1]
            if not up[i] and ub[i] > ub[i - 1]:
                ub[i] = ub[i - 1]
    p = up.astype(float)
    p[:period] = 0
    return p


# ---------------- PatternPy (keithorange) ----------------
_TP = None


def _patternpy_module():
    """The library writes strings into a float column (np.nan), which pandas 3 refuses; the one change made to it is
    starting those columns as None (object dtype). Logic untouched."""
    global _TP
    if _TP is None:
        import importlib.util
        import types
        spec = importlib.util.find_spec("tradingpatterns")
        src = open(spec.origin).read().replace("_pattern'] = np.nan", "_pattern'] = None")
        _TP = types.ModuleType("tradingpatterns_p3")
        exec(compile(src, spec.origin, "exec"), _TP.__dict__)
    return _TP


def patternpy(d, which):
    """Runs the library's own detector; it looks one bar ahead (shift(-1)), so the signal is used one bar later.
    Bullish pattern -> long, bearish -> flat, held until the opposite pattern (the library has no trading rules)."""
    tp = _patternpy_module()
    df = pd.DataFrame({"Open": d.open.to_numpy(), "High": d.high.to_numpy(), "Low": d.low.to_numpy(),
                       "Close": d.close.to_numpy()})
    if which == "head_shoulder":
        col, bull, bear = tp.detect_head_shoulder(df)["head_shoulder_pattern"], "Inverse Head and Shoulder", "Head and Shoulder"
    else:
        col, bull, bear = tp.detect_double_top_bottom(df)["double_pattern"], "Double Bottom", "Double Top"
    col = col.shift(1)
    p = _persist((col == bull).to_numpy(), (col == bear).to_numpy())
    return p


# ---------------- Alpaca example-scalping ----------------
def alpaca_scalper(d, n=20, tick=0.01, cancel_bars=2):
    """alpacahq/example-scalping on 1m bars (New York time): buy limit at the close when the close crosses above its
    20-bar SMA (cancelled after 2 minutes unfilled); then sell limit at max(cost + $0.01, price at fill); market
    bail-out at 15:55; no entries after 15:55."""
    o, h, lo, c = (_arr(d[k]) for k in ("open", "high", "low", "close"))
    m = _arr(sma(d.close, n))
    loc = d.index.tz_convert("America/New_York")
    mins = loc.hour * 60 + loc.minute
    late = mins >= 15 * 60 + 55
    day = loc.normalize()
    newday = np.r_[True, day[1:] != day[:-1]]
    trades = []
    i, N = n + 1, len(c)
    while i < N - 1:
        if late[i] or not (c[i - 1] < m[i - 1] and c[i] > m[i]):
            i += 1
            continue
        lim, fill = c[i], None
        for j in range(i + 1, min(i + 1 + cancel_bars, N)):
            if newday[j] or late[j]:
                break
            if lo[j] <= lim:
                fill = j
                break
        if fill is None:
            i += 1
            continue
        px_in = min(lim, o[fill])
        tgt = max(px_in + tick, c[fill])
        k = fill + 1
        out = None
        while k < N:
            if newday[k]:
                out = (k - 1, c[k - 1])
                break
            if late[k]:
                out = (k, o[k])
                break
            if h[k] >= tgt:
                out = (k, max(tgt, o[k]))
                break
            k += 1
        if out is None:
            out = (N - 1, c[-1])
        trades.append((fill, out[0], 1, px_in, out[1]))
        i = out[0] + 1
    return trades


# ---------------- conor19w Binance-Futures-Trading-Bot ----------------
def _tp_sl_trades(d, direction, tp=0.01, sl=0.015):
    """Enter at the next open in `direction` (1 long, -1 short, 0 none); exit at TP / SL inside later bars
    (stop first if a bar touches both); one position at a time."""
    o, h, lo, c = (_arr(d[k]) for k in ("open", "high", "low", "close"))
    N = len(c)
    trades, i = [], 60
    while i < N - 1:
        s = direction[i]
        if s == 0:
            i += 1
            continue
        j = i + 1
        px = o[j]
        tp_px, sl_px = (px * (1 + tp), px * (1 - sl)) if s > 0 else (px * (1 - tp), px * (1 + sl))
        k = j
        out = None
        while k < N:
            if s > 0:
                if lo[k] <= sl_px:
                    out = (k, min(sl_px, o[k]) if k > j else sl_px)
                elif h[k] >= tp_px:
                    out = (k, max(tp_px, o[k]) if k > j else tp_px)
            else:
                if h[k] >= sl_px:
                    out = (k, max(sl_px, o[k]) if k > j else sl_px)
                elif lo[k] <= tp_px:
                    out = (k, min(tp_px, o[k]) if k > j else tp_px)
            if out:
                break
            k += 1
        if out is None:
            out = (N - 1, c[-1])
        trades.append((j, out[0], s, px, out[1]))
        i = out[0]
    return trades


def _ta_stochrsi(c):
    from ta.momentum import StochRSIIndicator
    s = StochRSIIndicator(c)
    return _arr(s.stochrsi_k()), _arr(s.stochrsi_d())


def cw_triple_ema_stochrsi(d):
    """tripleEMAStochasticRSIATR (the repo's default): close > EMA8 > EMA14 > EMA50 and StochRSI %K crosses above %D
    -> long; mirror -> short. TP 1%, SL 1.5% (TP_SL_choice '%', TP_mult 1, SL_mult 1.5)."""
    c = d.close
    e50, e14, e8, x = _arr(ema(c, 50)), _arr(ema(c, 14)), _arr(ema(c, 8)), _arr(c)
    k, dd = _ta_stochrsi(c)
    kp, dp = np.r_[np.nan, k[:-1]], np.r_[np.nan, dd[:-1]]
    lg = (x > e8) & (e8 > e14) & (e14 > e50) & (k > dd) & (kp < dp)
    sh = (x < e8) & (e8 < e14) & (e14 < e50) & (k < dd) & (kp > dp)
    return _tp_sl_trades(d, np.where(lg, 1, np.where(sh, -1, 0)))


def cw_stochrsi_macd(d):
    """StochRSIMACD: the repo feeds ta's stoch (%K) as 'fastd' and stoch_signal (%D) as 'fastk'; RSI 14; MACD 12/26/9."""
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD
    st = StochasticOscillator(d.high, d.low, d.close)
    fd, fk = _arr(st.stoch()), _arr(st.stoch_signal())
    r = _arr(RSIIndicator(d.close).rsi())
    mc = MACD(d.close)
    M, S = _arr(mc.macd()), _arr(mc.macd_signal())
    n = len(r)
    sig = np.zeros(n, int)
    for i in range(4, n):
        lo = lambda j: fd[i - j] < 20 and fk[i - j] < 20  # noqa: E731
        hi = lambda j: fd[i - j] > 80 and fk[i - j] > 80  # noqa: E731
        up = r[i] > 50 and M[i] > S[i]
        dn = r[i] < 50 and M[i] < S[i]
        below80 = fd[i] < 80 and fk[i] < 80
        above20 = fd[i] > 20 and fk[i] > 20
        if up and ((lo(0) and M[i - 1] < S[i - 1]) or (lo(1) and M[i - 2] < S[i - 2] and below80) or
                   (lo(2) and M[i - 1] < S[i - 1] and below80) or (lo(3) and M[i - 2] < S[i - 2] and below80)):
            sig[i] = 1
        elif dn and ((hi(0) and M[i - 1] > S[i - 1]) or (hi(1) and M[i - 2] > S[i - 2] and above20) or
                     (hi(2) and M[i - 1] > S[i - 1] and above20) or (hi(3) and M[i - 2] > S[i - 2] and above20)):
            sig[i] = -1
    return _tp_sl_trades(d, sig)


def cw_triple_ema(d):
    """tripleEMA (EMA 5/20/50): EMA5 above both others for 4 bars then below both -> short; mirror -> long."""
    c = d.close
    a, b, e = _arr(ema(c, 5)), _arr(ema(c, 20)), _arr(ema(c, 50))
    above = (a > b) & (a > e)
    below = (a < b) & (a < e)
    n = len(a)
    sig = np.zeros(n, int)
    for i in range(4, n):
        if above[i - 4:i].all() and below[i]:
            sig[i] = -1
        if below[i - 4:i].all() and above[i]:
            sig[i] = 1
    return _tp_sl_trades(d, sig)


# ---------------- fmz classics ----------------
def _day_key(d, stock):
    idx = d.index.tz_convert("America/New_York") if stock else d.index
    return idx.normalize()


def r_breaker(d, stock=False, f1=0.35, f2=0.07, f3=0.25):
    """R-Breaker (fmzquant): levels from yesterday's high/low/close; trend: long above bBreak, short below sBreak;
    reversal: after today's high passes sSetup, short below sEnter (mirror for longs); flat at the day's last bar."""
    key = _day_key(d, stock)
    g = d.groupby(key)
    daily = pd.DataFrame({"H": g.high.max(), "L": g.low.min(), "C": g.close.last()}).shift()
    H, L, C = (daily[k].reindex(key).to_numpy() for k in ("H", "L", "C"))
    s_setup = H + f1 * (C - L)
    b_setup = L - f1 * (H - C)
    s_enter = (1 + f2) / 2 * (H + L) - f2 * L
    b_enter = (1 + f2) / 2 * (H + L) - f2 * H
    b_break = s_setup + f3 * (s_setup - b_setup)
    s_break = b_setup - f3 * (s_setup - b_setup)
    dh = d.high.groupby(key).cummax().to_numpy()
    dl = d.low.groupby(key).cummin().to_numpy()
    c = d.close.to_numpy()
    last = np.r_[key[1:] != key[:-1], True]
    n = len(c)
    p = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if np.isnan(H[i]):
            continue
        if cur == 0:
            if c[i] > b_break[i]:
                cur = 1.0
            elif c[i] < s_break[i]:
                cur = -1.0
        if cur >= 0 and dh[i] > s_setup[i] and c[i] < s_enter[i]:
            cur = -1.0
        elif cur <= 0 and dl[i] < b_setup[i] and c[i] > b_enter[i]:
            cur = 1.0
        if last[i]:
            cur = 0.0
        p[i] = cur
    return p


def dual_thrust(d, stock=False, k1=0.67, k2=0.62, tf_hours=4, n=20):
    """Dual Thrust (fmzquant pine version): Range = max(HH - LC, HC - LL) over the last 20 completed 4h bars;
    BuyLine = today's open + 0.67 Range, SellLine = open - 0.62 Range; reverse on a close crossing either line."""
    rs = d.resample(f"{tf_hours}h", origin="start_day").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    HH, LL = rs.high.rolling(n).max(), rs.low.rolling(n).min()
    HC, LC = rs.close.rolling(n).max(), rs.close.rolling(n).min()
    rng = np.maximum(HH - LC, HC - LL).shift()            # completed 4h bars only
    end = rs.index + pd.Timedelta(hours=tf_hours)
    rng.index = rs.index
    R = rng.reindex(d.index, method="ffill").to_numpy()
    key = _day_key(d, stock)
    o_day = d.open.groupby(key).transform("first").to_numpy()
    bl, sl = o_day + k1 * R, o_day - k2 * R
    c = d.close.to_numpy()
    cp = np.r_[np.nan, c[:-1]]
    blp, slp = np.r_[np.nan, bl[:-1]], np.r_[np.nan, sl[:-1]]
    lc = (c > bl) & (cp <= blp)
    sc = (c < sl) & (cp >= slp)
    s = pd.Series(np.where(lc, 1.0, np.where(sc, -1.0, np.nan))).ffill().fillna(0.0).to_numpy()
    del end
    return s


# ---------------- pskrunner14/trading-bot (DQN) ----------------
def _sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def dqn_agent(d, train_end, window=10, episodes=10, batch=32, gamma=0.95, seed=0):
    """DQN agent with the repo's state (sigmoid of the last 10 raw price changes), actions hold/buy/sell, reward =
    sale price - purchase price, gamma 0.95, epsilon 1 -> 0.01 (x0.995 per replay), 128-256-256-128 net.
    Differences: sklearn MLP instead of Keras (no TensorFlow here), 10 episodes instead of 50, replay every 4 steps,
    inventory capped at one unit (the repo can stack buys). Trained on bars before train_end, then run greedily."""
    from sklearn.neural_network import MLPRegressor
    rng = np.random.default_rng(seed)
    px = d.close.to_numpy(float)
    dif = _sig(np.diff(px, prepend=px[0]))
    S = np.lib.stride_tricks.sliding_window_view(np.r_[np.full(window - 1, 0.5), dif], window)[:, 1:]  # 9 diffs
    tr_end = int(np.searchsorted(d.index, pd.Timestamp(train_end, tz=d.index.tz)))
    net = MLPRegressor(hidden_layer_sizes=(128, 256, 256, 128), learning_rate_init=0.001, random_state=seed)
    net.partial_fit(S[:2], np.zeros((2, 3)))
    eps = 1.0
    mem = []
    for _ in range(episodes):
        inv = None
        for t in range(1, tr_end - 1):
            a = int(rng.integers(3)) if rng.random() <= eps else int(np.argmax(net.predict(S[t:t + 1])[0]))
            r = 0.0
            if a == 1 and inv is None:
                inv = px[t]
            elif a == 2 and inv is not None:
                r, inv = px[t] - inv, None
            mem.append((t, a, r, t == tr_end - 2))
            if len(mem) > 1000:
                mem.pop(0)
            if len(mem) > batch and t % 4 == 0:
                mb = [mem[k] for k in rng.choice(len(mem), batch, replace=False)]
                ts = np.array([m[0] for m in mb])
                q = net.predict(S[ts])
                qn = net.predict(S[ts + 1]).max(axis=1)
                for k, (_, a_, r_, done) in enumerate(mb):
                    q[k, a_] = r_ if done else r_ + gamma * qn[k]
                net.partial_fit(S[ts], q)
                eps = max(0.01, eps * 0.995)
    q = net.predict(S)
    act = q.argmax(axis=1)
    p = np.zeros(len(px))
    cur = 0.0
    for t in range(tr_end, len(px)):
        if act[t] == 1:
            cur = 1.0
        elif act[t] == 2:
            cur = 0.0
        p[t] = cur
    return p


# ---------------- CyberPunkMetalHead Binance-volatility-trading-bot ----------------
def volatility_bot(d, change=10.0, sl=5.0, tp=0.8, tsl=0.4, ttp=0.1):
    """Buy a coin that rose more than 10% from its recent low (the bot looks at a 2-minute window; the smallest bars
    here are 5m: close vs the lower of the previous close and this bar's low). Exit rules as in the repo, checked on
    bar closes: start with SL -5% / TP +0.8%; once the gain passes TP, TP = gain + 0.1 and SL = TP - 0.4 (trailing);
    sell when the gain falls below SL."""
    o, lo, c = (_arr(d[k]) for k in ("open", "low", "close"))
    base = np.minimum(np.r_[np.nan, c[:-1]], lo)
    jump = (c / base - 1) * 100 > change
    N = len(c)
    trades, i = [], 1
    while i < N - 1:
        if not jump[i]:
            i += 1
            continue
        j = i + 1
        px = o[j]
        s_l, t_p = -sl, tp
        k, out = j, None
        while k < N:
            g = (c[k] / px - 1) * 100
            if g > t_p:
                t_p = g + ttp
                s_l = t_p - tsl
            elif g < s_l:
                out = (k, c[k])
                break
            k += 1
        if out is None:
            out = (N - 1, c[-1])
        trades.append((j, out[0], 1, px, out[1]))
        i = out[0] + 1
    return trades
