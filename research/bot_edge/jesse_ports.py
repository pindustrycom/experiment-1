"""Jesse example strategies not covered in PR #22 (KDJstrategy, MAGen, TradingView_RSI), with Jesse's own indicator
functions (jesse.indicators) and an event loop that follows Jesse's order flow: signal at a candle's close, market
fill at the next open, stop / take-profit checked on each candle's range (stop first), update_position at closes.
Full stake per trade (the originals size by risk %, which only scales returns)."""
import sys
import os
import numpy as np
import pandas as pd
import jesse.indicators as ta
sys.path.insert(0, os.path.dirname(__file__))
import evalkit as ek

DATA, OUT = sys.argv[1], sys.argv[2]


def jc(d):
    ts = (d.index.view("int64") // 10**6).astype(float)
    return np.column_stack([ts, d.open, d.close, d.high, d.low, d.volume]).astype(float)


def crossed_above(x, lvl, i):
    return x[i - 1] <= lvl < x[i]


def crossed_below(x, lvl, i):
    return x[i - 1] >= lvl > x[i]


def run(d, name):
    C = jc(d)
    o, c, h, lo = C[:, 1], C[:, 2], C[:, 3], C[:, 4]
    n = len(c)
    if name == "KDJstrategy":
        k, dd, j = ta.kdj(C, sequential=True)
        atr = ta.atr(C, period=14, sequential=True)
    elif name == "MAGen":
        fast = ta.ma(C, period=5, matype=11, sequential=True)
        slow = ta.ma(C, period=20, matype=11, sequential=True)
        adx = ta.adx(C, period=8, sequential=True)
        atr = ta.atr(C, period=32, sequential=True)
    else:
        rsi = ta.rsi(C, 5, sequential=True)
    trades = []
    side, i0, p0, stop, tp = 0, 0, 0.0, None, None
    pending = None
    for i in range(50, n):
        # fills at this open
        if pending is not None:
            side, i0, p0 = pending[0], i, o[i]
            if name == "KDJstrategy":
                stop, tp = p0 - 2 * atr[i - 1], None
            elif name == "MAGen":
                stop = p0 - side * 2 * atr[i - 1]
                tp = p0 + side * 5 * atr[i - 1]
            else:
                stop, tp = p0 * 0.95, p0 * 1.10
            pending = None
        if side != 0:
            ex = None
            if stop is not None and ((side > 0 and lo[i] <= stop) or (side < 0 and h[i] >= stop)):
                ex = (min(o[i], stop) if side > 0 else max(o[i], stop)) if i > i0 else stop
            elif tp is not None and ((side > 0 and h[i] >= tp) or (side < 0 and lo[i] <= tp)):
                ex = (max(o[i], tp) if side > 0 else min(o[i], tp)) if i > i0 else tp
            if ex is not None:
                trades.append((i0, i, side, p0, ex)); side = 0
                continue
            # update_position at the close -> liquidate at next open
            liq = False
            if name == "KDJstrategy":
                inprofit = c[i] > p0
                if inprofit and j[i - 1] > j[i]:
                    liq = True
                elif j[i] < dd[i]:            # `J < (K and D)` in Python is J < D
                    liq = True
            elif name == "MAGen":
                if (side < 0 and fast[i] > slow[i] and adx[i] < 15) or (side > 0 and fast[i] < slow[i] and adx[i] < 15):
                    liq = True
            else:
                if crossed_below(rsi, 75, i) or crossed_below(rsi, 10, i):
                    liq = True
            if liq and i + 1 < n:
                trades.append((i0, i + 1, side, p0, o[i + 1])); side = 0
            continue
        if i + 1 >= n:
            break
        if name == "KDJstrategy":
            if j[i] > dd[i]:
                pending = (1,)
        elif name == "MAGen":
            ch = 1 if (fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]) else (-1 if (fast[i] < slow[i] and fast[i - 1] >= slow[i - 1]) else 0)
            if ch != 0 and adx[i] > 13:
                pending = (ch,)
        else:
            if crossed_above(rsi, 35, i):
                pending = (1,)
    if side != 0:
        trades.append((i0, n - 1, side, p0, c[-1]))
    return trades


rows = []
for asset in ("BTC", "ETH", "SPY", "QQQ"):
    crypto = asset in ("BTC", "ETH")
    fee = 0.001 if crypto else 0.0002
    for name, tf in (("KDJstrategy", "1d"), ("MAGen", "4h"), ("TradingView_RSI", "4h")):
        d = pd.read_feather(f"{DATA}/{asset}_USDT-{tf}.feather").set_index("date").sort_index()
        tr = run(d, name)
        net, gross, pos = ek.bar_returns_from_trades(d, tr, fee)
        r = ek.score(d, net, gross, pos, 365 if crypto else 252, len(tr))
        r.update({"source": "jesse example-strategies", "strategy": name, "asset": asset, "tf": tf, "native_tf": tf,
                  "note": "Jesse indicators, full stake", "win_rate": float(np.mean([(t[4] / t[3] - 1) * t[2] > 2 * fee for t in tr])) if tr else np.nan})
        rows.append(r)
        print(f"jesse {asset} {name:16} {tf} n {r['trades']} cagr {r['cagr']:+.1%} gross {r['gross_cagr']:+.1%} sh {r['sharpe']:.2f} hold {r['hold_sharpe']:.2f} p {r['p_random']:.3f} z {r['z_random']:+.1f} a_t {r['alpha_t']:+.1f}", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
