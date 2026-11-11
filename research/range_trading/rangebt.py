"""Range trading test (Camille van Merrienboer / "Trading Wizard" video, 2026-10-19).
He publishes no rules, so this is the textbook horizontal-range method crypto range traders describe:
  range  = highest high (RH) and lowest low (RL) of the last N bars, equilibrium EQ = middle.
  choppy = Kaufman efficiency ratio of those N closes below 0.3 (price went sideways, not trending).
Setups (all read at a bar's close, filled at the next bar's open, levels frozen at entry):
  fade      : close in the bottom 15% of a choppy range -> long; top 15% -> short.
              Stop 15% of the width beyond the edge, target EQ (or the opposite edge with tgt='edge').
  deviation : a close outside a choppy range, then within 6 bars a close back inside -> trade back into it.
              Stop just beyond the deviation's extreme, target EQ or the opposite edge. Deviations deeper
              than half the width are treated as a breakout and dropped.
  breakout  : (control) close outside a choppy range -> follow it, stop at EQ, target one width further.
Stops and targets are checked on each bar's high/low; if both are touched in one bar the stop counts.
A gap through a level fills at the open. Intraday mode flattens at each session's last bar.
Costs: per side, in basis points of notional. Each trade uses 1x the account, no leverage."""
import numpy as np
import pandas as pd


def efficiency(close, n):
    ch = close.diff(n).abs()
    vol = close.diff().abs().rolling(n).sum()
    return ch / vol


def run(d, setup, n=48, er_max=0.3, tgt='eq', cost_bp=0.3, intraday=True, max_hold=None, edge=0.15):
    """d: DataFrame with open/high/low/close and 'day'. Returns list of trades (entry_time, side, ret, bars)."""
    o, h, l, c = (d[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    rh = d.high.rolling(n).max().shift(1).to_numpy()      # range of the n bars before this one
    rl = d.low.rolling(n).min().shift(1).to_numpy()
    er = efficiency(d.close, n).shift(1).to_numpy()
    day = d.day.to_numpy()
    last = np.r_[day[1:] != day[:-1], True]                # last bar of each session
    idx = d.index
    trades = []
    pos, entry, stop, target, t0, k0 = 0, 0.0, 0.0, 0.0, None, 0
    dev = None                                             # pending deviation: (side, RL, RH, extreme, bar)
    N = len(d)
    for k in range(n + 1, N - 1):
        # manage an open position on bar k
        if pos:
            exit_px = None
            if pos > 0:
                if o[k] <= stop: exit_px = o[k]
                elif l[k] <= stop: exit_px = stop
                elif o[k] >= target: exit_px = o[k]
                elif h[k] >= target: exit_px = target
            else:
                if o[k] >= stop: exit_px = o[k]
                elif h[k] >= stop: exit_px = stop
                elif o[k] <= target: exit_px = o[k]
                elif l[k] <= target: exit_px = target
            if exit_px is None and ((intraday and last[k]) or (max_hold and k - k0 >= max_hold)):
                exit_px = c[k]
            if exit_px is not None:
                trades.append((t0, pos, pos * (exit_px / entry - 1) - 2 * cost_bp / 1e4, k - k0 + 1))
                pos = 0
            continue
        if intraday and last[k]:                           # no entry that would fill on the next session
            dev = None
            continue
        a, b = rl[k], rh[k]
        if not np.isfinite(a) or b <= a:
            continue
        w = b - a
        choppy = er[k] < er_max
        side = 0
        if setup == 'fade' and choppy:
            if a < c[k] <= a + edge * w: side = 1
            elif b - edge * w <= c[k] < b: side = -1
            if side:
                stop = a - edge * w if side > 0 else b + edge * w
                target = (a + b) / 2 if tgt == 'eq' else (b - 0.05 * w if side > 0 else a + 0.05 * w)
        elif setup == 'deviation':
            if dev is not None:
                s, da, db, ext, kd = dev
                dw = db - da
                ext = min(ext, l[k]) if s > 0 else max(ext, h[k])
                deep = (da - ext > 0.5 * dw) if s > 0 else (ext - db > 0.5 * dw)
                if deep or k - kd > 6:
                    dev = None
                elif (s > 0 and c[k] > da) or (s < 0 and c[k] < db):
                    side = s
                    stop = ext - 0.05 * dw if s > 0 else ext + 0.05 * dw
                    target = (da + db) / 2 if tgt == 'eq' else (db - 0.05 * dw if s > 0 else da + 0.05 * dw)
                    dev = None
                else:
                    dev = (s, da, db, ext, kd)
            if not side and dev is None and choppy:
                if c[k] < a: dev = (1, a, b, l[k], k)
                elif c[k] > b: dev = (-1, a, b, h[k], k)
        elif setup == 'breakout' and choppy:
            if c[k] > b: side, stop, target = 1, (a + b) / 2, b + w
            elif c[k] < a: side, stop, target = -1, (a + b) / 2, a - w
        if side:
            px = o[k + 1]
            # skip if the next open already sits beyond the stop or the target
            if (side > 0 and (px <= stop or px >= target)) or (side < 0 and (px >= stop or px <= target)):
                continue
            pos, entry, t0, k0 = side, px, idx[k + 1], k + 1
    return trades


def stats(trades, periods_per_year, index_days=None):
    if not trades:
        return {'trades': 0}
    t = pd.DataFrame(trades, columns=['t', 'side', 'ret', 'bars'])
    t['day'] = pd.to_datetime(t.t).dt.strftime('%Y-%m-%d')
    daily = t.groupby('day').ret.sum()
    if index_days is not None:
        daily = daily.reindex(index_days, fill_value=0.0)
    eq = (1 + daily).cumprod()
    yrs = len(daily) / periods_per_year
    r = t.ret.to_numpy()
    by_year = (1 + daily).groupby(daily.index.str[:4]).prod().sub(1).mul(100).round(1).to_dict()
    return dict(trades=len(r), per_year=round(len(r) / yrs, 1), win=round(100 * (r > 0).mean(), 1),
                avg_bp=round(1e4 * r.mean(), 2),
                pf=round(r[r > 0].sum() / -r[r < 0].sum(), 2) if (r < 0).any() else None,
                long_share=round(100 * (t.side > 0).mean(), 1),
                cagr=round(100 * (max(eq.iloc[-1], 0) ** (1 / yrs) - 1), 1),
                sharpe=round(daily.mean() / daily.std() * np.sqrt(periods_per_year), 2) if daily.std() > 0 else 0,
                maxdd=round(100 * (eq / eq.cummax() - 1).min(), 1), by_year=by_year)
