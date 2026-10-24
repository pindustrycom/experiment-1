"""One-asset, one-position bar backtester shaped like freqtrade's and Jesse's backtests.

A signal is read at a bar's close and filled at the next bar's open. While a trade is open, each bar first checks the
stop (fixed, trailing, or an ATR stop set at entry) on its low/high, then the freqtrade minimal_roi table on its
high/low, then the exit signal at its close. The stop wins when both could have hit inside one bar. The whole account is
in each trade (full stake), there is no leverage, and costs are paid on both sides in basis points of notional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Rules:
    entry_long: pd.Series
    exit_long: pd.Series | None = None
    entry_short: pd.Series | None = None
    exit_short: pd.Series | None = None
    roi: dict | None = None                 # freqtrade minimal_roi: {minutes: min profit}
    stoploss: float | None = None           # e.g. -0.10
    trailing: tuple | None = None           # (trailing_stop_positive, offset, only_offset_is_reached)
    exit_profit_only: bool = False
    stop_dist: pd.Series | None = None      # price distance of a stop fixed at entry (Turtle 2N, Dual Thrust ATR)
    flat_at: pd.Series | None = None        # close everything at this bar's close (intraday systems)
    skip_after_win: bool = False            # Turtle System 1: skip the next breakout after a winning trade
    extra: dict = field(default_factory=dict)


def simulate(d: pd.DataFrame, r: Rules, bar_min: float, cost_bp: float, cash: pd.Series | None = None):
    """Returns (per-bar strategy returns, trades). Cash (a per-bar return) is earned while flat."""
    o, h, lo, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)

    def arr(s, default=False):
        if s is None:
            return np.full(n, default)
        return s.reindex(d.index).fillna(default).to_numpy()

    el, xl, es, xs = arr(r.entry_long), arr(r.exit_long), arr(r.entry_short), arr(r.exit_short)
    flat = arr(r.flat_at)
    sd = arr(r.stop_dist, np.nan).astype(float)
    cr = arr(cash, 0.0).astype(float) if cash is not None else np.zeros(n)
    roi = sorted((float(k), float(v)) for k, v in (r.roi or {}).items())
    cost = cost_bp / 1e4
    ret = np.zeros(n)
    trades = []
    side = 0
    entry_px = ref = ext = stop = 0.0
    e_i = 0
    pending = 0          # +1/-1 enter at this bar's open, 2 exit at this bar's open
    last_win = False

    def roi_for(mins):
        thr = None
        for k, v in roi:
            if mins >= k:
                thr = v
        return thr

    for j in range(n):
        if pending == 2:
            px = o[j]
            ret[j] += side * (px / ref - 1) - cost
            pnl = side * (px / entry_px - 1) - 2 * cost
            trades.append((e_i, j, side, pnl))
            last_win = pnl > 0
            side, pending = 0, 0
        if pending in (1, -1):
            side, entry_px, ref, e_i = pending, o[j], o[j], j
            ext = entry_px
            stop = entry_px * (1 + side * r.stoploss) if r.stoploss is not None else np.nan
            if not math.isnan(sd[j - 1] if j else np.nan):
                stop = entry_px - side * sd[j - 1]
            ret[j] -= cost
            pending = 0
        if side == 0:
            ret[j] += cr[j]
        else:
            exit_px = None
            # 1. stop (uses the level known before this bar's extreme)
            if not math.isnan(stop):
                if side > 0 and lo[j] <= stop:
                    exit_px = min(o[j], stop) if j > e_i else stop
                elif side < 0 and h[j] >= stop:
                    exit_px = max(o[j], stop) if j > e_i else stop
            # 2. minimal_roi
            if exit_px is None and roi:
                thr = roi_for((j - e_i) * bar_min)
                if thr is not None:
                    tgt = entry_px * (1 + side * thr)
                    if side > 0 and h[j] >= tgt:
                        exit_px = max(o[j], tgt) if j > e_i else tgt
                    elif side < 0 and lo[j] <= tgt:
                        exit_px = min(o[j], tgt) if j > e_i else tgt
            if exit_px is not None:
                ret[j] += side * (exit_px / ref - 1) - cost
                pnl = side * (exit_px / entry_px - 1) - 2 * cost
                trades.append((e_i, j, side, pnl))
                last_win = pnl > 0
                side = 0
            else:
                ret[j] += side * (c[j] / ref - 1)
                ref = c[j]
                # trailing stop update with this bar's extreme
                ext = max(ext, h[j]) if side > 0 else min(ext, lo[j])
                if r.trailing is not None:
                    pos, off, only_off = r.trailing
                    gain = side * (ext / entry_px - 1)
                    if gain > off:
                        new = ext * (1 - side * pos)
                    elif not only_off and r.stoploss is not None:
                        new = ext * (1 + side * r.stoploss)
                    else:
                        new = np.nan
                    if not math.isnan(new):
                        stop = new if math.isnan(stop) else (max(stop, new) if side > 0 else min(stop, new))
                if flat[j]:
                    ret[j] -= cost
                    pnl = side * (c[j] / entry_px - 1) - 2 * cost
                    trades.append((e_i, j, side, pnl))
                    last_win = pnl > 0
                    side = 0
                elif j + 1 < n:
                    want_out = (xl[j] or es[j]) if side > 0 else (xs[j] or el[j])
                    if want_out and r.exit_profit_only and side * (c[j] / entry_px - 1) <= 2 * cost:
                        want_out = bool(es[j] if side > 0 else el[j])
                    if want_out:
                        pending = 2
        if side == 0 and pending == 0 and j + 1 < n and not flat[j]:
            sig = 1 if el[j] else (-1 if es[j] else 0)
            if sig and r.skip_after_win and last_win:
                last_win = False          # skip this breakout, take the next one
                sig = 0
            if sig:
                pending = sig
    if side != 0:   # mark the open trade to the last close
        trades.append((e_i, n - 1, side, side * (c[-1] / entry_px - 1) - cost))
    return pd.Series(ret, index=d.index), trades


def daily(ret: pd.Series) -> pd.Series:
    """Compound per-bar returns into calendar-day returns (index = date)."""
    g = np.log1p(ret).groupby(ret.index.normalize()).sum()
    return np.expm1(g)


def stats(r: pd.Series, bench: pd.Series, per_year: int, trades=None, index=None, bar_min=None) -> dict:
    r = r.dropna()
    if len(r) < 30:
        return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "max_dd": float("nan"), "beta": float("nan"),
                "alpha_ann": float("nan"), "alpha_t": float("nan"), "cagr_2019_22": float("nan"), "cagr_2023_26": float("nan")}
    b = bench.reindex(r.index).fillna(0.0)
    yrs = len(r) / per_year
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else float("nan")
    vol = r.std() * math.sqrt(per_year)
    sharpe = r.mean() / r.std() * math.sqrt(per_year) if r.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    X = np.column_stack([np.ones(len(b)), b.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, r.to_numpy(), rcond=None)
    resid = r.to_numpy() - X @ coef
    se = math.sqrt((resid @ resid) / max(len(r) - 2, 1) * np.linalg.pinv(X.T @ X)[0, 0])
    out = {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": dd, "beta": coef[1],
           "alpha_ann": coef[0] * per_year, "alpha_t": coef[0] / se if se > 0 else 0.0}
    by_year = (1 + r).groupby(r.index.year).prod() - 1
    out["by_year"] = {int(k): float(v) for k, v in by_year.items()}
    half = r.index < pd.Timestamp("2023-01-01")
    for name, m in (("cagr_2019_22", half), ("cagr_2023_26", ~half)):
        x = r[m]
        out[name] = float((1 + x).prod() ** (per_year / max(len(x), 1)) - 1) if len(x) else float("nan")
    if trades is not None:
        t = [p for p in trades if index is None or index[p[0]] >= r.index[0]]
        out["trades"] = len(t)
        out["win_pct"] = 100 * float(np.mean([p[3] > 0 for p in t])) if t else 0.0
        out["avg_trade_pct"] = 100 * float(np.mean([p[3] for p in t])) if t else 0.0
        if bar_min and t:
            out["avg_hold_days"] = float(np.mean([(p[1] - p[0] + 1) for p in t])) * bar_min / 1440
            out["time_in_market"] = float(sum(p[1] - p[0] + 1 for p in t)) / max(sum(index >= r.index[0]), 1)
    return {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in out.items()}
