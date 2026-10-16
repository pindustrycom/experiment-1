"""Ideas from Joshua Kyan Aalampour's ORIA design (an Alpaca "Learn" article), tested on the topological-arb universe.

    python -m exp2 oria --data topo_data --out oria_results

Sleeves, each market neutral or explicitly directional, chosen on 2012-2018 and frozen for 2019 onwards:
- graph_open    the graph-residual reversal, traded at the next open (signal at close T, fill at open T+1, held
                to open T+2), instead of at the close; also its intraday-only and overnight-only parts;
- heat          the same with heat-kernel diffusion, h = exp(-tL) x, and exposure scaled down when the residual
                cross-section has fat tails ("tail mass");
- etf_shock     after a sector ETF moves more than k of its own daily sds, trade its stocks' residuals against the
                ETF: follow-through (laggards catch up) or snapback, whichever the training years favour, held 5 days;
- etf_follow    the ETF itself in the direction (or against the direction) of its shock, held 5 days;
- vol_spy       SPY with exposure inversely proportional to a HAR forecast of its variance (the volatility finding
                from Experiment 3), against plain SPY;
- ensemble      every sleeve's returns with market beta removed, then decorrelated by Loewdin symmetric
                orthogonalisation (with shrinkage) and weighted by their training Sharpe (negative ones get 0).
Costs: 5 bp per unit traded. Paper backtest only.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import topo

log = logging.getLogger("exp2.oria")

SECTOR_ETF = {"tech": "XLK", "comm": "XLC", "discretionary": "XLY", "financials": "XLF", "health": "XLV", "staples": "XLP",
              "energy": "XLE", "industrials": "XLI", "utilities": "XLU"}
ETFS = sorted(set(SECTOR_ETF.values()))
COST_BP = 5.0


def load_panel(data: Path, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjusted closes and adjusted opens (open x adj_close / close), one column per ticker."""
    cl, op = {}, {}
    for t in tickers:
        f = data / "prices" / f"{t}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, parse_dates=["date"]).drop_duplicates("date").set_index("date").sort_index()
        cl[t] = d["adj_close"].astype(float)
        if "open" in d:
            op[t] = d["open"].astype(float) * d["adj_close"] / d["close"]
    return pd.DataFrame(cl), pd.DataFrame(op)


def _pnl(W: np.ndarray, ret: np.ndarray, cost_bp: float = COST_BP, trades_per_day: float = 1.0) -> np.ndarray:
    """P&L (in %) of weights W[t] (set at the close of t) earning ret[t] (a return realised after t);
    trades_per_day = 2 when the book is opened and closed within the day."""
    W = np.nan_to_num(W)
    turn = np.abs(np.diff(W, axis=0, prepend=0)).sum(1) if trades_per_day == 1 else 2 * np.abs(W).sum(1)
    return (W * np.nan_to_num(ret)).sum(1) - turn * cost_bp / 100


def hold(W: np.ndarray, days: int) -> np.ndarray:
    """Average of the last `days` signals: a book where each day's signal is held `days` sessions (overlapping tranches)."""
    W = np.nan_to_num(W)
    c = np.cumsum(W, 0)
    out = c.copy()
    out[days:] = c[days:] - c[:-days]
    return out / days


def heat_signal(R: pd.DataFrame, window: int, lookback: int, ts: tuple[float, ...]) -> dict:
    X = R.to_numpy()
    T, n = X.shape
    out = {f"heat_t{t}": np.full((T, n), np.nan) for t in ts}
    tail = np.full(T, np.nan)
    cum = np.cumsum(np.nan_to_num(X), 0)
    for i in range(window, T):
        Y = X[i - window + 1:i + 1]
        vol = Y.std(0) + 1e-9
        x = (cum[i] - cum[i - lookback]) / (vol * math.sqrt(lookback))
        lam, V = np.linalg.eigh(topo.graph_laplacian(np.corrcoef(Y.T)))
        xv = V.T @ x
        for t in ts:
            e = V @ ((1 - np.exp(-t * lam)) * xv)
            out[f"heat_t{t}"][i] = topo._neutral(-e)
            if t == ts[len(ts) // 2]:
                z = (e - e.mean()) / (e.std() + 1e-9)
                tail[i] = np.abs(z)[np.abs(z) > 2].sum() / np.abs(z).sum()
    return out, tail


def etf_shock(R: pd.DataFrame, E: pd.DataFrame, k: float, window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """On days a sector ETF's return exceeds k sds: residual weights within its sector (stock minus beta x ETF),
    traded towards the ETF (laggards up, leaders down), and the ETF's own shock sign. Returns stock weights
    (rows sum to 0) and ETF weights (+-1 per shocked ETF, scaled to gross 1)."""
    names = list(R.columns)
    sec = {t: next(s for s, v in topo.SECTORS.items() if t in v) for t in names}
    X, Ex = R.to_numpy(), E.reindex(R.index).to_numpy()
    etfs = list(E.columns)
    T = len(R)
    Ws = np.zeros((T, len(names)))
    We = np.zeros((T, len(etfs)))
    for j, etf in enumerate(etfs):
        cols = [i for i, t in enumerate(names) if SECTOR_ETF[sec[t]] == etf]
        if len(cols) < 2:
            continue
        e = Ex[:, j]
        for t in range(window, T):
            if np.isnan(e[t]):
                continue
            ew = e[t - window:t]
            sd = np.nanstd(ew)
            if not sd or np.isnan(sd) or abs(e[t]) < k * sd:
                continue
            Y = X[t - window:t][:, cols]
            beta = np.array([np.nanmean((Y[:, m] - np.nanmean(Y[:, m])) * (ew - np.nanmean(ew))) / sd ** 2 for m in range(len(cols))])
            u = X[t, cols] - beta * e[t]
            Ws[t, cols] += -(u - u.mean())
            We[t, j] = np.sign(e[t])
    gs = np.abs(Ws).sum(1, keepdims=True)
    ge = np.abs(We).sum(1, keepdims=True)
    return np.divide(Ws, gs, out=np.zeros_like(Ws), where=gs > 0), np.divide(We, ge, out=np.zeros_like(We), where=ge > 0)


def har_var(spy: pd.Series) -> pd.Series:
    """HAR forecast of next-day log variance, refitted each January on all earlier data (walk-forward)."""
    lv = np.log(spy ** 2 + 1e-4)
    f = pd.DataFrame({"d": lv, "w": np.log((spy ** 2).rolling(5).mean() + 1e-4), "m": np.log((spy ** 2).rolling(22).mean() + 1e-4)})
    y = lv.shift(-1)
    pred = pd.Series(np.nan, index=spy.index)
    for yr in sorted(set(spy.index.year))[2:]:
        tr = (spy.index < pd.Timestamp(f"{yr}-01-01")) & f.notna().all(axis=1) & y.notna()
        A = np.c_[np.ones(tr.sum()), f[tr]]
        b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        sel = spy.index.year == yr
        pred[sel] = np.c_[np.ones(sel.sum()), f[sel].fillna(0)] @ b
    return pred


def loewdin(P: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Symmetric orthogonalisation matrix S^(-1/2) of the sleeves' covariance (shrunk towards its diagonal)."""
    S = np.cov(P.T)
    S = (1 - shrink) * S + shrink * np.diag(np.diag(S))
    w, V = np.linalg.eigh(S)
    return V @ np.diag(1 / np.sqrt(np.clip(w, 1e-12, None))) @ V.T


def run(data: Path, out: Path, fetch: bool = True, train_end: str = topo.TRAIN_END, universe: list[str] = topo.UNIVERSE) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    rep = {"built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "warnings": [], "train_end": train_end}
    if fetch:
        rep["fetch"] = topo.update_prices(data, [*universe, topo.MARKET, *ETFS], need_open=True)
    cl, op = load_panel(data, [*universe, topo.MARKET, *ETFS])
    have = [t for t in universe if t in cl and t in op and op[t].notna().mean() > 0.95]
    keep = cl[have].dropna().index
    cl, op = cl.loc[keep], op.reindex(keep)
    lc, lo = 100 * np.log(cl), 100 * np.log(op)
    R = lc.diff()[have].iloc[1:]
    idx = R.index
    cc_next = R.shift(-1).to_numpy()                                      # close t -> close t+1
    oo_next = (lo.shift(-2) - lo.shift(-1))[have].reindex(idx).to_numpy()  # open t+1 -> open t+2
    id_next = (lc.shift(-1) - lo.shift(-1))[have].reindex(idx).to_numpy()  # open t+1 -> close t+1
    on_next = (lo.shift(-1) - lc)[have].reindex(idx).to_numpy()            # close t -> open t+1
    first = idx[max(topo.GRID["window"]) + 5]
    train = np.asarray((idx >= first) & (idx <= pd.Timestamp(train_end)))
    test = np.asarray(idx > pd.Timestamp(train_end))
    rep["sessions"] = [idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d"), len(idx)]

    def sr(p, sel):
        return topo.stats(pd.Series(p, index=idx)[sel]).get("sharpe", float("nan"))

    sleeves, trials = {}, {}
    # 1. execution timing for the graph residual (settings from the topo run's training choice)
    W = topo.signals(R, 120, 5, [(0.5, 3)])["graph_a0.5_J3"]
    for name, ret, tpd in (("graph_close", cc_next, 1), ("graph_open", oo_next, 1), ("graph_intraday", id_next, 2), ("graph_overnight", on_next, 2)):
        trials[name] = _pnl(W, ret, trades_per_day=tpd)
    for d in (5, 20):
        trials[f"graph_open_hold{d}"] = _pnl(hold(W, d), oo_next)
    log.info("graph timing done (%.0fs)", time.time() - t0)
    # 2. heat kernel, with a tail-mass gate
    H, tail = heat_signal(R, 120, 5, (0.5, 1.0, 2.0, 4.0))
    for k, Wh in H.items():
        trials[f"{k}_hold5"] = _pnl(hold(Wh, 5), oo_next)
    best_heat = max((k for k in trials if k.startswith("heat_")), key=lambda k: sr(trials[k], train))
    Wh = hold(H[best_heat.replace("_hold5", "")], 5)
    tz = pd.Series(tail, index=idx)
    gate = np.clip(1 - (tz - tz[train].mean()) / (tz[train].std() + 1e-9), 0, 2).fillna(1.0).to_numpy()
    tr_gate = _pnl(Wh * gate[:, None], oo_next)
    tr_inv = _pnl(Wh * (2 - gate)[:, None], oo_next)
    trials["heat_tailgate"] = tr_gate if sr(tr_gate, train) >= sr(tr_inv, train) else tr_inv
    rep["heat_choice"] = best_heat
    log.info("heat kernel done (%.0fs)", time.time() - t0)
    # 3. ETF shocks
    E = lc.diff()[[e for e in ETFS if e in lc]].reindex(idx)
    Eoo = (lo.shift(-2) - lo.shift(-1))[list(E.columns)].reindex(idx).to_numpy()
    for k in (1.5, 2.0, 2.5):
        Ws, We = etf_shock(R, E, k)
        for d in (1, 5):
            for name, Wx, ret in ((f"etf_resid_k{k}_h{d}", hold(Ws, d), oo_next), (f"etf_follow_k{k}_h{d}", hold(We, d), Eoo)):
                a, b = _pnl(Wx, ret), _pnl(-Wx, ret)          # follow-through or snapback, whichever trained better
                trials[name] = a if sr(a, train) >= sr(b, train) else b
        rep.setdefault("etf_shock_days", {})[str(k)] = int((np.abs(We).sum(1) > 0).sum())
    log.info("ETF shocks done (%.0fs)", time.time() - t0)
    # 4. vol-managed SPY (no shorting; exposure capped at 2)
    spy = lc[topo.MARKET].diff().reindex(idx)
    v = np.exp(har_var(spy))
    expo = (np.nanmedian(v[train]) / v).clip(upper=2.0).fillna(1.0)
    spy_next = spy.shift(-1).to_numpy()
    trials["spy"] = np.nan_to_num(spy_next)
    trials["vol_spy"] = _pnl(expo.to_numpy()[:, None], spy_next[:, None])

    choose = {"graph_open": "graph_open", "graph_open_hold5": "graph_open_hold5", "heat": best_heat, "heat_tailgate": "heat_tailgate",
              "etf_resid": max((k for k in trials if k.startswith("etf_resid")), key=lambda k: sr(trials[k], train)),
              "etf_follow": max((k for k in trials if k.startswith("etf_follow")), key=lambda k: sr(trials[k], train)),
              "vol_spy": "vol_spy"}
    sleeves = {k: trials[v] for k, v in choose.items()}
    rep["chosen"] = choose
    # 5. ensemble: remove market beta (fitted on training), Loewdin-orthogonalise, weight by training Sharpe
    names = list(sleeves)
    P = np.column_stack([sleeves[k] for k in names])
    m = np.nan_to_num(spy_next)
    ok = train & ~np.isnan(P).any(1)
    beta = np.array([np.cov(P[ok, j], m[ok])[0, 1] / np.var(m[ok]) for j in range(P.shape[1])])
    Pn = np.nan_to_num(P) - m[:, None] * beta[None, :]
    S = loewdin(Pn[ok])
    Q = Pn @ S
    w = np.array([max(0.0, Q[ok, j].mean() / Q[ok, j].std()) for j in range(Q.shape[1])])
    ens = Q @ w if w.sum() > 0 else np.zeros(len(Q))
    ens = ens * (np.std(Pn[ok].mean(1)) / (np.std(ens[ok]) + 1e-12))
    trials["ensemble"] = ens
    rep["ensemble"] = {"beta": dict(zip(names, np.round(beta, 3).tolist())), "orth_weights": dict(zip(names, np.round(w, 3).tolist())),
                       "corr_train": np.round(np.corrcoef(Pn[ok].T), 2).tolist()}

    all_train = [sr(p, train) / math.sqrt(252) for p in trials.values()]
    res = {}
    for k, p in trials.items():
        s = pd.Series(p, index=idx)
        r = {"train": topo.stats(s[train]), "test": topo.stats(s[test])}
        pt = s[test].dropna()
        if len(pt) > 30 and pt.std() > 0:
            r["test"]["deflated_sharpe_prob"] = topo.deflated_sharpe(pt.mean() / pt.std(), len(pt), all_train, float(pt.skew()), float(pt.kurt() + 3))
        r["by_year"] = {str(y): round(float(np.expm1(np.log1p(g / 100).sum())), 4) for y, g in s[s.index >= first].groupby(s.index[s.index >= first].year)}
        res[k] = r
    rep["results"] = res
    rep["n_trials"] = len(trials)
    rep["seconds"] = round(time.time() - t0, 1)
    pd.DataFrame(trials, index=idx).to_csv(out / "sleeves_daily.csv", float_format="%.5g")
    (out / "summary.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
