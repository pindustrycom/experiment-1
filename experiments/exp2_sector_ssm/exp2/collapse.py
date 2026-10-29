"""Collapse risk and a combined portfolio, on the S&P 500 price cache of exp2/papers.py.

Joaquin's goal: model the process by which a company may collapse, and make money from it. With prices only, a
"collapse" is a fall of 50% or more from today's price at some point in the next 12 months. A hazard model
(logistic, as in Campbell, Hilscher & Szilagyi, and boosted trees) gives each stock a monthly collapse probability.
That probability, the volatility forecast of exp2/volpred.py and a market regime model then drive seven strategy
families ("sleeves"):

  1. collapse      short the most collapse-prone stocks, long the safest; and the index without the riskiest names
  2. vol carry     a synthetic short variance swap on SPY (VIX^2 against next month's realised variance), timed
  3. construction  long-only stock portfolios: inverse volatility, minimum variance (Ledoit-Wolf), HRP
  4. pairs         cointegrated same-sector pairs traded on an Ornstein-Uhlenbeck z-score, skipping collapse-prone legs
  5. bet sizing    meta-labelling of the pair trades (a second model decides which trades to take)
  6. trend         time-series momentum on SPY and 11 sector baskets; SPY trend filter
  7. calendar and regimes   turn of the month, pre-holiday, a 2-state Markov-switching filter on SPY

The sleeves are then combined into one portfolio. Everything that is chosen (which variant of each family, sleeve
weights, leverage) is chosen on 2005-2018 only; 2019 to today is the untouched test. Costs: 10 bp per unit traded,
1% a year to borrow shorted stock, 0.5 vol point to roll the variance swap. Paper research only: nothing is traded.

The universe is the S&P 500 as it was each month (exp2/pit.py), so firms that went bankrupt or were taken over are in
it while they were members: the collapse model learns from real collapses, not only from survivors' drawdowns.
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

from . import papers, pit, volpred
from .papers import COST_BP, TRAIN_END, deciles, sstats

log = logging.getLogger("exp2.collapse")
CRASH = math.log(0.5)                       # a fall of 50% from the month-end price
BORROW_PCT_YEAR = 1.0
FIRST_YEAR = 2005
CX = ["f_gbt", "lv22", "lv252", "vol_jump", "mom1", "mom12_2", "ex12", "ivol3m", "beta12m", "ldvol", "lprice", "hi52",
      "max1m", "skew63", "lvix", "p_hi"]


# ---------------------------------------------------------------- regimes (technique 7)

def hmm2(r: np.ndarray, fit: np.ndarray, iters: int = 60) -> tuple[np.ndarray, dict]:
    """Two-state Gaussian hidden Markov model (calm / turbulent) fitted by Baum-Welch on the `fit` days; returns the
    filtered probability of the turbulent state at each day (uses data up to that day only) and the parameters."""
    x = r[fit]
    mu = np.array([x.mean(), x.mean()])
    sd = np.array([x.std() * 0.7, x.std() * 1.8])
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    pi = np.array([0.8, 0.2])

    def dens(v):
        return np.exp(-0.5 * ((v[:, None] - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)) + 1e-300

    for _ in range(iters):
        B = dens(x)
        n = len(x)
        al, c = np.empty((n, 2)), np.empty(n)
        al[0] = pi * B[0]
        c[0] = al[0].sum()
        al[0] /= c[0]
        for t in range(1, n):
            al[t] = (al[t - 1] @ A) * B[t]
            c[t] = al[t].sum()
            al[t] /= c[t]
        be = np.ones((n, 2))
        for t in range(n - 2, -1, -1):
            be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
        g = al * be
        g /= g.sum(1, keepdims=True)
        xi = (al[:-1, :, None] * A[None] * (B[1:] * be[1:])[:, None, :]) / c[1:, None, None]
        A = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)
        pi = g[0]
        mu = (g * x[:, None]).sum(0) / g.sum(0)
        sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / g.sum(0))
    hi = int(np.argmax(sd))
    p = np.empty(len(r))
    a = pi.copy()
    for t in range(len(r)):
        v = r[t] if not np.isnan(r[t]) else mu @ a
        b = np.exp(-0.5 * ((v - mu) / sd) ** 2) / sd
        a = (a @ A if t else a) * b
        a /= a.sum()
        p[t] = a[hi]
    return p, {"mu": mu.tolist(), "sd": sd.tolist(), "A": A.tolist(), "turbulent_state": hi}


# ---------------------------------------------------------------- data

def load_all(data: Path, out: Path, train_end: str):
    """Monthly stock panel with the volatility forecast (stage 1 of volpred), daily closes, membership and SPY."""
    cur = pd.read_csv(data / "sp500_current.csv")
    changes = pd.read_csv(data / "sp500_changes.csv", parse_dates=["date"])
    member = papers.membership(cur, changes, pd.date_range(papers.START, pd.Timestamp.today(), freq="ME"))
    tickers = sorted(member.columns)
    px = papers.load(data, tickers)
    base = papers.load(data, ["SPY", "^VIX", "^IRX"])
    spy, vix = base["SPY"]["adj_close"], base["^VIX"]["close"]
    sectors = papers.sectors_all(data)
    cache = out / "panel_stage1.pkl"
    if cache.exists() and cache.stat().st_mtime > max((data / "prices").stat().st_mtime, Path(volpred.__file__).stat().st_mtime):
        P = pd.read_pickle(cache)
    else:
        P = papers.panel(px, spy, sectors, member)
        P = P.merge(volpred.vol_features(px, spy, vix), on=["month", "ticker"], how="left")
        _, P = volpred.stage1(P, 2003, train_end)
        P.to_pickle(cache)
    C = pd.DataFrame({t: d["adj_close"] for t, d in px.items()}).sort_index()
    C = C[C.index >= spy.index[0]]
    rf = (base["^IRX"]["close"].reindex(spy.index).ffill() / 100 / 252).fillna(0.0)     # daily T-bill
    return P, C, member, spy, vix, rf, sectors


def month_ret(daily: pd.Series) -> pd.Series:
    """Daily simple returns -> monthly simple returns in %, indexed by month period."""
    return ((1 + daily).groupby(daily.index.to_period("M")).prod() - 1) * 100


# ---------------------------------------------------------------- 1. collapse model

def collapse_labels(P: pd.DataFrame, C: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    """crash12 = 1 if the stock trades 50% or more below its month-end price within the next 252 sessions."""
    lc = np.log(C)
    fwd_min = lc[::-1].rolling(252, min_periods=252).min()[::-1].shift(-1)
    # month-end rows of each stock
    me = lc.groupby(lc.index.to_period("M")).tail(1)
    drop = (fwd_min.loc[me.index] - me)
    lab = drop.stack(future_stack=True).rename("fwd_drop").reset_index()
    lab.columns = ["date", "ticker", "fwd_drop"]
    lab["month"] = lab["date"].dt.to_period("M")
    P = P.merge(lab[["month", "ticker", "fwd_drop"]], on=["month", "ticker"], how="left")
    P["crash12"] = np.where(P["fwd_drop"].isna(), np.nan, (P["fwd_drop"] <= CRASH).astype(float))
    r = np.log(C).diff()
    skew = r.rolling(63, min_periods=50).skew()
    sk = skew.groupby(skew.index.to_period("M")).last().stack(future_stack=True).rename("skew63").reset_index()
    sk.columns = ["month", "ticker", "skew63"]
    P = P.merge(sk, on=["month", "ticker"], how="left")
    s12 = np.log(spy).diff().rolling(252).sum()
    P["ex12"] = P.groupby("ticker")["ret"].transform(lambda s: s.rolling(12).sum()) - P["month"].map(s12.groupby(s12.index.to_period("M")).last())
    P["vol_jump"] = P["lv5"] - P["lv252"]                                   # a change-point flag for the stock's own volatility
    return P


def collapse_model(P: pd.DataFrame, train_end: str) -> tuple[dict, pd.DataFrame]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    P = P.copy()
    y = P["crash12"].to_numpy()
    X = P[CX].to_numpy(float)
    mons = P["month"].to_numpy()
    yrs = P["month"].dt.year.to_numpy()
    out = {k: np.full(len(P), np.nan) for k in ("logit", "gbt")}
    for yr in range(FIRST_YEAR, int(yrs.max()) + 1):
        te = yrs == yr
        cut = pd.Period(f"{yr - 1}-01", "M")                                # labels need 12 more months to be known
        tr = (mons < cut) & ~np.isnan(y)
        if not te.any() or tr.sum() < 3000 or y[tr].sum() < 30:
            continue
        mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-9
        Z = np.nan_to_num((X - mu) / sd)
        lr = LogisticRegression(C=0.1, max_iter=500).fit(Z[tr], y[tr])
        out["logit"][te] = lr.predict_proba(Z[te])[:, 1]
        g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=300, l2_regularization=1.0,
                                           random_state=0).fit(X[tr], y[tr])
        out["gbt"][te] = g.predict_proba(X[te])[:, 1]
    P["pc_logit"], P["pc_gbt"] = out["logit"], out["gbt"]
    P["pc_vol"] = P["f_gbt"]                                                # benchmark: rank by forecast volatility alone
    res = {}
    test = P["month_end"] > pd.Timestamp(train_end)
    for per, sel in (("before_test", ~test & (yrs >= FIRST_YEAR)), ("test", test)):
        row = {}
        for k in ("pc_logit", "pc_gbt", "pc_vol"):
            s = sel & P[k].notna() & P["crash12"].notna()
            if s.sum() < 100 or P.loc[s, "crash12"].nunique() < 2:
                continue
            g = P[s]
            top = g[g.groupby("month")[k].rank(pct=True) > 0.9]
            xs = g.groupby("month").apply(lambda d, k=k: roc_auc_score(d["crash12"], d[k]) if d["crash12"].nunique() == 2 else np.nan)
            # cross-sectional AUC: which stocks collapse, month by month; pooled AUC also asks when (market crashes)
            row[k] = {"auc_cross_section": float(xs.mean()), "auc_pooled": float(roc_auc_score(g["crash12"], g[k])), "base_rate": float(g["crash12"].mean()),
                      "rate_in_top_decile": float(top["crash12"].mean()), "n": int(s.sum()), "collapses": int(g["crash12"].sum())}
        res[per] = row
    return res, P


def collapse_sleeves(P: pd.DataFrame, rf_m: pd.Series) -> dict[str, pd.Series]:
    """Monthly % returns: distress long-short (long safest decile, short riskiest) and the equal-weighted index with
    and without the riskiest 10% (both long-only, in excess of T-bills)."""
    out = {}
    for k in ("pc_logit", "pc_gbt"):
        g = P[P[k].notna()].assign(safe=lambda d, k=k: -d[k])
        ls = deciles(g, "safe") - BORROW_PCT_YEAR / 12                         # borrow on the short leg
        out[f"collapse_ls_{k[3:]}"] = ls
        ew, flt = {}, {}
        for m, d in g.dropna(subset=["y"]).groupby("month"):
            r = (np.exp(d["y"]) - 1) * 100
            ew[m] = r.mean()
            flt[m] = r[d[k].rank(pct=True) <= 0.9].mean()
        ew, flt = pd.Series(ew), pd.Series(flt)
        nxt = (ew.index + 1)
        rf_n = rf_m.reindex(nxt).to_numpy()
        out[f"collapse_filter_{k[3:]}"] = flt - rf_n - 0.02                      # ~10% of the book changes a month, 10 bp
        out["equal_weight_index"] = ew - rf_n
    # sleeves are indexed by the formation month; returns accrue the next month
    return out


# ---------------------------------------------------------------- 2. variance carry

def vol_carry(spy: pd.Series, vix: pd.Series, p_hi: pd.Series, sdf: pd.DataFrame) -> dict[str, pd.Series]:
    """Short a one-month SPY variance swap at VIX^2 each month end; P&L in vega terms (K^2 - RV) / (2K) vol points,
    times a vega notional of 0.5% of capital per vol point, less 0.5 vol point to trade. Timed: only when VIX^2 is
    above the HAR forecast of next month's variance and the regime filter says calm."""
    r = np.log(spy).diff().dropna()
    g = r.index.to_period("M")
    rv = (r ** 2).groupby(g).mean() * 252 * 1e4                            # next month's realised variance, annual %^2
    K = vix.groupby(vix.index.to_period("M")).last()
    df = pd.DataFrame({"K": K, "rv_next": rv.shift(-1)}).dropna()
    V = 0.5
    pnl = V * ((df["K"] ** 2 - df["rv_next"]) / (2 * df["K"]) - 0.5)
    fvar = np.exp(2 * sdf["fvol"]) * 252 * 1e4
    ph = p_hi.groupby(p_hi.index.to_period("M")).last()
    on = (df["K"] ** 2 > fvar.reindex(df.index)) & (ph.reindex(df.index) < 0.5)
    return {"varswap_always": pnl, "varswap_timed": pnl.where(on, 0.0)}


# ---------------------------------------------------------------- 3. portfolio construction

def _minvar(S: np.ndarray) -> np.ndarray:
    n = len(S)
    keep = np.arange(n)
    for _ in range(10):                                                     # long-only by dropping negative weights
        Si = np.linalg.pinv(S[np.ix_(keep, keep)])
        w = Si @ np.ones(len(keep))
        w /= w.sum()
        if (w >= 0).all():
            break
        keep = keep[w > 0]
    out = np.zeros(n)
    out[keep] = np.clip(w, 0, None)
    return out / out.sum()


def _hrp(S: np.ndarray) -> np.ndarray:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    sd = np.sqrt(np.diag(S))
    corr = np.clip(S / np.outer(sd, sd), -1, 1)
    D = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    np.fill_diagonal(D, 0)
    order = leaves_list(linkage(squareform(D, checks=False), "single"))
    w = np.ones(len(S))
    stack = [order]
    while stack:
        c = stack.pop()
        if len(c) < 2:
            continue
        a, b = c[: len(c) // 2], c[len(c) // 2:]

        def cvar(ix):
            iv = 1 / np.diag(S)[ix]
            iv /= iv.sum()
            return iv @ S[np.ix_(ix, ix)] @ iv
        va, vb = cvar(a), cvar(b)
        al = 1 - va / (va + vb)
        w[a] *= al
        w[b] *= 1 - al
        stack += [a, b]
    return w / w.sum()


def construction(P: pd.DataFrame, C: pd.DataFrame, rf_m: pd.Series, risk_col: str = "pc_gbt", min_names: int = 50) -> dict[str, pd.Series]:
    """Long-only monthly portfolios of the members: inverse forecast volatility, minimum variance (Ledoit-Wolf
    covariance of the last 252 sessions), hierarchical risk parity; each also without the 10% most collapse-prone."""
    from sklearn.covariance import LedoitWolf
    R = np.log(C).diff()
    me_dates = R.groupby(R.index.to_period("M")).apply(lambda d: d.index[-1])
    books = {k: {} for k in ("invvol", "minvar", "hrp", "minvar_nocollapse", "hrp_nocollapse")}
    prev = {k: None for k in books}
    for m, g in P[P[risk_col].notna() & P["y"].notna()].groupby("month"):
        if m not in me_dates.index or len(g) < min_names:
            continue
        end = me_dates[m]
        W = R.loc[:end].iloc[-252:][g["ticker"]]
        ok = W.notna().sum() >= 240
        g = g[ok.to_numpy()]
        W = W.loc[:, ok].fillna(0.0)
        S = LedoitWolf().fit(W.to_numpy()).covariance_
        nxt = (np.exp(g["y"].to_numpy()) - 1) * 100
        safe = (g[risk_col].rank(pct=True) <= 0.9).to_numpy()
        ws = {"invvol": 1 / np.exp(g["f_gbt"].to_numpy()), "minvar": _minvar(S), "hrp": _hrp(S)}
        ws["invvol"] /= ws["invvol"].sum()
        ix = np.where(safe)[0]
        for k, base in (("minvar_nocollapse", _minvar), ("hrp_nocollapse", _hrp)):
            w = np.zeros(len(g))
            w[ix] = base(S[np.ix_(ix, ix)])
            ws[k] = w
        for k, w in ws.items():
            s = pd.Series(w, index=g["ticker"].to_numpy())
            turn = 1.0 if prev[k] is None else float((s.subtract(prev[k], fill_value=0)).abs().sum())
            prev[k] = s
            books[k][m] = float(w @ nxt) - turn * COST_BP / 100
    out = {}
    for k, b in books.items():
        s = pd.Series(b)
        out[f"port_{k}"] = s - rf_m.reindex(s.index + 1).to_numpy()
    return out


# ---------------------------------------------------------------- 4 + 5. pairs and meta-labelling

def pairs(P: pd.DataFrame, C: pd.DataFrame, sectors: dict, member: pd.DataFrame, top: int = 20, entry: float = 2.0, exit_: float = 0.5,
          stop: float = 4.0, first_year: int = 2003) -> pd.DataFrame:
    """Every month: in each sector, Engle-Granger test every pair of member stocks on 252 sessions of log prices;
    keep the `top` most strongly mean-reverting (ADF t < -3.34, half-life 3-40 sessions), at most 2 per stock.
    Next month, trade the spread's z-score: enter at |z| > 2, exit at |z| < 0.5, stop at |z| > 4, close at month end.
    Returns one row per trade with its features, net return and daily P&L path."""
    L = np.log(C)
    R = L.diff()
    risk = P.set_index(["month", "ticker"])["pc_gbt"]
    fvol = P.set_index(["month", "ticker"])["f_gbt"]
    me = L.groupby(L.index.to_period("M")).apply(lambda d: d.index[-1])
    mem = member.copy()
    mem.index = pd.DatetimeIndex(mem.index).to_period("M")
    trades = []
    months = [m for m in me.index if m.year >= first_year and m in mem.index]
    for m in months[:-1]:
        end = me[m]
        names = [t for t in mem.columns[mem.loc[m].to_numpy()] if t in L.columns]
        Wd = L.loc[:end].iloc[-252:]
        nxt = L.loc[(L.index > end) & (L.index <= me[m + 1])] if (m + 1) in me.index else None
        if nxt is None or len(nxt) == 0 or len(Wd) < 252:
            continue
        Wd = Wd[names]
        Wd = Wd.loc[:, Wd.notna().all()]
        cand = []
        bysec: dict[str, list[str]] = {}
        for t in Wd.columns:
            bysec.setdefault(sectors.get(t, "?"), []).append(t)
        for _, ts in bysec.items():
            if len(ts) < 2:
                continue
            X = Wd[ts].to_numpy()
            ia, ib = np.triu_indices(len(ts), 1)
            A, B = X[:, ia], X[:, ib]
            Bc, Ac = B - B.mean(0), A - A.mean(0)
            beta = (Ac * Bc).sum(0) / (Bc ** 2).sum(0)
            S = A - beta * B
            dS, S1 = np.diff(S, axis=0), S[:-1] - S[:-1].mean(0)
            phi = (S1 * (dS - dS.mean(0))).sum(0) / (S1 ** 2).sum(0)
            res = dS - dS.mean(0) - phi * S1
            se = np.sqrt((res ** 2).sum(0) / (len(dS) - 2) / (S1 ** 2).sum(0))
            tstat = phi / se
            hl = -math.log(2) / np.log1p(np.clip(phi, -0.99, -1e-6))
            ok = (tstat < -3.34) & (hl > 3) & (hl < 40) & (beta > 0)
            for j in np.where(ok)[0]:
                cand.append((tstat[j], ts[ia[j]], ts[ib[j]], beta[j], S[:, j].mean(), S[:, j].std(), hl[j]))
        cand.sort()
        used: dict[str, int] = {}
        chosen = []
        for c in cand:
            if used.get(c[1], 0) >= 2 or used.get(c[2], 0) >= 2:
                continue
            chosen.append(c)
            used[c[1]] = used.get(c[1], 0) + 1
            used[c[2]] = used.get(c[2], 0) + 1
            if len(chosen) == top:
                break
        for tstat, a, b, beta, mu, sd, hl in chosen:
            s = (nxt[a] - beta * nxt[b]).to_numpy()
            z = (s - mu) / sd
            z_prev = (L.loc[end, a] - beta * L.loc[end, b] - mu) / sd
            ra, rb = R.loc[nxt.index, a].to_numpy(), R.loc[nxt.index, b].to_numpy()
            leg = (ra - beta * rb) / (1 + beta)
            pos, path, entry_i, zs = 0, np.zeros(len(z)), None, np.r_[z_prev, z[:-1]]
            for i in range(len(z)):
                zi = zs[i]                                                   # decide on yesterday's close
                newpos = pos
                if pos == 0 and abs(zi) > entry and abs(zi) < stop:
                    newpos, entry_i = -int(np.sign(zi)), i
                elif pos != 0 and (abs(zi) < exit_ or abs(zi) > stop):
                    newpos = 0
                path[i] = newpos * leg[i] - abs(newpos - pos) * COST_BP / 1e4
                if newpos == 0 and pos != 0:
                    trades.append(_trade(m, a, b, tstat, hl, beta, zs[entry_i], nxt.index, entry_i, i, path, risk, fvol))
                    path = np.zeros(len(z))
                pos = newpos
            if pos != 0:
                path[-1] -= COST_BP / 1e4                                    # close at month end
                trades.append(_trade(m, a, b, tstat, hl, beta, zs[entry_i], nxt.index, entry_i, len(z), path, risk, fvol))
    return pd.DataFrame(trades)


def _trade(m, a, b, tstat, hl, beta, z0, idx, i0, i1, path, risk, fvol) -> dict:
    return {"month": m, "a": a, "b": b, "adf_t": tstat, "half_life": hl, "beta": beta, "z_entry": abs(z0), "start": idx[i0],
            "end": idx[min(i1, len(idx) - 1)], "days": i1 - i0, "ret": float(path.sum()),
            "risk_max": max(risk.get((m, a), np.nan), risk.get((m, b), np.nan)),
            "fvol_mean": (fvol.get((m, a), np.nan) + fvol.get((m, b), np.nan)) / 2,
            "path": pd.Series(path[path != 0], index=idx[np.where(path != 0)[0]])}


def pairs_sleeves(T: pd.DataFrame, p_hi: pd.Series, vix: pd.Series, risk_cut: pd.Series, top: int = 20) -> tuple[dict, dict]:
    """Daily P&L of the pairs book (each pair 1/top of capital) -> monthly %, plain, without collapse-prone legs, and
    with meta-labelling: a boosted-tree classifier, refit each January on trades closed before it, predicts whether
    a trade will make money; trades with probability < 0.5 are skipped and the rest sized by 2p - 1 (capped at 1)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    T = T.copy()
    T["p_hi"] = p_hi.reindex(T["start"]).to_numpy()
    T["lvix"] = np.log(vix.reindex(T["start"]).ffill().to_numpy())
    T["risky"] = T["risk_max"] > T["month"].map(risk_cut).astype(float)
    feats = ["adf_t", "half_life", "beta", "z_entry", "risk_max", "fvol_mean", "p_hi", "lvix"]
    T["win"] = (T["ret"] > 0).astype(float)
    T["meta_p"] = np.nan
    yrs = T["start"].dt.year.to_numpy()
    for yr in range(int(yrs.min()) + 2, int(yrs.max()) + 1):
        tr = (T["end"] < pd.Timestamp(f"{yr}-01-01")).to_numpy()
        te = yrs == yr
        if tr.sum() < 300 or not te.any():
            continue
        X = T[feats].to_numpy(float)
        g = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, min_samples_leaf=100, random_state=0).fit(X[tr], T["win"].to_numpy()[tr])
        T.loc[te, "meta_p"] = g.predict_proba(X[te])[:, 1]
    size = {"pairs": np.ones(len(T)), "pairs_nocollapse": (~T["risky"]).astype(float).to_numpy(),
            "pairs_meta": np.where(T["meta_p"].isna(), np.nan, np.clip(2 * T["meta_p"].fillna(0) - 1, 0, 1)) }
    out = {}
    for k, w in size.items():
        daily = pd.concat([p * w[i] / top for i, p in enumerate(T["path"]) if not np.isnan(w[i]) and w[i] > 0]).groupby(level=0).sum()
        out[k] = month_ret(daily).rename(None)
        if k == "pairs_meta":
            first = T.loc[T["meta_p"].notna(), "start"].min()
            out[k] = out[k][out[k].index >= first.to_period("M")]
    stats = {"trades": int(len(T)), "win_rate": float(T["win"].mean()), "avg_trade_pct": float(100 * T["ret"].mean()),
             "win_rate_risky_legs": float(T.loc[T["risky"], "win"].mean()) if T["risky"].any() else None,
             "avg_trade_risky_pct": float(100 * T.loc[T["risky"], "ret"].mean()) if T["risky"].any() else None,
             "meta_auc": None}
    s = T["meta_p"].notna()
    if s.sum() > 100:
        from sklearn.metrics import roc_auc_score
        stats["meta_auc"] = float(roc_auc_score(T.loc[s, "win"], T.loc[s, "meta_p"]))
    return out, stats


# ---------------------------------------------------------------- 6. trend

def trend(P: pd.DataFrame, C: pd.DataFrame, spy: pd.Series, rf: pd.Series, sectors: dict, member: pd.DataFrame) -> dict[str, pd.Series]:
    """Time-series momentum (Moskowitz, Ooi & Pedersen): each month hold each asset long or short by the sign of its
    past-12-month excess return, sized to 40% annual volatility / N; assets are SPY and 11 equal-weighted sector
    baskets of the members. Also a SPY trend filter (SPY when its 12-month excess return is positive, else bills)."""
    R = C.pct_change(fill_method=None)
    mem = member.copy()
    mem.index = pd.DatetimeIndex(mem.index).to_period("M")
    M = mem.reindex(R.index.to_period("M")).fillna(False).to_numpy()
    cols = [c for c in R.columns if c in mem.columns]
    Rm = R[cols].where(pd.DataFrame(M[:, [list(mem.columns).index(c) for c in cols]], index=R.index, columns=cols))
    assets = {"SPY": spy.pct_change()}
    for sec in sorted({sectors.get(c) for c in cols if isinstance(sectors.get(c), str)}):
        assets[sec] = Rm[[c for c in cols if sectors.get(c) == sec]].mean(axis=1)
    D = pd.DataFrame(assets).dropna(how="all")
    ex = D.sub(rf.reindex(D.index).fillna(0), axis=0)
    Mx = ex.groupby(ex.index.to_period("M")).apply(lambda d: (1 + d).prod() - 1) * 100
    vol = D.rolling(63).std().groupby(D.index.to_period("M")).last() * math.sqrt(252)
    sig = np.sign(Mx.rolling(12).sum())
    w = (sig * 0.40 / vol / Mx.shape[1]).clip(-2, 2)
    turn = w.diff().abs().sum(axis=1)
    tsmom = (w.shift(1) * Mx).sum(axis=1, min_count=3) - turn.shift(1) * COST_BP / 100
    on = (Mx["SPY"].rolling(12).sum() > 0).astype(float)
    spy_tr = on.shift(1) * Mx["SPY"] - on.diff().abs().shift(1) * COST_BP / 100
    return {"trend_tsmom": tsmom.iloc[13:], "trend_spy_filter": spy_tr.iloc[13:], "spy": Mx["SPY"]}


# ---------------------------------------------------------------- 7. calendar and regime

def calendar_regime(spy: pd.Series, rf: pd.Series, p_hi: pd.Series) -> dict[str, pd.Series]:
    r = spy.pct_change().dropna()
    f = rf.reindex(r.index).fillna(0)
    ex = r - f
    idx = r.index
    per = idx.to_period("M")
    pos_in_m = pd.Series(1, index=idx).groupby(per).cumcount().to_numpy()
    left = pd.Series(1, index=idx).groupby(per).cumcount(ascending=False).to_numpy()
    tom = ((left == 0) | (pos_in_m < 3)).astype(float)                     # last day + first three days of the month
    nxt = np.r_[idx[1:].to_numpy(), idx[-1].to_numpy()]
    gap = np.busday_count(idx.to_numpy().astype("datetime64[D]"), nxt.astype("datetime64[D]"))
    pre_h = (gap > 1).astype(float)                                         # the session before a weekday holiday
    pre_h[-1] = 0.0
    regime = (p_hi.reindex(idx).shift(1).fillna(0) < 0.5).astype(float)    # decided on yesterday's close
    out = {}
    for k, held in (("cal_turn_of_month", tom), ("cal_pre_holiday", pre_h), ("regime_spy", regime.to_numpy())):
        h = pd.Series(held, index=idx)
        trade = h.diff().abs().fillna(h.iloc[0])
        out[k] = month_ret(h * ex - trade * COST_BP / 1e4)
    return out


# ---------------------------------------------------------------- combination

def combine(sleeves: dict[str, pd.Series], train_end: str, target_vol: float = 10.0) -> tuple[dict, pd.DataFrame]:
    """Pick, on 2005-2018, the best variant of each family (if its Sharpe is above 0.3), weight the picks by inverse
    volatility, minimum variance or HRP estimated on 2005-2018, lever to 10% annual volatility capped at half Kelly,
    and report 2019 on."""
    from sklearn.covariance import LedoitWolf

    from .topo import deflated_sharpe
    D = pd.DataFrame(sleeves).sort_index()
    D = D[D.index.year >= FIRST_YEAR]
    cut = pd.Period(pd.Timestamp(train_end), "M")
    pre, post = D[D.index <= cut], D[D.index > cut]
    fam = lambda k: k.split("_")[0]                                         # noqa: E731
    pre_sr = {k: float(pre[k].mean() / pre[k].std() * math.sqrt(12)) for k in D if pre[k].count() > 36}
    picks = {}
    for k, s in pre_sr.items():
        if k in ("spy", "equal_weight_index"):
            continue
        f = fam(k)
        if s > 0.3 and (f not in picks or s > pre_sr[picks[f]]):
            picks[f] = k
    chosen = sorted(picks.values())
    res = {"pre_test_sharpe_by_sleeve": pre_sr, "chosen": chosen, "weights": {}, "portfolios": {}}
    books = {}
    trials = list(pre_sr.values())
    variants = []
    for tag, cols in (("", chosen), ("_no_varswap", [c for c in chosen if fam(c) != "varswap"])):
        X = pre[cols].dropna()
        S = LedoitWolf().fit(X.to_numpy()).covariance_
        variants += [(n + tag, cols, X, w) for n, w in (("inverse_vol", 1 / np.sqrt(np.diag(S))), ("min_variance", _minvar(S)), ("hrp", _hrp(S)))]
    for name, chosen, X, w in variants:
        w = w / w.sum()
        pr = X.to_numpy() @ w
        mu, sd = pr.mean(), pr.std()
        lev = min(target_vol / (sd * math.sqrt(12)), 0.5 * mu / sd ** 2 * 100 if mu > 0 else 0.0)
        res["weights"][name] = {k: round(float(v), 3) for k, v in zip(chosen, w)}
        book = (post[chosen].fillna(0.0).to_numpy() @ w) * lev
        books[name] = pd.Series(book, index=post.index)
        b = books[name]
        cum = np.log1p(b / 100).cumsum()
        res["portfolios"][name] = {"leverage": round(lev, 2), **sstats(b), "vol_pct_year": float(b.std() * math.sqrt(12)),
                                   "max_drawdown_pct": float(100 * (1 - np.exp((cum - cum.cummax()).min()))),
                                   "pre_test_sharpe": float(mu / sd * math.sqrt(12)),
                                   "deflated_prob": deflated_sharpe(b.mean() / b.std(), len(b), [t / math.sqrt(12) for t in trials])}
    spy = post["spy"].dropna()
    cum = np.log1p(spy / 100).cumsum()
    res["spy_test"] = {**sstats(spy), "max_drawdown_pct": float(100 * (1 - np.exp((cum - cum.cummax()).min())))}
    res["sleeves_test"] = {k: {**sstats(post[k]), "worst_month_pct": float(post[k].min())} for k in D}
    res["sleeves_pre_test_worst_month_pct"] = {k: float(pre[k].min()) for k in D}
    res["sleeve_corr_test"] = post[res["chosen"]].corr().round(2).to_dict() if res["chosen"] else {}
    return res, pd.DataFrame(books)


def run(data: Path, out: Path, train_end: str = TRAIN_END) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    rep = {"built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "train_end": train_end, "warnings": []}
    if pit.former(data) == set():
        rep["warnings"].append("No point-in-time membership in the cache: firms that went bankrupt or were taken over are missing, "
                               "so 'collapse' means a 50% fall of a survivor, and every long-only result is flattered.")
    P, C, member, spy, vix, rf, sectors = load_all(data, out, train_end)
    log.info("data (%.0fs)", time.time() - t0)
    r_spy = np.log(spy).diff()
    fit = (r_spy.index <= pd.Timestamp(train_end)) & r_spy.notna().to_numpy()
    ph, rep["hmm"] = hmm2(r_spy.fillna(0).to_numpy(), fit)
    p_hi = pd.Series(ph, index=spy.index)
    P["p_hi"] = P["month"].map(p_hi.groupby(p_hi.index.to_period("M")).last())
    P = collapse_labels(P, C, spy)
    rep["collapse_model"], P = collapse_model(P, train_end)
    log.info("collapse model (%.0fs)", time.time() - t0)
    rf_m = month_ret(rf)
    sleeves = collapse_sleeves(P, rf_m)
    sleeves = {k: _shift(v) for k, v in sleeves.items()}                    # formation month -> return month
    _, sdf = volpred.spy_timing(data, train_end)
    sleeves.update(vol_carry(spy, vix, p_hi, sdf))
    sleeves.update({k: _shift(v) for k, v in construction(P, C, rf_m).items()})
    log.info("construction (%.0fs)", time.time() - t0)
    T = pairs(P, C, sectors, member)
    risk_cut = P.groupby("month")["pc_gbt"].quantile(0.9)
    ps, rep["pairs"] = pairs_sleeves(T, p_hi, vix, risk_cut)
    sleeves.update(ps)
    log.info("pairs %d trades (%.0fs)", len(T), time.time() - t0)
    sleeves.update(trend(P, C, spy, rf, sectors, member))
    sleeves.update(calendar_regime(spy, rf, p_hi))
    # variance swap P&L is indexed by the month the swap is struck: it accrues the next month
    for k in ("varswap_always", "varswap_timed"):
        sleeves[k] = _shift(sleeves[k])
    rep["combined"], books = combine(sleeves, train_end)
    pd.DataFrame(sleeves).sort_index().to_csv(out / "sleeves_monthly.csv", float_format="%.4f")
    books.to_csv(out / "portfolio_test_monthly.csv", float_format="%.4f")
    P[["month", "ticker", "crash12", "pc_logit", "pc_gbt", "f_gbt"]].to_csv(out / "collapse_probabilities.csv", index=False, float_format="%.4g")
    T.drop(columns=["path"]).to_csv(out / "pair_trades.csv", index=False, float_format="%.5g")
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def _shift(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.PeriodIndex(s.index, freq="M") + 1
    return s
