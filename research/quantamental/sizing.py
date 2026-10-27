"""Slide 5 (sizing, Lopez de Prado): does sizing by volatility and covariance beat equal money across our pots?

Input: the pots' daily returns 2019-01-03 to 2026-10-15 (research/allocator's returns.csv, after each pot's own costs).
At each month's last close, weights come only from the trailing window; they drift with the pots until the next
month; 2 bp per unit moved. 2019 is used only to learn, results from 2020-01-02.

  python sizing.py returns.csv out/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import stats as st

POTS = ["trend", "reversal", "best_ideas", "conviction", "pelosi", "exp1", "exp2", "exp7"]
WINDOW = 126
COST = 0.0002
START = "2020-01-01"
CAP = 0.4


def shrunk_cov(R: np.ndarray, k: float = 0.5) -> np.ndarray:
    """Sample covariance pulled halfway to its diagonal (a fixed-weight stand-in for Ledoit-Wolf)."""
    S = np.cov(R, rowvar=False)
    return (1 - k) * S + k * np.diag(np.diag(S))


def cap(w: np.ndarray, c: float = CAP) -> np.ndarray:
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return np.full(len(w), 1 / len(w))
    w = w / w.sum()
    for _ in range(50):
        over = w > c + 1e-12
        if not over.any():
            break
        extra = (w[over] - c).sum()
        w[over] = c
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += extra * w[free] / w[free].sum()
    return w / w.sum()


def erc(S: np.ndarray, iters: int = 500) -> np.ndarray:
    """Equal risk contributions: every pot adds the same share of the book's variance."""
    n = len(S)
    w = 1 / np.sqrt(np.diag(S))
    w /= w.sum()
    for _ in range(iters):
        rc = w * (S @ w)
        w = w * (rc.mean() / np.maximum(rc, 1e-18)) ** 0.5
        w /= w.sum()
    return w if np.isfinite(w).all() else np.full(n, 1 / n)


def hrp(S: np.ndarray) -> np.ndarray:
    """Hierarchical risk parity (Lopez de Prado 2016): cluster by correlation, split money top-down by inverse variance."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    sd = np.sqrt(np.diag(S))
    C = S / np.outer(sd, sd)
    D = np.sqrt(np.clip((1 - C) / 2, 0, None))
    np.fill_diagonal(D, 0)
    order = list(leaves_list(linkage(squareform(D, checks=False), "single")))
    w = np.ones(len(S))
    items = [order]
    while items:
        items = [i[j:k] for i in items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for a, b in zip(items[::2], items[1::2]):
            va = _cluster_var(S, a)
            vb = _cluster_var(S, b)
            f = 1 - va / (va + vb)
            w[a] *= f
            w[b] *= 1 - f
    return w / w.sum()


def _cluster_var(S, idx):
    s = S[np.ix_(idx, idx)]
    iv = 1 / np.diag(s)
    iv /= iv.sum()
    return float(iv @ s @ iv)


def weights(method: str, R: np.ndarray) -> np.ndarray:
    n = R.shape[1]
    S = shrunk_cov(R)
    vol = np.sqrt(np.diag(S))
    mu = R.mean(axis=0)
    if method == "equal":
        return np.full(n, 1 / n)
    if method == "inverse_vol":
        return cap(1 / vol)
    if method == "risk_parity":
        return cap(erc(S))
    if method == "hrp":
        return cap(hrp(S))
    if method == "conviction_over_vol":   # slide: w ~ conviction / sigma, conviction = trailing Sharpe if positive
        return cap(np.clip(mu / vol, 0, None) / vol)
    if method == "mean_variance":         # slide: w* ~ Sigma^-1 (mu - rf), long only, capped
        return cap(np.linalg.solve(S, mu))
    raise ValueError(method)


METHODS = ["equal", "inverse_vol", "risk_parity", "hrp", "conviction_over_vol", "mean_variance"]


def simulate(ret: pd.DataFrame, method: str) -> tuple[pd.Series, pd.DataFrame]:
    ret = ret.fillna(0.0)
    month_end = ret.index.to_series().groupby(ret.index.to_period("M")).transform("max") == ret.index
    w = None
    out, book, cost = [], {}, 0.0
    for i, d in enumerate(ret.index):
        if w is not None:
            r = ret.iloc[i].values
            gross = float(w @ r)
            w = w * (1 + r) / (1 + gross)
            out.append((d, gross - cost))
            cost = 0.0
        if month_end.iloc[i] and i + 1 >= WINDOW:
            win = ret.iloc[i + 1 - WINDOW:i + 1]
            live = (win != 0).sum().values >= WINDOW // 2      # a pot counts once it has half a window of history
            new = np.zeros(ret.shape[1])
            new[live] = weights(method, win.values[:, live])
            cost = COST * (np.abs(new - w).sum() if w is not None else 1.0)
            w = new
            book[d] = new
    s = pd.Series(dict(out)).sort_index()
    return s, pd.DataFrame(book, index=ret.columns).T


def run(path: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    spy = d["SPY"]
    res, series = {}, {}
    for label, pots in (("all_8_pots", POTS), ("without_exp2", [p for p in POTS if p != "exp2"])):
        for m in METHODS:
            s, book = simulate(d[pots], m)
            s = s[START:]
            series[f"{label}|{m}"] = s
            row = st.summary(s, spy, ppy=252)
            row["avg_weights"] = book[START:].mean().round(3).to_dict()
            res[f"{label}|{m}"] = row
        base = series[f"{label}|equal"]
        for m in METHODS[1:]:
            diff = series[f"{label}|{m}"] - base
            res[f"{label}|{m}"]["vs_equal_ann"] = diff.mean() * 252
            res[f"{label}|{m}"]["vs_equal_t"] = diff.mean() / st.nw_se(diff.values, st.default_lags(len(diff)))
            # Sharpe is unchanged by leverage, so also compare at equal-weight's volatility
            k = base.std() / series[f"{label}|{m}"].std()
            res[f"{label}|{m}"]["ann_at_equal_vol"] = series[f"{label}|{m}"].mean() * k * 252
            dm = series[f"{label}|{m}"] * k - base
            res[f"{label}|{m}"]["at_equal_vol_t"] = dm.mean() / st.nw_se(dm.values, st.default_lags(len(dm)))
    srs = [s.mean() / s.std() for s in series.values()]
    var = float(np.var(srs, ddof=1))
    for k, s in series.items():
        res[k].update({f"dsr_{n}": st.deflated_sharpe(s, n, var)["dsr"] for n in (len(series), 300)})
    spy_s = spy[START:]
    res["SPY"] = st.summary(spy_s, ppy=252)
    pd.DataFrame(series).to_csv(out / "sizing_daily.csv", float_format="%.6g")
    (out / "sizing.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    r = run(sys.argv[1], Path(sys.argv[2]))
    cols = ["ann", "vol", "sharpe", "max_dd", "alpha_ann", "alpha_t", "beta", "vs_equal_ann", "vs_equal_t", "ann_at_equal_vol", "at_equal_vol_t", "dsr_12", "dsr_300"]
    print(pd.DataFrame(r).T.reindex(columns=cols).astype(float).round(3).to_string())
