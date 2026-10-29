"""The carousel as one process: steps 0 to 6 run in order to look for an edge, walk-forward, on a universe of
companies (S&P 500 stocks) or of markets (ETFs across sectors, countries, bonds and commodities).

0. Two camps    the analyst's checks (fundamentals, filings) and the quant's price signals, side by side.
1. The idea     every check becomes a number for every name, every month, then a z-score across the universe.
2. The formula  each signal's information coefficient (IC) and breadth: IR ~ IC x sqrt(breadth).
3. The signal   keep only signals whose IC was reliably non-zero in the past (t > 2, known at the time) and
                combine them weighted by that IC: score_i = sum_k w_k z_ik.
4. The data     point-in-time only: fundamentals from their filing day, returns strictly after.
5. Sizing       equal weights vs conviction / volatility vs a covariance-aware book (Sigma^-1 alpha).
6. The traps    out of sample only (each month uses the past alone), a hindsight run to show the overfitting
                gap, costs, deflated Sharpe over every trial.
Then an "edge card" for each company or market: where it ranks today and which signals drive that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest
import stats as st

MIN_TRAIN = 36      # months of IC history before a signal can be chosen
T_KEEP = 2.0        # a signal is used only while its past IC t-stat is above this


def zs(x: pd.Series) -> pd.Series:
    return backtest.zs(x.replace([np.inf, -np.inf], np.nan))


# ---------------------------------------------------------------- steps 0 and 1: ideas as numbers

STOCK_SIGNALS = {
    # analyst camp: the statements and the filings
    "value: book to price": ("analyst", lambda p: p["bm"]),
    "value: earnings yield": ("analyst", lambda p: p["ni_0"] / p["mcap"]),
    "value: cash-flow yield": ("analyst", lambda p: p["cfo_0"] / p["mcap"]),
    "Piotroski F-score": ("analyst", lambda p: p["fscore"]),
    "quality: profitability": ("analyst", lambda p: p["prof"]),
    "quality: growth": ("analyst", lambda p: p["growth"]),
    "quality: safety": ("analyst", lambda p: p["safety"]),
    "quality: payout": ("analyst", lambda p: p["payout"]),
    "low asset growth": ("analyst", lambda p: -(p["assets_0"] / p["assets_1"] - 1)),
    "low accruals": ("analyst", lambda p: (p["cfo_0"] - p["ni_0"]) / p["assets_0"]),
    "10-K less negative than last year (LM)": ("analyst", lambda p: -p["d_lm_neg_sh"] if "d_lm_neg_sh" in p else np.nan),
    # quant camp: prices
    "momentum 12-1": ("quant", lambda p: p["mom12_1"]),
    "1-month reversal": ("quant", lambda p: -p["rev1"]),
    "low volatility": ("quant", lambda p: -p["ivol"]),
    "low beta": ("quant", lambda p: -p["beta"]),
}


def signals(P: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, list[str]]:
    P = P.copy()
    names = []
    for k, (_, f) in spec.items():
        try:
            raw = f(P)
        except KeyError:
            continue
        if not isinstance(raw, pd.Series) or raw.notna().mean() < 0.3:
            continue
        P[k] = raw.replace([np.inf, -np.inf], np.nan).groupby(P["month"]).transform(zs)
        names.append(k)
    return P, names


# ---------------------------------------------------------------- step 2: IC and breadth

def ic_table(P: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = {}
    for m, g in P.groupby("month"):
        g = g[g["ret"].notna()]
        rows[m] = {k: g[k].corr(g["ret"], method="spearman") if g[k].notna().sum() >= max(8, len(g) // 3) else np.nan for k in names}
    return pd.DataFrame(rows).T.sort_index()


def ic_summary(ic: pd.DataFrame, P: pd.DataFrame, spec: dict) -> pd.DataFrame:
    out = {}
    for k in ic.columns:
        s = ic[k].dropna()
        n = P.dropna(subset=[k, "ret"]).groupby("month").size().mean()
        t = s.mean() / st.nw_se(s.values, st.default_lags(len(s))) if len(s) > 12 else np.nan
        out[k] = {"camp": spec[k][0], "ic": s.mean(), "ic_t": t, "names": n, "law_ir_naive": s.mean() * np.sqrt(12 * n),
                  "ir_of_ic": s.mean() / s.std() * np.sqrt(12), "months": len(s)}
    return pd.DataFrame(out).T


# ---------------------------------------------------------------- step 3: combine, walk forward

def weights_at(ic: pd.DataFrame, m: pd.Timestamp, hindsight: bool = False) -> pd.Series:
    """IC weights known at month m: from formation months strictly before m (their returns have arrived)."""
    past = ic if hindsight else ic[ic.index < m]
    w = {}
    for k in past.columns:
        s = past[k].dropna()
        if len(s) < MIN_TRAIN:
            continue
        t = s.mean() / st.nw_se(s.values, st.default_lags(len(s)))
        if abs(t) > T_KEEP:
            w[k] = s.mean()
    return pd.Series(w, dtype=float)


def composite(P: pd.DataFrame, ic: pd.DataFrame, start: str, hindsight: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, chosen = [], {}
    for m, g in P[P["month"] >= start].groupby("month"):
        w = weights_at(ic, m, hindsight)
        chosen[m] = w
        if w.empty:
            continue
        z = g[w.index].fillna(0.0)
        g = g.assign(score=zs(z @ w / w.abs().sum()), ic_combo=w.abs().sum() / np.sqrt(len(w)))
        rows.append(g)
    return (pd.concat(rows, ignore_index=True) if rows else P.iloc[:0].assign(score=np.nan)), pd.DataFrame(chosen).T


# ---------------------------------------------------------------- step 5: sizing

def _single_index_inverse(alpha: np.ndarray, beta: np.ndarray, resid: np.ndarray, mkt_var: float) -> np.ndarray:
    """Sigma^-1 alpha for Sigma = mkt_var * beta beta' + diag(resid^2) (Woodbury), so no 500 x 500 inverse is needed."""
    d = 1 / resid**2
    da, db = d * alpha, d * beta
    return da - db * (beta @ da) / (1 / mkt_var + beta @ db)


def book_weights(g: pd.DataFrame, how: str, long_only: bool) -> pd.Series:
    g = g.dropna(subset=["score"])
    sig = (g["ivol"].fillna(g["ivol"].median()).clip(lower=0.05) / np.sqrt(12)).values      # monthly residual vol
    alpha = g["ic_combo"].values * sig * g["score"].values                                    # Grinold: alpha = IC x vol x score
    if how == "equal":
        r = g["score"].rank(pct=True)
        w = (r >= 0.7).astype(float) - (0 if long_only else (r <= 0.3).astype(float))
        w = w.values
    elif how == "conviction_over_vol":
        w = alpha / sig**2
    else:   # covariance aware
        beta = g["beta"].fillna(1.0).clip(-1, 3).values
        w = _single_index_inverse(alpha, beta, sig, (0.15 / np.sqrt(12)) ** 2)
    w = pd.Series(w, index=g["ticker"].values)
    if long_only:
        w = w.clip(lower=0)
        w = w[w > 0]
        cap = 0.05 if len(g) > 60 else 0.25
        for _ in range(20):
            w = (w / w.sum()).clip(upper=cap)
        return w / w.sum()
    pos, neg = w.clip(lower=0), w.clip(upper=0)
    return pos / pos.sum() + (neg / -neg.sum() if neg.sum() < 0 else 0)


def run_books(C: pd.DataFrame, cost: float, borrow: float) -> dict[str, pd.Series]:
    out = {}
    for long_only in (False, True):
        for how in ("equal", "conviction_over_vol", "covariance"):
            res, prev = {}, pd.Series(dtype=float)
            for m, g in C.groupby("month"):
                g = g[g["ret"].notna()]
                if len(g) < 10:
                    continue
                w = book_weights(g, how, long_only)
                r = float((w * g.set_index("ticker")["ret"].reindex(w.index)).sum())
                r -= cost * w.sub(prev, fill_value=0).abs().sum() + (0 if long_only else borrow)
                res[m + pd.offsets.MonthEnd(1)] = r
                prev = w
            label = ("long only" if long_only else "long-short") + f", {how.replace('_', ' ')}"
            out[label] = pd.Series(res).sort_index()
    return out


# ---------------------------------------------------------------- step 6 + cards

def cards(C: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    m = C["month"].max()
    g = C[C["month"] == m].copy()
    w = chosen.loc[m].dropna()
    contrib = g[w.index].fillna(0.0).mul(w / w.abs().sum())
    g["drivers"] = contrib.apply(lambda r: ", ".join(f"{k} {v:+.2f}" for k, v in r.reindex(r.abs().sort_values(ascending=False).index[:3]).items()), axis=1)
    g["rank_pct"] = g["score"].rank(pct=True)
    g["exp_alpha_yr"] = g["ic_combo"] * g["ivol"] * g["score"]
    return g.sort_values("score", ascending=False)[["month", "ticker", "score", "rank_pct", "exp_alpha_yr", "drivers"]]


def run(P: pd.DataFrame, spec: dict, spy: pd.Series, start: str, cost: float, borrow: float, F=None) -> dict:
    P, names = signals(P, spec)
    ic = ic_table(P, names)
    C, chosen = composite(P, ic, start)
    S = run_books(C, cost, borrow)
    H, _ = composite(P, ic, start, hindsight=True)
    hind = run_books(H, cost, borrow)
    S["hindsight (trap): long-short, equal"] = hind["long-short, equal"]
    S["hindsight (trap): long only, equal"] = hind["long only, equal"]
    every = {}
    for m, g in P[P["month"] >= start].groupby("month"):
        g = g[g["ret"].notna()]
        every[m + pd.offsets.MonthEnd(1)] = g["ret"].mean() - cost * 0.1
    S["every name, equal weight (baseline)"] = pd.Series(every).sort_index()
    R = backtest.evaluate(S, spy, F, prior_trials=300 + len(names))
    use = chosen.notna().mean().sort_values(ascending=False) if len(chosen) else pd.Series(dtype=float)
    sign = np.sign(chosen.mean()) if len(chosen) else pd.Series(dtype=float)
    return {"ic": ic_summary(ic, P, spec), "results": R, "signal_use": pd.DataFrame({"share_of_months_used": use, "sign": sign}),
            "cards": cards(C, chosen) if len(C) else pd.DataFrame(), "series": S, "chosen": chosen}
