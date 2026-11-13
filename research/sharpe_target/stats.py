"""Performance stats, Newey-West alpha, and the Harvey-Liu (2015) Sharpe haircut for multiple testing."""
import math

import numpy as np
import pandas as pd
from scipy import stats as st

TD = 252


def perf(r: pd.Series, rf: pd.Series) -> dict:
    r = r.dropna()
    x = r - rf.reindex(r.index).fillna(0)
    eq = (1 + r).cumprod()
    yrs = len(r) / TD
    vol = r.std() * math.sqrt(TD)
    return {"end": float(eq.iloc[-1]), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol": float(vol),
            "sharpe": float(x.mean() * TD / vol) if vol else float("nan"),
            "max_dd": float((eq / eq.cummax() - 1).min()), "years": yrs}


def alpha_nw(r: pd.Series, m: pd.Series, rf: pd.Series, lags: int = 5) -> dict:
    df = pd.concat([r, m, rf], axis=1, keys=["r", "m", "rf"]).dropna()
    y = (df["r"] - df["rf"]).to_numpy()
    X = np.c_[np.ones(len(y)), (df["m"] - df["rf"]).to_numpy()]
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    u = X * (y - X @ b)[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        G = u[L:].T @ u[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtX @ S @ XtX))
    return {"beta": float(b[1]), "alpha": float(b[0] * TD), "alpha_t": float(b[0] / se[0])}


def haircut_sharpe(sr: float, years: float, trials: int, method: str = "bhy", rank: int = 1,
                   pvals_rest: np.ndarray | None = None) -> dict:
    """Harvey & Liu (2015) 'Backtesting': turn an annual Sharpe into a t-stat (SR * sqrt(years)), raise its p-value for
    the number of strategies tried, and turn the raised p back into the Sharpe it implies.
    method: 'bonferroni' (p*M), 'holm' (p*(M-rank+1)), 'bhy' (p*M*c(M)/rank, c(M)=sum 1/k). With only the count of
    trials known (the other p-values unknown) Harvey-Liu treat the strategy as rank 1, which we do too."""
    t = sr * math.sqrt(years)
    p = 2 * st.norm.sf(abs(t))
    M = trials
    if method == "bonferroni":
        padj = min(1.0, p * M)
    elif method == "holm":
        padj = min(1.0, p * (M - rank + 1))
    else:
        c = sum(1 / k for k in range(1, M + 1))
        padj = min(1.0, p * M * c / rank)
    t_adj = st.norm.isf(padj / 2) if padj < 1 else 0.0
    sr_adj = math.copysign(t_adj / math.sqrt(years), sr)
    return {"t": t, "p": p, "p_adj": padj, "sr_haircut": sr_adj,
            "haircut_pct": 100 * (1 - sr_adj / sr) if sr else float("nan")}
