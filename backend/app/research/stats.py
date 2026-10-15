"""Statistical building blocks. Every function takes data that already ends at the cutoff;
none of them looks beyond the last row it is given."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def align(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    df = df[(df["a"] > 0) & (df["b"] > 0)]
    return df.sort_index()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()


def window_corr(rets: pd.DataFrame, window: int) -> float | None:
    tail = rets.tail(window)
    if len(tail) < max(20, int(0.9 * window)):
        return None
    if tail["a"].std() == 0 or tail["b"].std() == 0:
        return None
    return float(tail["a"].corr(tail["b"]))


def cumulative_log_gap(prices: pd.DataFrame, window: int) -> float | None:
    """ln(PA_t/PA_{t-w}) - ln(PB_t/PB_{t-w}) using the last w+1 closes."""
    if len(prices) < window + 1:
        return None
    seg = prices.tail(window + 1)
    la = np.log(seg["a"].iloc[-1] / seg["a"].iloc[0])
    lb = np.log(seg["b"].iloc[-1] / seg["b"].iloc[0])
    return float(la - lb)


def rolling_corr_series(rets: pd.DataFrame, window: int) -> pd.Series:
    return rets["a"].rolling(window, min_periods=int(0.9 * window)).corr(rets["b"]).dropna()


@dataclass
class HedgeFit:
    beta: float
    alpha: float
    resid_mean: float
    resid_std: float
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    n: int


def fit_hedge_ratio(log_prices: pd.DataFrame) -> HedgeFit | None:
    """OLS of log PA on log PB over the supplied (formation) window only."""
    if len(log_prices) < 60:
        return None
    x, y = log_prices["b"].to_numpy(), log_prices["a"].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return HedgeFit(
        beta=float(coef[1]),
        alpha=float(coef[0]),
        resid_mean=float(resid.mean()),
        resid_std=float(resid.std(ddof=1)),
        formation_start=log_prices.index[0],
        formation_end=log_prices.index[-1],
        n=len(log_prices),
    )


def out_of_sample_zscores(prices: pd.DataFrame, formation: int = 252, gap: int = 63) -> tuple[HedgeFit | None, pd.Series]:
    """Fit the hedge ratio on [t-formation-gap, t-gap) and score the spread on the last `gap`
    days with the formation-period mean/std, so the z-scores are never in-sample."""
    lp = np.log(prices)
    if len(lp) < formation + gap:
        return None, pd.Series(dtype=float)
    form = lp.iloc[-(formation + gap):-gap]
    fit = fit_hedge_ratio(form)
    if fit is None or fit.resid_std == 0:
        return fit, pd.Series(dtype=float)
    test = lp.iloc[-gap:]
    spread = test["a"] - (fit.alpha + fit.beta * test["b"])
    return fit, (spread - fit.resid_mean) / fit.resid_std


def rolling_oos_zscore(prices: pd.DataFrame, formation: int = 252, step: int = 21) -> pd.DataFrame:
    """Walk-forward z-score series for charts: the hedge ratio is re-fitted every `step` days on the
    preceding `formation` days and applied only to the following `step` days."""
    lp = np.log(prices)
    rows = []
    for start in range(formation, len(lp), step):
        fit = fit_hedge_ratio(lp.iloc[start - formation:start])
        if fit is None or fit.resid_std == 0:
            continue
        seg = lp.iloc[start:start + step]
        spread = seg["a"] - (fit.alpha + fit.beta * seg["b"])
        for ts, s in spread.items():
            rows.append({"date": ts, "spread": float(s), "z": float((s - fit.resid_mean) / fit.resid_std), "beta": fit.beta})
    return pd.DataFrame(rows)


def engle_granger_pvalue(prices: pd.DataFrame) -> float | None:
    """Engle-Granger two-step cointegration test on log prices (statsmodels.coint, MacKinnon
    p-values). Cointegration is tested on price levels, never inferred from return correlation."""
    if len(prices) < 120:
        return None
    lp = np.log(prices)
    try:
        _, p, _ = coint(lp["a"], lp["b"], trend="c", autolag="aic")
    except Exception:  # numerical failure on degenerate input
        return None
    return float(p) if np.isfinite(p) else None


def beta_to_benchmark(rets: pd.Series, bench: pd.Series, window: int) -> float | None:
    df = pd.concat([rets.rename("x"), bench.rename("m")], axis=1, join="inner").dropna().tail(window)
    if len(df) < max(60, int(0.8 * window)) or df["m"].var() == 0:
        return None
    return float(df["x"].cov(df["m"]) / df["m"].var())


def annualized_vol(rets: pd.Series, window: int) -> float | None:
    tail = rets.dropna().tail(window)
    if len(tail) < max(60, int(0.8 * window)):
        return None
    return float(tail.std(ddof=1) * np.sqrt(252))


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
