"""Market regimes by K-Means, and how each strategy pot's replay did in each.

Each session gets two features of SPY known at its close: the past FEATURE_DAYS sessions' return and the log of their
realized volatility (annualized; the log keeps the 2020 crash from taking a group to itself). Both are standardized and
split into K groups by K-Means (k-means++ starts, the best of several runs, fixed seed). The groups are named by their
centre's volatility, lowest first: "Calm", "Trending", "Crisis". On SPY since 2017 the data gives exactly that shape:
quiet and rising, more volatile and rising faster, volatile and falling.

A strategy's return on a session is counted in the regime of the previous close, so the regime is known before the day
it is scored on. The clustering itself is fitted on the whole history (a description of the past, not a forecast); how
much that matters is checked by refitting each year on the data up to then and counting the sessions labelled the same.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_DAYS, K, RUNS, SEED = 21, 3, 10, 0
NAMES = ("Calm", "Trending", "Crisis")


def features(spy: pd.Series, days: int = FEATURE_DAYS) -> pd.DataFrame:
    r = np.log(spy).diff()
    return pd.DataFrame({"ret": r.rolling(days).sum() * 252 / days, "vol": np.log(r.rolling(days).std() * np.sqrt(252))}).dropna()


def kmeans(x: np.ndarray, k: int = K, runs: int = RUNS, seed: int = SEED, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """(centres, labels) of the run with the lowest within-group sum of squares."""
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for _ in range(runs):
        c = x[[rng.integers(len(x))]]
        while len(c) < k:                                  # k-means++: far points are likelier starts
            d2 = ((x[:, None, :] - c[None]) ** 2).sum(-1).min(1)
            c = np.vstack([c, x[rng.choice(len(x), p=d2 / d2.sum())]])
        for _ in range(iters):
            lab = ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
            new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        sse = float(((x - c[lab]) ** 2).sum())
        if sse < best[0]:
            best = (sse, c, lab)
    return best[1], best[2]


def name_groups(centres: np.ndarray) -> dict[int, str]:
    """centres in raw units (ret, log vol), named from the quietest to the most volatile."""
    return {int(j): NAMES[i] for i, j in enumerate(np.argsort(centres[:, 1]))}


def fit(f: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Regime name per session, and the centres in raw units."""
    mu, sd = f.mean(), f.std()
    c, lab = kmeans(((f - mu) / sd).to_numpy())
    raw = c * sd.to_numpy() + mu.to_numpy()
    names = name_groups(raw)
    cen = pd.DataFrame(raw, columns=["ret", "vol"], index=[names[j] for j in range(len(raw))])
    return pd.Series([names[j] for j in lab], index=f.index), cen


def stability(f: pd.DataFrame, labels: pd.Series, first_year: int) -> float:
    """Share of sessions from first_year on whose label from a fit on data up to the end of the previous year (assigned
    to the nearest of that fit's centres) matches the full-history label."""
    same, n = 0, 0
    for y in sorted({d.year for d in f.index if d.year >= first_year}):
        past = f[f.index.year < y]
        if len(past) < 250:
            continue
        _, cen = fit(past)
        mu, sd = past.mean(), past.std()
        z = ((f[f.index.year == y] - mu) / sd).to_numpy()
        cz = ((cen - mu) / sd).to_numpy()
        lab = np.array(cen.index)[((z[:, None, :] - cz[None]) ** 2).sum(-1).argmin(1)]
        same += int((lab == labels[f.index.year == y].to_numpy()).sum())
        n += len(lab)
    return same / n if n else float("nan")


def by_regime(returns: pd.DataFrame, labels: pd.Series) -> dict:
    """Per strategy and regime: sessions, yearly return (mean x 252), volatility, Sharpe, and share of sessions up."""
    lag = labels.shift(1).reindex(returns.index)          # the regime known at the previous close
    out = {}
    for k in returns.columns:
        r = returns[k]
        row = {}
        for name in NAMES:
            x = r[(lag == name) & r.notna()]
            if len(x) < 10:
                continue
            vol = x.std() * np.sqrt(252)
            row[name] = {"days": int(len(x)), "ann_ret_pct": round(100 * x.mean() * 252, 1), "vol_pct": round(100 * vol, 1),
                         "sharpe": round(float(x.mean() * 252 / vol), 2) if vol else 0.0, "up_pct": round(100 * float((x > 0).mean()), 1),
                         "total_pct": round(100 * float((1 + x).prod() - 1), 1)}
        out[k] = row
    return out


def run(spy: pd.Series, returns: pd.DataFrame, since: str) -> dict:
    f = features(spy)
    labels, cen = fit(f)
    since_lab = labels[labels.index >= since]
    episodes = []
    for _, g in since_lab.groupby((since_lab != since_lab.shift()).cumsum()):
        episodes.append({"regime": g.iloc[0], "from": str(g.index[0].date()), "to": str(g.index[-1].date()), "days": len(g)})
    spy_since = spy[spy.index >= since]
    return {
        "today": {"date": str(labels.index[-1].date()), "regime": labels.iloc[-1], "ret_pct": round(100 * f["ret"].iloc[-1], 1),
                  "vol_pct": round(100 * np.exp(f["vol"].iloc[-1]), 1)},
        "centres": {n: {"ret_pct": round(100 * cen.at[n, "ret"], 1), "vol_pct": round(100 * np.exp(cen.at[n, "vol"]), 1)} for n in NAMES if n in cen.index},
        "share_pct": {n: round(100 * float((labels == n).mean()), 1) for n in NAMES},
        "fit_from": str(f.index[0].date()), "sessions": len(f),
        "stable_pct": round(100 * stability(f, labels, int(since[:4])), 1),
        "by_regime": by_regime(returns, labels),
        "episodes": [x for x in episodes if x["days"] >= 5],
        "timeline": [{"date": str(d.date()), "spy": round(float(spy_since.get(d, np.nan)), 2), "regime": since_lab.at[d],
                      "ret": round(float(f.at[d, "ret"]), 3), "vol": round(float(np.exp(f.at[d, "vol"])), 3)} for d in since_lab.index],
        "scatter": [[round(float(a), 3), round(float(np.exp(b)), 3), lab] for (a, b), lab in zip(f[["ret", "vol"]].to_numpy()[::2], labels.to_numpy()[::2])],
    }
