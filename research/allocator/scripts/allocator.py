"""The allocator: a pot that holds no stocks of its own and instead splits its money across the other pots, by regime.

At each rebalance (every week's last close, and any close where the regime changes) it looks at the market regime known
at that close (regimes.py, fitted only on data up to the previous year end) and, for every pot, at how that pot did in
that same regime on all the days before. Pots that made money in this regime get money, in proportion to their return
per unit of variance there (a Kelly-style score), each capped at CAP; pots that lost in this regime get nothing. If none
made money, the pot sits in cash. Until a pot has MIN_DAYS days in a regime, its record over all regimes is used.

Moving money between pots costs SWITCH_COST per unit traded, on top of each pot's own costs (already in its returns).
Nothing here peeks ahead: the regime, the scores and the weights at a close come only from earlier data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import regimes

CAP, MIN_DAYS, SWITCH_COST, WARMUP = 0.4, 40, 0.0002, 250


def point_in_time_regimes(spy: pd.Series) -> pd.Series:
    """Regime at each close, from a K-Means fit on the features up to the end of the previous year (no look-ahead)."""
    f = regimes.features(spy)
    out = pd.Series(index=f.index, dtype=object)
    for y in sorted(set(f.index.year)):
        past = f[f.index.year < y]
        if len(past) < 500:
            continue
        _, cen = regimes.fit(past)
        mu, sd = past.mean(), past.std()
        z = ((f[f.index.year == y] - mu) / sd).to_numpy()
        cz = ((cen - mu) / sd).to_numpy()
        out[f.index.year == y] = np.array(cen.index)[((z[:, None, :] - cz[None]) ** 2).sum(-1).argmin(1)]
    return out.dropna()


def scores(hist: pd.DataFrame, lab: pd.Series, regime: str | None, use_regime: bool = True) -> pd.Series:
    """Return per unit of variance of each pot, in this regime (or overall), from the rows given."""
    out = {}
    for k in hist.columns:
        r = hist[k].dropna()
        x = r[lab.reindex(r.index) == regime] if use_regime and regime is not None else r
        if len(x) < MIN_DAYS:
            x = r
        if len(x) < MIN_DAYS or x.var() <= 0:
            continue
        out[k] = x.mean() / x.var()
    return pd.Series(out, dtype=float)


def weights(sc: pd.Series, cap: float = CAP) -> dict[str, float]:
    """Positive scores to weights summing to at most 1, none above cap (the rest is cash)."""
    pos = sc[sc > 0]
    if pos.empty:
        return {}
    w = pos / pos.sum()
    for _ in range(10):                       # cap and hand the excess to the uncapped
        over = w > cap
        if not over.any():
            break
        extra = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        if not free.any():
            break
        w[free] += extra * w[free] / w[free].sum()
    return {k: float(v) for k, v in w.items() if v > 1e-6}


def run(rets: pd.DataFrame, lab: pd.Series, start: str, use_regime: bool = True, oracle: bool = False) -> dict:
    """Daily returns of the allocator from start. rets: one column per pot (fractions, after the pot's own costs).
    lab: regime per close (point in time). oracle: use the whole history's scores (look-ahead, for reference only)."""
    dates = rets.index[rets.index >= start]
    full = {g: scores(rets, lab, g, use_regime) for g in regimes.NAMES} if oracle else None
    w: dict[str, float] = {}
    last_reg, out, books = None, [], []
    for i, d in enumerate(dates):
        ret = sum(x * rets.at[d, k] for k, x in w.items() if pd.notna(rets.at[d, k]))
        reg = lab.get(d)
        if i == 0 or reg != last_reg or (i + 1 < len(dates) and dates[i + 1].isocalendar()[1] != d.isocalendar()[1]):
            sc = full[reg] if oracle else scores(rets.loc[:d], lab.loc[:d], reg, use_regime)
            nw = weights(sc)
            ret -= SWITCH_COST * sum(abs(nw.get(k, 0) - w.get(k, 0)) for k in set(nw) | set(w))
            w, last_reg = nw, reg
            books.append({"date": str(d.date()), "regime": reg, "w": {k: round(v, 3) for k, v in w.items()}})
        out.append((d, ret))
    return {"returns": pd.Series(dict(out)), "books": books}
