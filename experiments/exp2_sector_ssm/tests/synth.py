"""Synthetic prices for tests: correlated random walks over business days."""
import numpy as np
import pandas as pd

from exp2.config import DEFAULT


def prices(n=3200, seed=0, start="2010-01-04"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    mkt = rng.normal(0.0003, 0.01, n)
    out = {}
    for k, t in enumerate([DEFAULT.target, *DEFAULT.constituents, *DEFAULT.related]):
        beta = 0.5 + (k % 5) * 0.2
        r = beta * mkt + rng.normal(0, 0.008, n)
        p = 50 * np.exp(np.cumsum(r))
        out[t] = pd.DataFrame({"close": p, "adj_close": p, "adj_volume": rng.integers(1e6, 5e6, n).astype(float),
                               "div_cash": 0.0, "split_factor": 1.0}, index=dates)
        out[t].index.name = "date"
    return out
