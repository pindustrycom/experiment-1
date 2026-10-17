import numpy as np
import pandas as pd

from exp2 import tech


def prices(n=2000, seed=0, vol=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2004-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, vol, n))), index=idx)


def test_vol_timing_tracks_volatility():
    p = prices()
    vi = pd.Series(16.0, index=p.index)
    df = tech.vol_timing(p, vi)
    f = df["fvol"].dropna()
    assert len(f) > 20 and abs(np.exp(f).median() - 0.01) < 0.003


def test_held_when_calm_is_cash_when_turbulent():
    p = prices()
    rf = pd.Series(0.0, index=p.index)
    calm = tech.held_when_calm(p, rf, pd.Series(0.0, index=p.index))
    out = tech.held_when_calm(p, rf, pd.Series(1.0, index=p.index))
    assert calm.abs().sum() > 0 and out.iloc[1:].abs().max() < 1e-9
