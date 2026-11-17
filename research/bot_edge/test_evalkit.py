import numpy as np
import pandas as pd

import evalkit as ek


def _bars(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2019-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": np.r_[100, c[:-1]], "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0}, idx)


def test_shift_test_detects_foresight_and_not_noise():
    d = _bars()
    r = d.close.pct_change().fillna(0).to_numpy()
    _, p_good, z_good = ek.shift_test(np.sign(r), r, 24)          # knows the bar's move
    rng = np.random.default_rng(1)
    ps = [ek.shift_test(rng.integers(0, 2, len(r)).astype(float), r, 24)[1] for _ in range(40)]
    assert p_good < 0.01 and z_good > 10
    assert 0.2 < np.mean(ps) < 0.8                                 # noise: p-values spread around 0.5


def test_trade_replay_matches_compounded_trade_returns():
    d = _bars(500)
    trades = [(10, 50, 1, d.open.iloc[10], d.open.iloc[50]), (100, 140, -1, d.open.iloc[100], d.open.iloc[140])]
    net, gross, pos = ek.bar_returns_from_trades(d, trades, 0.001)
    want = (d.open.iloc[50] / d.open.iloc[10]) * (2 - d.open.iloc[140] / d.open.iloc[100]) * (1 - 0.001) ** 4
    assert abs((1 + net).prod() - want) < 1e-9
    assert pos.iloc[11:51].eq(1).all() and pos.iloc[101:141].eq(-1).all()


def test_holm():
    adj = ek.holm(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06])
