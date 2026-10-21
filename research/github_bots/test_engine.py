import numpy as np
import pandas as pd

import strategies
from engine import Rules, daily, simulate, stats


def bars(n=400, seed=1, freq="1h"):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.001, n))
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.004, n)))
    idx = pd.date_range("2019-01-01", periods=n, freq=freq)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": rng.uniform(1, 10, n)}, index=idx)


def flat(n):
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)


def test_fill_next_open_and_signal_exit():
    d = flat(6)
    d.loc[d.index[2], ["open", "high", "close"]] = [110, 110, 110]
    d.loc[d.index[3], ["open", "high", "low", "close"]] = [120, 121, 119, 120]
    e = pd.Series(False, index=d.index)
    e.iloc[0] = True
    x = pd.Series(False, index=d.index)
    x.iloc[2] = True
    r, t = simulate(d, Rules(e, x), 1440, 0.0)
    assert t[0][:3] == (1, 3, 1) and abs(t[0][3] - 0.2) < 1e-12                       # in at bar 1 open (100), out at bar 3 open (120)
    assert abs((1 + r).prod() - 1.2) < 1e-12


def test_stop_beats_roi_and_costs():
    d = flat(4)
    d.loc[d.index[2], ["high", "low"]] = [106, 94]
    e = pd.Series([True, False, False, False], index=d.index)
    r, t = simulate(d, Rules(e, roi={0: 0.05}, stoploss=-0.05), 1440, 10.0)
    assert len(t) == 1 and abs(t[0][3] - (-0.05 - 0.002)) < 1e-12
    assert abs((1 + r).prod() - (1 - 0.001) * (0.95 - 0.001)) < 1e-9


def test_roi_table_by_age():
    d = flat(5)
    d.loc[d.index[3], "high"] = 101.5
    e = pd.Series([True, False, False, False, False], index=d.index)
    _, t = simulate(d, Rules(e, roi={0: 0.05, 2880: 0.01}), 1440, 0.0)
    assert t[0][1] == 3 and abs(t[0][3] - 0.01) < 1e-12


def test_short_and_flat_at():
    d = flat(4)
    d.loc[d.index[2], ["low", "close"]] = [90, 90]
    es = pd.Series([True, False, False, False], index=d.index)
    fl = pd.Series([False, False, True, False], index=d.index)
    _, t = simulate(d, Rules(pd.Series(False, index=d.index), entry_short=es, flat_at=fl), 1440, 0.0)
    assert t[0][:3] == (1, 2, -1) and abs(t[0][3] - 0.1) < 1e-12 and len(t) == 1


def test_every_strategy_runs():
    for name, (build, _, _, mk) in strategies.CATALOG.items():
        d = bars(3000, freq="5min" if any("5m" in v for v in mk.values()) else "1h")
        r, t = simulate(d, build(d), 60, 2.0)
        s = stats(daily(r), daily(d.close.pct_change().fillna(0)), 365, t, d.index, 60)
        assert np.isfinite(s["sharpe"]), name
