import math

import numpy as np
import pandas as pd

from exp2 import topo


def test_rips_square_has_one_loop():
    # a unit square: the loop is born when the 4 sides appear (1) and dies when a diagonal fills it (sqrt 2)
    P = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    h0, h1 = topo.rips_persistence(D, eps_max=2.0)
    assert np.allclose(sorted(h0), [1, 1, 1])
    assert h1.shape == (1, 2) and np.allclose(h1[0], [1, math.sqrt(2)])


def test_rips_circle_vs_blob():
    rng = np.random.default_rng(0)
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    circ = np.c_[np.cos(a), np.sin(a)] + rng.normal(0, 0.02, (24, 2))
    blob = rng.normal(0, 0.5, (24, 2))
    pers = []
    for P in (circ, blob):
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        _, h1 = topo.rips_persistence(D, eps_max=3.0)
        pers.append((h1[:, 1] - h1[:, 0]).max())
    assert pers[0] > 0.8 and pers[0] > 3 * pers[1]       # one big loop on the circle, only small ones in the blob


def _returns(T=900, seed=0):
    """Two correlated clusters plus idiosyncratic noise that partly reverses the next day."""
    rng = np.random.default_rng(seed)
    names = topo.UNIVERSE[:12]
    f = rng.normal(0, 1, (T, 2))
    e = rng.normal(0, 1, (T, 12))
    e[1:] -= 0.3 * e[:-1]
    X = np.where(np.arange(12) < 6, f[:, :1], f[:, 1:]) + e
    return pd.DataFrame(X, index=pd.bdate_range("2012-01-02", periods=T), columns=names)


def test_graph_residual_trades_planted_reversal():
    R = _returns()
    W = topo.signals(R, 60, 1, [(0.5, 3)])
    for k in ("xs", "pca", "graph_a0.5_J3", "graphres_a0.5_J3"):
        w = W[k][100]
        assert abs(w.sum()) < 1e-9 and abs(np.abs(w).sum() - 1) < 1e-9
    p = topo.backtest(W["graph_a0.5_J3"], R, cost_bp=0)
    assert topo.stats(p.iloc[100:])["sharpe"] > 2       # the planted reversal is found
    # the graph separates the two clusters better than the cross-sectional mean
    assert topo.stats(p.iloc[100:])["sharpe"] > topo.stats(topo.backtest(W["xs"], R, cost_bp=0).iloc[100:])["sharpe"]


def test_backtest_timing_and_costs():
    R = pd.DataFrame({"A": [0.0, 1.0, 2.0, 3.0], "B": [0.0, -1.0, -2.0, -3.0]}, index=pd.bdate_range("2020-01-01", periods=4))
    W = np.array([[0.5, -0.5]] * 4)
    p = topo.backtest(W, R, cost_bp=0)
    assert list(p) == [0.0, 1.0, 2.0, 3.0]                  # weights set at close t earn session t+1
    pc = topo.backtest(W, R, cost_bp=10)
    assert math.isclose(p.iloc[1] - pc.iloc[1], 0.1)        # 1 unit traded at 10 bp = 0.1%


def test_run_end_to_end(tmp_path):
    R = _returns(T=700)
    lp = R.cumsum() / 100
    (tmp_path / "prices").mkdir()
    for c in R.columns:
        pd.DataFrame({"date": R.index.strftime("%Y-%m-%d"), "close": 50 * np.exp(lp[c]), "adj_close": 50 * np.exp(lp[c])}).to_csv(tmp_path / "prices" / f"{c}.csv", index=False)
    spy = R.mean(axis=1).cumsum() / 100
    pd.DataFrame({"date": R.index.strftime("%Y-%m-%d"), "close": np.exp(spy), "adj_close": np.exp(spy)}).to_csv(tmp_path / "prices" / "SPY.csv", index=False)
    grid = {"window": (60,), "lookback": (1,), "alpha": (0.5,), "J": (1, 3)}
    rep = topo.run(tmp_path, tmp_path / "out", fetch=False, grid=grid, train_end="2013-06-30", universe=list(R.columns))
    assert {"xs", "graph", "graph_topo", "graph_res_topo_vol", "spy"} <= set(rep["results"])
    assert rep["results"]["graph"]["test"]["sharpe"] > 1
    assert (tmp_path / "out" / "index.html").exists()
