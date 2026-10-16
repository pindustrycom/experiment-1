import numpy as np
import pandas as pd

from exp2 import oria, topo


def test_hold_averages_last_signals():
    W = np.eye(3)[[0, 1, 2, 0]]
    H = oria.hold(W, 2)
    assert np.allclose(H[1], [0.5, 0.5, 0]) and np.allclose(H[3], [0.5, 0, 0.5])


def test_loewdin_decorrelates():
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    P = np.c_[a + 0.3 * rng.normal(size=2000), a + 0.3 * rng.normal(size=2000), rng.normal(size=2000)]
    Q = P @ oria.loewdin(P, shrink=0.0)
    C = np.corrcoef(Q.T)
    assert np.abs(C - np.eye(3)).max() < 1e-6


def test_etf_shock_buys_laggards():
    rng = np.random.default_rng(1)
    names = topo.SECTORS["tech"][:4]
    idx = pd.bdate_range("2015-01-01", periods=200)
    e = rng.normal(0, 1, 200)
    X = e[:, None] + 0.2 * rng.normal(size=(200, 4))
    e[150], X[150] = 6.0, [8.0, 6.0, 6.0, 4.0]          # a shock: the first stock overshoots, the last lags
    R = pd.DataFrame(X, index=idx, columns=names)
    Ws, We = oria.etf_shock(R, pd.DataFrame({"XLK": e}, index=idx), k=3.0)
    assert We[150, 0] == 1 and Ws[150, 0] < 0 < Ws[150, 3]
    assert np.abs(We[:150]).sum() <= 3                   # only a few ordinary days pass a 3-sd threshold


def test_run_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    names = topo.SECTORS["tech"][:6] + topo.SECTORS["financials"][:6]
    idx = pd.bdate_range("2012-01-02", periods=800)
    (tmp_path / "prices").mkdir()
    f = rng.normal(0, 1, (800, 1))
    for t in [*names, "SPY", "XLK", "XLF"]:
        r = 0.01 * (f[:, 0] + rng.normal(0, 1, 800))
        c = 50 * np.exp(np.cumsum(r))
        o = c * np.exp(0.003 * rng.normal(size=800))
        pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "open": o, "close": c, "adj_close": c}).to_csv(tmp_path / "prices" / f"{t}.csv", index=False)
    rep = oria.run(tmp_path, tmp_path / "out", fetch=False, train_end="2013-12-31", universe=names)
    assert {"graph_open", "graph_close", "etf_resid_k2.0_h5", "vol_spy", "spy", "ensemble"} <= set(rep["results"])
    assert (tmp_path / "out" / "summary.json").exists()
