import numpy as np
import pandas as pd

from exp2 import volpred


def _write(tmp_path, T=2900, n=120, seed=0):
    """Stocks with persistent, stock-specific volatility regimes (so volatility is forecastable) and a market."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2001-01-01", periods=T)
    mkt = rng.normal(0, 0.01, T)
    d = tmp_path / "prices"
    d.mkdir()
    for i in range(n):
        lv = np.zeros(T)
        for t in range(1, T):
            lv[t] = 0.98 * lv[t - 1] + rng.normal(0, 0.12)
        sig = 0.015 * np.exp(lv / 2) * (0.5 + i / n)
        r = mkt + rng.normal(0, 1, T) * sig
        c = 50 * np.exp(np.cumsum(r))
        o = c * np.exp(rng.normal(0, 0.003, T))
        pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "open": o, "close": c, "adj_close": c, "volume": 1e6}).to_csv(d / f"S{i}.csv", index=False)
    spy = 50 * np.exp(np.cumsum(mkt))
    rv = pd.Series(mkt).rolling(21).std().bfill().to_numpy() * np.sqrt(252) * 100
    for t, c in (("SPY", spy), ("^VIX", rv * 1.2), ("^IRX", np.full(T, 2.0))):
        pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "open": c, "close": c, "adj_close": c, "volume": 1e6}).to_csv(d / f"{t}.csv", index=False)
    pd.DataFrame({"ticker": [f"S{i}" for i in range(n)], "sector": ["a", "b", "c"] * (n // 3)}).to_csv(tmp_path / "sp500_current.csv", index=False)
    pd.DataFrame({"date": ["2000-01-01"], "added": ["S0"], "removed": ["ZZZ"]}).to_csv(tmp_path / "sp500_changes.csv", index=False)


def test_two_stage_end_to_end(tmp_path):
    _write(tmp_path)
    rep = volpred.run(tmp_path, tmp_path / "out", train_end="2009-06-30", first_years=(2004, 2006))
    s1 = rep["stage1_vol_forecast"]["test"]
    # volatility is planted as persistent, so both fitted forecasters should rank next month's vol well
    assert s1["f_gbt"]["rank_ic"] > 0.5 and s1["f_har"]["rank_ic"] > 0.5
    assert s1["f_gbt"]["mse_log"] < 1.2 * s1["f_last"]["mse_log"]
    assert {"base", "plus_vol", "risk_adj_target", "vol_only"} <= set(rep["stage2_ranking"]["test"])
    assert {"before_test", "test"} <= set(rep["low_vol"]) and "mom12_2_vol_scaled_book" in rep["vol_momentum"]["test"]
    assert "vol_target_forecast" in rep["spy"]["test"]
    assert (tmp_path / "out" / "vol_forecasts.csv").exists()
