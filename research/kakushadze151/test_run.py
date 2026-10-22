import numpy as np
import pandas as pd

import run
import strategies as S
from engine import backtest


def fake(seed=0, start="2015-01-01", end="2026-10-20"):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start, end)
    ohlc = {}
    names = list(dict.fromkeys(run.SP100 + run.SECTORS + run.MULTI + run.COMMOD + run.FUT + run.OTHER))
    names = [n for n in names if n != "BTC-USD"] + ["BTC", "EUR", "GBP", "AUD", "JPY", "CAD", "CHF"]
    for k, n in enumerate(names):
        c = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(cal))))
        o = c * (1 + rng.normal(0, 0.003, len(cal)))
        d = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c}, index=cal)
        if n in ("PLTR", "XLC"):
            d.iloc[:1200] = np.nan        # late listing
        ohlc[n] = d
    ohlc["^VIX3M"]["close"] = ohlc["^VIX"]["close"] * (1 + rng.normal(0.02, 0.05, len(cal)))
    m = pd.date_range("2014-01-01", end, freq="MS")
    rates = pd.DataFrame({c: rng.uniform(0, 0.05, len(m)) for c in run.RATES}, index=m)
    return {"ohlc": ohlc, "rates": rates}


def test_engine_drift_and_costs():
    idx = pd.bdate_range("2020-01-01", periods=4)
    ret = pd.DataFrame({"A": [0.0, 0.10, 0.0, 0.0], "B": [0.0, -0.10, 0.0, 0.0]}, index=idx)
    W = pd.DataFrame({"A": [0.5, np.nan, np.nan, np.nan], "B": [0.5, np.nan, np.nan, np.nan]}, index=idx)
    bt = backtest(W, ret, pd.Series(0.0, index=idx), cost_bp=10.0, borrow_yr=0.0)
    assert abs(bt.ret.iloc[0] + 0.001) < 1e-12           # 10 bp on 100% bought
    assert abs(bt.ret.iloc[1]) < 1e-12                    # +5% -5%
    assert bt.turnover.iloc[1:].sum() == 0                # drifts, no rebalance


def test_engine_short_and_cash():
    idx = pd.bdate_range("2020-01-01", periods=3)
    ret = pd.DataFrame({"A": [0.0, 0.10, 0.0]}, index=idx)
    rf = pd.Series(0.001, index=idx)
    W = pd.DataFrame({"A": [-1.0, np.nan, np.nan]}, index=idx)
    bt = backtest(W, ret, rf, cost_bp=0.0, borrow_yr=0.252)
    assert abs(bt.ret.iloc[1] - (-0.10 + 2 * 0.001 - 0.001)) < 1e-12   # short loses 10%, earns rf on 2x cash, pays borrow


def test_lag_shifts_trades():
    idx = pd.bdate_range("2020-01-01", periods=3)
    ret = pd.DataFrame({"A": [0.0, 0.10, 0.10]}, index=idx)
    W = pd.DataFrame({"A": [1.0, np.nan, np.nan]}, index=idx)
    z = pd.Series(0.0, index=idx)
    assert abs(backtest(W, ret, z, 0.0, lag=0).ret.iloc[1] - 0.1) < 1e-12
    assert abs(backtest(W, ret, z, 0.0, lag=1).ret.iloc[1]) < 1e-12


def test_rules_no_lookahead():
    rng = np.random.default_rng(3)
    p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800))), index=pd.bdate_range("2020-01-01", periods=800))
    full = S.two_ma(p)
    cut = S.two_ma(p.iloc[:600])
    assert full.iloc[:600].equals(cut)
    hp_full = S.hp_trend(p, win=300)
    hp_cut = S.hp_trend(p.iloc[:500], win=300)
    assert np.allclose(hp_full.iloc[:500].dropna(), hp_cut.dropna())


def test_fomc_days_hold_one_day():
    cal = pd.bdate_range("2024-01-01", "2024-12-31")
    w = S.pre_fomc(cal).SPY
    assert w.sum() == 8 and w["2024-01-30"] == 1 and w["2024-01-31"] == 0


def test_full_pipeline_on_fake_data(tmp_path):
    rows, six = run.run_all(fake())
    run.report(rows, six, str(tmp_path))
    assert len(rows) > 40 and all(np.isfinite(r["net"]["sharpe"]) for r in rows)
    assert (tmp_path / "table.md").exists()
