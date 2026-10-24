import json

import numpy as np
import pandas as pd

from exp2 import openbb_ideas


def _write(d, t, close, idx):
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "open": close, "close": close, "adj_close": close, "volume": 1e6}).to_csv(d / f"{t}.csv", index=False)


def _data(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2002-01-01", "2026-09-30")
    n = len(idx)
    vol = 0.01 * np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.2).clip(0.3, 3)
    r = rng.normal(0.0003, vol)
    d = tmp_path / "prices"
    d.mkdir(parents=True)
    _write(d, "SPY", 100 * np.exp(np.cumsum(r)), idx)
    vix = (vol * np.sqrt(252) * 100 * 1.15).clip(9, 80)
    _write(d, "^VIX", vix, idx)
    _write(d, "^IRX", np.full(n, 2.0), idx)
    late = idx >= "2011-10-03"
    short_vol = np.where(late, -1.5 * np.r_[0, np.diff(np.log(vix))] + 0.0005, 0)
    _write(d, "SVXY", 50 * np.exp(np.cumsum(short_vol))[late], idx[late])
    _write(d, "VIXY", 50 * np.exp(np.cumsum(-short_vol))[late], idx[late])
    # VX curve at month ends: front and second contract, second a little above the front (contango)
    ends = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last()
    rows = []
    for m, day in ends.items():
        if day < pd.Timestamp("2010-01-01"):
            continue
        v = float(vix[idx.get_loc(day)])
        for k in (1, 2, 3):
            rows.append({"date": day.strftime("%Y-%m-%d"), "expiration": str(m + k), "price": v + k, "symbol": f"VX{k}"})
    pd.DataFrame(rows).to_csv(tmp_path / "vx_curve_monthly.csv", index=False)
    spread = np.where((idx >= "2006-06-01") & (idx < "2007-06-01") | (idx >= "2022-11-01") & (idx < "2024-11-01"), -0.5, 1.5)
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "month_3": 0.02, "year_10": 0.02 + spread / 100}).to_csv(tmp_path / "treasury_curve.csv", index=False)
    months = pd.period_range("2002-01", "2026-09", freq="M")
    ref = pd.DataFrame({"spy": rng.normal(0.8, 4, len(months)), "regime_spy": rng.normal(0.6, 3, len(months)),
                        "trend_tsmom": rng.normal(0.4, 2, len(months)), "varswap_always": rng.normal(0.3, 2, len(months)),
                        "varswap_timed": rng.normal(0.5, 1, len(months))}, index=months.astype(str))
    ref.to_csv(tmp_path / "sleeves.csv")
    return tmp_path


def test_vx_roll_follows_one_contract():
    c = pd.DataFrame({"date": ["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"],
                      "expiration": ["2024-02", "2024-03", "2024-03", "2024-04"], "price": [14.0, 15.5, 14.5, 16.0]})
    roll = openbb_ideas.vx_roll(c)
    assert roll.index.tolist() == [pd.Period("2024-02", "M")] and roll.iloc[0] == 1.0     # sold 15.5, bought back 14.5


def test_inversion_episodes():
    idx = pd.date_range("2020-01-31", periods=30, freq="ME")
    spread = pd.Series(np.where((np.arange(30) >= 5) & (np.arange(30) < 9), -0.3, 1.0), index=idx)
    spy = pd.Series(np.linspace(100, 130, 30), index=idx)
    ep = openbb_ideas.inversion_episodes(spread, spy)
    assert len(ep) == 1 and ep[0]["from"] == "2020-06" and ep[0]["months"] == 4 and ep[0]["spy_next_12m_from_start_pct"] > 0


def test_chain_summary():
    rows = []
    for dte in (10, 31, 95):
        for k in (90, 95, 100, 105, 110):
            for typ, delta in (("call", (110 - k) / 25), ("put", -(k - 90) / 25)):
                iv = 0.2 + (0.05 if typ == "put" and k < 100 else 0) + dte / 1000
                rows.append({"underlying_price": 100.0, "dte": dte, "strike": k, "option_type": typ, "implied_volatility": iv,
                             "delta": delta, "open_interest": 10 if typ == "call" else 20, "volume": 5})
    s = openbb_ideas.chain_summary(pd.DataFrame(rows))
    assert s["spot"] == 100 and abs(s["iv30_atm"] - 0.23) < 0.01 and s["iv90_atm"] > s["iv30_atm"]
    assert s["skew30"] > 0 and s["pc_oi"] == 2.0 and s["dte_30_used"] == 31


def test_run_end_to_end(tmp_path):
    d = _data(tmp_path)
    rep = openbb_ideas.run(d, tmp_path / "out", d / "sleeves.csv", fetch=False)
    for k in ("svxy_timed", "shortvixy_timed", "vxfut_timed", "varswap_timed"):
        assert rep["vol_products"][k]["test"]["n"] > 50
    assert set(rep["portfolio_with"]) == {"svxy", "shortvixy", "vxfut"}
    assert rep["curve"]["episodes"] and rep["curve"]["vol_forecast"]["test"]["n"] > 50
    assert "varswap_timedcurve" in rep["curve"]["sleeves"]
    json.loads((tmp_path / "out" / "summary.json").read_text())
