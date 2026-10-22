import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kakushadze151"))
from test_run import fake

import premia
import premia_data


def test_parse_french():
    txt = ("This file was created by ...\n\n,Mkt-RF,SMB,HML,RF\n19630701,   -0.67,    0.02,   -0.35,    0.012\n"
           "19630702,    0.79,   -0.28,    0.28,    0.012\n\n Copyright 2026 Eugene F. Fama and Kenneth R. French\n")
    d = premia_data.parse_french(txt)
    assert list(d.columns) == ["Mkt-RF", "SMB", "HML", "RF"] and len(d) == 2 and abs(d.iloc[0, 0] + 0.0067) < 1e-12


def fake_data():
    f = fake(start="2009-01-01", end="2026-09-30")["ohlc"]
    rng = np.random.default_rng(5)
    raw = {t: f[t] for t in premia.ETFS if t in f}
    for t, (cur, inv) in premia.k151.FX.items():
        raw[t] = f[cur]
    cal = pd.bdate_range("1988-01-01", "2026-08-31")
    ff = pd.DataFrame(rng.normal(0.0002, 0.006, (len(cal), 7)), index=cal, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "UMD"])
    ff["RF"] = 0.0001
    put = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.007, len(cal)))), index=cal)
    return {"french": ff, "cboe": {"PUT": put, "BXM": put * 1.01}, "raw": raw}


def test_pipeline(tmp_path):
    data = fake_data()
    a = premia.part_a(data["french"], data["cboe"])
    assert "HML" in a and np.isfinite(a["HML"]["sharpe_after_pub"])
    sl, rf = premia.sleeves(data)
    assert {"FX carry", "VIX carry", "Trend", "HML", "PUT"} <= set(sl.columns)
    assert sl["Trend"].first_valid_index() > sl.index[200]        # no cash-only history before the first signal
    b = premia.part_b(sl, rf, "2019-01-02")
    assert np.isfinite(b["portfolios"]["Risk parity at 10% vol (max 2x)"]["sharpe"])
    assert b["window"].endswith("2026-08-31")                       # stops where the French data stops
    md = premia.report(a, [b], str(tmp_path))
    assert "Hold SPY" in md


def test_clean_drops_bad_prints():
    s = pd.Series([100.0, 101, 1010, 102, 103], index=pd.bdate_range("2020-01-01", periods=5))
    assert list(premia_data.clean(s, "X")) == [100.0, 101, 102, 103]
