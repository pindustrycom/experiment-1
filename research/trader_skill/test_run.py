import numpy as np
import pandas as pd

import run


def prices():
    cal = pd.bdate_range("2020-01-01", periods=300)
    return pd.DataFrame({"SPY": np.linspace(100, 130, 300), "AAA": np.linspace(10, 20, 300), "BBB": 50.0}, index=cal)


def test_hold_series_buys_and_holds_between_events():
    px = prices()
    ev = [(px.index[10], {"AAA": 0.5, "BBB": 0.5}), (px.index[100], {"AAA": 1.0})]
    r = run.hold_series(px, ev, cost_bp=0.0)
    assert r.index[0] == px.index[11]
    v0 = 0.5 * px.AAA.iloc[100] / px.AAA.iloc[10] + 0.5
    assert np.isclose((1 + r.loc[:px.index[100]]).prod(), v0)
    assert np.isclose((1 + r.loc[px.index[101]:]).prod(), px.AAA.iloc[-1] / px.AAA.iloc[100])


def test_hold_series_charges_turnover():
    px = prices()
    ev = [(px.index[10], {"BBB": 1.0})]
    r = run.hold_series(px, ev, cost_bp=10.0)
    assert np.isclose(r.iloc[0], -0.001)


def test_filer_events_enter_after_the_filing_day_and_skip_unpriced():
    px = prices()
    q = [{"period": "2020-01-31", "filed": str(px.index[20].date()),
          "top": [{"cusip": "1", "value": 3.0}, {"cusip": "2", "value": 1.0}, {"cusip": "3", "value": 4.0}]}]
    ev, cov = run.filer_events(q, {"1": "AAA", "2": "BBB", "3": "ZZZ"}, px, top=10, when="filed")
    assert ev[0][0] == px.index[21]
    assert ev[0][1] == {"AAA": 0.75, "BBB": 0.25}
    assert cov == [0.5]
    ev, _ = run.filer_events(q, {"1": "AAA"}, px, top=10, when="period")
    assert ev[0][0] == pd.Timestamp("2020-01-31")


def test_event_study_and_portfolio():
    px = prices()
    rows = pd.DataFrame({"ticker": ["AAA", "BBB", "NOPE"], "filed": [px.index[5]] * 3, "date": [px.index[0]] * 3})
    ev = run.event_study(rows, px, "filed")
    assert len(ev) == 2 and ev.entry.iloc[0] == px.index[6]
    i = 6
    assert np.isclose(ev.x21.iloc[0], px.AAA.iloc[i + 21] / px.AAA.iloc[i] - px.SPY.iloc[i + 21] / px.SPY.iloc[i])
    r = run.congress_portfolio(rows, ev, px, hold=10)
    assert r.index[0] == px.index[7]
    s = run.summarize_events(pd.concat([ev] * 6, ignore_index=True))
    assert s["21d"]["n"] == 12


def test_stats_on_benchmark_itself():
    px = prices()
    spy = px.SPY.pct_change().dropna()
    s = run.stats(spy, spy)
    assert np.isclose(s["excess_cagr"], 0.0) and np.isclose(s["beta"], 1.0)
