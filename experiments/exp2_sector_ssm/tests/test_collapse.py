import numpy as np
import pandas as pd

from exp2 import collapse


def test_hmm_finds_the_turbulent_regime():
    rng = np.random.default_rng(0)
    state = (np.arange(4000) // 300) % 2
    r = rng.normal(0, np.where(state, 0.025, 0.007))
    p, prm = collapse.hmm2(r, np.arange(4000) < 3000)
    assert prm["sd"][prm["turbulent_state"]] > 2 * min(prm["sd"])
    assert p[state == 1].mean() > 0.7 and p[state == 0].mean() < 0.3


def test_min_variance_and_hrp_weights():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 6)) * np.array([1, 1, 1, 2, 3, 4]) / 100
    S = np.cov(X.T)
    for w in (collapse._minvar(S), collapse._hrp(S)):
        assert np.isclose(w.sum(), 1) and (w >= 0).all()
        assert w[:3].sum() > w[3:].sum()          # the calm assets get more weight


def test_pairs_trade_a_planted_mean_reverting_spread():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2003-01-01", periods=900)
    common = np.cumsum(rng.normal(0, 0.01, 900))
    ou = np.zeros(900)
    for t in range(1, 900):
        ou[t] = 0.85 * ou[t - 1] + rng.normal(0, 0.01)
    noise = np.cumsum(rng.normal(0, 0.01, (900, 2)), axis=0)
    C = pd.DataFrame(np.exp(np.c_[common + ou, common, noise] + 3), index=idx, columns=["A", "B", "X", "Y"])
    months = pd.date_range("2002-12-31", idx[-1], freq="ME")
    member = pd.DataFrame(True, index=months, columns=list(C))
    P = pd.DataFrame({"month": pd.PeriodIndex([], freq="M"), "ticker": [], "pc_gbt": [], "f_gbt": []})
    T = collapse.pairs(P, C, {t: "s" for t in C}, member, top=1)
    assert len(T) >= 5
    assert {"A", "B"} == set(T["a"]) | set(T["b"])
    assert T["ret"].mean() > 0


def test_calendar_marks_turn_of_month_and_pre_holiday():
    idx = pd.bdate_range("2020-01-01", "2020-12-31").drop(pd.Timestamp("2020-07-03"))
    spy = pd.Series(np.geomspace(100, 400, len(idx)), index=idx)
    rf = pd.Series(0.0, index=idx)
    out = collapse.calendar_regime(spy, rf, pd.Series(0.0, index=idx))
    assert {"cal_turn_of_month", "cal_pre_holiday", "regime_spy"} <= set(out)
    assert out["cal_pre_holiday"].loc[pd.Period("2020-07", "M")] > 0      # July 2 is the session before a holiday
