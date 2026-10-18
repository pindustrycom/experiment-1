import json

import numpy as np
import pandas as pd
import pytest

from exp2 import improve, live

PLAN = {"weights": {"AAA": 0.6, "BBB": 0.4}, "beta": 0.5, "idio": 0.006, "K": 20.0, "fvar": 300.0, "on": True,
        "p_hi_end": 0.1, "n_days": 21, "formed": "2024-01-31"}


def history(n=400, seed=0, settings=None, plans=None):
    """A replayed log over synthetic SPY and stock returns, as live.backfill builds it."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-06-01", periods=n)
    df = pd.DataFrame({"r": rng.normal(0.0004, 0.01, n), "vix": 16 + rng.normal(0, 1, n)}, index=idx)
    sig = pd.Series(0.01, index=idx)
    p_hi = pd.Series(rng.uniform(0, 1, n), index=idx)
    plans = plans or {str(m): {**PLAN, "fvar": 300.0 + 150 * (i % 3), "p_hi_end": 0.2 + 0.1 * (i % 6)}
                      for i, m in enumerate(pd.period_range("2018-06", "2020-12", freq="M"))}
    for p in plans.values():
        p["on"] = improve.swap_on(p, improve.DEFAULTS)
    R = pd.DataFrame({"AAA": df["r"] + rng.normal(0, 0.005, n), "BBB": df["r"]})
    return live.backfill(df, sig, p_hi, plans, R, None, settings=settings), plans


def test_shadow_reproduces_the_log_for_any_settings():
    L, plans = history()
    assert np.allclose(improve.shadow(L, plans, improve.DEFAULTS).to_numpy(), L["real_ret"].astype(float).to_numpy(), atol=1e-9)
    s = improve.settings_of({"leverage": 1.7, "regime_cut": 0.3, "vol_size": 1.5, "vol_bar": 1.25, "stock_size": 0.5})
    L2, _ = history(settings=s, plans=plans)
    assert np.allclose(improve.shadow(L, plans, s).to_numpy(), L2["real_ret"].astype(float).to_numpy(), atol=1e-9)
    assert json.loads(L2["settings"].iloc[0]) == s


def test_settings_are_capped_and_logged():
    s = improve.settings_of('{"leverage": 3, "vol_size": 9, "junk": 1}')
    assert s["leverage"] == 1.7 and s["vol_size"] == 1.5 and "junk" not in s
    assert improve.settings_of(float("nan")) == improve.DEFAULTS
    p = live.predict(PLAN, 0.01, 0.1, False, 1.0, 1.0)
    assert json.loads(p["settings"]) == improve.DEFAULTS
    big = live.predict(PLAN, 0.01, 0.1, False, 1.0, 1.0, {"vol_size": 1.5})
    assert big["exp_varswap"] == pytest.approx(1.5 * p["exp_varswap"])
    assert not live.predict(PLAN, 0.01, 0.45, False, 1.0, 1.0, {"regime_cut": 0.4})["regime_in"]


def test_trials_take_turns_and_alternate():
    st = improve.load(__import__("pathlib").Path("/nonexistent"))
    seen = [improve.next_trial(st) for _ in range(6)]
    assert [t["key"] for t in seen] == improve.ORDER + ["vol_size"]
    assert seen[0] == {"key": "vol_size", "from": 1.0, "to": 1.25} and seen[5]["to"] == 0.75   # up first, then down


def live_log(n_weeks, ret=0.1, start="2026-10-19"):
    days = pd.bdate_range(start, periods=5 * n_weeks)
    L, plans = history(len(days) + 200)
    x = L.tail(len(days)).copy()
    x["date"] = [str(d.date()) for d in days]
    x["source"] = "live"
    if ret is not None:
        x["real_ret"] = ret
    plans = {**plans, **{str(m): plans["2019-06"] for m in pd.period_range("2026-10", "2027-12", freq="M")}}
    return x, plans


def test_weekly_review_runs_once_a_week_and_judges_after_12_weeks():
    L, plans = live_log(14)
    st = improve.load(__import__("pathlib").Path("/nonexistent"))
    assert improve.weekly(L, plans, st, "2026-10-23")["started"]["key"] == "vol_size"     # first Friday close
    assert st["trial"]["start"] == "2026-10-26"
    assert improve.weekly(L, plans, st, "2026-10-23") == {"review": "not due"}
    assert improve.weekly(L, plans, st, "2026-10-26") == {"review": "not due"}           # Monday: last week already done
    assert improve.weekly(L, plans, st, "2026-11-06")["trial_weeks_done"] == 2
    out = improve.weekly(L, plans, st, "2027-01-15")                                      # 12 weeks after 2026-10-26
    assert out["judged"]["event"] in ("trial won", "trial lost") and st["trial"] is None or st["pending"]
    if st["pending"]:
        assert "started" not in out                                                       # one change waits at a time
        improve.decide(st, "approve", "2027-01-18")
        assert st["settings"]["vol_size"] == 1.25 and st["previous"]["vol_size"] == 1.0 and st["pending"] is None


def test_drawdown_reverts_and_pauses():
    L, plans = live_log(6, ret=None)
    L["real_ret"] = -4.0                                                                   # a steady slide
    st = improve.load(__import__("pathlib").Path("/nonexistent"))
    st.update(settings={**improve.DEFAULTS, "vol_size": 1.5}, previous=dict(improve.DEFAULTS), adopted="2026-10-19")
    assert improve.weekly(L, plans, st, "2026-10-30")["review"] == "reverted"
    assert st["settings"] == improve.DEFAULTS and st["paused"]
    assert improve.weekly(L, plans, st, "2026-11-06")["review"] == "paused"
    improve.decide(st, "resume", "2026-11-09")
    assert "started" in improve.weekly(L, plans, st, "2026-11-13")


def test_state_round_trip_and_page(tmp_path):
    L, plans = live_log(3)
    st = improve.load(tmp_path)
    improve.weekly(L, plans, st, "2026-10-23")
    improve.save(tmp_path, st)
    st2 = improve.load(tmp_path)
    assert st2["trial"] == st["trial"]
    from exp2 import live_site
    html = live_site.loop_panel({**st2, "trial_so_far": improve.progress(L, plans, st2)})
    assert "Weekly improvement loop" in html and "volatility trade size" in html and "×1.25" in html
