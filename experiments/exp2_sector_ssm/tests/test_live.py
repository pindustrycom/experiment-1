from datetime import UTC, datetime

import numpy as np
import pandas as pd

from exp2 import live, live_site

PLAN = {"weights": {"AAA": 0.6, "BBB": 0.4}, "beta": 0.5, "idio": 0.006, "K": 20.0, "fvar": 300.0, "on": True,
        "p_hi_end": 0.1, "n_days": 21, "formed": "2024-01-31"}


def spy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-06-01", periods=n)
    return pd.DataFrame({"r": rng.normal(0.0004, 0.01, n), "vix": 16 + rng.normal(0, 1, n)}, index=idx)


def test_prediction_matches_simulation():
    p = live.predict(PLAN, sigma_raw=0.01, p_hi=0.1, first_day=False, vol_mult=1.0, band_mult=1.0)
    rng = np.random.default_rng(1)
    r = rng.normal(live.MU, 0.01, 200_000)
    e = rng.normal(0, PLAN["idio"], 200_000)
    sims = [live.realize(p, PLAN, x, pd.Series({"AAA": PLAN["beta"] * x + y, "BBB": PLAN["beta"] * x + y}))["real_ret"]
            for x, y in zip(r[:20000], e[:20000], strict=True)]
    assert abs(np.mean(sims) - p["exp_ret"]) < 0.01
    assert abs(np.std(sims) / p["sd"] - 1) < 0.05
    assert p["regime_in"] and p["varswap_on"] and p["band_lo"] < p["exp_ret"] < p["band_hi"]


def test_regime_and_swap_switches():
    off = live.predict({**PLAN, "on": False}, 0.01, p_hi=0.9, first_day=True, vol_mult=1.0, band_mult=1.0)
    assert not off["regime_in"] and off["exp_varswap"] == 0 and off["exp_regime"] == 0
    roll = live.predict(PLAN, 0.01, 0.1, first_day=True, vol_mult=1.0, band_mult=1.0)
    assert roll["exp_varswap"] < live.predict(PLAN, 0.01, 0.1, first_day=False, vol_mult=1.0, band_mult=1.0)["exp_varswap"]


def test_learning_scales_towards_realised():
    rows = [{"status": "scored", "spy_ret": 2.0, "sigma_raw": 0.01, "error_pp": 1.0, "sd": 0.5}] * 200
    vol, band = live.learned(pd.DataFrame(rows))
    assert 1.8 < vol < 2.0 and 1.8 < band < 2.0          # 120 days of evidence against 20 pseudo-days
    assert live.learned(pd.DataFrame(columns=live.LOG_COLS)) == (1.0, 1.0)


def test_har_forecast_is_positive_and_scaled():
    df = spy(1500)
    s = live.sigma_next(df, live.fit_har(df)).dropna()
    assert (s > 0).all() and 0.005 < s.median() < 0.02


def test_morning_then_evening_then_catch_up():
    df = spy()
    sig = pd.Series(0.01, index=df.index)
    p_hi = pd.Series(0.1, index=df.index)
    plans = {str(m): PLAN for m in pd.period_range("2018-06", "2020-12", freq="M")}
    R = pd.DataFrame({"AAA": df["r"], "BBB": df["r"]})
    full, seen = df, df.iloc[:-3]
    L = live.backfill(seen, sig, p_hi, plans, R, None)
    assert (L["source"] == "backfill").all() and len(L) == (seen.index >= live.BACKFILL_FROM).sum()
    today = full.index[-3]
    m = live.morning(L, seen, sig, p_hi, plans, today, datetime(2020, 1, 1, tzinfo=UTC))
    L = pd.concat([L, pd.DataFrame([m["row"]])], ignore_index=True)
    assert live.morning(L, seen, sig, p_hi, plans, today, datetime(2020, 1, 1, tzinfo=UTC))["skipped"] == "already predicted"
    L, done = live.evening(L, full, plans, R, datetime(2020, 1, 1, tzinfo=UTC))
    row = L[L["date"] == str(today.date())].iloc[0]
    assert done == [str(today.date())] and row["status"] == "scored"
    assert abs(row["error_pp"] - (row["real_ret"] - row["exp_ret"])) < 1e-12
    L = live.backfill(full, sig, p_hi, plans, R, L, "catch-up")
    assert list(L["source"].tail(2)) == ["catch-up", "catch-up"] and L["date"].is_unique


def test_evening_scores_a_ledger_read_back_with_empty_text_columns():
    df = spy()
    sig, p_hi = pd.Series(0.01, index=df.index), pd.Series(0.1, index=df.index)
    plans = {str(m): PLAN for m in pd.period_range("2018-06", "2020-12", freq="M")}
    R = pd.DataFrame({"AAA": df["r"], "BBB": df["r"]})
    today = df.index[-3]
    m = live.morning(pd.DataFrame(columns=live.LOG_COLS), df.iloc[:-3], sig, p_hi, plans, today, datetime(2020, 1, 1, tzinfo=UTC))
    L = pd.DataFrame([m["row"]], columns=live.LOG_COLS)
    L["evaluated_at"] = float("nan")                                   # what read_csv gives for a column never filled
    L, done = live.evening(L, df, plans, R, datetime(2020, 1, 1, tzinfo=UTC))
    assert done == [str(today.date())] and L.at[0, "evaluated_at"].startswith("2020-01-01")


def test_morning_ignores_todays_live_quote():
    df = spy()
    today = df.index[-1]
    seen = live.closed_sessions(df, today)
    assert seen.index[-1] < today and len(seen) == len(df) - 1
    sig = pd.Series(0.01, index=df.index)
    p_hi = pd.Series(0.1, index=df.index)
    plans = {str(today.to_period("M")): PLAN}
    m = live.morning(pd.DataFrame(columns=live.LOG_COLS), seen, sig, p_hi, plans, today, datetime(2020, 1, 1, tzinfo=UTC))
    assert m["row"]["last_close"] == str(seen.index[-1].date())


def test_market_closed_day():
    df = spy()
    L = pd.DataFrame([{**{c: None for c in live.LOG_COLS}, "date": "2019-07-04", "source": "live", "status": "pending"}])
    L, done = live.evening(L, df[df.index != "2019-07-04"], {}, pd.DataFrame(), datetime(2020, 1, 1, tzinfo=UTC))
    assert L.at[0, "status"] == "market closed"


def test_site_builds(tmp_path):
    df = spy()
    sig = pd.Series(0.01, index=df.index)
    plans = {str(m): PLAN for m in pd.period_range("2018-06", "2020-12", freq="M")}
    R = pd.DataFrame({"AAA": df["r"], "BBB": df["r"]})
    L = live.backfill(df, sig, pd.Series(0.1, index=df.index), plans, R, None)
    rep = live_site.build(tmp_path, L, plans, df)
    assert rep["scored"] == len(L)
    page = (tmp_path / "index.html").read_text()
    assert "Experiment 2 · volatility-timed portfolio" in page and "Prediction log" in page
    assert (tmp_path / "data" / "summary.json").exists()


def test_yahoo_latest_session_without_adjusted_close(monkeypatch):
    from exp2 import papers

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"chart": {"result": [{"timestamp": [1760535000, 1760621400, 1760707800],
                                          "indicators": {"quote": [{"open": [1, 1, 1], "close": [10.0, 11.0, 12.0], "volume": [1, 1, 1]}],
                                                         "adjclose": [{"adjclose": [9.0, 9.9, None]}]}}]}}

    monkeypatch.setattr(papers.requests, "get", lambda *a, **k: R())
    df = papers.fetch_yahoo("SPY")
    assert len(df) == 3 and abs(df["adj_close"].iloc[-1] - 10.8) < 1e-9


def test_yahoo_quote_adds_a_lagging_session():
    from exp2 import papers
    day = 86400
    t0 = 1760535000                                                     # a Wednesday, 09:30 New York
    payload = {"chart": {"result": [{"timestamp": [t0, t0 + day],
                                     "meta": {"regularMarketTime": t0 + 2 * day + 6.5 * 3600, "regularMarketPrice": 12.0},
                                     "indicators": {"quote": [{"open": [1, 1], "close": [10.0, 11.0], "volume": [1, 1]}],
                                                    "adjclose": [{"adjclose": [9.0, 9.9]}]}}]}}
    df = papers.parse_yahoo_chart(payload)
    assert len(df) == 3 and abs(df["adj_close"].iloc[-1] - 10.8) < 1e-9
    payload["chart"]["result"][0]["meta"]["regularMarketTime"] = t0 + 2 * day + 3600   # mid-session: not a close yet
    assert len(papers.parse_yahoo_chart(payload)) == 2


def test_waits_until_the_stock_book_is_priced():
    df = spy()
    R = pd.DataFrame({"AAA": df["r"], "BBB": df["r"]})
    d = df.index[-1]
    R.loc[d, "AAA"] = np.nan                                            # 60% of the book has no price yet
    L = pd.DataFrame([{**{c: None for c in live.LOG_COLS}, "date": str(d.date()), "source": "live", "status": "pending"}])
    plans = {str(d.to_period("M")): PLAN}
    L2, done = live.evening(L, df, plans, R, datetime(2020, 1, 1, tzinfo=UTC))
    assert done == [] and L2.at[0, "status"] == "pending"
