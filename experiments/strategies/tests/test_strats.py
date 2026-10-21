import json

import numpy as np
import pandas as pd

from strats import cli, live, rules, site


def prices(n=600, seed=0, tickers=None):
    rng = np.random.default_rng(seed)
    tickers = tickers or [*rules.TREND_ETFS, *(f"S{i:02d}" for i in range(40))]
    idx = pd.bdate_range("2018-01-02", periods=n)
    drift = np.where(np.arange(len(tickers)) % 2 == 0, 0.0008, -0.0008)
    r = rng.normal(drift, 0.01, (n, len(tickers)))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=tickers)


def test_trend_follows_the_sign_of_the_year_and_caps_gross():
    px = prices()
    w = rules.trend_weights(px, px.index[-1])
    for t, x in w.items():
        mom = px[t].iloc[-1] / px[t].iloc[-1 - rules.TREND_LOOKBACK] - 1
        assert np.sign(x) == np.sign(mom)
    assert 0 < sum(abs(x) for x in w.values()) <= rules.TREND_GROSS + 1e-9
    assert rules.trend_weights(px.iloc[:100], px.index[99]) == {}       # not a year of data yet


def test_reversal_buys_the_worst_week():
    px = prices()
    members = [c for c in px.columns if c.startswith("S")]
    d = px.index[-1]
    w = rules.reversal_weights(px, d, members)
    r = px[members].iloc[-1] / px[members].iloc[-1 - rules.REVERSAL_DAYS] - 1
    assert set(w) == set(r.nsmallest(rules.REVERSAL_N).index) and abs(sum(w.values()) - 1) < 1e-9


def test_event_rules():
    cal = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "date": ["2024-01-02"] * 4,
                        "epsActual": [1.10, 1.02, None, 0.02], "epsEstimate": [1.00, 1.00, 1.0, 0.001]})
    assert list(rules.earnings_beats(cal)["symbol"]) == ["A"]       # B too small, C no result, D estimate ~0
    tx = pd.DataFrame({"symbol": ["X"] * 4 + ["Y"] * 3, "name": ["a", "b", "c", "c", "a", "b", "b"],
                       "transactionCode": ["P", "P", "P", "P", "P", "P", "S"], "change": [10, 5, 3, 3, 1, 1, -1],
                       "filingDate": ["2024-03-01", "2024-03-10", "2024-03-20", "2024-03-21", "2024-03-01", "2024-03-02", "2024-03-03"]})
    assert rules.insider_clusters(tx, pd.Timestamp("2024-03-25")) == ["X"]
    assert rules.insider_clusters(tx, pd.Timestamp("2024-05-25")) == []      # outside the 30-day window
    assert rules.event_weights({"A": 3}) == {"A": 1 / rules.EVENT_SLOTS}
    assert sum(rules.event_weights({f"T{i}": 5 for i in range(40)}).values()) == 1.0


def test_calendar_helpers():
    assert rules.is_last_session_of_month(pd.Timestamp("2026-10-30")) and not rules.is_last_session_of_month(pd.Timestamp("2026-10-29"))
    assert rules.is_last_session_of_week(pd.Timestamp("2026-10-23")) and not rules.is_last_session_of_week(pd.Timestamp("2026-10-22"))
    idx = pd.bdate_range("2026-01-26", "2026-02-06")
    assert list(idx[rules.month_end(idx).values]) == [pd.Timestamp("2026-01-30")]


def test_step_marks_books_and_holds_events():
    px = prices()
    members = [c for c in px.columns if c.startswith("S")]
    d0, d1, d2 = px.index[-3], px.index[-2], px.index[-1]
    earn = pd.DataFrame({"symbol": ["S01"], "date": [str((d0 - pd.Timedelta(days=1)).date())], "epsActual": [2.0], "epsEstimate": [1.0]})
    tx = pd.DataFrame({"symbol": ["S02"] * 3, "name": ["a", "b", "c"], "transactionCode": ["P"] * 3, "change": [1, 1, 1],
                       "filingDate": [str(d0.date())] * 3})
    st = {}
    live.step(st, px.loc[:d0], d0, members, earn, tx)
    assert st["start"] == str(d0.date()) and st["events"]["pead"] == {"S01": rules.PEAD_HOLD}
    assert st["events"]["insider"] == {"S02": rules.INSIDER_HOLD}
    assert all(len(v) == 1 for v in st["pots"].values())
    rep = live.step(st, px.loc[:d1], d1, members, earn, tx)
    pe = st["pots"]["pead"]
    want = rules.book_return(pe[0]["w"], px, d0, d1)
    assert abs(pe[1]["ret"] - want) < 1e-12 and st["events"]["pead"] == {"S01": rules.PEAD_HOLD - 1}   # same event not bought twice
    assert live.step(st, px.loc[:d1], d1, members, earn, tx)["trend"].startswith("already")
    # the trend book is kept until the month ends; the index compounds the day returns
    live.step(st, px, d2, members, earn, tx)
    tr = st["pots"]["trend"]
    assert tr[1]["w"] == tr[0]["w"] or rules.is_last_session_of_month(d1)
    assert abs(tr[-1]["index"] - np.prod([1 + r["ret"] for r in tr])) < 1e-12
    assert "insider" in rep


def test_missed_rebalance_is_caught_up():
    rows = [{"date": "2026-09-30", "reb": True}, {"date": "2026-10-28", "reb": False}]
    assert live.due_now(rows, pd.Timestamp("2026-11-03"), "M", rules.is_last_session_of_month)        # October's was missed
    rows[0]["date"] = "2026-10-30"
    assert not live.due_now(rows, pd.Timestamp("2026-11-03"), "M", rules.is_last_session_of_month)


def test_replay_and_site(tmp_path):
    px = prices(700)
    members = [c for c in px.columns if c.startswith("S")]
    rp = live.replay_summary(live.replay(px, members, start="2019-03-01"))
    assert {"trend", "reversal", "SPY"} <= set(rp) and rp["trend"]["vol"] > 0
    st = {}
    live.step(st, px, px.index[-1], members, pd.DataFrame(), pd.DataFrame())
    out = site.write(tmp_path, st, rp, {})
    assert out["sessions"] == 1
    head = (tmp_path / "data" / "daily.csv").read_text().splitlines()[0]
    assert head == "date,trend,pead,insider,reversal"
    s = json.loads((tmp_path / "data" / "summary.json").read_text())
    assert s["strategies"]["pead"]["category"] == rules.LONG_TERM and "cash" in s["strategies"]["pead"]["position"]
    page = (tmp_path / "index.html").read_text()
    assert '<main class="wrap">' in page and "</head>" in page and "Weekly reversal" in page


def test_cli_offline(tmp_path):
    px = prices(400)
    (tmp_path / "data" / "prices").mkdir(parents=True)
    for t in px.columns:
        px[t].rename("adj_close").to_csv(tmp_path / "data" / "prices" / f"{t}.csv", index_label="date")
    pd.DataFrame({"ticker": [c for c in px.columns if c.startswith("S")]}).to_csv(tmp_path / "data" / "sp500.csv", index=False)
    rep = cli.run(tmp_path / "data", tmp_path / "state" / "state.json", tmp_path / "site", fetch=False)
    assert rep["last_close"] == str(px.index[-1].date()) and (tmp_path / "site" / "index.html").exists()
    assert json.loads((tmp_path / "state" / "state.json").read_text())["pots"]["reversal"]
