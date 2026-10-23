import json

import numpy as np
import pandas as pd
import pytest

from strats import cli, copy, crypto, data, live, rules, site, stops


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
    assert head == "date,trend,pead,insider,best_ideas,conviction,pelosi,reversal,btc_donchian"
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


def copy_data(px):
    d = px.index
    q = lambda per, filed, tops, acc: {"period": per, "filed": filed, "accession": acc, "total": 100.0, "stock_total": 100.0,  # noqa: E731
                                       "top": [{"cusip": c, "issuer": c, "title": "COM", "value": v} for c, v in tops]}
    f1 = {"cik": "1", "name": "A", "quarters": [q("2019-12-31", str(d[30].date()), [("cS01", 40), ("cS02", 30), ("cS03", 8)], "a1"),
                                                q("2020-03-31", str(d[90].date()), [("cS04", 50), ("cS01", 10)], "a2")]}
    f2 = {"cik": "2", "name": "B", "quarters": [q("2019-12-31", str(d[31].date()), [("cSPY", 60), ("cS02", 20)], "b1")]}
    cusips = {f"c{t}": t for t in px.columns}
    pel = copy.pelosi_frame([{"ticker": "S05", "type": "P", "date": str(d[5].date()), "filed": str(d[20].date())},
                             {"ticker": "S05", "type": "P", "date": str(d[5].date()), "filed": str(d[20].date())},
                             {"ticker": "S05", "type": "S (partial)", "date": str(d[50].date()), "filed": str(d[70].date())},
                             {"ticker": "S06", "type": "P", "date": str(d[60].date()), "filed": str(d[61].date())}])
    return {"filers": [f1, f2], "cusips": cusips, "pelosi": pel}


def test_copy_pots():
    px = prices()
    d = px.index
    cp = copy_data(px)
    assert copy.best_ideas_weights(cp, px, d[30]) == {}                     # filed that day: not public before the close
    assert copy.best_ideas_weights(cp, px, d[31]) == {"S01": 1.0}
    assert copy.best_ideas_weights(cp, px, d[40]) == {"S01": 0.5, "S02": 0.5}   # B's top is SPY, an index fund: next one
    assert copy.best_ideas_weights(cp, px, d[91])["S04"] == 0.5
    # conviction: 7.5%+ names of the 2019-12-31 books, from the first session on or after 2020-02-16
    reb = d[d.searchsorted(pd.Timestamp("2019-12-31") + pd.Timedelta(days=copy.CONV_LAG_DAYS))]
    w = copy.conviction_weights(cp, px, reb)
    assert list(w) == ["S02", "S01", "S03"] and abs(sum(w.values()) - 1) < 1e-12   # S02 is in both books, so first
    assert copy.conviction_weights(cp, px, reb - pd.Timedelta(days=1)) == {}
    # pelosi: shares + calls on one day are one buy; held from the close after the filing until the sale is disclosed
    assert copy.pelosi_weights(cp, px, d[20]) == {}
    assert copy.pelosi_weights(cp, px, d[21]) == {"S05": 1.0}
    assert copy.pelosi_weights(cp, px, d[62]) == {"S05": 0.5, "S06": 0.5}
    assert copy.pelosi_weights(cp, px, d[71]) == {"S06": 1.0}
    assert copy.pelosi_weights(cp, px, d[62 + copy.PELOSI_HOLD]) == {}
    assert {"S01", "S02", "S03", "S04", "S05", "S06"} <= copy.tickers(cp, since="2000-01-01") and "SPY" not in copy.tickers(cp, "2000")


def test_copy_pots_step_and_replay():
    px = prices()
    cp = copy_data(px)
    st = {}
    live.step(st, px.loc[:px.index[100]], px.index[100], [], pd.DataFrame(), pd.DataFrame(), cp)
    assert st["pots"]["pelosi"][-1]["w"] == {"S06": 1.0} and st["pots"]["best_ideas"][-1]["w"]
    st2 = {}
    rep = live.step(st2, px, px.index[-1], [], pd.DataFrame(), pd.DataFrame())
    assert "pelosi" not in st2["pots"]                                     # no data: the pot waits
    assert rep["pelosi"].startswith("no filings")
    rp = live.replay(px, [], start="2019-03-01", cp=cp)
    assert {"best_ideas", "conviction", "pelosi"} <= set(rp.columns)
    s = live.replay_summary(rp)
    assert "pelosi" in s["corr_spy"]


def test_copy_load(tmp_path):
    d = tmp_path / "copy"
    (d / "13f").mkdir(parents=True)
    (d / "ptr").mkdir()
    (d / "13f" / "1.json").write_text(json.dumps({"cik": "1", "name": "A", "quarters": []}))
    (d / "cusips.json").write_text("{}")
    (d / "ptr" / "2024.json").write_text(json.dumps({"year": 2024, "docs": {
        "1": {"last": "Pelosi", "filed": "2024-01-05", "rows": [{"ticker": "nvda", "type": "P", "date": "2024-01-02"}]},
        "2": {"last": "Smith", "filed": "2024-01-05", "rows": [{"ticker": "X", "type": "P", "date": "2024-01-02"}]}}}))
    cp = copy.load(tmp_path)
    assert list(cp["pelosi"]["ticker"]) == ["NVDA"] and copy.has_data(cp)


def test_live_quote_of_an_open_session_is_dropped():
    px = prices(10)
    today = px.index[-1]
    ny = lambda h: (today + pd.Timedelta(hours=h)).tz_localize("America/New_York")  # noqa: E731
    assert cli.closed(px, ny(11)).index[-1] == px.index[-2]
    assert cli.closed(px, ny(17)).index[-1] == today


def btc_bars(n=400, seed=3, start="2026-08-01"):
    rng = np.random.default_rng(seed)
    c = 60000 * np.exp(np.cumsum(rng.normal(0.002, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range(start, periods=n, freq="4h")
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c}, index=idx)


def test_btc_rule_trades_breakouts_at_the_next_open():
    d = btc_bars()
    eq = crypto.run_rule(d, cost=0.0)
    hi = d.high.rolling(crypto.CHANNEL).max().shift()
    sma = d.close.rolling(crypto.TREND).mean()
    first = d.index[((d.close > hi) & (d.close > sma)).to_numpy()][0]
    j = d.index.get_loc(first)
    assert not eq["held"].iloc[j] and eq["after"].iloc[j] and eq["held"].iloc[j + 1]
    assert eq["equity"].iloc[j + 1] == pytest.approx(d.close.iloc[j + 1] / d.open.iloc[j + 1])
    assert (eq["equity"].iloc[: j + 1] == 1.0).all()
    assert crypto.run_rule(d)["equity"].iloc[-1] < eq["equity"].iloc[-1]       # costs bite


def test_btc_mark_bar_is_the_last_bar_closed_by_the_us_close():
    eq = crypto.run_rule(btc_bars())
    # 2026-09-10 16:00 New York = 20:00 UTC: the bar opening 16:00 UTC is the last one closed
    assert crypto.mark_bar(eq, pd.Timestamp("2026-09-10")) == pd.Timestamp("2026-09-10 16:00")
    assert crypto.mark_bar(eq, pd.Timestamp("2026-01-05")) is None


def test_btc_pot_steps_with_the_rule_equity():
    d = btc_bars()
    px = prices()
    st = {}
    days = [pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-09"), pd.Timestamp("2026-09-10")]
    for day in days:
        live.step(st, px, day, [], pd.DataFrame(), pd.DataFrame(), None, d)
    rows = st["pots"]["btc_donchian"]
    eq = crypto.run_rule(d, stop=stops.STOPS["btc_donchian"])["equity"]
    assert [r["date"] for r in rows] == ["2026-09-08", "2026-09-09", "2026-09-10"] and rows[0]["ret"] == 0.0
    b0, b2 = pd.Timestamp(rows[0]["bar"]), pd.Timestamp(rows[2]["bar"])
    assert rows[2]["index"] == pytest.approx(eq[b2] / eq[b0])
    assert "BTC" in site.position_text("btc_donchian", rows[-1], {}) or "cash" in site.position_text("btc_donchian", rows[-1], {})
    st2 = {}
    live.step(st2, px, days[0], [], pd.DataFrame(), pd.DataFrame(), None, None)
    assert "btc_donchian" not in st2["pots"]                 # no bars: the pot waits


def test_pots_added_later_are_replayed_from_the_origin():
    px = prices()
    members = [c for c in px.columns if c.startswith("S")]
    o, d = px.index[-6], px.index[-1]
    earn = pd.DataFrame({"symbol": ["S01"], "date": [str((px.index[-4] - pd.Timedelta(days=1)).date())], "epsActual": [2.0], "epsEstimate": [1.0]})
    # stepped by hand, close by close, from the origin
    ref = {}
    for s in px.index[-6:]:
        live.step(ref, px.loc[:s], s, members, earn, pd.DataFrame(), origin=str(s.date()))
    # a state whose pots only started at the last session but one is rebuilt from the origin on the next run
    st = {}
    live.step(st, px.loc[:px.index[-2]], px.index[-2], members, earn, pd.DataFrame(), origin=str(px.index[-2].date()))
    rep = live.step(st, px, d, members, earn, pd.DataFrame(), origin=str(o.date()))
    assert rep["trend_replayed"] == f"replayed 5 sessions from {o.date()}" and st["start"] == str(o.date())
    for k in ("trend", "pead", "insider", "reversal"):
        rows = st["pots"][k]
        assert [r["date"] for r in rows] == [r["date"] for r in ref["pots"][k]]
        assert rows[-1]["index"] == pytest.approx(ref["pots"][k][-1]["index"])
        assert all(r.get("replayed") for r in rows[:-1]) and not rows[-1].get("replayed")
    assert st["events"]["pead"] == ref["events"]["pead"]
    # once a pot reaches back to the origin it is not rebuilt again
    assert not any(k.endswith("_replayed") for k in live.step(st, px, d, members, earn, pd.DataFrame(), origin=str(o.date())))
    out = site.page({"strategies": {"trend": {"name": "Trend following", "category": rules.LONG_TERM, "what": "", "position": ""}}}, st["pots"])
    assert f"replayed with the same rule up to the {px.index[-2].date()} close" in out


def ohlc(px, low_cut=None, high_pop=None, gap=None):
    """Daily bars around the closes; low_cut/high_pop/gap: {(date, ticker): fraction of the previous close}."""
    op, lo, hi = px.shift().fillna(px), px * 0.995, px * 1.005
    lo, hi = np.minimum(lo, op), np.maximum(hi, op)
    for (d, t), f in (low_cut or {}).items():
        lo.at[d, t] = px.shift().at[d, t] * f
    for (d, t), f in (high_pop or {}).items():
        hi.at[d, t] = px.shift().at[d, t] * f
    for (d, t), f in (gap or {}).items():
        op.at[d, t] = lo.at[d, t] = px.shift().at[d, t] * f
    return {"open": op, "low": lo, "high": hi}


def test_stop_sells_at_the_stop_or_the_gap_and_shorts_use_the_high():
    px = prices(30)
    d0, d1, d2 = px.index[10], px.index[11], px.index[12]
    a, b, c = "S00", "S01", "S02"
    bars = ohlc(px, low_cut={(d1, a): 0.90}, high_pop={(d1, b): 1.10}, gap={(d1, c): 0.85})
    w = {a: 0.5, b: -0.25, c: 0.25}
    r, hit = stops.held_return(w, px, bars, d0, d1, 0.08)
    p0 = px.loc[d0]
    want = 0.5 * -0.08 - 0.25 * 0.08 + 0.25 * -0.15
    assert sorted(hit) == [a, b, c] and r == pytest.approx(want)
    assert stops.held_return(w, px, bars, d0, d1, None) == (pytest.approx(rules.book_return(w, px, d0, d1)), [])
    # over two sessions the stop is checked against each previous close; a stopped position earns nothing after
    r2, hit2 = stops.held_return({a: 1.0}, px, bars, d0, d2, 0.08)
    assert hit2 == [a] and r2 == pytest.approx(-0.08)
    assert p0[a] > 0


def test_stopped_position_is_cash_at_the_close_and_bought_back():
    px = prices()
    members = [f"S{i:02d}" for i in range(40)]
    d0, d1 = px.index[-2], px.index[-1]
    st = {}
    live.step(st, px.loc[:d0], d0, members, pd.DataFrame(), pd.DataFrame(), origin=str(d0.date()))
    held = st["pots"]["reversal"][-1]["w"]
    t = sorted(held)[0]
    bars = ohlc(px, low_cut={(d1, t): 0.80})
    rep = live.step(st, px, d1, members, pd.DataFrame(), pd.DataFrame(), origin=str(d0.date()), bars=bars)
    row = st["pots"]["reversal"][-1]
    assert row["stopped"] == [t] and "stopped out" in rep["reversal"]
    plain = rules.book_return(held, px, d0, d1)
    lost = held[t] * (px.at[d1, t] / px.at[d0, t] - 1)
    rebuy = live.COST * rules.turnover({k: v for k, v in held.items() if k != t}, row["w"])     # back in at the close
    assert row["ret"] == pytest.approx(plain - lost + held[t] * -stops.REVERSAL_CAP - live.COST * held[t] - rebuy, abs=1e-12)


def test_btc_stop_sells_and_waits_for_the_us_close():
    d = btc_bars()
    base = crypto.run_rule(d, cost=0.0)
    j = int(np.argmax(base["held"].to_numpy())) + 2           # a bar while the rule holds BTC
    d2 = d.copy()
    d2.iloc[j, d2.columns.get_loc("low")] = d2.close.iloc[j - 1] * 0.5
    eq = crypto.run_rule(d2, cost=0.0, stop=0.06)
    assert eq["held"].iloc[j] and not eq["after"].iloc[j]
    mark = crypto.us_close_bars(d2.index)
    k = j + int(np.argmax(mark[j:]))                           # back in after the next US close, if the rule still wants it
    assert not eq["held"].iloc[j + 1: k + 1].any()
    assert eq["equity"].iloc[j] < base["equity"].iloc[j]
    assert crypto.run_rule(d, stop=None)["equity"].equals(crypto.run_rule(d)["equity"])


def test_yahoo_bars_are_adjusted_like_the_close():
    ts = [int(pd.Timestamp(f"2026-10-0{i} 20:00", tz="UTC").timestamp()) for i in (1, 2)]
    payload = {"chart": {"result": [{"timestamp": ts, "indicators": {
        "quote": [{"open": [10.0, 11.0], "low": [9.0, 10.0], "high": [11.0, 12.0], "close": [10.0, 11.5]}],
        "adjclose": [{"adjclose": [5.0, 5.75]}]}}]}}
    b = data.parse_yahoo_bars(payload)
    assert list(b["adj_low"]) == [4.5, 5.0] and list(b["adj_open"]) == [5.0, 5.5]


def test_stop_backtest_and_explain_run():
    px = prices()
    members = [f"S{i:02d}" for i in range(40)]
    res = stops.backtest(px, ohlc(px), members, str(px.index[300].date()), {"reversal": [0.0, 0.05], "trend": [0.0]}, live.COST)
    assert set(res) == {"reversal stop 0%", "reversal stop 5%", "trend stop 0%"}
    st = {}
    for s in px.index[-3:]:
        live.step(st, px.loc[:s], s, members, pd.DataFrame(), pd.DataFrame(), origin=str(px.index[-3].date()))
    ex = stops.explain(st, px, px.index[-1])
    assert "reversal" in ex and len(ex["reversal"]["worst"]) == 5
