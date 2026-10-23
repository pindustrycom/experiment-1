import json

import pytest

import build_hub
import portfolio

MAIN = '<html><body><main class="wrap">{b}</main></body></html>'
EXP2_COLS = "date,source,status,real_ret,regime_in,varswap_on,K\n"


@pytest.fixture(autouse=True)
def fx(monkeypatch):
    rates = {"2026-10-16": 1.40, "2026-10-19": 1.40, "2026-10-20": 1.414}
    monkeypatch.setattr(portfolio, "fetch_fx", lambda start: dict(rates))
    return rates


def make_site(site, rows, direction="a_advantaged", z=0.3):
    d = site / "exp1" / "data"
    d.mkdir(parents=True, exist_ok=True)
    (site / "exp1" / "index.html").write_text(MAIN.format(b="pairs"))
    (d / "pairs.json").write_text(json.dumps([
        {"slug": "ko-pep", "company_a": {"ticker": "KO"}, "company_b": {"ticker": "PEP"}, "direction": direction},
        {"slug": "v-ma", "company_a": {"ticker": "V"}, "company_b": {"ticker": "MA"}, "direction": "uncertain"}]))
    (d / "pair_prices.json").write_text(json.dumps([
        {"slug": "ko-pep", "a": "KO", "b": "PEP", "last_date": rows[-1][0], "z_last": z,
         "rows": [{"date": r[0], "a": r[1], "b": r[2]} for r in rows]},
        {"slug": "v-ma", "a": "V", "b": "MA", "last_date": rows[-1][0], "z_last": None,
         "rows": [{"date": r[0], "a": 100, "b": 100} for r in rows]}]))


def write_exp2(site, lines):
    d = site / "exp2" / "data"
    d.mkdir(parents=True, exist_ok=True)
    (site / "exp2" / "index.html").write_text(MAIN.format(b="exp2"))
    (d / "predictions.csv").write_text(EXP2_COLS + "".join(x + "\n" for x in lines))


def test_exp1_book_rules():
    pairs = [{"slug": "a", "company_a": {"ticker": "A"}, "company_b": {"ticker": "B"}, "direction": "b_advantaged"},
             {"slug": "c", "company_a": {"ticker": "C"}, "company_b": {"ticker": "D"}, "direction": "neutral"},
             {"slug": "e", "company_a": {"ticker": "E"}, "company_b": {"ticker": "F"}, "direction": "uncertain"}]
    prices = [{"slug": "a", "z_last": 3}, {"slug": "c", "z_last": 2.5}, {"slug": "e", "z_last": -1}]
    book = portfolio.exp1_book(pairs, prices)
    # direction wins; otherwise a stretched spread is bet to close (short the rich A side); else cash
    assert [(b["long"], b["short"], b["kind"]) for b in book] == [("B", "A", "bias"), ("D", "C", "convergence")]


def test_three_portfolios_over_two_days(tmp_path, fx):
    site, state = tmp_path / "site", tmp_path / "state" / "portfolio.json"
    # Friday close before the start: the first Exp1 book is formed (long KO / short PEP), nothing earned yet
    make_site(site, [("2026-10-15", 100, 100), ("2026-10-16", 100, 100)])
    r = build_hub.build(site, state, update=True)
    assert r["latest"]["combined"] == 1000.0
    # Monday: KO +2%, PEP -1% -> the KO/PEP slice makes 3%; it is the only position, so Exp1 makes 3%
    make_site(site, [("2026-10-16", 100, 100), ("2026-10-19", 102, 99)])
    write_exp2(site, ["2026-10-16,backfill,scored,5,True,True,18", "2026-10-19,live,scored,1.0,True,True,18",
                      "2026-10-20,live,pending,,True,True,18"])
    build_hub.build(site, state, update=True)
    t = json.loads((site / "data" / "portfolio.json").read_text())
    last = t["rows"][-1]
    assert last["date"] == "2026-10-19"
    assert last["exp1"] == pytest.approx(1030.0) and last["exp2"] == pytest.approx(1010.0)   # the replayed 10-16 is ignored
    assert last["combined"] == pytest.approx(1020.0)
    # Tuesday: Exp1 flat, Exp2 +0.5%; the US dollar gains 1% against CAD, which lifts every pot by 1%
    write_exp2(site, ["2026-10-19,live,scored,1.0,True,True,18", "2026-10-20,live,scored,0.5,False,False,18"])
    make_site(site, [("2026-10-19", 102, 99), ("2026-10-20", 102, 99)])
    build_hub.build(site, state, update=True)
    last = json.loads((site / "data" / "portfolio.json").read_text())["rows"][-1]
    assert last["exp1"] == pytest.approx(1030 * 1.01) and last["exp2"] == pytest.approx(1000 * 1.01 * 1.005 * 1.01)
    hub = (site / "index.html").read_text()
    assert "+" in hub and "CAD" in hub and "long KO / short PEP" in hub and "SPY out (cash)" in hub
    e1 = (site / "exp1" / "index.html").read_text()
    assert "Exp1 paper portfolio" in e1 and e1.count("<!--portfolio-->") == 1
    assert "Exp2 paper portfolio" in (site / "exp2" / "index.html").read_text()


def test_publish_only_does_not_move_the_book(tmp_path):
    site, state = tmp_path / "site", tmp_path / "portfolio.json"
    make_site(site, [("2026-10-16", 100, 100), ("2026-10-19", 100, 100)])
    build_hub.build(site, state, update=True)
    before = state.read_text()
    make_site(site, [("2026-10-19", 100, 100), ("2026-10-20", 110, 100)])
    build_hub.build(site, state, update=False)
    assert state.read_text() == before
    build_hub.build(site, state, update=True)
    build_hub.build(site, state, update=True)          # repeating a run changes nothing
    rows = json.loads(state.read_text())["exp1"]
    assert [r["date"] for r in rows] == ["2026-10-19", "2026-10-20"] and rows[-1]["index"] == pytest.approx(1.1)


def test_missing_fx_keeps_going(tmp_path, monkeypatch):
    def boom(start):
        raise OSError("offline")
    monkeypatch.setattr(portfolio, "fetch_fx", boom)
    site = tmp_path / "site"
    make_site(site, [("2026-10-16", 100, 100), ("2026-10-19", 100, 100)], direction="neutral", z=None)
    r = build_hub.build(site, tmp_path / "p.json", update=True)
    assert "fx_warning" in r and r["latest"]["combined"] == 1000.0
    assert "In cash" in (site / "index.html").read_text() and "exclude currency moves" in (site / "index.html").read_text()


def test_glass_theme_on_every_page_once(tmp_path):
    site = tmp_path / "site"
    make_site(site, [("2026-10-16", 100, 100), ("2026-10-19", 100, 100)])
    write_exp2(site, [])
    for f in ("exp1/index.html", "exp2/index.html"):
        (site / f).write_text("<html><head><title>x</title></head><body><main class=\"wrap\">x</main></body></html>")
    build_hub.build(site, tmp_path / "p.json", update=True)
    build_hub.build(site, tmp_path / "p.json")
    for f in ("index.html", "exp1/index.html", "exp2/index.html"):
        assert "backdrop-filter" in (site / f).read_text()
    assert (site / "exp1" / "index.html").read_text().count("<!--glass-->") == 1


def test_exp7_pot_is_separate_from_the_combined_one(tmp_path, fx):
    site, state = tmp_path / "site", tmp_path / "p.json"
    make_site(site, [("2026-10-16", 100, 100), ("2026-10-19", 100, 100)], direction="neutral", z=None)
    d = site / "exp7" / "data"
    d.mkdir(parents=True)
    (site / "exp7" / "index.html").write_text("<html><head></head><body><main class=\"wrap\">exp7</main></body></html>")
    (d / "daily.csv").write_text("date,ret,trades\n2026-10-16,0.5,3\n2026-10-19,0.02,3\n")   # the day before START is ignored
    (d / "summary.json").write_text(json.dumps({"position": "2026-10-19: 3 trades, day +2.00%; flat overnight"}))
    build_hub.build(site, state, update=True)
    last = json.loads((site / "data" / "portfolio.json").read_text())["rows"][-1]
    assert last["exp7"] == pytest.approx(1020.0) and last["combined"] == pytest.approx(1000.0)
    hub = (site / "index.html").read_text()
    assert "Exp7 VWAP" in hub and "Short-term" in hub and 'href="exp7/index.html">Exp7 VWAP</a>' in hub
    page = (site / "exp7" / "index.html").read_text()
    assert "Exp7 paper portfolio" in page and "3 trades" in page and "backdrop-filter" in page


def test_no_exp7_card_without_exp7(tmp_path):
    site = tmp_path / "site"
    make_site(site, [("2026-10-16", 100, 100), ("2026-10-19", 100, 100)])
    build_hub.build(site, tmp_path / "p.json", update=True)
    assert "Exp7" not in (site / "index.html").read_text()


def test_strategy_pots_split_long_and_short(tmp_path, fx):
    site, state = tmp_path / "site", tmp_path / "p.json"
    make_site(site, [("2026-10-16", 100, 100), ("2026-10-19", 100, 100), ("2026-10-20", 100, 100)], direction="neutral", z=None)
    write_exp2(site, ["2026-10-19,live,scored,0.0,True,True,18", "2026-10-20,live,scored,0.0,True,True,18"])
    d = site / "strategies" / "data"
    d.mkdir(parents=True)
    (site / "strategies" / "index.html").write_text("<html><head></head><body><main class=\"wrap\">pots</main></body></html>")
    # trend joins on 10-19 (the first close only forms the book) and makes 1% on 10-20; reversal joins on 10-20
    (d / "daily.csv").write_text("date,trend,pead,insider,reversal\n2026-10-19,0.0,,,\n2026-10-20,0.01,,,0.0\n")
    (d / "summary.json").write_text(json.dumps({"strategies": {
        "trend": {"name": "Trend following", "category": "Long-term", "what": "ETFs by trend", "position": "long SPY 9%"},
        "reversal": {"name": "Weekly reversal", "category": "Short-term", "what": "losers", "position": "20 stocks"},
        "pead": {"name": "Earnings drift", "category": "Long-term", "what": "beats", "position": "cash"}}}))
    build_hub.build(site, state, update=True)
    last = json.loads((site / "data" / "portfolio.json").read_text())["rows"][-1]
    assert last["trend"] == pytest.approx(1000 * 1.01 * 1.414 / 1.40) and last["reversal"] == pytest.approx(1000.0)   # joined on 10-20, after the currency move
    assert "pead" not in last and last["combined"] == pytest.approx(last["exp1"] / 2 + last["exp2"] / 2)
    hub = (site / "index.html").read_text()
    assert 'id="long"' in hub and 'id="short"' in hub and 'href="#markets">← Markets' in hub
    lo, sh = hub.index('id="long"'), hub.index('id="short"')
    assert "Trend following" in hub[lo:sh] and "Weekly reversal" not in hub[lo:sh] and "Weekly reversal" in hub[sh:]
    assert "4 × 1,000 CAD" in hub and "long SPY 9%" in hub and 'href="strategies/index.html#trend"' in hub
    assert "backdrop-filter" in (site / "strategies" / "index.html").read_text()
    # the best % return on its 1,000 CAD leads the Markets page, the table is ranked, and the overview door names it
    best = hub.index("Best pot so far")
    assert best < hub.index("All paper pots") and 'href="strategies/index.html#trend">Trend following</a> <span class="pos">+2.01%' in hub
    assert "from 1,000 CAD on 2026-10-19" in hub and "a lead this young is mostly luck" in hub
    table = hub[hub.index("<table class=\"mk-pots\">"):]
    assert '<tr class=lead><td class="mono">1</td><td><a href="strategies/index.html#trend">' in table
    assert table.index("Trend following") < table.index("Weekly reversal")
    assert 'Best pot: <b>Trend following</b>' in hub[:hub.index('class="open"')]


def test_no_best_pot_when_nothing_is_up():
    rows = [{"date": "2026-10-16", "a": 1000.0, "b": 1000.0}, {"date": "2026-10-19", "a": 990.0, "b": 1000.0}]
    pots = [("a", "A", "Long-term", "a.html"), ("b", "B", "Short-term", "b.html")]
    assert "None ahead yet" in build_hub.best_pot(rows, pots)
    assert [p[0] for p in build_hub.ranked(rows, pots)] == ["b", "a"]
    assert build_hub.pot_since(rows, "a") == ("2026-10-16", 1) and build_hub.pot_since(rows, "b") == (None, 0)


def test_exp1_pair_that_loses_the_stop_sits_out_one_session(tmp_path):
    site, state = tmp_path / "site", {}
    rows = [("2026-10-16", 50, 100), ("2026-10-19", 50, 100)]
    make_site(site, rows)
    portfolio.update_exp1(state, site)
    portfolio.update_exp1(state, site)
    make_site(site, rows + [("2026-10-20", 46, 100)])          # long KO / short PEP: -8% in the session
    rep = portfolio.update_exp1(state, site)
    assert "stopped: ko-pep" in rep["exp1"]
    book = state["exp1"][-1]["book"]
    assert book[0]["stopped"] and "stopped out" in book[0]["why"]
    make_site(site, rows + [("2026-10-20", 46, 100), ("2026-10-21", 40, 100)])
    portfolio.update_exp1(state, site)
    assert state["exp1"][-1]["ret"] == 0.0                     # in cash while KO fell further
    assert not state["exp1"][-1]["book"][0].get("stopped")     # and back in the next book
    assert "in cash" in build_hub.exp1_positions({"exp1_book": book, "exp1_book_date": "2026-10-20"})


def test_exp2_session_after_a_stop_is_cash():
    r = [("a", 0.01), ("b", -0.025), ("c", 0.03), ("d", -0.01), ("e", 0.02)]
    assert portfolio.with_day_stop(r, 0.02) == [("a", 0.01), ("b", -0.025), ("c", 0.0), ("d", -0.01), ("e", 0.02)]
