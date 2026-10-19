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
    assert "Exp7 · VWAP + 9 EMA" in hub and 'href="exp7/index.html">Experiment 7' in hub
    page = (site / "exp7" / "index.html").read_text()
    assert "Exp7 paper portfolio" in page and "3 trades" in page and "backdrop-filter" in page


def test_no_exp7_card_without_exp7(tmp_path):
    site = tmp_path / "site"
    make_site(site, [("2026-10-16", 100, 100), ("2026-10-19", 100, 100)])
    build_hub.build(site, tmp_path / "p.json", update=True)
    assert "Exp7" not in (site / "index.html").read_text()
