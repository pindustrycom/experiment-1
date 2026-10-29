import numpy as np
import pandas as pd

from exp2 import papers


def test_membership_walks_changes_backwards():
    cur = pd.DataFrame({"ticker": ["A", "B", "C"], "sector": "x"})
    changes = pd.DataFrame({"date": pd.to_datetime(["2020-06-15", "2015-03-02"]), "added": ["C", "B"], "removed": ["D", "E"]})
    months = pd.date_range("2014-01-31", "2021-12-31", freq="ME")
    m = papers.membership(cur, changes, months)
    assert m.loc["2021-01-31", ["A", "B", "C"]].all() and not m.loc["2021-01-31", "D"]
    assert m.loc["2019-01-31", "D"] and not m.loc["2019-01-31", "C"] and m.loc["2019-01-31", "B"]
    assert m.loc["2014-06-30", "E"] and not m.loc["2014-06-30", "B"]


def test_sv_filter_follows_variance_regimes():
    rng = np.random.default_rng(0)
    lv = np.where((np.arange(3000) // 250) % 2, np.log(4.0), np.log(0.25))
    r = rng.normal(size=3000) * np.exp(lv / 2)
    h = papers.sv_filter(r, np.arange(3000) < 2000)
    assert np.corrcoef(h[:-1], lv[1:])[0, 1] > 0.8


def _px(T=1500, n=80, seed=0):
    """Stocks whose next-month return is planted on their past 12-month momentum."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2003-01-01", periods=T)
    mkt = rng.normal(0, 0.01, T)
    px = {}
    drift = rng.normal(0, 0.0008, n)
    for i in range(n):
        r = mkt + drift[i] + rng.normal(0, 0.015, T)
        c = 50 * np.exp(np.cumsum(r))
        o = c * np.exp(rng.normal(0, 0.003, T))
        px[f"S{i}"] = pd.DataFrame({"open": o, "close": c, "adj_close": c, "volume": 1e6}, index=idx)
    spy = pd.Series(50 * np.exp(np.cumsum(mkt)), index=idx)
    return px, spy


def test_panel_and_deciles():
    px, spy = _px()
    months = pd.date_range("2003-01-31", "2008-12-31", freq="ME")
    mem = pd.DataFrame(True, index=months, columns=list(px))
    P = papers.panel(px, spy, {}, mem)
    assert set(papers.FEATURES) <= set(P.columns) and P["y"].notna().sum() > 3000
    # persistent drifts make momentum a real signal here: top-decile momentum should beat bottom
    s = papers.deciles(P, "mom12_2")
    assert s.mean() > 0
    o = papers.overnight(P, train_end="2005-12-31")
    assert "on12" in o["test_2019_on"] or o["1999_2018"]["on12"]["total"]["n"] > 10


def test_run_end_to_end(tmp_path):
    px, spy = _px(T=2600, n=150)
    d = tmp_path / "prices"
    d.mkdir()
    for t, df in px.items():
        df.rename_axis("date").reset_index().assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_csv(d / f"{t}.csv", index=False)
    idx = spy.index.strftime("%Y-%m-%d")
    rv = spy.pct_change().rolling(21).std().bfill() * np.sqrt(252) * 100
    for t, c in (("SPY", spy.to_numpy()), ("^VIX", rv.to_numpy() * 1.2), ("^IRX", np.full(len(spy), 2.0))):
        pd.DataFrame({"date": idx, "open": c, "close": c, "adj_close": c, "volume": 1e6}).to_csv(d / f"{t}.csv", index=False)
    pd.DataFrame({"ticker": list(px), "sector": ["a", "b", "c"] * 50}).to_csv(tmp_path / "sp500_current.csv", index=False)
    pd.DataFrame({"date": ["2001-01-01"], "added": ["S0"], "removed": ["ZZZ"]}).to_csv(tmp_path / "sp500_changes.csv", index=False)
    rep = papers.run(tmp_path, tmp_path / "out", fetch=False, train_end="2009-12-31")
    assert not rep["warnings"], rep["warnings"]
    assert {"h1", "h3", "h6"} <= set(rep["vrp"])
    g = rep["gkx"]
    assert {"ols3", "ridge", "gbt", "nn"} <= set(g)
    # momentum is planted, so every model should rank stocks profitably before costs dominate
    assert g["ridge"]["test_2019_on"]["ls_equal"]["mean_pct_month"] > 0


def test_parses_wikipedia_tables():
    html = """<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
    <tr><td>MMM</td><td>3M</td><td>Industrials</td></tr><tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td></tr></table>
    <table><tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
    <tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
    <tr><td>June 23, 2025</td><td>MMM</td><td>3M</td><td>OLD</td><td>Old Co</td><td>x</td></tr></table>"""
    cur, ch = papers.sp500_tables(html)
    assert list(cur["ticker"]) == ["MMM", "BRK.B"] and cur["sector"][0] == "Industrials"
    assert ch["added"][0] == "MMM" and ch["removed"][0] == "OLD" and ch["date"][0] == pd.Timestamp("2025-06-23")
    assert papers.ysym("BRK.B") == "BRK-B"


def test_membership_from_dates_added():
    html = """<table><tr><th>Symbol</th><th>GICS Sector</th><th>Date added</th></tr>
    <tr><td>OLD</td><td>Energy</td><td>1957-03-04</td></tr><tr><td>NEW</td><td>Energy</td><td>2021-06-01</td></tr></table>"""
    cur, ch = papers.sp500_tables(html)
    m = papers.membership(cur, ch, pd.date_range("2019-01-31", "2022-12-31", freq="ME"))
    assert m.loc["2020-01-31", "OLD"] and not m.loc["2020-01-31", "NEW"] and m.loc["2022-01-31", "NEW"]


def test_point_in_time_tables(tmp_path):
    from exp2 import pit

    iv = pd.DataFrame({"ticker": ["A", "B", "OLD-201905", "C"],
                       "start": pd.to_datetime(["1990-01-02", "2015-03-02", "2001-01-02", "2010-01-04"]),
                       "end": pd.to_datetime([None, None, "2019-05-20", "2017-01-03"])})
    (tmp_path / "pit").mkdir()
    iv.to_parquet(tmp_path / "pit" / "membership_intervals.parquet")
    days = pd.bdate_range("2000-01-03", "2020-12-31")
    rng = np.random.default_rng(0)
    xlk = np.cumsum(rng.normal(0, 0.01, len(days)))
    rows = []
    for t, x in (("XLK", xlk), ("XLE", np.cumsum(rng.normal(0, 0.01, len(days)))), ("A", xlk), ("B", xlk), ("C", xlk),
                 ("OLD-201905", xlk + np.cumsum(rng.normal(0, 0.002, len(days))))):
        p = 50 * np.exp(x)
        if t == "OLD-201905":
            p = p[days < "2019-05-21"]
        rows.append(pd.DataFrame({"date": days[: len(p)], "ticker": t, "open": p, "close": p, "adj_close": p, "volume": 1e6}))
    pd.concat(rows).to_parquet(tmp_path / "pit" / "prices.parquet")
    rep = pit.refresh(tmp_path, "2005-01-01", gics=pd.DataFrame({"ticker": ["A"], "sector": ["Health Care"]}), fetch=False)
    assert rep["members_today"] == 2 and rep["former_members"] == 2 and rep["former_priced"] == 2
    assert (tmp_path / "prices" / "OLD-201905.csv").exists() and not (tmp_path / "prices" / "A.csv").exists()
    sec = pit.sectors(tmp_path)
    assert sec["A"] == "Health Care" and sec["OLD-201905"] == "Information Technology"
    cur = pd.read_csv(tmp_path / "sp500_current.csv")
    ch = pd.read_csv(tmp_path / "sp500_changes.csv", parse_dates=["date"])
    m = papers.membership(cur, ch, pd.date_range("2008-01-31", "2020-12-31", freq="ME"))
    assert m.loc["2016-06-30", ["A", "B", "C", "OLD-201905"]].all()
    assert not m.loc["2018-06-30", "C"] and m.loc["2018-06-30", "OLD-201905"] and not m.loc["2019-05-31", "OLD-201905"]
    assert not m.loc["2014-12-31", "B"]
    assert pit.former(tmp_path) == {"C", "OLD-201905"}
    out = papers.update(tmp_path, ["OLD-201905"], pause=0)
    assert out == {"fetched": 0, "failed": []}       # former members are never sent to Yahoo
