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
