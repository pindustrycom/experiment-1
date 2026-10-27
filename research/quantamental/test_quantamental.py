import numpy as np
import pandas as pd

import backtest
import sec
import stats as st
import text


def fact(start, end, val, filed, form="10-K"):
    d = {"end": end, "val": val, "filed": filed, "form": form, "fy": 0, "fp": "FY", "accn": "x"}
    if start:
        d["start"] = start
    return d


def facts_doc():
    g = lambda *rows: {"units": {"USD": list(rows)}}
    return {"facts": {"us-gaap": {
        "NetIncomeLoss": g(fact("2019-01-01", "2019-12-31", 10, "2020-02-20"), fact("2020-01-01", "2020-12-31", 12, "2021-02-20"),
                           fact("2020-01-01", "2020-12-31", 99, "2022-02-20"),            # restated a year later
                           fact("2020-01-01", "2020-03-31", 3, "2020-05-01", "10-Q")),
        "Assets": g(fact(None, "2019-12-31", 100, "2020-02-20"), fact(None, "2020-12-31", 110, "2021-02-20")),
    }, "dei": {"EntityPublicFloat": {"units": {"USD": [fact(None, "2020-06-30", 500, "2021-02-20")]}}}}}


def test_annual_table_is_point_in_time():
    a = sec.annual_table(facts_doc())
    y20 = a[a["end"] == "2020-12-31"].iloc[0]
    assert y20["avail"] == pd.Timestamp("2021-02-20")
    assert y20["ni_0"] == 12          # the 2022 restatement is not known in 2021
    assert y20["ni_1"] == 10 and y20["assets_1"] == 100
    assert y20["float_usd"] == 500
    assert len(a) == 2                # the quarter is not a year


def test_piotroski_counts_checks():
    f = pd.DataFrame([{"ni_0": 10, "ni_1": 5, "cfo_0": 20, "assets_0": 100, "assets_1": 100, "assets_2": 100, "ltd_0": 10, "ltd_1": 20,
                       "ca_0": 3, "cl_0": 1, "ca_1": 2, "cl_1": 1, "shares_0": 10, "shares_1": 10, "gp_0": 50, "gp_1": 40, "rev_0": 100,
                       "rev_1": 100, "cogs_0": np.nan, "cogs_1": np.nan, "ltd_2": 0}])
    p = backtest.piotroski(f)
    assert p["fscore"].iloc[0] == 8   # all but asset turnover (flat)


def test_word_counts():
    lists = {"lm_neg": frozenset({"LOSS"}), "h4_neg": frozenset({"LIABILITY", "TAX"})}
    c = text.count(text.clean("<p>Deferred tax liability; no loss.</p><ix:header>LOSS LOSS</ix:header>"), lists)
    assert c == {"words": 5, "lm_neg": 1, "h4_neg": 2}


def test_word_lists_load():
    w = text.word_lists()
    assert "LOSS" in w["lm_neg"] and "LIABILITY" not in w["lm_neg"]
    assert "LIABILITY" in w["h4_neg"] or "TAX" in w["h4_neg"]


def test_deflated_sharpe_penalises_many_trials():
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.04, 180))
    one = st.deflated_sharpe(r, 1, 0.01)["dsr"]
    many = st.deflated_sharpe(r, 300, 0.01)["dsr"]
    assert many < one


def synthetic(tmp, n=40):
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2008-01-01", "2026-10-15")
    (tmp / "prices").mkdir()
    spy = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(days))))
    pd.DataFrame({"date": days, "close": spy, "adj_close": spy}).to_csv(tmp / "prices" / "SPY.csv", index=False)
    tick = [f"T{i}" for i in range(n)]
    rows = []
    for i, t in enumerate(tick):
        px = 50 * np.exp(np.cumsum(rng.normal(0.0002 + 0.0001 * (i % 3), 0.015, len(days))))
        pd.DataFrame({"date": days, "close": px, "adj_close": px}).to_csv(tmp / "prices" / f"{t}.csv", index=False)
        for y in range(2008, 2026):
            r = {"ticker": t, "cik": i + 1, "end": pd.Timestamp(f"{y}-12-31"), "avail": pd.Timestamp(f"{y + 1}-02-25"),
                 "float_usd": 1e9 * (1 + i), "float_date": pd.Timestamp(f"{y}-06-30")}
            for lag in range(5):
                for it in sec.ITEMS:
                    r[f"{it}_{lag}"] = abs(rng.normal(100, 20))
            rows.append(r)
    pd.DataFrame({"ticker": tick, "sector": "X"}).to_csv(tmp / "sp500_current.csv", index=False)
    pd.DataFrame(columns=["date", "added", "removed"]).to_csv(tmp / "sp500_changes.csv", index=False)
    pd.DataFrame(rows).to_csv(tmp / "fund.csv", index=False)
    texts = pd.DataFrame([{"cik": i + 1, "accn": f"{i}-{y}", "filed": f"{y + 1}-02-25", "words": 50000, "lm_neg": rng.integers(300, 900),
                           "lm_pos": 200, "lm_unc": 300, "h4_neg": rng.integers(2000, 3000), "h4_pos": 2500}
                          for i in range(n) for y in range(2008, 2026)])
    texts.to_csv(tmp / "text.csv", index=False)


def test_backtest_end_to_end(tmp_path, monkeypatch):
    synthetic(tmp_path)
    monkeypatch.setattr(backtest, "french_factors", lambda: None)
    rep = backtest.run(tmp_path, tmp_path / "fund.csv", tmp_path / "text.csv", tmp_path / "out")
    res = rep["results"]
    assert "QMJ: top 30% minus bottom 30%, value weighted" in res and "SPY" in res
    assert "10-K negativity falling, finance list (LM): top minus bottom 20%" in res
    # random signals: no long-short book should look like an edge
    assert all(abs(v["alpha_t"]) < 3.5 for k, v in res.items() if "minus" in k)
    assert rep["filing_reaction"]["filings"] > 100


def test_edge_pipeline_end_to_end(tmp_path, monkeypatch):
    import markets
    import pipeline

    synthetic(tmp_path)
    fund = pd.read_csv(tmp_path / "fund.csv", parse_dates=["end", "avail", "float_date"])
    P, spy, _ = backtest.build_panel(tmp_path, fund, pd.read_csv(tmp_path / "text.csv"))
    res = pipeline.run(P, pipeline.STOCK_SIGNALS, spy, "2013-01-31", 0.001, 0.0004)
    assert "momentum 12-1" in res["ic"].index
    assert "long-short, covariance" in res["results"].index
    # walk-forward weights only use months before each formation month
    assert all(pipeline.weights_at(pd.DataFrame({"a": [0.5] * 40}, index=pd.date_range("2010-01-31", periods=40, freq="ME")),
                                   pd.Timestamp("2012-12-31")).empty for _ in [0])

    rng = np.random.default_rng(3)
    days = pd.bdate_range("2003-01-01", "2026-10-15")

    def fake(t, cache):
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(days)))), index=days, name="adj_close")

    monkeypatch.setattr(markets, "fetch", fake)
    Pm, spy_m = markets.panel(tmp_path / "m")
    rm = pipeline.run(Pm, markets.SIGNALS, spy_m, "2010-01-31", 0.0003, 0.00025)
    assert Pm["ticker"].nunique() == sum(len(v) for v in markets.UNIVERSE.values())
    assert "every name, equal weight (baseline)" in rm["results"].index
