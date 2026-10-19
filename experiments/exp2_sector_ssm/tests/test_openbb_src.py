import os

import pandas as pd
import pytest

from exp2 import openbb_src, papers


def _series(close, start="2020-01-01"):
    d = pd.bdate_range(start, periods=len(close)).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": d, "open": close, "close": close, "adj_close": close, "volume": None})


def test_frame_matches_the_papers_layout():
    raw = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-03"]), "open": [1, 2, 3], "close": [13.0, 12.5, 13.5]})
    df = openbb_src.frame(raw)
    assert list(df.columns) == ["date", "open", "close", "adj_close", "volume"]
    assert df["date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert df["adj_close"].tolist() == [12.5, 13.5]


def test_agreement_uses_each_tickers_units():
    old = _series([20.0] * 100)
    assert openbb_src.agrees("^VIX", old, _series([20.1] * 100))          # 0.5%
    assert not openbb_src.agrees("^VIX", old, _series([21.0] * 100))      # 5%
    rates = _series([4.0] * 100)
    assert openbb_src.agrees("^IRX", rates, _series([4.1] * 100))         # 0.1 points
    assert not openbb_src.agrees("^IRX", rates, _series([0.04] * 100))    # decimals, not percent
    assert openbb_src.agrees("^VIX", None, _series([21.0] * 100))
    assert not openbb_src.agrees("^VIX", old, _series([20.0] * 100, start="2030-01-01"))   # no overlap


@pytest.fixture
def yahoo_down(monkeypatch):
    def fail(t, start=papers.START):
        raise RuntimeError("yahoo down")
    monkeypatch.setattr(papers, "fetch_yahoo", fail)


def test_update_falls_back_to_openbb_for_vix_only(tmp_path, monkeypatch, yahoo_down):
    calls = []
    monkeypatch.setattr(openbb_src, "fetch", lambda t, start: calls.append(t) or _series([20.0] * 100))
    out = papers.update(tmp_path, ["^VIX", "AAPL"], pause=0)
    assert calls == ["^VIX"] and out["openbb"] == ["^VIX"] and out["failed"] == ["AAPL"] and out["fetched"] == 1
    assert len(papers.load(tmp_path, ["^VIX"])["^VIX"]) == 100


def test_update_keeps_the_stored_file_when_openbb_disagrees(tmp_path, monkeypatch, yahoo_down):
    (tmp_path / "prices").mkdir()
    f = tmp_path / "prices" / "^IRX.csv"
    _series([4.0] * 100).to_csv(f, index=False)
    os.utime(f, (0, 0))                                                    # stale, so update() refetches
    monkeypatch.setattr(openbb_src, "fetch", lambda t, start: _series([0.04] * 100))
    out = papers.update(tmp_path, ["^IRX"], pause=0)
    assert out["failed"] == ["^IRX"] and out["openbb_refused"] == ["^IRX"]
    assert pd.read_csv(f)["close"].iloc[0] == 4.0


def test_update_survives_openbb_errors(tmp_path, monkeypatch, yahoo_down):
    def boom(t, start):
        raise ImportError("no openbb")
    monkeypatch.setattr(openbb_src, "fetch", boom)
    out = papers.update(tmp_path, ["^VIX"], pause=0)
    assert out["failed"] == ["^VIX"] and out["openbb_refused"] == ["^VIX"]
