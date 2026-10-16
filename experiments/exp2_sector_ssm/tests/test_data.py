import pandas as pd

from exp2 import data


def test_rate_limited_tickers_are_retried(tmp_path):
    calls = {}

    def fetch(t, since, key):
        calls[t] = calls.get(t, 0) + 1
        if t == "XLK" and calls[t] == 1:
            raise data.ProviderError("HTTP 429")
        return pd.DataFrame([{"date": "2026-01-02", "close": 1.0, "adj_close": 1.0, "adj_volume": 1.0,
                              "div_cash": 0.0, "split_factor": 1.0}], columns=data.PRICE_COLS)

    out = data.update_prices(tmp_path, ["XLK", "AAPL"], "2010-01-04", "k", fetch=fetch, retry_wait=0)
    assert out["failed"] == {} and sorted(out["updated"]) == ["AAPL", "XLK"]
    assert calls == {"XLK": 2, "AAPL": 1}


def test_yahoo_payload_parses_dividends_and_splits():
    day = 86400
    t0 = 1704204000  # 2024-01-02 14:00 UTC
    payload = {"chart": {"result": [{
        "timestamp": [t0, t0 + day, t0 + 2 * day],
        "indicators": {"quote": [{"close": [10.0, None, 12.0], "volume": [100, 200, 300]}],
                       "adjclose": [{"adjclose": [9.5, None, 12.0]}]},
        "events": {"dividends": {str(t0): {"amount": 0.25, "date": t0}},
                   "splits": {str(t0 + 2 * day): {"numerator": 2, "denominator": 1, "date": t0 + 2 * day}}}}]}}
    df = data.parse_yahoo(payload)
    assert list(df.columns) == data.PRICE_COLS
    assert df["date"].tolist() == ["2024-01-02", "2024-01-04"]
    assert df["div_cash"].tolist() == [0.25, 0.0]
    assert df["split_factor"].tolist() == [1.0, 2.0]


def test_rate_limited_tiingo_falls_back_to_yahoo(monkeypatch):
    calls = []

    def tiingo(t, s, k):
        calls.append(("tiingo", t))
        raise data.ProviderError("https://api.tiingo.com/x: HTTP 429")

    monkeypatch.setattr(data, "fetch_tiingo", tiingo)
    monkeypatch.setattr(data, "fetch_yahoo", lambda t, s, k=None: calls.append(("yahoo", t)) or "ok")
    monkeypatch.setattr(data, "_tiingo_limited", False)
    assert data.fetch_prices("AAPL", "2020-01-01", "k") == "ok"
    assert data.fetch_prices("MSFT", "2020-01-01", "k") == "ok"
    assert calls == [("tiingo", "AAPL"), ("yahoo", "AAPL"), ("yahoo", "MSFT")]
