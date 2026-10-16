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
