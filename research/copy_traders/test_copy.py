import numpy as np
import pandas as pd

import activist13d as a13
import backtest as bt

MASTER = """Description: Master Index of EDGAR Dissemination Feed
CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
111|ACME CORP|SC 13D|2024-03-04|edgar/data/111/0001.txt
222|STARBOARD VALUE LP|SC 13D|2024-03-04|edgar/data/111/0001.txt
111|ACME CORP|SC 13D/A|2024-05-01|edgar/data/111/0002.txt
333|BETA INC|SCHEDULE 13D|2025-06-02|edgar/data/333/0003.txt
444|SMITH JOHN|SCHEDULE 13D|2025-06-02|edgar/data/333/0003.txt
555|PRIVATE CO|SC 13D|2025-06-03|edgar/data/555/0004.txt
666|SOMEONE|SC 13D|2025-06-03|edgar/data/555/0004.txt
"""


def prices(n=400, tickers=("SPY", "AAA", "BBB")):
    idx = pd.bdate_range("2024-01-02", periods=n)
    r = np.random.default_rng(1).normal(0.0005, 0.01, (n, len(tickers)))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=list(tickers))


def test_parse_and_pair():
    rows = a13.parse_master(MASTER)
    assert len(rows) == 6 and "SC 13D/A" not in set(rows["form"])          # amendments are not new stakes
    ev = a13.pair_up(rows, {111: "ACME", 333: "BETA", 222: "SBV"})
    assert list(ev["ticker"]) == ["ACME", "BETA"]                            # PRIVATE CO has no ticker
    assert ev.loc[0, "activist"] == "Starboard" and pd.isna(ev.loc[1, "activist"])


def test_event_study_and_portfolio():
    px = prices()
    ev = pd.DataFrame({"filed": [px.index[10], px.index[300]], "ticker": ["AAA", "BBB"], "activist": ["X", "Y"]})
    es = a13.event_study(ev, px)
    d0 = px.index[11]
    want = px.at[px.index[11 + 21], "AAA"] / px.at[d0, "AAA"] - 1 - (px.at[px.index[32], "SPY"] / px.at[d0, "SPY"] - 1)
    assert abs(es.loc[0, "x21"] - want) < 1e-12 and pd.isna(es.loc[1].get("x252"))
    r = a13.portfolio(es, px, 126)
    assert r.index[0] == px.index[12] and (r.loc[px.index[140]:px.index[300]] == 0).all()   # in cash between the two


def test_pelosi_sells_on_disclosed_sale():
    px = prices(tickers=("SPY", "NVDA"))
    d = px.index
    rows = pd.DataFrame({"last": ["Pelosi"] * 3, "ticker": ["NVDA"] * 3, "type": ["P", "P", "S (partial)"],
                         "asset_type": ["OP", "ST", "ST"], "date": [d[5], d[5], d[50]], "filed": [d[20], d[20], d[70]]})
    r, info = bt.pelosi(rows, px, hold=252)
    assert info["buys"] == 1                                                 # shares + calls the same day = one signal
    t = info["trades"][0]
    assert t["entry"] == str(d[21].date()) and t["exit"] == str(d[71].date())
    assert r.index[0] == d[22] and r.loc[d[72]] < 0 and (r.loc[d[73]:] == 0).all()   # exit cost, then cash
