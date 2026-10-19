import numpy as np
import pandas as pd

from exp7 import backtest, strategy


def _day(prices, day="2024-03-04", vol=100.0):
    ts = pd.date_range(f"{day} 09:30", periods=len(prices), freq="1min", tz="America/New_York")
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": p + 0.01, "low": p - 0.01, "close": p, "volume": vol}, index=ts)


def test_vwap_resets_each_day():
    d = pd.concat([_day([100] * 390), _day([200] * 390, "2024-03-05")])
    d = strategy.indicators(strategy.resample(d, 1))
    assert abs(d.vwap.iloc[0] - 100) < 0.02 and abs(d.vwap.iloc[390] - 200) < 0.02


def test_cross_trades_next_bar_open_and_flat_at_close():
    # falls for an hour, then rises: EMA crosses above VWAP once it has climbed
    p = list(np.linspace(100, 99, 60)) + list(np.linspace(99, 103, 330))
    d = strategy.indicators(strategy.resample(_day(p), 1))
    tr = strategy.trades(d, "long")
    assert len(tr) == 1
    k = d.index.get_loc(tr.entry_time.iloc[0])
    assert d.ema.iloc[k - 1] > d.vwap.iloc[k - 1] and d.ema.iloc[k - 2] <= d.vwap.iloc[k - 2]   # signal on the previous bar's close
    assert tr.entry.iloc[0] == d.open.iloc[k]
    assert tr.exit.iloc[0] == d.close.iloc[-1] and tr.ret.iloc[0] > 0


def test_first_variant_trades_at_most_once():
    rng = np.random.default_rng(0)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 390)))
    d = strategy.indicators(strategy.resample(_day(p), 3))
    assert strategy.trades(d, "first").shape[0] <= 1
    assert strategy.trades(d, "flip").shape[0] >= strategy.trades(d, "first").shape[0]


def test_costs_lower_returns():
    p = list(np.linspace(100, 99, 60)) + list(np.linspace(99, 103, 330))
    d = strategy.indicators(strategy.resample(_day(p), 1))
    tr = strategy.trades(d, "flip")
    days = pd.Index(sorted(d.day.unique()))
    assert strategy.daily(tr, days, 1.0).sum() < strategy.daily(tr, days, 0.0).sum()
    assert backtest.stats(strategy.daily(tr, days, 0.0), tr)["trades"] == len(tr)


def test_live_replay_starts_at_start_and_charges_costs(tmp_path):
    from exp7 import live
    p = list(np.linspace(100, 99, 60)) + list(np.linspace(99, 103, 330))
    d1 = pd.concat([_day(p, "2026-10-16"), _day(p, "2026-10-19")])
    daily, tr = live.replay(d1, "2026-10-19")
    assert list(daily.date) == ["2026-10-19"] and set(tr.day) == {"2026-10-19"}
    gross = strategy.trades(strategy.indicators(strategy.resample(d1, 3)).query("day == '2026-10-19'"), "flip").ret.sum()
    assert abs(daily.ret.iloc[0] - (gross - len(tr) * 2 * 0.3 / 1e4)) < 1e-12


def test_live_run_writes_site(tmp_path):
    from exp7 import live
    p = list(np.linspace(100, 99, 60)) + list(np.linspace(99, 103, 330))
    data = tmp_path / "data" / "SPY"
    data.mkdir(parents=True)
    d = _day(p, "2026-10-19").reset_index(names="date")
    d["date"] = d["date"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    d.to_csv(data / "2026-10.csv", index=False)
    r = live.run(tmp_path / "data", tmp_path / "site")
    assert r["sessions"] == 1
    html = (tmp_path / "site" / "index.html").read_text()
    assert '<main class="wrap">' in html and "</head>" in html
    assert (tmp_path / "site" / "data" / "daily.csv").read_text().startswith("date,ret,trades")
