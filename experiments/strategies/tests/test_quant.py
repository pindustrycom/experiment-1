import json

import numpy as np
import pandas as pd
from test_strats import prices

from strats import events, live, quant, regimes, rules, surface


def test_implied_vol_inverts_black():
    F, K, T, D = 100.0, np.array([70, 90, 100, 110, 140.0]), 0.5, 0.98
    vol = np.array([0.35, 0.24, 0.2, 0.18, 0.22])
    for call in (False, True):
        p = surface.black(np.full(5, F), K, np.full(5, T), np.full(5, D), vol, np.full(5, call))
        got = surface.implied_vol(p, np.full(5, F), K, np.full(5, T), np.full(5, D), np.full(5, call))
        assert np.allclose(got, vol, atol=1e-6)
    # a price below intrinsic value has no vol
    assert np.isnan(surface.implied_vol([0.5], [100.0], [80.0], [0.5], [1.0], [True])[0])


def chain(spot=500.0, day="2026-10-23", skew=0.3, step=5.0, nd=4):
    rows = []
    for dte in (14, 30, 60, 120, 250):
        T, D = dte / 365, np.exp(-0.04 * dte / 365)
        F = spot * np.exp(0.03 * T)
        for K in np.arange(spot * 0.7, spot * 1.25, step):
            iv = 0.18 - skew * np.log(K / F) + 0.5 * np.log(K / F) ** 2
            for cp in ("call", "put"):
                p = float(surface.black(np.array([F]), np.array([K]), np.array([T]), np.array([D]), np.array([iv]), np.array([cp == "call"]))[0])
                rows.append({"expiration": pd.Timestamp(day) + pd.Timedelta(days=dte), "option_type": cp, "strike": K,
                             "bid": round(p * 0.99, nd), "ask": round(p * 1.01, nd), "iv": iv, "open_interest": 1, "volume": 1})
    return pd.DataFrame(rows)


def test_surface_recovers_the_forward_and_the_skew():
    s = surface.surface(chain(), 500.0, "2026-10-23")
    e = next(x for x in s["expiries"] if x["dte"] == 30)
    assert abs(e["forward"] - 500 * np.exp(0.03 * 30 / 365)) < 0.05 and abs(e["discount"] - np.exp(-0.04 * 30 / 365)) < 1e-4
    assert abs(s["atm_30d"] - 0.18) < 0.003 and s["skew_30d"] > 0.05           # puts below the market cost more
    assert s["cboe_median_gap"] < 0.003


def test_one_dollar_strikes_quoted_in_cents_keep_every_expiry():
    c = chain(spot=760.0, step=1.0, nd=2)
    noise = np.random.default_rng(3).normal(0, 0.03, len(c))         # quotes a few cents off fair, as real ones are
    mid = (c["bid"] + c["ask"]) / 2 + noise
    c["bid"], c["ask"] = (mid - 0.03).round(2).clip(lower=0.01), (mid + 0.03).round(2)
    s = surface.surface(c, 760.0, "2026-10-23")
    assert [x["dte"] for x in s["expiries"]] == [14, 30, 60, 120, 250] and s["dropped_expiries"] == {"no forward": 0, "few quotes": 0}
    assert abs(s["atm_30d"] - 0.18) < 0.004


def test_parse_cboe_reads_occ_symbols():
    ch, spot, day = surface.parse_cboe({"timestamp": "2026-10-23 16:15:00", "data": {"current_price": 742.5, "options": [
        {"option": "SPY261120C00745000", "bid": 10.1, "ask": 10.3, "iv": 0.15, "open_interest": 5, "volume": 2},
        {"option": "SPY261120P00700000", "bid": 3.0, "ask": 3.1, "iv": 0.2, "open_interest": 5, "volume": 2}]}})
    assert spot == 742.5 and day == "2026-10-23"
    assert list(ch["option_type"]) == ["call", "put"] and list(ch["strike"]) == [745.0, 700.0]
    assert ch["expiration"].iloc[0] == pd.Timestamp("2026-11-20")


def test_regimes_are_named_by_volatility():
    rng = np.random.default_rng(1)
    calm = rng.normal(0.0006, 0.005, 400)
    trend = rng.normal(0.0015, 0.011, 400)
    crash = rng.normal(-0.004, 0.03, 200)
    r = np.r_[calm, trend, crash, calm]
    spy = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2017-01-02", periods=len(r)))
    f = regimes.features(spy)
    labels, cen = regimes.fit(f)
    assert list(cen.sort_values("vol").index) == list(regimes.NAMES)
    assert labels.iloc[300] == "Calm" and labels.iloc[900] == "Crisis"
    out = regimes.by_regime(pd.DataFrame({"x": pd.Series(r, index=spy.index)}), labels)
    assert out["x"]["Crisis"]["ann_ret_pct"] < 0 < out["x"]["Calm"]["ann_ret_pct"]


def test_event_loop_matches_the_replay_with_its_assumptions_and_never_sees_ahead():
    px = prices(n=500)
    members = [c for c in px.columns if c.startswith("S")]
    rep = live.replay(px, members, start="2019-03-01")
    res = events.compare(px, None, members, "2019-03-01", rep, keys=("reversal",))
    assert res["reversal"]["check"]["corr"] > 0.99
    # the strategy only ever gets the history up to the session it is called for
    seen = []
    orig = rules.reversal_weights

    def spy(h, d, m):
        seen.append(h.index[-1] == d)
        return orig(h, d, m)
    rules.reversal_weights = spy
    try:
        events.run("reversal", px, None, members, "2019-03-01", events.SETTINGS[1])
    finally:
        rules.reversal_weights = orig
    assert seen and all(seen)


def test_whole_shares_and_costs_lower_the_result():
    px = prices(n=500)
    members = [c for c in px.columns if c.startswith("S")]
    free = events.run("reversal", px, None, members, "2019-03-01", ("free", "open", 0.0, (0.0, 0.0), 0.0, None))
    paid = events.run("reversal", px, None, members, "2019-03-01", events.SETTINGS[2])
    assert paid["end"] < free["end"] and paid["cost_pct_per_year"] > 0 and free["cost_pct_per_year"] == 0


def test_quant_page_builds_offline(tmp_path):
    d = tmp_path / "data"
    (d / "prices").mkdir(parents=True)
    px = prices(n=700)
    for t in px.columns:
        pd.DataFrame({"adj_close": px[t], "adj_open": px[t] * 0.999}).to_csv(d / "prices" / f"{t}.csv", index_label="date")
    pd.DataFrame({"ticker": [c for c in px.columns if c.startswith("S")]}).to_csv(d / "sp500.csv", index=False)
    day = d / "chains" / "2026-10-23"
    day.mkdir(parents=True)
    chain().to_csv(day / "SPY.csv.gz", index=False)
    (day / "spot.json").write_text(json.dumps({"SPY": 500.0}))
    rep = quant.run(d, None, tmp_path / "site", fetch=False)
    page = (tmp_path / "site" / "index.html").read_text()
    assert "Quant lab" in page and "const Q=" in page and rep["surface"]["SPY"]["contracts"] > 50
    assert set(rep["events"]) == {"trend", "reversal"} and rep["regimes"]["today"]["regime"] in regimes.NAMES
