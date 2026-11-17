"""Backtest the hand-ported bots (hummingbot, Superalgos, huseinzol05, quant-trading) and score them."""
import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import evalkit as ek
import ports as P

DATA = sys.argv[1]
OUT = sys.argv[2]
FEE = {"crypto": 0.001, "crypto_perp": 0.0005, "stock": 0.0002}


def load(asset, tf):
    d = pd.read_feather(f"{DATA}/{asset}_USDT-{tf}.feather").set_index("date").sort_index()
    return d


def resample_min(d1, m):
    loc = d1.tz_convert("America/New_York")
    g = loc.groupby(pd.Grouper(freq=f"{m}min", origin="start_day", offset="30min")).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.tz_convert("UTC")


rows = []


def add(src, name, asset, tf, d, trades, fee, per_year, start="2019-01-01", note=""):
    net, gross, pos = ek.bar_returns_from_trades(d, trades, fee)
    r = ek.score(d, net, gross, pos, per_year, len([t for t in trades if d.index[t[0]].tz_localize(None) >= pd.Timestamp(start)] if d.index.tz is not None else trades), start=start)
    r.update({"source": src, "strategy": name, "asset": asset, "tf": tf, "native_tf": tf, "note": note,
              "win_rate": float(np.mean([(t[4] / t[3] - 1) * t[2] > 2 * fee for t in trades])) if trades else np.nan})
    rows.append(r)
    print(f"{src:14} {asset} {name:24} {tf:4} n {r['trades']:5d} cagr {r['cagr']:+.1%} gross {r['gross_cagr']:+.1%} sh {r['sharpe']:.2f} "
          f"hold {r['hold_sharpe']:.2f} p {r['p_random']:.3f} z {r['z_random']:+.1f} a_t {r['alpha_t']:+.1f}", flush=True)


ONLY_EXTRA = len(sys.argv) > 3 and sys.argv[3] == "extra"

for asset in ("BTC", "ETH", "SPY", "QQQ"):
    crypto = asset in ("BTC", "ETH")
    py = 365 if crypto else 252
    fee = FEE["crypto"] if crypto else FEE["stock"]
    for tf in ("1d", "1h"):
        d = load(asset, tf)
        add("backtesting.py", "SmaCross 10/20 (README)", asset, tf, d, P.signal_trades(d, P.btpy_smacross(d).to_numpy()), fee, py)
    if ONLY_EXTRA:
        continue
    crypto = asset in ("BTC", "ETH")
    py = 365 if crypto else 252
    fee = FEE["crypto"] if crypto else FEE["stock"]
    dd = load(asset, "1d")
    # ---- hummingbot (directional V2 controllers, defaults) ----
    if crypto:
        hb, hbtf, tl, cd, eod = load(asset, "1h"), "1h", 15, 1, None          # 45 min on 3m candles = 15 bars; scaled to 1h bars
        hbfee = FEE["crypto_perp"]
        note = "3m defaults scaled to 1h bars (time limit 15 bars)"
    else:
        hb = resample_min(load(asset, "1m"), 3)
        hbtf, tl, cd, hbfee, note = "3m", 15, 2, FEE["stock"], "native 3m, 45 min limit, flat at the close"
        day = hb.index.tz_convert("America/New_York").normalize()
        eod = np.r_[day[1:] != day[:-1], True]
    for name, fn in (("bollinger_v1", P.hb_bollinger_v1), ("macd_bb_v1", P.hb_macd_bb_v1), ("supertrend_v1", P.hb_supertrend_v1)):
        sig = fn(hb)
        tr = P.triple_barrier(hb, sig, sl=0.03, tp=0.02, time_bars=tl, cooldown_bars=cd, allow_short=True, flat_eod=eod)
        add("hummingbot", name, asset, hbtf, hb, tr, hbfee, py, note=note)
    # ---- Superalgos (BTC/USDT 1h systems, short-side) ----
    if crypto:
        h1 = load(asset, "1h")
        h1n = h1.copy(); h1n.index = h1n.index.tz_localize(None)
        for name, fn in (("Weak Hands Buster", P.sa_whb), ("BB Top Bounce", P.sa_bbtb)):
            tr = fn(h1n)
            add("Superalgos", name, asset, "1h", h1, tr, fee, py, note="approximate port, short side")
    # ---- huseinzol05 agents (daily, as in the notebooks) ----
    yr = 365 if crypto else 252
    for name, pos in (("turtle-agent", P.hz_turtle(dd, int(0.10 * yr))),
                      ("moving-average-agent", P.hz_moving_average(dd, int(0.025 * yr), int(0.05 * yr))),
                      ("signal-rolling-agent", P.hz_signal_rolling(dd, 5))):
        add("huseinzol05", name, asset, "1d", dd, P.signal_trades(dd, pos.to_numpy()), fee, py)
    for seed in range(5):
        pos = P.hz_es_agent(dd, pd.Timestamp("2023-01-01", tz="UTC"), fee, seed=seed, iters=200)
        add("huseinzol05", f"evolution-strategy-agent s{seed}", asset, "1d", dd, P.signal_trades(dd, pos.to_numpy()), fee, py,
            start="2023-01-01", note="trained 2019-2022, scored 2023+ only")
    # ---- quant-trading extras (daily) ----
    for name, pos in (("awesome_oscillator", P.qt_awesome(dd)), ("parabolic_sar", P.qt_parabolic_sar(dd)),
                      ("heikin_ashi", P.qt_heikin_ashi(dd)), ("rsi_pattern", P.qt_rsi_pattern(dd)),
                      ("bollinger_w", P.qt_bollinger_w(dd))):
        add("quant-trading", name, asset, "1d", dd, P.signal_trades(dd, pos.to_numpy()), fee, py)

pd.DataFrame(rows).to_csv(OUT, index=False)
