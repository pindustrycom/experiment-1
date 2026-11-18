"""Round 2 ports (gekko, zenbot, vnpy, pycryptobot, crypto-trading-bot, smart-money-concepts, binance-trade-bot).

  python run_ports2.py local DATA rows.csv           # BTC/ETH 1h, SPY/QQQ at each bot's own bar size (from 1-minute)
  python run_ports2.py ci DATA BASKET_DATA rows.csv  # CI: BTC/ETH at the bots' own sub-hour bar sizes (Binance 5m),
                                                     #     binance-trade-bot on the 12-coin basket
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import evalkit as ek  # noqa: E402
import ports as P  # noqa: E402
import ports2 as Q  # noqa: E402

FEE = {"spot": 0.001, "perp": 0.0005, "stock": 0.0002}
MODE = sys.argv[1]
DATA = sys.argv[2]
OUT = sys.argv[-1]

# (source, name, function, kind, native bar size): kind "pos" -> position series, "trades" -> trade list;
# shorts=True -> long/short, perpetual fees on crypto
BOTS = [
    ("gekko", "MACD", Q.gk_macd, "pos", "1h", False),
    ("gekko", "PPO", Q.gk_ppo, "pos", "1h", False),
    ("gekko", "RSI", Q.gk_rsi, "pos", "1h", False),
    ("gekko", "StochRSI", Q.gk_stochrsi, "pos", "1h", False),
    ("gekko", "CCI", Q.gk_cci, "pos", "1h", False),
    ("gekko", "DEMA", Q.gk_dema, "pos", "1h", False),
    ("gekko", "TMA", Q.gk_tma, "pos", "1h", False),
    ("gekko", "TSI", Q.gk_tsi, "pos", "1h", False),
    ("gekko", "UO", Q.gk_uo, "pos", "1h", False),
    ("zenbot", "trend_ema (default)", Q.zb_trend_ema, "pos", "2m", False),
    ("zenbot", "macd", Q.zb_macd, "pos", "1h", False),
    ("zenbot", "bollinger", Q.zb_bollinger, "pos", "1h", False),
    ("zenbot", "sar", Q.zb_sar, "pos", "2m", False),
    ("zenbot", "trust_distrust", Q.zb_trust_distrust, "pos", "30m", False),
    ("vnpy", "DoubleMaStrategy", Q.vn_double_ma, "pos", "1m", True),
    ("vnpy", "AtrRsiStrategy", Q.vn_atr_rsi, "trades", "1m", True),
    ("vnpy", "BollChannelStrategy", Q.vn_boll_channel, "trades", "15m", True),
    ("vnpy", "KingKeltnerStrategy", Q.vn_king_keltner, "trades", "5m", True),
    ("vnpy", "TurtleSignalStrategy", Q.vn_turtle, "trades", "1m", True),
    ("pycryptobot", "default (EMA12/26 + MACD + golden cross)", Q.pcb_default, "pos", "1h", False),
    ("crypto-trading-bot", "macd", Q.ctb_macd, "pos", "15m", True),
    ("crypto-trading-bot", "awesome_oscillator_cross_zero", Q.ctb_ao, "pos", "15m", True),
    ("crypto-trading-bot", "cci", Q.ctb_cci, "pos", "15m", True),
    ("crypto-trading-bot", "parabolicsar", Q.ctb_psar, "pos", "15m", True),
    ("smart-money-concepts", "BOS/CHoCH swing 50", Q.smc_structure, "pos", "1h", True),
]
MIN = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
rows = []


def load(base, tf, data=DATA):
    return pd.read_feather(f"{data}/{base}_USDT-{tf}.feather").set_index("date").sort_index()


def resample(d, tf, stock):
    if stock:
        return _rs_stock(d, MIN[tf])
    return d.resample(tf.replace("m", "min")).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                                    "volume": "sum"}).dropna()


def _rs_stock(d1, m):
    if m == 1:
        return d1
    loc = d1.tz_convert("America/New_York")
    g = loc.groupby(pd.Grouper(freq=f"{m}min", origin="start_day", offset="30min")).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.tz_convert("UTC")


def add(src, name, asset, tf, native, d, trades, fee, per_year, note=""):
    net, gross, pos = ek.bar_returns_from_trades(d, trades, fee)
    t0 = pd.Timestamp("2019-01-01", tz=d.index.tz)
    n_tr = sum(1 for t in trades if d.index[t[0]] >= t0)
    r = ek.score(d, net, gross, pos, per_year, n_tr)
    r.update({"source": src, "strategy": name, "asset": asset, "tf": tf, "native_tf": native, "note": note, "fee": fee,
              "win_rate": float(np.mean([(t[4] / t[3] - 1) * t[2] > 2 * fee for t in trades])) if trades else np.nan})
    rows.append(r)
    print(f"{src:20} {asset} {name[:28]:28} {tf:4} n {r['trades']:6d} cagr {r['cagr']:+.1%} gross {r['gross_cagr']:+.1%} "
          f"sh {r['sharpe']:.2f} hold {r['hold_sharpe']:.2f} p {r['p_random']:.3f} z {r['z_random']:+.1f} "
          f"a_t {r['alpha_t']:+.1f}", flush=True)


def run_bot(src, name, fn, kind, native, shorts, asset, d, tf, stock, note=""):
    fee = FEE["stock"] if stock else (FEE["perp"] if shorts else FEE["spot"])
    py = 252 if stock else 365
    out = fn(d)
    trades = P.signal_trades(d, out) if kind == "pos" else out
    add(src, name, asset, tf, native, d, trades, fee, py, note)


def basket_rotation(bdata, coins, tf="5m"):
    """binance-trade-bot on the basket: one coin held at a time; scored against the equal-weight basket."""
    closes = pd.concat({c: load(c, tf, bdata).close for c in coins}, axis=1).dropna()
    closes = closes[closes.index >= pd.Timestamp("2022-10-01", tz="UTC")]
    W = Q.btb_rotation(closes, fee=FEE["spot"], scout_multiplier=5.0, start_coin="BTC")
    R = closes.pct_change().fillna(0.0)
    Wh = W.shift().fillna(0.0)                              # decided at the close, held over the next bar
    turn = W.diff().abs().sum(axis=1).fillna(1.0)           # a jump sells one coin and buys another: 2 sides
    gross = (Wh * R).sum(axis=1)
    net = gross - turn.shift().fillna(0.0) * FEE["spot"]
    bench = R.mean(axis=1)
    dn, dg, db = ek.daily(net), ek.daily(gross), ek.daily(bench)
    out = ek.perf(dn, 365)
    out["gross_cagr"] = ek.perf(dg, 365)["cagr"]
    hp = ek.perf(db, 365)
    out.update({"hold_cagr": hp["cagr"], "hold_sharpe": hp["sharpe"], "hold_dd": hp["max_dd"]})
    b, t = ek.nw_t(dn.to_numpy(), np.column_stack([np.ones(len(dn)), db.reindex(dn.index).fillna(0).to_numpy()]))
    out.update({"alpha_ann": float(b[0] * 365), "alpha_t": float(t[0]), "beta": float(b[1])})
    # random timing: the same time shift applied to every coin's weight column (keeps jumps, holding times, costs)
    Wn, Rn = Wh.to_numpy(), R.to_numpy()
    n = len(Rn)
    bpd = 288
    ms = 20 * bpd

    def shift_stats(a, b_):
        S = np.zeros(len(a))
        for j in range(a.shape[1]):
            S += np.fft.irfft(np.conj(np.fft.rfft(a[:, j])) * np.fft.rfft(b_[:, j]), len(a))
        act = float((a * b_).sum())
        kk = np.arange(len(a))
        kp = (kk >= ms) & (kk <= len(a) - ms) & (kk != 0)
        s = S[kp]
        return (1 + np.sum(s >= act - 1e-12)) / (1 + len(s)), (act - s.mean()) / s.std()
    out["p_random"], out["z_random"] = shift_stats(Wn, Rn)
    mid = n // 2
    out["p_random_h1"], out["z_random_h1"] = shift_stats(Wn[:mid], Rn[:mid])
    out["p_random_h2"], out["z_random_h2"] = shift_stats(Wn[mid:], Rn[mid:])
    out["trades"] = int((W.diff().abs().sum(axis=1) > 0).sum())
    out["exposure"] = 1.0
    out["years"] = len(dn) / 365
    out.update({"source": "binance-trade-bot", "strategy": "ratio jumping (scout_multiplier 5)", "asset": f"basket of {len(coins)}",
                "tf": tf, "native_tf": "seconds", "fee": FEE["spot"],
                "note": f"sample {closes.index[0].date()} to {closes.index[-1].date()}, halves split at the sample midpoint; "
                        "benchmark = equal-weight basket"})
    rows.append(out)
    print({k_: out[k_] for k_ in ("trades", "cagr", "sharpe", "hold_cagr", "hold_sharpe", "p_random", "z_random",
                                  "z_random_h1", "z_random_h2", "alpha_t")}, flush=True)


if MODE == "local":
    for asset in ("BTC", "ETH", "SPY", "QQQ"):
        stock = asset in ("SPY", "QQQ")
        h1 = load(asset, "1h")
        m1 = load(asset, "1m") if stock else None
        cache = {}
        for src, name, fn, kind, native, shorts in BOTS:
            if stock:
                tf = native
                if tf not in cache:
                    cache[tf] = _rs_stock(m1, MIN[tf]) if tf != "1h" else h1
                d, note = cache[tf], ""
            else:
                tf, d = "1h", h1
                note = "" if native == "1h" else f"native {native}, run on 1h here (own bar size in CI)"
            run_bot(src, name, fn, kind, native, shorts, asset, d, tf, stock, note)
        if not stock:
            d1 = load(asset, "1d")
            run_bot("smart-money-concepts", "BOS/CHoCH swing 50", Q.smc_structure, "pos", "1h", True, asset, d1, "1d", False,
                    "daily bars")
    pd.DataFrame(rows).to_csv(OUT, index=False)
elif MODE == "ci":
    bdata = sys.argv[3]
    for asset in ("BTC", "ETH"):
        b5 = load(asset, "5m")
        cache = {}
        for src, name, fn, kind, native, shorts in BOTS:
            if native == "1h":
                continue                       # done locally on the same hourly bars
            tf = {"1m": "5m", "2m": "5m"}.get(native, native)
            if tf not in cache:
                cache[tf] = b5 if tf == "5m" else resample(b5, tf, False)
            note = "" if tf == native else f"native {native}, smallest bars here are 5m"
            run_bot(src, name, fn, kind, native, shorts, asset, cache[tf], tf, False, note)
    coins = os.environ.get("BASKET", "").replace("USDT", "").split(",")
    basket_rotation(bdata, [c for c in coins if c])
    pd.DataFrame(rows).to_csv(OUT, index=False)
