"""Round 3 ports (see ports3.py) scored like every other row.

  python run_ports3.py local DATA rows.csv     # BTC/ETH 1h + 1d, SPY/QQQ at 1m/5m/1h/1d
  python run_ports3.py ci DATA rows.csv        # BTC/ETH at the bots' own sub-hour bar sizes (Binance 5m)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evalkit as ek  # noqa: E402
import ports as P  # noqa: E402
import ports3 as R  # noqa: E402

PATTERNPY = os.environ.get("PATTERNPY", "")          # folder holding tradingpatterns.py (keithorange/PatternPy)
if PATTERNPY:
    sys.path.insert(0, PATTERNPY)

FEE = {"spot": 0.001, "perp": 0.0005, "stock": 0.0002}
rows = []


def load(data, base, tf):
    return pd.read_feather(f"{data}/{base}_USDT-{tf}.feather").set_index("date").sort_index()


def fee_for(stock, shorts):
    return FEE["stock"] if stock else (FEE["perp"] if shorts else FEE["spot"])


def emit(r, src, name, asset, tf, native, fee, note, win_rate=np.nan):
    r.update({"source": src, "strategy": name, "asset": asset, "tf": tf, "native_tf": native, "fee": fee, "note": note,
              "win_rate": win_rate})
    rows.append(r)
    print(f"{src:22} {asset:8} {name[:30]:30} {tf:4} n {r['trades']:6d} cagr {r['cagr']:+.1%} gross {r['gross_cagr']:+.1%} "
          f"sh {r['sharpe']:.2f} hold {r['hold_sharpe']:.2f} p {r['p_random']:.3f} z {r['z_random']:+.1f} "
          f"h1 {r['z_random_h1']:+.1f} h2 {r['z_random_h2']:+.1f} a_t {r['alpha_t']:+.1f}", flush=True)


def run(src, name, kind, fn, asset, d, tf, native, stock, shorts, note="", start="2019-01-01"):
    fee = fee_for(stock, shorts)
    py = 252 if stock else 365
    try:
        out = fn(d)
    except Exception as e:                     # noqa: BLE001
        print(f"{src} {name} {asset} {tf} FAILED {e!r}", flush=True)
        return
    t0 = pd.Timestamp(start, tz=d.index.tz)
    if kind == "weight":
        net, gross, pos = ek.bar_returns_from_position(d, out, fee)
        w = out.reindex(d.index).fillna(0.0)
        n_tr = int((w.diff().abs() > 1e-12)[d.index >= t0].sum())
        wr = np.nan
    else:
        trades = P.signal_trades(d, out) if kind == "pos" else out
        net, gross, pos = ek.bar_returns_from_trades(d, trades, fee)
        n_tr = sum(1 for t in trades if d.index[int(t[0])] >= t0)
        wr = float(np.mean([(t[4] / t[3] - 1) * t[2] > 2 * fee for t in trades])) if trades else np.nan
    r = ek.score(d, net, gross, pos, py, n_tr, start=start)
    emit(r, src, name, asset, tf, native, fee, note, wr)


def run_pair(src, name, asset_a, asset_b, a, b, stock, method):
    """Spread trades: +1 = long A / short B. Capital split across the legs (half each for the ratio rule; by the
    Kalman hedge ratio for Chan's filter). Benchmark = holding A."""
    fee = fee_for(stock, True)
    py = 252 if stock else 365
    j = a.index.intersection(b.index)
    a, b = a.loc[j], b.loc[j]
    ra, rb = a.close.pct_change().fillna(0.0), b.close.pct_change().fillna(0.0)
    if method == "ratio":
        p = R.kq_ratio_z(a.close, b.close)
        rs = 0.5 * (ra - rb)
    else:
        p, beta = R.chan_kalman(a.close, b.close)
        bl = pd.Series(beta, j).shift().fillna(1.0)
        rs = (ra - bl * rb) / (1 + bl.abs())
    syn = pd.DataFrame({"close": (1 + rs).cumprod()}, index=j)
    syn["open"] = syn.close
    w = pd.Series(p, j).shift().fillna(0.0)
    net, gross, pos = ek.bar_returns_from_position(syn, w, fee)
    hold = ek.daily(ra[j >= pd.Timestamp("2019-01-01", tz=j.tz)])
    n_tr = int((w.diff().abs() > 0)[j >= pd.Timestamp("2019-01-01", tz=j.tz)].sum())
    r = ek.score(syn, net, gross, pos, py, n_tr, bench_daily=hold)
    emit(r, src, name, f"{asset_a}/{asset_b}", "1d", "1d", fee,
         "spread return per unit of capital; benchmark = holding " + asset_a + "; hedge-ratio rebalancing costs left out")


def volatility_basket(bdata, coins, slots=7):
    """Binance-volatility-trading-bot on the 12-coin basket at 5m: each coin trades on its own rule, one seventh of the
    pot per position (MAX_COINS 7). Random timing: the same time shift applied to every coin's positions."""
    fee = FEE["spot"]
    closes = pd.concat({c: load(bdata, c, "5m").close for c in coins}, axis=1).dropna()
    net_l, gross_l, pos_l, ntr = [], [], [], 0
    for c in coins:
        d = load(bdata, c, "5m").reindex(closes.index).ffill()
        tr = R.volatility_bot(d)
        ntr += len(tr)
        n_, g_, p_ = ek.bar_returns_from_trades(d, tr, fee)
        net_l.append(n_)
        gross_l.append(g_)
        pos_l.append(p_)
    net = pd.concat(net_l, axis=1).sum(axis=1) / slots
    gross = pd.concat(gross_l, axis=1).sum(axis=1) / slots
    Pm = pd.concat(pos_l, axis=1).to_numpy() / slots
    Rm = closes.pct_change().fillna(0.0).to_numpy()
    bench = closes.pct_change().fillna(0.0).mean(axis=1)
    dn, dg, db = ek.daily(net), ek.daily(gross), ek.daily(bench)
    out = ek.perf(dn, 365)
    out["gross_cagr"] = ek.perf(dg, 365)["cagr"]
    hp = ek.perf(db, 365)
    out.update({"hold_cagr": hp["cagr"], "hold_sharpe": hp["sharpe"], "hold_dd": hp["max_dd"]})
    b, t = ek.nw_t(dn.to_numpy(), np.column_stack([np.ones(len(dn)), db.reindex(dn.index).fillna(0).to_numpy()]))
    out.update({"alpha_ann": float(b[0] * 365), "alpha_t": float(t[0]), "beta": float(b[1])})
    ms = 20 * 288

    def stats(a, r):
        S = np.zeros(len(a))
        for j in range(a.shape[1]):
            S += np.fft.irfft(np.conj(np.fft.rfft(a[:, j])) * np.fft.rfft(r[:, j]), len(a))
        act = float((a * r).sum())
        k = np.arange(len(a))
        s = S[(k >= ms) & (k <= len(a) - ms)]
        if not np.any(a) or s.std() == 0:
            return 1.0, 0.0
        return (1 + np.sum(s >= act - 1e-12)) / (1 + len(s)), (act - s.mean()) / s.std()
    out["p_random"], out["z_random"] = stats(Pm, Rm)
    mid = len(Rm) // 2
    out["p_random_h1"], out["z_random_h1"] = stats(Pm[:mid], Rm[:mid])
    out["p_random_h2"], out["z_random_h2"] = stats(Pm[mid:], Rm[mid:])
    out.update({"trades": ntr, "exposure": float(np.mean(np.abs(Pm).sum(axis=1) > 0)), "years": len(dn) / 365})
    emit(out, "Binance-volatility-trading-bot", "+10% jump buy, TP 0.8% trailing 0.4/0.1, SL 5%",
         f"basket of {len(coins)}", "5m", "2m window", fee,
         f"sample {closes.index[0].date()} to {closes.index[-1].date()}, halves split at the sample midpoint; "
         "benchmark = equal-weight basket; exits checked on 5m closes")


def bots(stock):
    """(source, name, kind, fn, native tf, shorts)"""
    st = stock
    B = [
        ("gekko-strategies", "RSI_BULL_BEAR", "pos", R.gk_rsi_bull_bear, "5m", False),
        ("gekko-strategies", "RSI_BULL_BEAR_ADX", "pos", R.gk_rsi_bull_bear_adx, "5m", False),
        ("gekko-strategies", "NEO", "pos", R.gk_neo, "5m", False),
        ("gekko-strategies", "BBRSI", "pos", R.gk_bbrsi, "5m", False),
        ("gekko-strategies", "TEMA (SMA200 safety)", "pos", R.gk_tema, "5m", False),
        ("grid bot (51bitquant/opentrader)", "grid +-10%, 80 levels, monthly re-centre", "weight", R.grid_bot, "5m", False),
        ("DCA bot (opentrader)", "base + 5 safety orders -2%, TP +2% (assumed defaults)", "weight", R.dca_bot, "5m", False),
        ("supertrend-crypto-bot", "supertrend 7 x 3 (long/flat)", "pos", R.htm_supertrend, "1m", False),
        ("Binance-Futures-Trading-Bot", "tripleEMAStochasticRSIATR (default), TP1% SL1.5%", "trades", R.cw_triple_ema_stochrsi, "1m", True),
        ("Binance-Futures-Trading-Bot", "StochRSIMACD, TP1% SL1.5%", "trades", R.cw_stochrsi_macd, "1m", True),
        ("Binance-Futures-Trading-Bot", "tripleEMA, TP1% SL1.5%", "trades", R.cw_triple_ema, "1m", True),
        ("fmzquant", "R-Breaker (0.35/0.07/0.25, intraday)", "pos", lambda d: R.r_breaker(d, st), "1m", True),
        ("fmzquant", "Dual Thrust (0.67/0.62, 4h x 20)", "pos", lambda d: R.dual_thrust(d, st), "1h", True),
    ]
    if PATTERNPY:
        B += [("PatternPy", "head and shoulders (one bar late)", "pos", lambda d: R.patternpy(d, "head_shoulder"), "1h", False),
              ("PatternPy", "double top/bottom (one bar late)", "pos", lambda d: R.patternpy(d, "double"), "1h", False)]
    return B


def daily_bots(d, stock, asset):
    py = 252 if stock else 365
    run("PyTrendFollow", "EWMAC 8/16/32/64, 12.5% vol target", "weight", lambda x: R.ewmac(x, per_year=py), asset, d,
        "1d", "1d", stock, True)
    run("pskrunner14/trading-bot", "DQN agent (trained to 2020, tested 2021+)", "pos",
        lambda x: R.dqn_agent(x, "2021-01-01"), asset, d, "1d", "1d", stock, False,
        "sklearn MLP, 10 episodes, one unit max; test 2021+", start="2021-01-01")
    if PATTERNPY:
        run("PatternPy", "head and shoulders (one bar late)", "pos", lambda x: R.patternpy(x, "head_shoulder"), asset, d,
            "1d", "1h", stock, False, "daily bars")
        run("PatternPy", "double top/bottom (one bar late)", "pos", lambda x: R.patternpy(x, "double"), asset, d, "1d",
            "1h", stock, False, "daily bars")


if __name__ == "__main__":
    mode, data, outp = sys.argv[1], sys.argv[2], sys.argv[3]       # ci: python run_ports3.py ci DATA rows.csv BASKET_DATA
    if mode == "local":
        for asset in ("BTC", "ETH", "SPY", "QQQ"):
            stock = asset in ("SPY", "QQQ")
            frames = {tf: load(data, asset, tf) for tf in (("1m", "5m", "1h", "1d") if stock else ("1h", "1d"))}
            for src, name, kind, fn, native, shorts in bots(stock):
                if stock:
                    tf = "1h" if native == "1h" else ("1m" if native == "1m" and "supertrend" in src else "5m")
                else:
                    tf = "1h"
                note = "" if tf == native else f"native {native}, run on {tf} here" + ("" if stock else " (own bar size in CI)")
                run(src, name, kind, fn, asset, frames[tf], tf, native, stock, shorts, note)
            if stock:
                run("alpaca example-scalping", "SMA20 cross, +$0.01 target, 15:55 bail-out", "trades", R.alpaca_scalper,
                    asset, frames["1m"], "1m", "1m", True, False)
            daily_bots(frames["1d"], stock, asset)
        for a, b, stock in (("ETH", "BTC", False), ("QQQ", "SPY", True)):
            A, Bd = load(data, a, "1d"), load(data, b, "1d")
            if stock:
                A.index, Bd.index = A.index.normalize(), Bd.index.normalize()
            run_pair("KidQuant pairs", "ratio z (MA5-MA60)/sd60, +-1 in, 0.75 out", a, b, A, Bd, stock, "ratio")
            run_pair("Chan (ep-chan-book)", "Kalman hedge ratio, +-sqrt(Q) bands", a, b, A, Bd, stock, "kalman")
    else:
        bdata = sys.argv[4] if len(sys.argv) > 4 else ""
        if bdata:
            coins = [c for c in os.environ.get("BASKET", "").replace("USDT", "").split(",") if c]
            volatility_basket(bdata, coins)
        for asset in ("BTC", "ETH"):
            b5 = load(data, asset, "5m")
            for src, name, kind, fn, native, shorts in bots(False):
                if native == "1h":
                    continue
                note = "" if native == "5m" else f"native {native}, smallest bars here are 5m"
                run(src, name, kind, fn, asset, b5, "5m", native, False, shorts, note)
    pd.DataFrame(rows).to_csv(outp, index=False)
