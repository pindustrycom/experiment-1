"""Can machine learning cut a bot's trades down to the ones that clear fees? (meta-labeling, Lopez de Prado ch. 3)

For each bot: take every trade it would make, describe the market at the moment of the entry decision (only data up
to that bar's close), and label the trade 1 if it made money after fees. A gradient-boosted classifier trained only
on trades that had closed before each test year decides which trades of that year to take. Walk-forward, yearly,
2021 onwards; the threshold is chosen inside the training years (last quarter held out). Skipped trades leave the
bot flat. The filtered trade list is scored exactly like every other row (random-timing test, halves, alpha, costs).

  python meta.py local DATA rows.csv           # BTC/ETH 1h, SPY/QQQ at the bots' own bar sizes
  python meta.py ci DATA rows.csv              # BTC/ETH at 5m/15m/30m (CI)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(__file__))
import evalkit as ek  # noqa: E402
import ports as P  # noqa: E402
import ports2 as Q  # noqa: E402

FEE = {"spot": 0.001, "perp": 0.0005, "stock": 0.0002}
TEST_YEARS = range(2021, 2027)

# bots with a gross (before-fee) gain that fees destroyed, plus a few controls
BOTS = [
    ("zenbot", "trend_ema (default)", Q.zb_trend_ema, "pos", "2m", False),
    ("zenbot", "sar", Q.zb_sar, "pos", "2m", False),
    ("zenbot", "macd", Q.zb_macd, "pos", "1h", False),
    ("gekko", "MACD", Q.gk_macd, "pos", "1h", False),
    ("gekko", "PPO", Q.gk_ppo, "pos", "1h", False),
    ("gekko", "StochRSI", Q.gk_stochrsi, "pos", "1h", False),
    ("vnpy", "DoubleMaStrategy", Q.vn_double_ma, "pos", "1m", True),
    ("vnpy", "AtrRsiStrategy", Q.vn_atr_rsi, "trades", "1m", True),
    ("vnpy", "KingKeltnerStrategy", Q.vn_king_keltner, "trades", "5m", True),
    ("vnpy", "TurtleSignalStrategy", Q.vn_turtle, "trades", "1m", True),
    ("crypto-trading-bot", "macd", Q.ctb_macd, "pos", "15m", True),
    ("crypto-trading-bot", "parabolicsar", Q.ctb_psar, "pos", "15m", True),
    ("backtesting.py", "SmaCross 10/20 (README)", P.btpy_smacross, "pos", "1h", False),
]
MIN = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "1h": 60}


def features(d):
    c, v = d.close, d.volume
    lr = np.log(c).diff()
    bpd = max(1, int(round(len(d) / max((d.index[-1] - d.index[0]).days, 1))))     # bars per day (rough)
    f = pd.DataFrame(index=d.index)
    for k in (1, 3, 12, bpd, 5 * bpd):
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    f["vol_s"] = lr.rolling(max(12, bpd // 4)).std()
    f["vol_d"] = lr.rolling(5 * bpd).std()
    f["vol_ratio"] = f.vol_s / f.vol_d
    f["rsi14"] = Q.rsi(c, 14)
    f["dist_sma200"] = np.log(c / c.rolling(200).mean())
    f["dist_sma_long"] = np.log(c / c.rolling(50 * bpd).mean())
    f["range_pos"] = (c - d.low.rolling(5 * bpd).min()) / (d.high.rolling(5 * bpd).max() - d.low.rolling(5 * bpd).min())
    f["vol_z"] = (v - v.rolling(5 * bpd).mean()) / v.rolling(5 * bpd).std()
    f["hour"] = d.index.hour
    f["dow"] = d.index.dayofweek
    return f


def trade_table(d, trades, fee, f):
    t = pd.DataFrame(trades, columns=["i0", "i1", "side", "p0", "p1"])
    t["net"] = np.where(t.side > 0, t.p1 / t.p0, 2 - t.p1 / t.p0) * (1 - fee) ** 2 - 1
    t["y"] = (t.net > 0).astype(int)
    t["t_entry"] = d.index[t.i0.to_numpy()]
    t["t_exit"] = d.index[t.i1.to_numpy()]
    X = f.iloc[(t.i0 - 1).clip(lower=0).to_numpy()].reset_index(drop=True)      # known at the decision close
    X["side"] = t.side.to_numpy()
    prev = t.net.shift()                      # the bot's previous trade result (closed before this entry)
    X["prev_net"] = prev.where(t.t_exit.shift() <= t.t_entry).to_numpy()
    X["prev_win_rate"] = t.y.shift().rolling(20, min_periods=5).mean().to_numpy()
    return t, X


def pick_threshold(p, net):
    best_t, best = 0.5, -np.inf
    for th in np.arange(0.3, 0.81, 0.025):
        k = p >= th
        if k.sum() < 10:
            continue
        s = np.log1p(net[k]).sum()
        if s > best:
            best_t, best = th, s
    return best_t


def meta_filter(t, X, tz):
    keep = np.zeros(len(t), bool)
    info = []
    for y in TEST_YEARS:
        t0, t1 = pd.Timestamp(f"{y}-01-01", tz=tz), pd.Timestamp(f"{y + 1}-01-01", tz=tz)
        tr = (t.t_exit < t0).to_numpy()
        te = ((t.t_entry >= t0) & (t.t_entry < t1)).to_numpy()
        if tr.sum() < 100 or te.sum() == 0:
            continue
        Xtr, ytr, ntr = X[tr], t.y[tr].to_numpy(), t.net[tr].to_numpy()
        cut = int(len(Xtr) * 0.75)
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                           min_samples_leaf=40, random_state=0)
        m.fit(Xtr.iloc[:cut], ytr[:cut])
        th = pick_threshold(m.predict_proba(Xtr.iloc[cut:])[:, 1], ntr[cut:]) if len(Xtr) - cut >= 40 else 0.5
        m.fit(Xtr, ytr)
        p = m.predict_proba(X[te])[:, 1]
        keep[np.where(te)[0][p >= th]] = True
        info.append(f"{y}:{th:.2f}")
    return keep, " ".join(info)


rows = []


def run(src, name, fn, kind, shorts, asset, d, tf, stock, note=""):
    fee = FEE["stock"] if stock else (FEE["perp"] if shorts else FEE["spot"])
    py = 252 if stock else 365
    out = fn(d)
    trades = P.signal_trades(d, out) if kind == "pos" else out
    if len(trades) < 200:
        return
    f = features(d)
    t, X = trade_table(d, trades, fee, f)
    keep, info = meta_filter(t, X, d.index.tz)
    start = "2021-01-01"
    t0 = pd.Timestamp(start, tz=d.index.tz)
    base = [tr for tr, te in zip(trades, (t.t_entry >= t0).to_numpy()) if te]
    kept = [tr for tr, k in zip(trades, keep) if k]
    for label, tl in (("all trades", base), ("ML-filtered", kept)):
        net, gross, pos = ek.bar_returns_from_trades(d, tl, fee)
        r = ek.score(d, net, gross, pos, py, len(tl), start=start)
        wr = float(np.mean([(x[4] / x[3] - 1) * x[2] > 2 * fee for x in tl])) if tl else np.nan
        r.update({"source": f"{src} + ML filter" if label == "ML-filtered" else f"{src} (2021+ baseline)",
                  "strategy": name, "asset": asset, "tf": tf, "fee": fee, "win_rate": wr,
                  "note": f"{note} walk-forward yearly 2021+, thresholds {info}".strip() if label == "ML-filtered"
                  else f"{note} same bot, all trades, 2021+ (comparison for the ML row)".strip()})
        rows.append(r)
        print(f"{src:18} {asset} {name[:22]:22} {tf:4} {label:11} n {r['trades']:6d} cagr {r['cagr']:+.1%} "
              f"gross {r['gross_cagr']:+.1%} sh {r['sharpe']:.2f} hold {r['hold_sharpe']:.2f} p {r['p_random']:.3f} "
              f"z {r['z_random']:+.1f} a_t {r['alpha_t']:+.1f}", flush=True)


def load(data, base, tf):
    return pd.read_feather(f"{data}/{base}_USDT-{tf}.feather").set_index("date").sort_index()


def rs_stock(d1, m):
    if m == 1:
        return d1
    loc = d1.tz_convert("America/New_York")
    g = loc.groupby(pd.Grouper(freq=f"{m}min", origin="start_day", offset="30min")).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.tz_convert("UTC")


if __name__ == "__main__":
    mode, data, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "local":
        for asset in ("BTC", "ETH", "SPY", "QQQ"):
            stock = asset in ("SPY", "QQQ")
            h1 = load(data, asset, "1h")
            m1 = load(data, asset, "1m") if stock else None
            cache = {}
            for src, name, fn, kind, native, shorts in BOTS:
                if stock:
                    if native not in cache:
                        cache[native] = rs_stock(m1, MIN[native]) if native != "1h" else h1
                    run(src, name, fn, kind, shorts, asset, cache[native], native, True)
                else:
                    run(src, name, fn, kind, shorts, asset, h1, "1h", False,
                        "" if native == "1h" else f"native {native}, run on 1h")
    else:
        for asset in ("BTC", "ETH"):
            b5 = load(data, asset, "5m")
            cache = {}
            for src, name, fn, kind, native, shorts in BOTS:
                if native == "1h":
                    continue
                tf = {"1m": "5m", "2m": "5m"}.get(native, native)
                if tf not in cache:
                    cache[tf] = b5 if tf == "5m" else load(data, asset, tf)
                run(src, name, fn, kind, shorts, asset, cache[tf], tf, False,
                    "" if tf == native else f"native {native}, smallest bars here are 5m")
    pd.DataFrame(rows).to_csv(outp, index=False)
