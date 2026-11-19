"""whchien/ai-trader classic strategies, run in backtrader with the repo's own classes, on daily bars.

  python run_aitrader.py AI_TRADER_REPO DATA rows.csv

Each strategy only buys and closes (long/flat). The position it holds after each bar is recorded; the position that
bar's decision leads to (filled at the next open) is then scored like every other row.
"""
import os
import sys

import backtrader as bt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evalkit as ek  # noqa: E402
import ports as P  # noqa: E402

REPO, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, REPO)
from ai_trader.backtesting.strategies import classic as C  # noqa: E402

FEE = {"spot": 0.001, "stock": 0.0002}
NAMES = [n for n in C.__all__ if n != "BuyHoldStrategy"]


def load(base, tf="1d"):
    return pd.read_feather(f"{DATA}/{base}_USDT-{tf}.feather").set_index("date").sort_index()


def positions(cls, d):
    held = []

    class Rec(cls):
        def next(self):
            held.append(1.0 if self.position.size > 0 else 0.0)
            super().next()

    cer = bt.Cerebro(stdstats=False)
    feed = d.tz_convert(None) if d.index.tz is not None else d
    cer.adddata(bt.feeds.PandasData(dataname=feed[["open", "high", "low", "close", "volume"]]))
    cer.addstrategy(Rec)
    cer.broker.setcash(1e12)
    cer.addsizer(bt.sizers.FixedSize, stake=1)
    cer.run()
    h = np.zeros(len(d))
    h[len(d) - len(held):] = held                      # warm-up bars before the first next() stay flat
    return np.r_[h[1:], h[-1]]                         # position decided at close t = held from open t+1


rows = []
for asset in ("BTC", "ETH", "SPY", "QQQ"):
    stock = asset in ("SPY", "QQQ")
    fee, py = (FEE["stock"], 252) if stock else (FEE["spot"], 365)
    d = load(asset)
    for n in NAMES:
        try:
            pos = positions(getattr(C, n), d)
        except Exception as e:                         # noqa: BLE001
            print(f"{n} {asset} FAILED {e!r}", flush=True)
            continue
        trades = P.signal_trades(d, pos)
        net, gross, p = ek.bar_returns_from_trades(d, trades, fee)
        t0 = pd.Timestamp("2019-01-01", tz=d.index.tz)
        r = ek.score(d, net, gross, p, py, sum(1 for t in trades if d.index[t[0]] >= t0))
        r.update({"source": "ai-trader (whchien)", "strategy": n.replace("Strategy", ""), "asset": asset, "tf": "1d",
                  "native_tf": "1d", "fee": fee, "note": "repo's backtrader class, default params",
                  "win_rate": float(np.mean([(t[4] / t[3] - 1) > 2 * fee for t in trades])) if trades else np.nan})
        rows.append(r)
        print(f"{n[:28]:28} {asset} n {r['trades']:4d} cagr {r['cagr']:+.1%} sh {r['sharpe']:.2f} hold {r['hold_sharpe']:.2f} "
              f"p {r['p_random']:.3f} z {r['z_random']:+.1f} a_t {r['alpha_t']:+.1f}", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
