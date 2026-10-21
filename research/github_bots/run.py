"""Run every ported GitHub strategy from 2019 to the latest data, with and without costs, against buy-and-hold.

  python run.py CRYPTO_CACHE EXP7_DATA_DIR OUT_DIR

Costs per side: crypto 10 bp (Binance spot taker fee), stocks 2 bp (commission-free broker, spread and slippage on
SPY/QQQ). Sharpe uses calendar-day returns for crypto (365) and trading days for stocks (252). Daily stock strategies
earn T-bills (BIL) while flat; intraday and crypto strategies earn nothing while flat.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

import allocation
import data
from engine import daily, simulate, stats
from strategies import CATALOG

FROM = pd.Timestamp("2019-01-01")
COST = {"crypto": 10.0, "stock": 2.0}
TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
CRYPTO = ("BTCUSDT", "ETHUSDT")
STOCKS = ("SPY", "QQQ")
BARS_PER_DAY = {"1d": 1, "1h": 7, "5m": 78}       # regular session: hourly bars 9:30-10:30 ... 15:30-16:00


def edge(s: dict, h: dict) -> bool:
    """A real edge: after costs, a better Sharpe than holding, a significant alpha, and positive in both halves."""
    return s["sharpe"] > h["sharpe"] and s["alpha_t"] > 2 and s["cagr_2019_22"] > 0 and s["cagr_2023_26"] > 0


def main(cache: str, exp7: str, out: str) -> None:
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(market, sym, name, tf, src, horizon, r_net, r_gross, bench, per_year, hold, trades=None, idx=None, bar_min=None):
        s = stats(r_net, bench, per_year, trades, idx, bar_min)
        g = stats(r_gross, bench, per_year)
        row = {"market": market, "symbol": sym, "strategy": name, "tf": tf, "horizon": horizon, "source": src,
               **s, "gross_cagr": g["cagr"], "gross_sharpe": g["sharpe"],
               "hold_cagr": hold["cagr"], "hold_sharpe": hold["sharpe"], "hold_dd": hold["max_dd"], "edge": edge(s, hold)}
        rows.append(row)
        print(f"{market:6} {sym:8} {name:24} {tf:3} cagr {s['cagr']:+.1%} (gross {g['cagr']:+.1%}) sharpe {s['sharpe']:.2f} "
              f"dd {s['max_dd']:.0%} alpha {s['alpha_ann']:+.1%} t {s['alpha_t']:.1f} trades {s.get('trades', 0):.0f} | "
              f"hold {hold['cagr']:+.1%} sh {hold['sharpe']:.2f}{'  EDGE' if row['edge'] else ''}", flush=True)

    # ---- stocks
    px = {t: data.yahoo_ohlc(t) for t in ("SPY", "QQQ", "EFA", "AGG", "BIL")}
    closes = pd.DataFrame({t: d.close for t, d in px.items()}).dropna()
    bil = closes["BIL"].pct_change()
    for sym in STOCKS:
        bench_all = px[sym].close.pct_change()
        bench = bench_all[bench_all.index >= FROM]
        hold = stats(bench, bench, 252)
        rows.append({"market": "stock", "symbol": sym, "strategy": "buy_and_hold", "tf": "1d", "horizon": "long", **hold})
        bars = {"1d": px[sym]}
        for tf, m in (("1h", 60), ("5m", 5)):
            try:
                bars[tf] = data.minute_bars(f"{exp7}/{sym}", m)
            except ValueError:
                print(f"no exp7 bars for {sym}", flush=True)
        for name, (build, src, horizon, mk) in CATALOG.items():
            for tf in mk.get("stock", []):
                if tf not in bars:
                    continue
                d = bars[tf]
                t0 = time.time()
                rules = build(d)
                cash = bil.reindex(d.index).fillna(0.0) if tf == "1d" else None
                res = {}
                for c in (COST["stock"], 0.0):
                    r, trades = simulate(d, rules, TF_MIN[tf] if tf != "1d" else 1440, c, cash)
                    rd = daily(r)
                    res[c] = (rd[rd.index >= FROM], trades)
                add("stock", sym, name, tf, src, horizon, res[COST["stock"]][0], res[0.0][0], bench, 252, hold,
                    res[COST["stock"]][1], d.index, 1440 / BARS_PER_DAY[tf])
                print(f"   ({time.time() - t0:.0f}s)", flush=True)
        # monthly allocation rules
        for name, fn, src in (("faber_10m", lambda c, s=sym: allocation.faber_10m(closes, c, s), "Faber 10-month SMA timing"),
                              ("gem_dual_momentum", lambda c: allocation.gem(closes, c), "Antonacci GEM (SPY/EFA/AGG)")):
            if name == "gem_dual_momentum" and sym != "SPY":
                continue
            rn, rg = fn(COST["stock"]), fn(0.0)
            add("stock", sym, name, "1mo", src, "long", rn[rn.index >= FROM], rg[rg.index >= FROM], bench, 252, hold)

    # ---- crypto
    for sym in CRYPTO:
        dd = data.binance(cache, sym, "1d", "2018-01")
        bench = dd.close.pct_change()
        bench = bench[bench.index >= FROM]
        hold = stats(bench, bench, 365)
        rows.append({"market": "crypto", "symbol": sym, "strategy": "buy_and_hold", "tf": "1d", "horizon": "long", **hold})
        tfs = sorted({tf for _, _, _, mk in CATALOG.values() for tf in mk.get("crypto", [])}, key=TF_MIN.get)
        for tf in tfs:
            d = data.binance(cache, sym, tf, "2018-01" if TF_MIN[tf] >= 60 else "2018-12")
            for name, (build, src, horizon, mk) in CATALOG.items():
                if tf not in mk.get("crypto", []):
                    continue
                t0 = time.time()
                rules = build(d)
                res = {}
                for c in (COST["crypto"], 0.0):
                    r, trades = simulate(d, rules, TF_MIN[tf], c)
                    rd = daily(r)
                    res[c] = (rd[rd.index >= FROM], trades)
                add("crypto", sym, name, tf, src, horizon, res[COST["crypto"]][0], res[0.0][0], bench, 365, hold,
                    res[COST["crypto"]][1], d.index, TF_MIN[tf])
                print(f"   ({time.time() - t0:.0f}s)", flush=True)
            del d

    (outp / "results.json").write_text(json.dumps(rows, indent=1, default=float))
    write_table(rows, outp / "table.md")
    print((outp / "table.md").read_text())


def write_table(rows, path):
    head = ("| market | symbol | strategy | tf | horizon | CAGR net | gross | Sharpe | max DD | alpha/yr (t) | trades | "
            "2019-22 | 2023-26 | hold CAGR | hold Sharpe | edge |")
    lines = [head, "|" + "---|" * 16]
    for r in sorted(rows, key=lambda r: (r["market"], r["symbol"], -(r["sharpe"] - r.get("hold_sharpe", r["sharpe"])))):
        lines.append(f"| {r['market']} | {r['symbol']} | {r['strategy']} | {r['tf']} | {r['horizon']} | {r['cagr']:+.1%} | "
                     f"{r.get('gross_cagr', r['cagr']):+.1%} | {r['sharpe']:.2f} | {r['max_dd']:.0%} | "
                     f"{r['alpha_ann']:+.1%} ({r['alpha_t']:.1f}) | {r.get('trades', '')} | {r['cagr_2019_22']:+.1%} | "
                     f"{r['cagr_2023_26']:+.1%} | {r.get('hold_cagr', r['cagr']):+.1%} | {r.get('hold_sharpe', r['sharpe']):.2f} | "
                     f"{'yes' if r.get('edge') else ''} |")
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
