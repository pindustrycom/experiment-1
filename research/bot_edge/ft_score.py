"""Score freqtrade's own backtest results (exported trades) with evalkit."""
import glob, json, sys, zipfile, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import evalkit as ek

DATA = "/tmp/claude-0/ft_ud/data/binance"
_bars = {}


def bars(asset, tf, fut=False):
    k = (asset, tf)
    if k not in _bars:
        d = pd.read_feather(f"{DATA}/{asset}_USDT-{tf}.feather").set_index("date").sort_index()
        _bars[k] = d
    return _bars[k]


def load(zpath):
    z = zipfile.ZipFile(zpath)
    name = [n for n in z.namelist() if n.endswith(".json") and not n.endswith("_config.json")][0]
    return json.loads(z.read(name))["strategy"]


def score_dir(zpath, asset, fee, run_tf, native_tf, source="freqtrade", strat_only=None):
    rows = []
    for strat, s in load(zpath).items():
        if strat_only and strat != strat_only:
            continue
        b = bars(asset, run_tf)
        idx = b.index
        tr = []
        for t in s["trades"]:
            i0 = idx.get_indexer([pd.Timestamp(t["open_date"])])[0]
            i1 = idx.get_indexer([pd.Timestamp(t["close_date"])])[0]
            if i0 < 0:
                i0 = idx.searchsorted(pd.Timestamp(t["open_date"]))
            if i1 < 0:
                i1 = min(idx.searchsorted(pd.Timestamp(t["close_date"])), len(idx) - 1)
            tr.append((i0, i1, -1 if t["is_short"] else 1, t["open_rate"], t["close_rate"]))
        net, gross, pos = ek.bar_returns_from_trades(b, tr, fee)
        per_year = 365 if asset in ("BTC", "ETH") else 252
        row = ek.score(b, net, gross, pos, per_year, len(tr))
        row.update({"source": source, "strategy": strat, "asset": asset, "tf": run_tf, "native_tf": native_tf,
                    "ft_profit_total": s["profit_total"], "ft_cagr": s.get("cagr"), "win_rate": float(np.mean([t["profit_ratio"] > 0 for t in s["trades"]])) if s["trades"] else np.nan})
        rows.append(row)
    return rows


if __name__ == "__main__":
    out = []
    meta = {}
    for done in glob.glob("ftout/*.json.done"):
        a, f, cls = os.path.basename(done)[:-len(".json.done")].split("_", 2)
        meta[(a, cls)] = json.load(open(done))
    for z in sorted(glob.glob("ftout/res_*.zip")):
        tag = os.path.basename(z).split("-")[0]          # res_BTC_0 / res_BTC_F
        asset = tag.split("_")[1]
        fut = tag.endswith("_F")
        fee = (0.0005 if fut else 0.001) if asset in ("BTC", "ETH") else 0.0002
        for strat, s in load(z).items():
            m = meta.get((asset, strat), {"native_tf": s["timeframe"]})
            rows = score_dir(z, asset, fee, s["timeframe"], m["native_tf"], "freqtrade" + (" futures" if fut else ""), strat_only=strat)
            out += rows
            for x in rows:
                print(f"{asset} {x['strategy']:28} {x['tf']:4} trades {x['trades']:5d} cagr {x['cagr']:+.1%} gross {x['gross_cagr']:+.1%} "
                      f"(ft {x['ft_profit_total']:+.0%}) sh {x['sharpe']:.2f} hold {x['hold_sharpe']:.2f} p_rand {x['p_random']:.3f} z {x['z_random']:+.1f} a_t {x['alpha_t']:+.1f}", flush=True)
    pd.DataFrame(out).to_csv(sys.argv[1] if len(sys.argv) > 1 else "ft_rows.csv", index=False)
