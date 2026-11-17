"""Score freqtrade's own trade exports with evalkit (one row per strategy and market).

  python ft_score.py DATADIR OUT.csv RESULT_DIR [RESULT_DIR ...]

Each zip in a result dir is one freqtrade backtest (one strategy, one pair). The trades are replayed bar by bar on
the bars freqtrade used (entry and exit at freqtrade's fill prices, its fee on both sides), so the CAGR here matches
freqtrade's own total profit; the bars also give the position series for the random-timing test.
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as ek  # noqa: E402

CRYPTO = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "LINK", "AVAX", "DOT", "TRX", "ATOM"}
_bars = {}


def bars(datadir, base, tf):
    k = (datadir, base, tf)
    if k not in _bars:
        _bars[k] = pd.read_feather(f"{datadir}/{base}_USDT-{tf}.feather").set_index("date").sort_index()
    return _bars[k]


def load(zpath):
    z = zipfile.ZipFile(zpath)
    name = [n for n in z.namelist() if n.endswith(".json") and not n.endswith("_config.json")][0]
    return json.loads(z.read(name))["strategy"]


def trades_on(b, trades):
    idx = b.index
    out = []
    for t in trades:
        i0 = idx.searchsorted(pd.Timestamp(t["open_date"]))
        i1 = min(idx.searchsorted(pd.Timestamp(t["close_date"])), len(idx) - 1)
        out.append((min(i0, len(idx) - 1), i1, -1 if t["is_short"] else 1, t["open_rate"], t["close_rate"]))
    return out


def metas(dirs):
    m = {}
    for d in dirs:
        for f in glob.glob(f"{d}/*.done"):
            j = json.load(open(f))
            if "strategy" in j:
                m[(j["asset"], j["mode"], j["strategy"])] = j
            else:   # first sandbox batch: <ASSET>_<fee>_<Strategy>.json.done
                a, fee, cls = os.path.basename(f)[:-len(".json.done")].split("_", 2)
                m[(a, "F" if fee.startswith("F") else "S", cls)] = j
    return m


def score_zip(z, datadir, meta):
    tag = os.path.basename(z).split("-")[0].split("_")      # res_BTC_S_1h / res_BTC_0 (old) / res_BTC_F (old)
    asset = tag[1]
    mode = "F" if tag[2] == "F" else "S"
    rows = []
    for strat, s in load(z).items():
        if asset.startswith("BASKET"):
            continue                                            # scored by basket_score.py
        crypto = asset in CRYPTO
        fee = s["trades"][0]["fee_open"] if s["trades"] else (0.001 if crypto else 0.0002)
        b = bars(datadir, asset, s["timeframe"])
        tr = trades_on(b, s["trades"])
        net, gross, pos = ek.bar_returns_from_trades(b, tr, fee)
        r = ek.score(b, net, gross, pos, 365 if crypto else 252, len(tr))
        mt = meta.get((asset, mode, strat), {})
        r.update({"source": "freqtrade" + (" (shorts allowed)" if mode == "F" else ""), "strategy": strat, "asset": asset,
                  "tf": s["timeframe"], "native_tf": mt.get("native_tf", s["timeframe"]), "fee": fee,
                  "ft_profit_total": s["profit_total"],
                  "win_rate": float(np.mean([t["profit_ratio"] > 0 for t in s["trades"]])) if s["trades"] else np.nan,
                  "lookahead_flag": strat in {"DevilStra", "GodStraNew", "Zeus", "wtc"}})
        rows.append(r)
    return rows


if __name__ == "__main__":
    datadir, out, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    meta = metas(dirs)
    rows = []
    for d in dirs:
        for z in sorted(glob.glob(f"{d}/res_*.zip")):
            for x in score_zip(z, datadir, meta):
                rows.append(x)
                print(f"{x['asset']:4} {x['strategy']:28} {x['tf']:4} n {x['trades']:5d} cagr {x['cagr']:+.1%} gross {x['gross_cagr']:+.1%} "
                      f"(ft {x['ft_profit_total']:+.0%}) sh {x['sharpe']:.2f} hold {x['hold_sharpe']:.2f} p {x['p_random']:.3f} "
                      f"z {x['z_random']:+.1f} a_t {x['alpha_t']:+.1f}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
