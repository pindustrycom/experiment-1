"""Run freqtrade strategies with freqtrade's own backtester, one strategy and one market per call.

  python ft_batch.py --strategies DIR --assets BTC,ETH --fee 0.001 --tfs 1h,4h,12h,1d --fallback 1h --out ftout
                     [--only-native 5m,15m] [--futures] [--timerange 20190101-] [--names A,B]

A strategy runs on its own timeframe when we have bars for it, otherwise on --fallback (crypto from the sandbox has
hourly bars only, so 5m/15m crypto strategies run on 1h there and at their own timeframe in the CI job).
Results: freqtrade's zipped trade exports in OUT (res_<ASSET>_<S|F>_<tf>-<timestamp>.zip) and one .done file each.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = {"DoesNothingStrategy",            # template, never trades
        "FixedRiskRewardLoss", "CustomStoplossWithPSAR",   # stop-loss templates: one entry, then hold
        "BreakEven", "Freqtrade_backtest_validation_freqtrade1"}  # break-even demo / engine validation helper
ap = argparse.ArgumentParser()
ap.add_argument("--strategies", nargs="+", required=True)
ap.add_argument("--assets", required=True)
ap.add_argument("--fee", type=float, required=True)
ap.add_argument("--tfs", required=True)
ap.add_argument("--fallback", default="1h")
ap.add_argument("--only-native", default="")
ap.add_argument("--futures", action="store_true")
ap.add_argument("--timerange", default="20190101-")
ap.add_argument("--names", default="")
ap.add_argument("--out", default="ftout")
ap.add_argument("--datadir", required=True)
ap.add_argument("--max-open-trades", type=int, default=1)
ap.add_argument("--tag", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tfs = set(a.tfs.split(","))
only = set(filter(None, a.only_native.split(",")))
names = set(filter(None, a.names.split(",")))

jobs = []
for d in a.strategies:
    for f in sorted(glob.glob(f"{d}/*.py")):
        src = open(f).read()
        mm = re.search(r"^class (\w+)\(\s*IStrategy", src, re.M) or re.search(r"^class (\w+)\(", src, re.M)
        if not mm:
            continue
        cls = mm.group(1)
        shorts = bool(re.search(r"^\s+can_short\s*(:\s*bool)?\s*=\s*True", src, re.M))   # class attribute, not self.can_short
        if cls in SKIP or (names and cls not in names) or shorts != a.futures:
            continue
        m = re.search(r"^\s+timeframe\s*=\s*['\"](\w+)['\"]", src, re.M)
        tf = m.group(1) if m else "5m"
        if only and tf not in only:
            continue
        jobs.append((cls, os.path.abspath(d), tf, tf if tf in tfs else a.fallback))

cfg = json.load(open(f"{HERE}/ft_config.json"))
cfg.update({"fee": a.fee, "max_open_trades": a.max_open_trades})
cfg.pop("datadir", None)
cfg.pop("user_data_dir", None)
if a.futures:
    cfg.update({"trading_mode": "futures", "margin_mode": "isolated"})
for asset in a.assets.split(","):
    pairs = [f"{x}/USDT" + (":USDT" if a.futures else "") for x in asset.split("+")]
    cfg["exchange"]["pair_whitelist"] = pairs
    mode = "F" if a.futures else "S"
    label = asset.replace("+", "") if len(pairs) == 1 else f"BASKET{len(pairs)}"
    cpath = f"{a.out}/cfg_{label}_{mode}.json"
    json.dump(cfg, open(cpath, "w"), indent=1)
    for cls, sdir, tf, run_tf in jobs:
        done = f"{a.out}/{label}_{mode}_{cls}{a.tag}.done"
        if os.path.exists(done):
            continue
        cmd = [sys.executable, f"{HERE}/ftrun.py", "backtesting", "-c", cpath, "-s", cls, "--strategy-path", sdir,
               "--timeframe", run_tf, "--timerange", a.timerange, "--export", "trades", "--cache", "none",
               "--backtest-directory", f"{a.out}/res_{label}_{mode}_{run_tf}", "--datadir", os.path.abspath(a.datadir),
               "--userdir", os.path.abspath(a.out)]
        env = dict(os.environ, FT_BASES=",".join(x for x in asset.split("+")))
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        meta = {"strategy": cls, "asset": label, "pairs": pairs, "native_tf": tf, "run_tf": run_tf, "mode": mode,
                "fee": a.fee, "ok": r.returncode == 0, "err": "" if r.returncode == 0 else r.stderr[-1500:]}
        open(done, "w").write(json.dumps(meta))
        print(label, mode, cls, tf, run_tf, "OK" if meta["ok"] else "FAIL " + r.stderr[-300:].replace("\n", " "), flush=True)
