"""Walk-forward runs: yearly expanding-window refits, test years 2021-2026, every model and ablation, 3 seeds.

    python run.py DATA_DIR OUT_DIR [--quick]

Each job trains on rows before the test year (last 6 months held out for early stopping) and writes its hourly test
predictions (mu, var) to OUT_DIR/preds/<job>.pkl. evaluate.py scores them.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import baselines as bl
import data
import models

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
ABL_YEARS = [2023, 2024, 2025, 2026]       # side-channel ablation: DVOL exists from 2021-03, so these folds train from
ABL_START = "2021-04-01"                   # 2021-04 for every input set (same rows, only the inputs differ)
SEEDS = [0, 1, 2]
_G = {}


def prepare(folder):
    d = data.load(folder)
    return d, data.features(d)


def fold(d, year, start=None):
    ix = d.index
    tr0 = max(ix.searchsorted(pd.Timestamp(start)) if start else 0, 31 * 24)
    te0 = ix.searchsorted(pd.Timestamp(f"{year}-01-01"))
    te1 = ix.searchsorted(pd.Timestamp(f"{year + 1}-01-01"))
    va0 = ix.searchsorted(pd.Timestamp(f"{year}-01-01") - pd.DateOffset(months=6))
    return tr0, va0, te0, te1


def vol_baselines(d, tr0, te0):
    r = d.r.to_numpy()
    hours = d.index.hour.to_numpy()
    ok = (~d.gap).to_numpy()
    train = np.zeros(len(r), bool)
    train[tr0:te0] = True
    v_e, pe = bl.fit_ewma(r, hours, train, ok)
    v_g, pg = bl.fit_garch(r, hours, train, ok)
    return {"hist": bl.hist_var(r), "ewma": v_e, "garch": v_g}, {"ewma": pe, "garch": pg}


def job(args):
    name, kind, inputs, year, start, seed = args[:6]
    dist = args[6] if len(args) > 6 else "gauss"
    d, f = _G["d"], _G["f"]
    out = _G["out"]
    path = f"{out}/preds/{name}.pkl"
    if os.path.exists(path):
        return name, "cached"
    tr0, va0, te0, te1 = fold(d, year, start)
    vb, _ = vol_baselines(d, tr0, te0)
    lv = np.log(vb["ewma"])
    z = data.standardize(f[data.SETS[inputs]], d.index[va0])
    lvz = (lv - np.nanmean(lv[tr0:va0])) / np.nanstd(lv[tr0:va0])
    z["lv"] = np.nan_to_num(lvz)
    r = d.r.to_numpy()
    ok = (~d.gap).to_numpy() & np.isfinite(r)
    t = time.time()
    mu, var, info = models.fit_predict(kind, r, z.to_numpy(np.float32), np.nan_to_num(lv), ok, tr0, va0, te0, te1,
                                       seed=seed, max_epochs=_G["epochs"], dist=dist)
    p = pd.DataFrame({"mu": mu, "var": var}, index=d.index[te0:te1])
    p.attrs = dict(info, name=name, kind=kind, inputs=inputs, year=year, start=start or "", seed=seed, dist=dist)
    os.makedirs(f"{out}/preds", exist_ok=True)
    p.to_pickle(path)
    return name, f"{time.time() - t:.0f}s epochs {info['epochs']} val {info['best_val']:.4f}"


def tdist_jobs():
    """Follow-up 3: the same GRU and hybrid with a fat-tailed (Student-t) predictive distribution."""
    return [(f"tdist_{kind}_{y}_s{s}", kind, "price+rv", y, None, s, "t")
            for y in YEARS for kind in ("hyb_unc", "gru") for s in SEEDS]


def jobs(quick=False):
    seeds = SEEDS[:1] if quick else SEEDS
    years = YEARS[-2:] if quick else YEARS
    out = []
    for y in years:
        for kind in ("kf_fixed", "kf_feat"):
            out.append((f"main_{kind}_{y}_s0", kind, "price+rv", y, None, 0))
        for kind in ("gru", "hyb_dyn", "hyb_unc", "hyb_both"):
            for s in seeds:
                out.append((f"main_{kind}_{y}_s{s}", kind, "price+rv", y, None, s))
    if not quick:
        for y in ABL_YEARS:
            for kind in ("gru", "hyb_both"):
                for inp in data.SETS:
                    for s in seeds:
                        out.append((f"abl_{kind}_{inp}_{y}_s{s}", kind, inp, y, ABL_START, s))
    return out


def init(folder, out, epochs):
    d, f = prepare(folder)
    _G.update(d=d, f=f, out=out, epochs=epochs)


if __name__ == "__main__":
    folder, out = sys.argv[1], sys.argv[2]
    quick = "--quick" in sys.argv
    os.makedirs(f"{out}/preds", exist_ok=True)
    js = tdist_jobs() if "--tdist" in sys.argv else jobs(quick)
    # longest jobs first so the pool stays busy
    js.sort(key=lambda j: (j[1] == "kf_fixed" or j[1] == "gru", j[4] is not None, -j[3]))
    with Pool(int(os.environ.get("WORKERS", os.cpu_count())), initializer=init,
              initargs=(folder, out, 6 if quick else 40)) as pool:
        for i, (name, msg) in enumerate(pool.imap_unordered(job, js), 1):
            print(f"[{i}/{len(js)}] {name}: {msg}", flush=True)
    if "--tdist" in sys.argv:
        sys.exit(0)
    d, _ = prepare(folder)
    params = {}
    for y in YEARS:
        tr0, va0, te0, te1 = fold(d, y)
        params[y] = vol_baselines(d, tr0, te0)[1]
    with open(f"{out}/baseline_params.json", "w") as fh:
        json.dump(params, fh, indent=1, default=float)
