"""Run Kronos out of sample: at each forecast origin, sample S paths of the next H hourly candles.

The model sees only the 512 finished hourly bars before the origin (bars stamped origin-512h .. origin-1h, the last one
closing exactly at the origin), normalised per window exactly as KronosPredictor does. Sampling uses the predictor's
defaults (T=1.0, top_p=0.9). We call the repo's own auto_regressive_inference with sample_count=1 on a batch where each
window is repeated S times, so every sampled path is kept (the stock predictor averages them away).

Usage: python forecast.py DATA_DIR KRONOS_REPO OUT_DIR --shard i --shards n
Writes OUT_DIR/paths_<i>.npz: symbol, origin (ns), close0, and log(close/close0), log(high/close0), log(low/close0)
for every sampled path, shape (origins, S, H)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

LOOKBACK, H, S = 512, 24, 8
TEST_START = pd.Timestamp("2025-09-01")          # after Kronos's public release (HF upload and arXiv, Jul-Aug 2025)
STEP = {"BTCUSDT": 4, "ETHUSDT": 8}              # hours between forecast origins
COLS = ["open", "high", "low", "close", "volume", "amount"]
MODEL, TOKENIZER = "NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"


def origins(d: pd.DataFrame, step: int) -> list[pd.Timestamp]:
    """Origins are bar-open stamps o; the context is the LOOKBACK bars stamped before o, and o's bar is the first one
    forecast. Only origins with a full context and H future bars in the data, and a gap-free hourly grid, are kept."""
    idx = d.index
    out = []
    for o in pd.date_range(TEST_START, idx[-1] - pd.Timedelta(hours=H - 1), freq=f"{step}h"):
        i = idx.searchsorted(o)
        if i < LOOKBACK or i + H > len(idx) or idx[i] != o:
            continue
        w = idx[i - LOOKBACK:i + H]
        if (w[-1] - w[0]) != pd.Timedelta(hours=LOOKBACK + H - 1):
            continue
        out.append(o)
    return out


def stamps(ts: pd.DatetimeIndex) -> np.ndarray:
    return np.stack([ts.minute, ts.hour, ts.weekday, ts.day, ts.month], axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    import torch
    from model import Kronos, KronosTokenizer
    from model.kronos import auto_regressive_inference

    torch.set_num_threads(max(1, torch.get_num_threads()))
    local = os.environ.get("KRONOS_LOCAL")          # a folder with tokenizer/ and model/ (offline smoke test only)
    tok = KronosTokenizer.from_pretrained(f"{local}/tokenizer" if local else TOKENIZER).eval()
    mdl = Kronos.from_pretrained(f"{local}/model" if local else MODEL).eval()
    try:
        from huggingface_hub import HfApi
        info = {m: {"sha": i.sha, "created": str(i.created_at), "modified": str(i.last_modified)}
                for m in (MODEL, TOKENIZER) for i in [HfApi().model_info(m)]}
    except Exception as e:  # noqa: BLE001
        info = {"error": str(e)}
    if a.shard == 0:
        with open(f"{a.out}/model_info.json", "w") as f:
            json.dump(info, f, indent=1)

    jobs = []
    data = {}
    for sym, step in STEP.items():
        if not os.path.exists(f"{a.data}/{sym}_1h.csv"):
            print("no data for", sym)
            continue
        d = pd.read_csv(f"{a.data}/{sym}_1h.csv", index_col="t")[COLS]
        d.index = pd.to_datetime(d.index, format="ISO8601")
        data[sym] = d
        jobs += [(sym, o) for o in origins(d, step)]
    jobs = jobs[a.shard::a.shards]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"shard {a.shard}/{a.shards}: {len(jobs)} origins", flush=True)

    res = {"symbol": [], "origin": [], "close0": [], "c": [], "h": [], "l": []}
    t0 = time.time()
    for b in range(0, len(jobs), a.batch):
        chunk = jobs[b:b + a.batch]
        xs, xst, yst, mu, sd, c0 = [], [], [], [], [], []
        for sym, o in chunk:
            d = data[sym]
            i = d.index.searchsorted(o)
            ctx = d.iloc[i - LOOKBACK:i]
            x = ctx.to_numpy(np.float32)
            m, s = x.mean(0), x.std(0)
            xs.append(np.clip((x - m) / (s + 1e-5), -5, 5))
            xst.append(stamps(ctx.index))
            yst.append(stamps(pd.date_range(o, periods=H, freq="h")))
            mu.append(m)
            sd.append(s)
            c0.append(float(ctx.close.iloc[-1]))
        rep = lambda z: torch.from_numpy(np.repeat(np.stack(z), S, axis=0))  # noqa: E731
        torch.manual_seed(1000003 * a.shard + b)
        z = auto_regressive_inference(tok, mdl, rep(xs), rep(xst), rep(yst), LOOKBACK, H, clip=5, T=1.0, top_k=0,
                                      top_p=0.9, sample_count=1)[:, -H:, :]
        z = z.reshape(len(chunk), S, H, -1) * (np.stack(sd)[:, None, None, :] + 1e-5) + np.stack(mu)[:, None, None, :]
        c0a = np.array(c0)[:, None, None]
        for k, (sym, o) in enumerate(chunk):
            res["symbol"].append(sym)
            res["origin"].append(o.value)
        res["close0"] += c0
        for key, col in (("c", 3), ("h", 1), ("l", 2)):
            res[key].append(np.log(np.maximum(z[..., col], 1e-9) / c0a).astype(np.float32))
        done = b + len(chunk)
        el = time.time() - t0
        print(f"{done}/{len(jobs)} origins, {el / 60:.1f} min, eta {el / done * (len(jobs) - done) / 60:.0f} min",
              flush=True)
        if (b // a.batch) % 10 == 0 or done == len(jobs):
            np.savez_compressed(f"{a.out}/paths_{a.shard}.npz", symbol=np.array(res["symbol"]),
                                origin=np.array(res["origin"], dtype=np.int64), close0=np.array(res["close0"]),
                                c=np.concatenate(res["c"]), h=np.concatenate(res["h"]), l=np.concatenate(res["l"]))


if __name__ == "__main__":
    main()
