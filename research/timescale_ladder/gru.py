"""GRU correction on top of the ladder: one small GRU, shared by all five scales, learns each scale's remaining error.

Input per period (all known before the period starts): the ladder's inputs (own filter states, HAR terms, the
parent's forecast, the child's latest state), the ladder's forecast itself, the last realized log rv and
standardized return, the time-of-day slot and a one-hot scale id, over the last 32 periods of that scale.
Output: a correction to the ladder's log-variance forecast and a forecast of the standardized return (direction).
Sharing one network across scales gives the month and week scales, which have only ~50-200 training periods,
the shapes learned from hours and minutes; the scale id lets it differ where the data say so.

Train 2019-2021, early-stop on 2022, test 2023+. Three seeds.

    python gru.py OUT_DIR   (with ladder.pkl from ladder.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ladder import FLOORS, LEVELS, TEST, nw_t, qlike

VAL = pd.Timestamp("2022-01-01")
SEQ, CAP = 32, 60_000  # sequence length; most training sequences drawn per scale
MODEL = "HAR + full ladder"


def features(S: dict, F: dict) -> dict:
    D = {}
    for i, lvl in enumerate(LEVELS):
        f, L = S[lvl], F[lvl]
        h_hat = L["h_hat_full"]
        v = np.exp(h_hat)
        y_full = np.log(f["rv"].to_numpy() + FLOORS[lvl])
        z = f["r"].to_numpy() / np.sqrt(v)
        X = np.column_stack([np.nan_to_num(L["Xh"][:, 1:]), np.nan_to_num(L["Xm"][:, 1:] / np.sqrt(np.nanmedian(v))),
                             h_hat, np.r_[np.nan, y_full[:-1]], np.r_[np.nan, z[:-1]],
                             np.sin(2 * np.pi * f["slot"] / 390), np.cos(2 * np.pi * f["slot"] / 390)])
        X = np.nan_to_num(X)
        tr = (f["start"] < VAL).to_numpy()
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        X = (X - mu) / sd
        X = np.column_stack([X, np.tile(np.eye(5)[i], (len(X), 1))])
        D[lvl] = {"X": X.astype(np.float32), "t_h": (y_full - h_hat).astype(np.float32), "t_z": np.clip(z, -6, 6).astype(np.float32),
                  "start": f["start"].to_numpy(), "h_hat": h_hat}
    width = max(d["X"].shape[1] for d in D.values())
    for d in D.values():  # scale one-hot stays last; pad the inputs in front of it
        k = width - d["X"].shape[1]
        d["X"] = np.column_stack([d["X"][:, :-5], np.zeros((len(d["X"]), k), np.float32), d["X"][:, -5:]])
    return D


def windows(d: dict, idx: np.ndarray) -> tuple:
    w = np.stack([d["X"][i - SEQ + 1:i + 1] for i in idx])
    return torch.from_numpy(w), torch.from_numpy(d["t_h"][idx]), torch.from_numpy(d["t_z"][idx])


class Net(nn.Module):
    def __init__(self, k: int, hdim: int = 32):
        super().__init__()
        self.gru = nn.GRU(k, hdim, batch_first=True)
        self.head = nn.Linear(hdim, 2)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1])


def run(D: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    split = {}
    for lvl, d in D.items():
        n = len(d["X"])
        ok = np.arange(SEQ, n)
        ok = ok[np.isfinite(d["t_h"][ok]) & np.isfinite(d["t_z"][ok])]
        st = d["start"][ok]
        tr, va = ok[st < VAL], ok[(st >= VAL) & (st < TEST)]
        split[lvl] = (rng.permutation(tr)[:CAP], va[:: max(1, len(va) // 20_000)])
    net = Net(next(iter(D.values()))["X"].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    # Each scale's loss is weighted equally, so minutes do not drown out months
    batches = {lvl: windows(D[lvl], split[lvl][0]) for lvl in D}
    vals = {lvl: windows(D[lvl], split[lvl][1]) for lvl in D if len(split[lvl][1])}

    def loss_on(b):
        x, th, tz = b
        p = net(x)
        return ((p[:, 0] - th) ** 2).mean() + ((p[:, 1] - tz) ** 2).mean()

    best, state, bad = np.inf, None, 0
    for ep in range(60):
        net.train()
        for lvl, (x, th, tz) in batches.items():
            perm = torch.randperm(len(x))
            for j in range(0, len(x), 512):
                k = perm[j:j + 512]
                opt.zero_grad()
                l = loss_on((x[k], th[k], tz[k]))
                l.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            v = np.mean([loss_on(b).item() for b in vals.values()])
        if v < best - 1e-4:
            best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(state)
    out = {}
    with torch.no_grad():
        for lvl, d in D.items():
            n = len(d["X"])
            ph, pz = np.full(n, np.nan), np.full(n, np.nan)
            idx = np.arange(SEQ, n)
            for j in range(0, len(idx), 20_000):
                x, _, _ = windows(d, idx[j:j + 20_000])
                p = net(x).numpy()
                ph[idx[j:j + 20_000]], pz[idx[j:j + 20_000]] = p[:, 0], p[:, 1]
            out[lvl] = (ph, pz)
    return out


def main(out: Path) -> None:
    L = pd.read_pickle(out / "ladder.pkl")
    S, F = L["S"], L["models"][MODEL]
    D = features(S, F)
    runs = [run(D, s) for s in (1, 2, 3)]
    rows = []
    for lvl in LEVELS:
        f, d = S[lvl], D[lvl]
        ph = np.nanmean([r[lvl][0] for r in runs], axis=0)
        pz = np.nanmean([r[lvl][1] for r in runs], axis=0)
        tr = (f["start"] < VAL).to_numpy() & np.isfinite(ph)
        te = (f["start"] >= TEST).to_numpy() & np.isfinite(ph) & np.isfinite(d["h_hat"])
        rv = f["rv"].to_numpy() + FLOORS[lvl]
        base = np.exp(d["h_hat"])
        h_new = d["h_hat"] + ph
        h_new += np.log(np.mean(rv[tr] / np.exp(h_new[tr])))
        new = np.exp(h_new)
        r = f["r"].to_numpy()
        dlt = qlike(rv[te], new[te]) - qlike(rv[te], base[te])
        rows.append({"scale": lvl, "test_periods": int(te.sum()), "qlike_ladder": qlike(rv[te], base[te]).mean(),
                     "qlike_ladder+gru": qlike(rv[te], new[te]).mean(),
                     "change_%": 100 * (qlike(rv[te], new[te]).mean() / qlike(rv[te], base[te]).mean() - 1),
                     "dm_t": nw_t(dlt, 30 if lvl == "minute" else 5),
                     "gru_dir_hit_%": 100 * np.mean(np.sign(pz[te]) == np.sign(r[te])),
                     "up_share_%": 100 * np.mean(r[te] > 0), "gru_dir_ic": np.corrcoef(pz[te], r[te])[0, 1]})
        S[lvl]["gru_h"], S[lvl]["gru_z"] = h_new, pz
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))
    R.to_csv(out / "gru_eval.csv", index=False)
    pd.to_pickle({lvl: S[lvl][["start", "r", "rv", "gru_h", "gru_z"]] for lvl in LEVELS}, out / "gru_preds.pkl")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
