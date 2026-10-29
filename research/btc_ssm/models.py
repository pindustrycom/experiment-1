"""Kalman-filter backbone, standalone GRU, and the hybrid KF + GRU estimators (research notes, sections 5-6).

State x_k = [l_k, m_k]: log price (percent) and latent hourly trend, A = [[1, 1], [0, phi]], C = [1, 0].
    x_{k+1|k} = A x_{k|k} + [B u_k + g_l, g_m],   P_{k+1|k} = A P A' + diag(q_l, q_m),   S = P_ll + R
The predictive for the next hourly return is N(mu, S) with mu = x_{k+1|k,l} - y_k. To stay exact in float32 the filter
carries e_k = x_{k|k,l} - y_k instead of the price level (after the update, e = (K_l - 1) * innovation).

Variants (same filter code, different switches):
  kf_fixed  fixed phi, Q, R, no inputs                      (notes: "linear Kalman baseline with fixed A,B,C,Q,R")
  kf_feat   + B u_k drift and q_l scaled by the seasonal EWMA variance (model-derived vol feature, conventional SSM)
  hyb_dyn   kf_feat + GRU dynamics correction g(h_k)         (branch 1: learn missing dynamics)
  hyb_unc   kf_feat + GRU multipliers on q_l, q_m, R          (branch 2: learn time-varying uncertainty)
  hyb_both  both heads
  gru       standalone GRU on the same inputs -> (mu, log var), similar size

The GRU sees the latest innovation only after it is observed: h_k = GRU(h_{k-1}, [nu_k / sqrt(S_k), log S_k, u_k]),
and its heads at k shape the prediction for k+1 (the causal ordering in the notes; no same-step circularity).
The innovation scale log S_k is the variance the filter itself predicted for k, so no future information enters h_k.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

KINDS = ("kf_fixed", "kf_feat", "hyb_dyn", "hyb_unc", "hyb_both", "gru")


class Model(nn.Module):
    def __init__(self, kind: str, p: int, hidden: int = 32, var_mean: float = 0.5, dist: str = "gauss"):
        super().__init__()
        self.kind = kind
        self.dist = dist
        self.log_nu2 = nn.Parameter(torch.tensor(math.log(4.0)))   # Student-t: nu = 2 + exp(.), starts at 6
        self.feat = kind != "kf_fixed"
        self.dyn = kind in ("hyb_dyn", "hyb_both")
        self.unc = kind in ("hyb_unc", "hyb_both")
        self.rnn = kind.startswith("hyb") or kind == "gru"
        lv = math.log(var_mean)
        self.phi_raw = nn.Parameter(torch.tensor(2.0))              # phi = sigmoid -> 0.88
        self.log_ql = nn.Parameter(torch.tensor(lv - 0.1))
        self.log_qm = nn.Parameter(torch.tensor(lv - 9.0))
        self.log_r = nn.Parameter(torch.tensor(lv - 4.0))
        if self.feat:
            self.B = nn.Linear(p, 1)
            nn.init.zeros_(self.B.weight)
            nn.init.zeros_(self.B.bias)
            self.w_lv = nn.Parameter(torch.tensor(1.0))              # q_l = exp(c + w * log vhat)
        if self.rnn:
            n_in = p + (2 if kind != "gru" else 0)
            self.cell = nn.GRUCell(n_in, hidden)
            self.head = nn.Linear(hidden, 5 if kind != "gru" else 2)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            if kind == "gru":
                with torch.no_grad():
                    self.head.bias[1] = lv
        self.hidden = hidden

    def init_state(self, n: int):
        z = torch.zeros(n)
        big = torch.full((n,), 10.0)
        return {"e": z.clone(), "m": z.clone(), "p11": big, "p12": z.clone(), "p22": torch.full((n,), 1e-3),
                "h": torch.zeros(n, self.hidden) if self.rnn else None, "z": z.clone(), "ls": z.clone()}

    def forward(self, r, u, lv, ok, st):
        """r[T,N] next-step returns, u[T,N,p] inputs at k, lv[T,N] log EWMA variance forecast for k+1, ok[T,N] score mask.
        Step k predicts r[k] from information through the close of row k-1 (the caller shifts). Returns mu, var [T,N]."""
        T = r.shape[0]
        mus, vs = [], []
        if self.kind == "gru":
            h = st["h"]
            for k in range(T):
                h = self.cell(u[k], h)
                o = self.head(h)
                mus.append(o[:, 0])
                vs.append(torch.exp(o[:, 1]))
            st = dict(st, h=h)
            return torch.stack(mus), torch.stack(vs), st
        phi = torch.sigmoid(self.phi_raw)
        e, m, p11, p12, p22, h = st["e"], st["m"], st["p11"], st["p12"], st["p22"], st["h"]
        z, ls = st["z"], st["ls"]
        for k in range(T):
            if self.rnn:   # absorb the last innovation (observed at the previous close) and the inputs known now
                h = self.cell(torch.cat([z[:, None], ls[:, None], u[k]], 1), h)
            log_ql = self.log_ql + (self.w_lv * (lv[k] - 0.0) if self.feat else 0.0)
            log_qm, log_r = self.log_qm, self.log_r
            drift = self.B(u[k]).squeeze(-1) if self.feat else 0.0
            gl = gm = 0.0
            if self.rnn:
                o = self.head(h)
                if self.dyn:
                    gl, gm = o[:, 0], 0.01 * o[:, 1]
                if self.unc:
                    log_ql, log_qm, log_r = log_ql + o[:, 2], log_qm + o[:, 3], log_r + o[:, 4]
            ql, qm, R = torch.exp(log_ql), torch.exp(log_qm), torch.exp(log_r)
            mtrend = phi * m + gm
            mu = e + m + drift + gl                      # x_{k+1|k,l} - y_k
            b11 = p11 + 2 * p12 + p22 + ql
            b12 = phi * (p12 + p22)
            b22 = phi * phi * p22 + qm
            S = b11 + R
            mus.append(mu)
            vs.append(S)
            nu = r[k] - mu
            k1, k2 = b11 / S, b12 / S
            okk = ok[k]
            e = torch.where(okk, (k1 - 1) * nu, mu - r[k])
            m = torch.where(okk, mtrend + k2 * nu, mtrend)
            p11 = torch.where(okk, b11 - b11 * b11 / S, b11)
            p12 = torch.where(okk, b12 - b11 * b12 / S, b12)
            p22 = torch.where(okk, b22 - b12 * b12 / S, b22)
            z = torch.where(okk, nu / torch.sqrt(S), torch.zeros_like(nu)).clamp(-10, 10)
            ls = torch.log(S)
        st = {"e": e, "m": m, "p11": p11, "p12": p12, "p22": p22, "h": h, "z": z, "ls": ls}
        return torch.stack(mus), torch.stack(vs), st


def gauss_nll(r, mu, v, ok):
    x = 0.5 * (torch.log(v) + (r - mu) ** 2 / v)
    return (x * ok).sum() / ok.sum().clamp(min=1)


def nll(model, r, mu, v, ok):
    """Gaussian or Student-t (fat-tailed, same variance v) negative log likelihood, averaged over scored rows."""
    if model.dist == "gauss":
        return gauss_nll(r, mu, v, ok)
    nu = 2 + torch.exp(model.log_nu2)
    a = v * (nu - 2) / nu                                          # t scale^2 so that the variance is v
    x = -(torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(math.pi * nu * a)
          - (nu + 1) / 2 * torch.log1p((r - mu) ** 2 / (nu * a))) - 0.5 * math.log(2 * math.pi)
    return (x * ok).sum() / ok.sum().clamp(min=1)


def _detach(st):
    return {k: (v.detach() if v is not None else None) for k, v in st.items()}


def _tensors(r, u, lv, ok, a, b):
    """Rows a..b-1 as the step inputs: step k sees u/lv of row k-1 and predicts r of row k."""
    return (torch.tensor(r[a:b], dtype=torch.float32), torch.tensor(u[a - 1:b - 1], dtype=torch.float32),
            torch.tensor(lv[a - 1:b - 1], dtype=torch.float32), torch.tensor(ok[a:b]))


def fit_predict(kind, r, u, lv, ok, tr0, va0, te0, te1, seed=0, streams=48, seg=168, max_epochs=25, patience=4, log=None,
                dist="gauss"):
    """Train on rows [tr0, va0), early-stop on [va0, te0), predict rows [te0, te1). Inputs are full-length numpy arrays
    (r: returns, u: standardized inputs, lv: log EWMA variance forecast, ok: rows to score). Returns mu, var, info."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    lv_c = lv - np.nanmean(lv[tr0:va0])                       # centre the vol input on training rows
    lv_c = np.nan_to_num(lv_c)
    var_mean = float(np.mean(r[tr0:va0][ok[tr0:va0]] ** 2))
    model = Model(kind, u.shape[1], var_mean=var_mean, dist=dist)
    with torch.no_grad():
        model.log_ql.fill_(math.log(var_mean))
    net = [p for n, p in model.named_parameters() if n.startswith(("cell", "head"))]
    kf = [p for n, p in model.named_parameters() if not n.startswith(("cell", "head"))]
    opt = torch.optim.Adam([{"params": kf, "lr": 2e-2}, {"params": net, "lr": 3e-3}])   # filter scalars move faster
    L = (va0 - tr0 - 1) // streams
    starts = tr0 + 1 + np.arange(streams) * L
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        st = model.init_state(streams)
        tot = 0.0
        for s in range(0, L, seg):
            n = min(seg, L - s)
            idx = starts[:, None] + s + np.arange(n)[None, :]          # [streams, n]
            rr = torch.tensor(r[idx].T, dtype=torch.float32)
            uu = torch.tensor(u[idx - 1].transpose(1, 0, 2), dtype=torch.float32)
            ll = torch.tensor(lv_c[idx - 1].T, dtype=torch.float32)
            oo = torch.tensor(ok[idx].T)
            if s == 0:
                oo[:48] = False                                          # filter burn-in
            mu, v, st = model(rr, uu, ll, oo, st)
            loss = nll(model, rr, mu, v, oo)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            st = _detach(st)
            tot += loss.item() * n
        # validation: run one stream from a month before va0 through te0
        model.eval()
        with torch.no_grad():
            a = max(tr0 + 1, va0 - 30 * 24)
            rr, uu, ll, oo = _tensors(r, u, lv_c, ok, a, te0)
            mu, v, _ = model(rr[:, None], uu[:, None], ll[:, None], oo[:, None], model.init_state(1))
            k = va0 - a
            val = nll(model, rr[k:, None], mu[k:], v[k:], oo[k:, None]).item()
        hist.append((ep, tot / L, val))
        if log:
            log(f"{kind} seed{seed} ep{ep} train {tot / L:.4f} val {val:.4f} ({time.time() - t0:.0f}s)")
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        a = max(tr0 + 1, va0 - 60 * 24)                                   # warm up the filter / GRU on recent rows
        rr, uu, ll, oo = _tensors(r, u, lv_c, ok, a, te1)
        mu, v, _ = model(rr[:, None], uu[:, None], ll[:, None], oo[:, None], model.init_state(1))
    k = te0 - a
    info = {"epochs": len(hist), "best_val": best, "phi": float(torch.sigmoid(model.phi_raw).detach()), "secs": time.time() - t0,
            "nu": float(2 + torch.exp(model.log_nu2).detach()) if dist == "t" else None}
    return mu[k:, 0].numpy().astype(float), v[k:, 0].numpy().astype(float), info
