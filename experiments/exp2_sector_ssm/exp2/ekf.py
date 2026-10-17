"""EKF + SSM: an extended Kalman filter over a learned state-space model of the sector.

The daily vector (returns, sector state, news) is treated as a noisy *measurement* of a small latent
sector state, rather than as the input of a deterministic network:

    x_t = f(x_{t-1}) + w_t,   w ~ N(0, Q)       f(x) = A x + U tanh(V x + c)
    z_t = H x_t + b + v_t,    v ~ N(0, R)       R diagonal

A is block diagonal with 2x2 damped rotations, the real form of S4D's complex diagonal modes
(r_k e^{+-i theta_k}); the tanh term lets the dynamics be nonlinear, which is why the filter is an
*extended* Kalman filter: the covariance is propagated with the Jacobian F = A + U diag(1 - tanh^2) V.

Each session the filter predicts, then updates with that day's measurements. Features that weren't
observed (news before collection began) are left out of the update, not treated as zeros. The
5-session targets are read out from the filtered state, y = C x_{t|t} + c0, with variance
diag(C P_{t|t} C^T) + s^2, so the state's uncertainty carries into the forecast.

All parameters (dynamics, Q, R, H, readout) are fitted together by backpropagating through the filter,
maximising the one-step-ahead measurement likelihood plus the targets' likelihood.

The update uses the information form, which only inverts d x d matrices (d = state size) however many
features are measured:
    P+ = (P^-1 + H^T R^-1 H)^-1,  x+ = x + P+ H^T R^-1 e.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class EKFSSM(nn.Module):
    def __init__(self, n_obs: int, n_targets: int, d_state: int = 8, d_hidden: int = 16):
        super().__init__()
        assert d_state % 2 == 0
        k = d_state // 2
        self.d, self.m = d_state, n_obs
        # damped rotations: r in (0.5, 1), periods spread from days to a year (S4D-Lin-like)
        self.rho = nn.Parameter(torch.linspace(1.0, 5.0, k))                 # r = 0.5 + 0.5 sigmoid(rho)
        self.theta = nn.Parameter(2 * math.pi / torch.logspace(math.log10(4), math.log10(250), k))
        self.U = nn.Parameter(torch.randn(d_state, d_hidden) * 0.05)
        self.V = nn.Parameter(torch.randn(d_hidden, d_state) * 0.3)
        self.c = nn.Parameter(torch.zeros(d_hidden))
        self.log_q = nn.Parameter(torch.full((d_state,), -2.0))
        self.H = nn.Parameter(torch.randn(n_obs, d_state) * 0.3)
        self.b = nn.Parameter(torch.zeros(n_obs))
        self.log_r = nn.Parameter(torch.zeros(n_obs))
        self.C = nn.Parameter(torch.zeros(n_targets, d_state))
        self.c0 = nn.Parameter(torch.zeros(n_targets))
        self.log_s = nn.Parameter(torch.zeros(n_targets))

    def A(self) -> torch.Tensor:
        r = 0.5 + 0.5 * torch.sigmoid(self.rho)
        cs, sn = r * torch.cos(self.theta), r * torch.sin(self.theta)
        blocks = torch.stack([torch.stack([cs, -sn], -1), torch.stack([sn, cs], -1)], -2)  # (k,2,2)
        return torch.block_diag(*blocks)

    def f(self, x: torch.Tensor, A: torch.Tensor):
        """State transition and its Jacobian, batched: x (B,d) -> (B,d), (B,d,d)."""
        h = torch.tanh(x @ self.V.T + self.c)                                  # (B,hid)
        fx = x @ A.T + h @ self.U.T
        J = A + torch.einsum("ij,bj,jk->bik", self.U, 1 - h ** 2, self.V)
        return fx, J

    def forward(self, z: torch.Tensor, mask: torch.Tensor):
        """Run the filter over z (B,L,m) with observation mask (B,L,m) (1 = observed).

        Returns the readout mean and variance per step (B,L,T), the one-step measurement negative
        log-likelihood per observed value (scalar), and the filtered means (B,L,d)."""
        B, L, _ = z.shape
        d = self.d
        A = self.A()
        Q = torch.diag(self.log_q.exp())
        rinv = (-self.log_r).exp()                                             # 1/R, (m,)
        eye = torch.eye(d, dtype=z.dtype)
        x = z.new_zeros(B, d)
        P = eye.expand(B, d, d) * 1.0
        nll, n_obs = z.new_zeros(()), z.new_zeros(())
        xs, Ps = [], []
        for t in range(L):
            if t > 0:
                x, F = self.f(x, A)
                P = F @ P @ F.transpose(1, 2) + Q
            msk = mask[:, t]                                                   # (B,m)
            w = rinv * msk                                                     # R^-1 on observed dims, 0 elsewhere
            e = (z[:, t] - (x @ self.H.T + self.b)) * msk                      # innovation
            HtW = self.H.T * w[:, None, :]                                     # (B,d,m) = H^T R^-1
            Pinv = torch.cholesky_inverse(torch.linalg.cholesky(P))
            info = Pinv + HtW @ self.H                                         # P+^-1
            Lc = torch.linalg.cholesky(info)
            Pp = torch.cholesky_inverse(Lc)
            g = (HtW @ e[..., None]).squeeze(-1)                               # H^T R^-1 e
            # log N(e; 0, S) with S = H P H^T + R, via the matrix determinant lemma and Woodbury
            logdetS = (self.log_r * msk).sum(-1) + torch.logdet(P) + 2 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
            quad = (e * e * w).sum(-1) - (g[:, None, :] @ Pp @ g[..., None]).squeeze(-1).squeeze(-1)
            nll = nll + 0.5 * (logdetS + quad + msk.sum(-1) * math.log(2 * math.pi)).sum()
            n_obs = n_obs + msk.sum()
            x = x + (Pp @ g[..., None]).squeeze(-1)
            P = 0.5 * (Pp + Pp.transpose(1, 2))
            xs.append(x)
            Ps.append(P)
        X, PP = torch.stack(xs, 1), torch.stack(Ps, 1)                          # (B,L,d), (B,L,d,d)
        mu = X @ self.C.T + self.c0
        var = torch.einsum("td,blde,te->blt", self.C, PP, self.C) + (2 * self.log_s).exp()
        return mu, var, nll / n_obs.clamp_min(1), X


def target_nll(mu, var, y):
    mask = ~torch.isnan(y)
    y = torch.nan_to_num(y)
    nll = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
    return (nll * mask).sum() / mask.sum().clamp_min(1)
