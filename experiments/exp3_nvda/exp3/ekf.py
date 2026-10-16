"""EKF + SSM: an extended Kalman filter over a learned state-space model (copied from Experiment 2).

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
    """norm: how much the input entering each 2x2 mode is scaled.
      "none"       as in Experiment 2;
      "h2"         by sqrt(1 - r^2), so a mode's state power no longer grows like 1/(1 - r^2) as r -> 1
                   (the H2 / white-input case of the resonance index; LRU's normalisation);
      "resonance"  by I_k^(-1/2), the resonance index of the mode under the measured input spectrum
                   (Philco, Eq. 21, with |g| = 1), so every mode carries the same output power and the
                   gradient reaching it is not dominated by the resonant ones.
    A mode's resonance index is also the energy of its impulse response, i.e. of the gradient of its
    output with respect to past inputs, summed over lags and weighted by the input spectrum; that is
    why the same quantity sets both how much a mode matters and how its gradient fades.
    radius: "sigmoid" r = 0.5 + 0.5 sigmoid(rho) (Experiment 2), or "exp" r = exp(-exp(nu)), initialised
    on a ring near the unit circle so slow modes start slow (as in LRU)."""

    def __init__(self, n_obs: int, n_targets: int, d_state: int = 8, d_hidden: int = 16, norm: str = "none", radius: str = "sigmoid",
                 r_min: float = 0.9, r_max: float = 0.995, sv: tuple[int, int] | None = None):
        """sv = (index of the daily return measurement, index of its log-square measurement): adds a stochastic
        volatility state h (log variance of the daily return) after the oscillating modes. h follows an AR(1)
        driven, like every other state, by the tanh coupling (so news and the other states can move it); it is
        measured by log(r^2) with slope 1 (Harvey-Ruiz-Shephard), it scales the return's measurement variance
        by exp(h), and it scales the forecast variance by exp(kappa h)."""
        super().__init__()
        assert d_state % 2 == 0 and norm in ("none", "h2", "resonance") and radius in ("sigmoid", "exp")
        k = d_state // 2
        self.k, self.sv = k, sv
        d_state = d_state + (1 if sv else 0)
        self.d, self.m, self.norm, self.radius = d_state, n_obs, norm, radius
        if sv:
            self.phi_raw = nn.Parameter(torch.tensor(3.0))                       # phi = sigmoid: persistence ~0.95
            self.kappa = nn.Parameter(torch.zeros(n_targets))
            hm = torch.ones(n_obs, d_state)
            hm[:, -1] = 0.0
            hm[sv[1], :] = 0.0
            hf = torch.zeros(n_obs, d_state)
            hf[sv[1], -1] = 1.0
            self.register_buffer("h_mask", hm)
            self.register_buffer("h_fix", hf)
        if radius == "sigmoid":
            # damped rotations: r in (0.5, 1), periods spread from days to a year (S4D-Lin-like)
            self.rho = nn.Parameter(torch.linspace(1.0, 5.0, k))                 # r = 0.5 + 0.5 sigmoid(rho)
        else:
            r0 = torch.linspace(r_min, r_max, k)
            self.rho = nn.Parameter(torch.log(-torch.log(r0)))                   # r = exp(-exp(rho))
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
        self.register_buffer("spectrum", torch.ones(64))      # input power spectrum on a frequency grid, mean 1

    def r(self) -> torch.Tensor:
        return 0.5 + 0.5 * torch.sigmoid(self.rho) if self.radius == "sigmoid" else torch.exp(-torch.exp(self.rho))

    def resonance(self) -> torch.Tensor:
        """Resonance index of each mode with unit gain under self.spectrum (both poles of the real 2x2 block)."""
        S = self.spectrum
        L = S.shape[0]
        r = self.r().clamp(max=1 - 1 / L)[:, None]
        nu = 2 * math.pi * torch.arange(L, dtype=r.dtype) / L
        th = self.theta[:, None]
        den = lambda t: 1 + r ** 2 - 2 * r * torch.cos(nu - t)  # noqa: E731
        return (S * 0.5 * (1 / den(th) + 1 / den(-th))).mean(-1)

    def gain(self) -> torch.Tensor:
        """Input gain per state dimension (each mode's value repeated for its two coordinates)."""
        if self.norm == "none":
            g = torch.ones_like(self.theta)
        elif self.norm == "h2":
            g = torch.sqrt(1 - self.r() ** 2)
        else:
            g = self.resonance().rsqrt()
        g = g.repeat_interleave(2)
        if self.sv:
            gh = torch.ones(1) if self.norm == "none" else torch.sqrt(1 - self.phi() ** 2).reshape(1)
            g = torch.cat([g, gh])
        return g

    def phi(self) -> torch.Tensor:
        return torch.sigmoid(self.phi_raw)

    def Hm(self) -> torch.Tensor:
        return self.H * self.h_mask + self.h_fix if self.sv else self.H

    def A(self) -> torch.Tensor:
        r = self.r()
        cs, sn = r * torch.cos(self.theta), r * torch.sin(self.theta)
        blocks = torch.stack([torch.stack([cs, -sn], -1), torch.stack([sn, cs], -1)], -2)  # (k,2,2)
        if self.sv:
            return torch.block_diag(*blocks, self.phi().reshape(1, 1))
        return torch.block_diag(*blocks)

    def f(self, x: torch.Tensor, A: torch.Tensor, g: torch.Tensor | None = None):
        """State transition and its Jacobian, batched: x (B,d) -> (B,d), (B,d,d)."""
        U = self.U if g is None else g[:, None] * self.U
        h = torch.tanh(x @ self.V.T + self.c)                                  # (B,hid)
        fx = x @ A.T + h @ U.T
        J = A + torch.einsum("ij,bj,jk->bik", U, 1 - h ** 2, self.V)
        return fx, J

    def forward(self, z: torch.Tensor, mask: torch.Tensor, x0: torch.Tensor | None = None, P0: torch.Tensor | None = None, record: bool = False):
        """Run the filter over z (B,L,m) with observation mask (B,L,m) (1 = observed).

        Returns the readout mean and variance per step (B,L,T), the one-step measurement negative
        log-likelihood per observed value (scalar), and the filtered means (B,L,d)."""
        B, L, _ = z.shape
        d = self.d
        A = self.A()
        g = self.gain()
        Q = torch.diag(self.log_q.exp() * g ** 2)
        eye = torch.eye(d, dtype=z.dtype)
        x = z.new_zeros(B, d) if x0 is None else x0
        P = eye.expand(B, d, d) * 1.0 if P0 is None else P0
        self.trace = []
        nll, n_obs = z.new_zeros(()), z.new_zeros(())
        H = self.Hm()
        xs, Ps = [], []
        for t in range(L):
            if t > 0:
                x, F = self.f(x, A, g)
                P = F @ P @ F.transpose(1, 2) + Q
            msk = mask[:, t]                                                   # (B,m)
            logr = self.log_r.expand(B, -1)
            if self.sv:     # the return's measurement variance is exp(h) times its base level (predicted h, plug-in)
                logr = logr + torch.nn.functional.one_hot(torch.tensor(self.sv[0]), self.m).to(z.dtype) * x[:, -1:]
            w = (-logr).exp() * msk                                            # R^-1 on observed dims, 0 elsewhere
            e = (z[:, t] - (x @ H.T + self.b)) * msk                           # innovation
            HtW = H.T * w[:, None, :]                                          # (B,d,m) = H^T R^-1
            Pinv = torch.cholesky_inverse(torch.linalg.cholesky(P))
            info = Pinv + HtW @ H                                              # P+^-1
            Lc = torch.linalg.cholesky(info)
            Pp = torch.cholesky_inverse(Lc)
            gi = (HtW @ e[..., None]).squeeze(-1)                              # H^T R^-1 e
            # log N(e; 0, S) with S = H P H^T + R, via the matrix determinant lemma and Woodbury
            logdetS = (logr * msk).sum(-1) + torch.logdet(P) + 2 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
            quad = (e * e * w).sum(-1) - (gi[:, None, :] @ Pp @ gi[..., None]).squeeze(-1).squeeze(-1)
            nll = nll + 0.5 * (logdetS + quad + msk.sum(-1) * math.log(2 * math.pi)).sum()
            n_obs = n_obs + msk.sum()
            if record:      # d x_{t|t} / d x_{t|t-1} = I - P+ H^T R^-1 H, and the transition Jacobian into it
                self.trace.append((F if t > 0 else None, eye - Pp @ HtW @ H))
            x = x + (Pp @ gi[..., None]).squeeze(-1)
            P = 0.5 * (Pp + Pp.transpose(1, 2))
            xs.append(x)
            Ps.append(P)
        self.last_x, self.last_P = x, P
        X, PP = torch.stack(xs, 1), torch.stack(Ps, 1)                          # (B,L,d), (B,L,d,d)
        mu = X @ self.C.T + self.c0
        noise = 2 * self.log_s
        if self.sv:
            noise = noise + self.kappa * X[..., -1:]
        var = torch.einsum("td,blde,te->blt", self.C, PP, self.C) + noise.exp()
        return mu, var, nll / n_obs.clamp_min(1), X


def target_nll(mu, var, y):
    mask = ~torch.isnan(y)
    y = torch.nan_to_num(y)
    nll = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
    return (nll * mask).sum() / mask.sum().clamp_min(1)
