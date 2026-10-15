"""A small S4D network (diagonal state-space layers) with mode pruning.

Each S4D layer runs H independent linear state-space systems, one per channel, each with N
complex diagonal modes:

    x_k = exp(dt * A) x_{k-1} + B u_k,     y_k = 2 Re(C x_k) + D u_k,   A = -exp(a_re) + i a_im

discretised with zero-order hold and initialised as S4D-Lin (A_n = -1/2 + i pi n). In training the
layer is applied as a causal convolution with the kernel K_l = 2 Re(sum_n C_n B_n exp(dt A_n l)),
computed with an FFT; `step` runs the same system as a recurrence, which is how it is checked.

Pruning: a mode's contribution to its channel's output is bounded by its kernel's L1 norm,
|C_n B_n| / (1 - |exp(dt A_n)|). Modes are ranked by that bound within each layer, the weakest are
masked off, the rest fine-tuned, and the pruned network is kept only if it validates as well.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


class S4D(nn.Module):
    def __init__(self, h: int, n: int, dt_min: float, dt_max: float, dropout: float):
        super().__init__()
        self.h, self.n = h, n
        log_dt = torch.rand(h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.a_re = nn.Parameter(torch.log(0.5 * torch.ones(h, n)))
        self.a_im = nn.Parameter(math.pi * torch.arange(n).float().repeat(h, 1))
        c = torch.randn(h, n, dtype=torch.cfloat) * (0.5 ** 0.5)
        self.c = nn.Parameter(torch.view_as_real(c))
        self.d = nn.Parameter(torch.randn(h))
        self.register_buffer("mask", torch.ones(h, n))
        self.drop = nn.Dropout(dropout)

    def _disc(self):
        dt = self.log_dt.exp().unsqueeze(-1)                         # (H,1)
        a = -self.a_re.exp() + 1j * self.a_im                        # (H,N)
        c = torch.view_as_complex(self.c) * self.mask                # (H,N)
        dta = dt * a
        cb = c * (dta.exp() - 1.0) / a                               # ZOH: B folded into C
        return dta, cb

    def kernel(self, length: int) -> torch.Tensor:
        dta, cb = self._disc()
        pos = torch.arange(length, device=dta.device)
        return 2 * (cb.unsqueeze(-1) * torch.exp(dta.unsqueeze(-1) * pos)).sum(1).real   # (H,L)

    def forward(self, u: torch.Tensor) -> torch.Tensor:              # u: (B,L,H)
        L = u.shape[1]
        k = self.kernel(L)
        uf = torch.fft.rfft(u.transpose(1, 2), n=2 * L)
        kf = torch.fft.rfft(k, n=2 * L)
        y = torch.fft.irfft(uf * kf, n=2 * L)[..., :L].transpose(1, 2)
        return self.drop(y + u * self.d)

    @torch.no_grad()
    def step_all(self, u: torch.Tensor) -> torch.Tensor:
        """Same output as forward (in eval mode), computed recurrently. Used in tests."""
        dta, cb = self._disc()
        lam = dta.exp()
        x = torch.zeros(u.shape[0], self.h, self.n, dtype=torch.cfloat)
        ys = []
        for t in range(u.shape[1]):
            x = lam * x + u[:, t, :, None]
            ys.append(2 * (cb * x).sum(-1).real + self.d * u[:, t])
        return torch.stack(ys, 1)

    @torch.no_grad()
    def poles(self):
        """(g, r, theta) per mode: gain g_n = C_n B_n (ZOH), and the discrete pole r_n e^{i theta_n}."""
        dta, cb = self._disc()
        lam = dta.exp()
        return cb, lam.abs(), lam.angle()

    @torch.no_grad()
    def importance(self) -> torch.Tensor:
        """Kernel L1 bound |g| / (1 - r)."""
        g, r, _ = self.poles()
        return g.abs() / (1 - r).clamp_min(1e-4)      # (H,N)

    @torch.no_grad()
    def scores(self, criterion: str, spectrum: torch.Tensor | None = None) -> torch.Tensor:
        """Per-mode pruning scores (H,N), higher = keep.

        resonance: Philco's resonance index, the mode's output power when driven by the measured
            input spectrum S_h (Eq. 21): |g|^2/L sum_j S_h(nu_j) / (1 + r^2 - 2 r cos(nu_j - theta)),
            with r clamped to 1 - 1/L.
        flat: the same with S_h replaced by a flat spectrum of equal power (isolates spectral shape).
        h2: |g|^2 / (1 - r^2), the white-input special case (AIRE-Prune).
        hinf: |g|^2 / (1 - r)^2, the worst-case-input special case (LAST).
        l1: |g| / (1 - r), the kernel L1 bound.
        """
        g, r, th = self.poles()
        g2 = g.abs() ** 2
        if criterion in ("resonance", "flat"):
            L = spectrum.shape[-1]
            r = r.clamp(max=1 - 1 / L)
            if criterion == "flat":
                return spectrum.mean(-1, keepdim=True) * g2 / (1 - r ** 2)
            nu = 2 * math.pi * torch.arange(L) / L
            denom = 1 + r[..., None] ** 2 - 2 * r[..., None] * torch.cos(nu - th[..., None])   # (H,N,L)
            return g2 * (spectrum[:, None, :] / denom).sum(-1) / L
        if criterion == "h2":
            return g2 / (1 - r ** 2).clamp_min(1e-8)
        if criterion == "hinf":
            return g2 / (1 - r).clamp_min(1e-4) ** 2
        if criterion == "l1":
            return self.importance()
        raise ValueError(criterion)


class SectorSSM(nn.Module):
    """Features -> linear encoder -> stacked S4D blocks -> mean and log-variance for each target."""

    def __init__(self, n_features: int, n_targets: int, cfg: ModelConfig):
        super().__init__()
        self.enc = nn.Linear(n_features, cfg.d_model)
        self.layers = nn.ModuleList(S4D(cfg.d_model, cfg.n_modes, cfg.dt_min, cfg.dt_max, cfg.dropout) for _ in range(cfg.n_layers))
        self.mix = nn.ModuleList(nn.Linear(cfg.d_model, cfg.d_model) for _ in range(cfg.n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers))
        self.head = nn.Linear(cfg.d_model, 2 * n_targets)
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor):
        z = self.enc(x)
        for ssm, mix, norm in zip(self.layers, self.mix, self.norms):
            z = norm(z + mix(nn.functional.gelu(ssm(z))))
        out = self.head(z)
        mu, logvar = out[..., : self.n_targets], out[..., self.n_targets:].clamp(-6, 8)
        return mu, logvar

    def param_groups(self, cfg: ModelConfig):
        ssm = [p for layer in self.layers for name, p in layer.named_parameters() if name in ("log_dt", "a_re", "a_im")]
        ids = {id(p) for p in ssm}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": ssm, "lr": cfg.ssm_lr, "weight_decay": 0.0}, {"params": rest, "lr": cfg.lr, "weight_decay": cfg.weight_decay}]

    @torch.no_grad()
    def layer_inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Each S4D layer's input sequence (B,L,H) for input x, in eval mode."""
        was = self.training
        self.eval()
        z, out = self.enc(x), []
        for ssm, mix, norm in zip(self.layers, self.mix, self.norms):
            out.append(z)
            z = norm(z + mix(nn.functional.gelu(ssm(z))))
        self.train(was)
        return out

    @torch.no_grad()
    def input_spectra(self, x: torch.Tensor, seg: int = 256) -> list[torch.Tensor]:
        """Per layer, each channel's input power spectrum (H,seg): periodogram averaged over
        non-overlapping length-`seg` segments of x (step 2 of the resonance-index algorithm)."""
        out = []
        for u in self.layer_inputs(x):
            u = u[0] if u.dim() == 3 else u                       # (L,H)
            n = u.shape[0] // seg
            segs = u[: n * seg].reshape(n, seg, -1)               # (B,seg,H)
            p = (torch.fft.fft(segs, dim=1).abs() ** 2).mean(0) / seg  # (seg,H)
            out.append(p.T.contiguous())
        return out

    @torch.no_grad()
    def prune(self, keep: float, criterion: str = "l1", spectra: list[torch.Tensor] | None = None) -> None:
        """Mask all but the highest-scoring `keep` fraction of modes in each layer (already-masked modes stay off)."""
        for k_layer, layer in enumerate(self.layers):
            sc = layer.scores(criterion, spectra[k_layer] if spectra is not None else None) * layer.mask
            k = max(1, round(keep * sc.numel()))
            thresh = sc.flatten().topk(k).values.min()
            layer.mask.copy_((sc >= thresh).float() * layer.mask)

    def modes(self) -> tuple[int, int]:
        active = sum(int(layer.mask.sum()) for layer in self.layers)
        return active, sum(layer.mask.numel() for layer in self.layers)


def gaussian_nll(mu, logvar, y):
    """Mean Gaussian negative log-likelihood over the targets that are known (not NaN)."""
    mask = ~torch.isnan(y)
    y = torch.nan_to_num(y)  # zeros where unknown; masked out below, and keeps NaN out of the gradient
    nll = 0.5 * (logvar + (y - mu) ** 2 / logvar.exp())
    return (nll * mask).sum() / mask.sum().clamp_min(1)
