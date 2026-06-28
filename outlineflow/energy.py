"""Energy critic for energy-guided sampling (spec-compliant: the GENERATOR stays a
flow-matching model; the energy is only a sampling-time guidance term).

E(x_t, t, outline) is a time-conditioned critic trained to separate REAL layout
tokens from flow-GENERATED ones (both noised with the flow's forward process).
At sample time we add  +guidance * d/dx logit_real  to the velocity, steering the
ODE toward the real-layout manifold -- classifier/discriminator guidance, in the
model's own token space (so it is fully differentiable, no non-diff decode).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GaussianFourierProjection, OutlineEncoder, AdaLNZeroBlock


class EnergyCritic(nn.Module):
    def __init__(self, cfg, n_layers: int = 3):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.embed_tok = nn.Linear(cfg.d, d)
        self.gfp = GaussianFourierProjection(d)
        self.t_embed = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.outline_enc = OutlineEncoder(d)
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(d, cfg.n_heads, cfg.mlp_ratio) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 1))

    def forward(self, x, t, outline):
        """x:[B,N,D] t:[B] outline:[B,P,4] -> logit[B] (high = looks REAL)."""
        h = self.embed_tok(x)
        c = F.silu(self.t_embed(self.gfp(t)) + self.outline_enc(outline)[0])
        for blk in self.blocks:
            h = blk(h, c)
        h = self.norm(h)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)  # perm-invariant
        return self.head(pooled).squeeze(-1)
