"""OutlineFlow: a conditional set-Transformer velocity field for rectified flow.

    v_theta( xt[B,N,D], t[B], outline[B,P,4] ) -> v[B,N,D]

Design (per the judged design dossier):
  * room tokens carry NO positional encoding (a plan is a SET);
  * the outline is encoded by a permutation-invariant PointNet-lite;
  * time + outline are injected through DiT-style AdaLN-Zero blocks, so training
    starts as the identity (gates zero-init) -- the most stable conditioning;
  * full bidirectional self-attention over the room tokens is what lets the model
    learn a JOINT, non-overlapping, outline-filling layout (an MLP cannot).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """Fixed random Fourier features for the scalar time t (vendored, device-safe)."""

    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: [B]
        proj = t[:, None] * self.W[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [B, embed_dim]


class OutlineEncoder(nn.Module):
    """PointNet-lite over boundary points -> pooled vector + per-point tokens."""

    def __init__(self, d_model: int):
        super().__init__()
        self.pt = nn.Sequential(
            nn.Linear(4, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, d_model),
        )
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, outline: torch.Tensor):                  # [B,P,4]
        h = self.pt(outline)                                   # [B,P,d_model] per-point tokens
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)
        return self.proj(pooled), h                            # ([B,d_model], [B,P,d_model])


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZeroBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, cross: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        self.cross = cross
        if cross:                                    # room tokens attend to outline tokens
            self.ln_x = nn.LayerNorm(d_model, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.ada_x = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))
            nn.init.zeros_(self.ada_x[-1].weight)
            nn.init.zeros_(self.ada_x[-1].bias)

    def forward(self, h, c, ctx=None):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.ada(c).chunk(6, dim=-1)
        x = modulate(self.ln1(h), s_msa, sc_msa)
        a, _ = self.attn(x, x, x, need_weights=False)
        h = h + g_msa.unsqueeze(1) * a
        if self.cross and ctx is not None:           # cross-attend to per-point outline
            s_x, sc_x, g_x = self.ada_x(c).chunk(3, dim=-1)
            x = modulate(self.ln_x(h), s_x, sc_x)
            a, _ = self.cross_attn(x, ctx, ctx, need_weights=False)
            h = h + g_x.unsqueeze(1) * a
        x = modulate(self.ln2(h), s_mlp, sc_mlp)
        h = h + g_mlp.unsqueeze(1) * self.mlp(x)
        return h


class OutlineFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.embed_tok = nn.Linear(cfg.d, d)
        self.gfp = GaussianFourierProjection(d)
        self.t_embed = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.outline_enc = OutlineEncoder(d)
        # optional absolute-scale conditioning ([log area, log w, log h] -> d), so the
        # model can vary room COUNT with outline size (sample_outline_points drops scale)
        self.use_scale = getattr(cfg, "use_scale", False)
        if self.use_scale:
            self.scale_embed = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.cross_attn = getattr(cfg, "cross_attn", False)
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(d, cfg.n_heads, cfg.mlp_ratio, cross=self.cross_attn)
             for _ in range(cfg.n_layers)]
        )
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.head = nn.Linear(d, cfg.d)
        for m in (self.ada_out[-1], self.head):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xt, t, outline, scale=None):
        h = self.embed_tok(xt)                       # [B,N,d]
        pooled, otok = self.outline_enc(outline)     # [B,d], [B,P,d]
        c = self.t_embed(self.gfp(t)) + pooled       # [B,d]
        if self.use_scale and scale is not None:
            c = c + self.scale_embed(scale)
        c = F.silu(c)
        ctx = otok if self.cross_attn else None
        for blk in self.blocks:
            h = blk(h, c, ctx)
        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        h = modulate(self.norm_out(h), shift, scale)
        return self.head(h)                          # [B,N,D]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
