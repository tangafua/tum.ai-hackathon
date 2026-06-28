"""OutlineFlow: a conditional set-Transformer velocity field for rectified flow.

    v_theta( xt[B,N,D], t[B], outline[B,P,4] ) -> v[B,N,D]

Design (per the judged design dossier):
  * room tokens carry NO positional encoding (a plan is a SET);
  * the outline is encoded by a PointNet-lite into BOTH a global conditioning
    vector AND a per-boundary-point feature sequence;
  * time + a GLOBAL outline summary are injected through DiT-style AdaLN-Zero
    blocks (gates zero-init -> training starts as the identity, the most stable
    conditioning);
  * each block ALSO cross-attends room tokens -> boundary points, so a token can
    "see" which part of the outline it sits near and learn a spatially-grounded,
    outline-filling layout (a single pooled vector cannot encode this -- it is the
    structural ceiling on Density).
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


class FourierFeatures2D(nn.Module):
    """Fixed random Fourier features for 2D boundary-point coordinates.

    Maps (x_n, y_n) in [-1,1] to [sin, cos] of random projections so the outline
    encoder can represent fine, high-frequency boundary geometry instead of seeing
    only a low-capacity linear map of the raw coordinates.
    """

    def __init__(self, n_freqs: int, scale: float = 10.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(2, n_freqs) * scale, requires_grad=False)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:  # [...,2]
        proj = 2.0 * np.pi * (xy @ self.W)                # [...,n_freqs]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [...,2*n_freqs]


class OutlineEncoder(nn.Module):
    """PointNet-lite over boundary points -> (global vector, per-point features).

    The global vector (max+mean pooled) feeds AdaLN conditioning; the per-point
    feature sequence is the K/V the room tokens cross-attend to.
    """

    def __init__(self, d_model: int, n_freqs: int = 16):
        super().__init__()
        self.ff = FourierFeatures2D(n_freqs)
        in_dim = 4 + 2 * n_freqs                # raw (x,y,nx,ny) + fourier(x,y)
        self.pt = nn.Sequential(
            nn.Linear(in_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, d_model),
        )
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, outline: torch.Tensor):                  # [B,P,4]
        ff = self.ff(outline[..., :2])                          # [B,P,2*n_freqs]
        h = self.pt(torch.cat([outline, ff], dim=-1))           # [B,P,d_model]
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)
        return self.proj(pooled), h                             # [B,d_model], [B,P,d_model]


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZeroBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, use_cross_attn: bool = True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # cross-attention: room tokens (Q) attend to boundary points (K/V)
        if use_cross_attn:
            self.ln_cross = nn.LayerNorm(d_model, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        # AdaLN modulation for self-attn + mlp (shift/scale/gate each = 6*d)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        # dedicated zero-init gate for the cross-attn residual (AdaLN-Zero style)
        if use_cross_attn:
            self.ada_cross = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model))
            nn.init.zeros_(self.ada_cross[-1].weight)
            nn.init.zeros_(self.ada_cross[-1].bias)

    def forward(self, h, c, ctx):
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.ada(c).chunk(6, dim=-1)
        # self-attention over room tokens
        x = modulate(self.ln1(h), s_msa, sc_msa)
        a, _ = self.attn(x, x, x, need_weights=False)
        h = h + g_msa.unsqueeze(1) * a
        # cross-attention: tokens -> boundary points (zero-gated -> identity at init)
        if self.use_cross_attn:
            g_cross = self.ada_cross(c)
            a_cross, _ = self.cross_attn(self.ln_cross(h), ctx, ctx, need_weights=False)
            h = h + g_cross.unsqueeze(1) * a_cross
        # mlp
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
        self.outline_enc = OutlineEncoder(d, getattr(cfg, "outline_fourier_freqs", 16))
        # global outline-scale -> conditioning (room-count control); zero-init last
        # layer so training starts identical to the no-count-condition model.
        n_cond = getattr(cfg, "n_cond", 0)
        self.cond_embed = None
        if n_cond:
            self.cond_embed = nn.Sequential(nn.Linear(n_cond, d), nn.SiLU(), nn.Linear(d, d))
            nn.init.zeros_(self.cond_embed[-1].weight)
            nn.init.zeros_(self.cond_embed[-1].bias)
        use_xattn = getattr(cfg, "use_cross_attn", True)
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(d, cfg.n_heads, cfg.mlp_ratio, use_xattn) for _ in range(cfg.n_layers)]
        )
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.head = nn.Linear(d, cfg.d)
        for m in (self.ada_out[-1], self.head):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xt, t, outline, cond=None):
        h = self.embed_tok(xt)                       # [B,N,d]
        outline_vec, outline_pts = self.outline_enc(outline)   # [B,d], [B,P,d]
        c = self.t_embed(self.gfp(t)) + outline_vec            # [B,d]
        if self.cond_embed is not None and cond is not None:
            c = c + self.cond_embed(cond)            # global outline-scale -> room count
        c = F.silu(c)
        for blk in self.blocks:
            h = blk(h, c, outline_pts)
        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        h = modulate(self.norm_out(h), shift, scale)
        return self.head(h)                          # [B,N,D]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
