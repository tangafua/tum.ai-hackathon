"""Rectified-flow / conditional-OT velocity matching + ODE sampler.

CONVENTION: noise at t=0, DATA at t=1.  Path x_t = (1-t)*x0 + t*x1, with
x0 ~ N(0,I), so the target velocity is the constant v = x1 - x0.  This is
written fresh and deliberately does NOT reuse the score-style loss in the
sibling Energy-weighted-flow-matching repo (that one puts data at t=0 with a
per-token t -- copying it would be a silent time-reversal bug).
"""
from __future__ import annotations
import copy
import torch


def make_weight(x1, cfg):
    """Per-channel x per-slot loss weights -> W [B,N,D].

    Geometry/type of PADDING slots is meaningless, so it is down-weighted; but
    the PRESENCE channel is supervised at full weight on EVERY slot (otherwise
    padding slots drift positive at sampling and the model over-generates rooms).
    """
    D = cfg.d
    cw = torch.ones(D, device=x1.device)
    cw[1:7] = cfg.w_geometry
    cw[7:] = cfg.w_type
    present = (x1[..., 0] > 0).float()                       # [B,N]
    slot_w = present + (1.0 - present) * cfg.w_pad_slot
    W = cw[None, None, :] * slot_w[:, :, None]
    W[..., 0] = cfg.w_presence                               # presence: all slots
    return W


def fm_loss(model, x1, outline, cfg, cond=None, cfg_drop=0.0):
    B = x1.shape[0]
    if getattr(cfg, "t_dist", "uniform") == "logitnormal":
        # SD3-style: emphasize mid timesteps (t=sigmoid(m+s*z)) -> better FM training
        m = getattr(cfg, "t_logit_mean", 0.0)
        s = getattr(cfg, "t_logit_std", 1.0)
        t = torch.sigmoid(m + s * torch.randn(B, device=x1.device))
    else:
        t = torch.rand(B, device=x1.device)                 # per-SAMPLE scalar
    x0 = torch.randn_like(x1)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    v_target = x1 - x0
    if cfg_drop > 0.0:                                       # classifier-free guidance
        # drop the WHOLE outline condition (boundary + scale) for a random subset, so
        # the model also learns an unconditional field usable at guided sampling.
        keep = (torch.rand(B, device=x1.device) >= cfg_drop).float()[:, None, None]
        outline = outline * keep
        if cond is not None:
            cond = cond * keep[:, :, 0]
    v_pred = model(xt, t, outline, cond)
    W = make_weight(x1, cfg)
    return (W * (v_pred - v_target) ** 2).mean()


@torch.no_grad()
def sample(model, outline, cfg, steps=None, heun_last=5, generator=None,
           cond=None, guidance=1.0, churn=0.0):
    """Integrate the velocity field from noise (t=0) to data (t=1).

    Pass ``generator`` (a torch.Generator) to make the initial noise -- and hence
    the whole sample -- reproducible INDEPENDENTLY of how much global RNG model
    construction / caching consumed.  generate() uses this so identical
    (outline, seed) always yields identical room polygons (brief: fixed seed 42).

    ``cond`` is the global outline-scale conditioning (room-count control).
    ``guidance`` > 1 applies classifier-free guidance: v = v_uncond + g*(v_cond-v_uncond),
    pushing samples to follow the outline more strongly (needs a model trained with
    cfg_drop > 0).
    """
    steps = steps or cfg.sample_steps
    B = outline.shape[0]
    dev = outline.device
    if generator is not None:
        # draw on the generator's device (CPU generator -> CPU noise), then move
        x = torch.randn(B, cfg.n_max, cfg.d, generator=generator,
                        device=generator.device).to(dev)
    else:
        x = torch.randn(B, cfg.n_max, cfg.d, device=dev)

    null_outline = torch.zeros_like(outline)
    null_cond = None if cond is None else torch.zeros_like(cond)

    def vel(xc, tc):
        if guidance is not None and guidance != 1.0:
            v_c = model(xc, tc, outline, cond)
            v_u = model(xc, tc, null_outline, null_cond)
            return v_u + guidance * (v_c - v_u)
        return model(xc, tc, outline, cond)

    dt = 1.0 / steps
    for i in range(steps):
        tval = i * dt
        t = torch.full((B,), tval, device=dev)
        v = vel(x, t)
        if i >= steps - heun_last:                           # Heun corrector near t=1
            t2 = torch.full((B,), min((i + 1) * dt, 1.0), device=dev)
            v2 = vel(x + v * dt, t2)
            x = x + 0.5 * (v + v2) * dt
        else:
            x = x + v * dt
        if churn > 0.0 and i < steps - heun_last:            # SDE-style noise injection
            # noise scaled by remaining time -> more diversity early, settle near t=1
            # (jiahua's Coverage lever: escapes the L2 mean-trajectory collapse)
            x = x + churn * (1.0 - tval) * (dt ** 0.5) * torch.randn_like(x)
    return x


# ----------------------------------------------------------------------------
# M4: EDM (Karras et al. 2022) -- preconditioned denoiser objective + Heun sampler.
# Same network F = model(xt, t, outline, cond); EDM wraps it as a denoiser
# D(x;sigma) = c_skip*x + c_out*F(c_in*x; c_noise), trained to predict clean data.
# ----------------------------------------------------------------------------

def edm_denoise(model, x, sigma, outline, cfg, cond=None):
    """D(x;sigma) with EDM preconditioning. sigma: [B]."""
    sd = cfg.sigma_data
    s2 = sigma ** 2
    c_skip = (sd ** 2 / (s2 + sd ** 2))[:, None, None]
    c_out = (sigma * sd / (s2 + sd ** 2).sqrt())[:, None, None]
    c_in = (1.0 / (s2 + sd ** 2).sqrt())[:, None, None]
    c_noise = 0.25 * sigma.log()                             # [B]; the network's "t" input
    F = model(c_in * x, c_noise, outline, cond)
    return c_skip * x + c_out * F


def edm_loss(model, x1, outline, cfg, cond=None, cfg_drop=0.0):
    B = x1.shape[0]
    ln_sigma = cfg.edm_p_mean + cfg.edm_p_std * torch.randn(B, device=x1.device)
    sigma = ln_sigma.exp()                                   # [B]
    x = x1 + sigma[:, None, None] * torch.randn_like(x1)
    if cfg_drop > 0.0:
        keep = (torch.rand(B, device=x1.device) >= cfg_drop).float()[:, None, None]
        outline = outline * keep
        if cond is not None:
            cond = cond * keep[:, :, 0]
    D = edm_denoise(model, x, sigma, outline, cfg, cond)
    W = make_weight(x1, cfg)
    lam = ((sigma ** 2 + cfg.sigma_data ** 2) / (sigma * cfg.sigma_data) ** 2)
    return (lam[:, None, None] * W * (D - x1) ** 2).mean()


@torch.no_grad()
def edm_sample(model, outline, cfg, steps=None, cond=None, guidance=1.0, generator=None):
    """EDM Heun sampler: integrate sigma_max -> 0 on the Karras rho-schedule."""
    steps = steps or cfg.sample_steps
    B = outline.shape[0]
    dev = outline.device
    s_min, s_max, rho = cfg.edm_sigma_min, cfg.edm_sigma_max, cfg.edm_rho
    i = torch.arange(steps, device=dev, dtype=torch.float32)
    sig = (s_max ** (1 / rho) + i / (steps - 1) *
           (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros(1, device=dev)])       # append sigma=0
    if generator is not None:
        x = torch.randn(B, cfg.n_max, cfg.d, generator=generator,
                        device=generator.device).to(dev)
    else:
        x = torch.randn(B, cfg.n_max, cfg.d, device=dev)
    x = x * sig[0]
    null_outline = torch.zeros_like(outline)
    null_cond = None if cond is None else torch.zeros_like(cond)

    def denoise(xc, sigma_scalar):
        sb = torch.full((B,), float(sigma_scalar), device=dev)
        if guidance is not None and guidance != 1.0:
            d_c = edm_denoise(model, xc, sb, outline, cfg, cond)
            d_u = edm_denoise(model, xc, sb, null_outline, cfg, null_cond)
            return d_u + guidance * (d_c - d_u)
        return edm_denoise(model, xc, sb, outline, cfg, cond)

    for j in range(steps):
        s_cur, s_next = sig[j], sig[j + 1]
        D = denoise(x, s_cur)
        d = (x - D) / s_cur
        x_next = x + (s_next - s_cur) * d
        if s_next > 0:                                       # Heun 2nd-order correction
            D2 = denoise(x_next, s_next)
            d2 = (x_next - D2) / s_next
            x_next = x + (s_next - s_cur) * 0.5 * (d + d2)
        x = x_next
    return x


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: p.detach().clone()
                       for k, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for k, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def make_model(self, model):
        """Return a deepcopy of `model` with the EMA weights loaded (for sampling)."""
        m = copy.deepcopy(model)
        sd = m.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)
        m.load_state_dict(sd)
        m.eval()
        return m

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}
