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


def fm_loss(model, x1, outline, cfg):
    B = x1.shape[0]
    t = torch.rand(B, device=x1.device)                     # per-SAMPLE scalar
    x0 = torch.randn_like(x1)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    v_target = x1 - x0
    v_pred = model(xt, t, outline)
    W = make_weight(x1, cfg)
    return (W * (v_pred - v_target) ** 2).mean()


@torch.no_grad()
def sample(model, outline, cfg, steps=None, heun_last=5, generator=None):
    """Integrate the velocity field from noise (t=0) to data (t=1).

    Pass ``generator`` (a torch.Generator) to make the initial noise -- and hence
    the whole sample -- reproducible INDEPENDENTLY of how much global RNG model
    construction / caching consumed.  generate() uses this so identical
    (outline, seed) always yields identical room polygons (brief: fixed seed 42).
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
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * dt, device=dev)
        v = model(x, t, outline)
        if i >= steps - heun_last:                           # Heun corrector near t=1
            t2 = torch.full((B,), min((i + 1) * dt, 1.0), device=dev)
            v2 = model(x + v * dt, t2, outline)
            x = x + 0.5 * (v + v2) * dt
        else:
            x = x + v * dt
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
