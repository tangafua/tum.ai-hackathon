"""A full DDPM (denoising-diffusion) stack — the alternative to flow matching.

Same set-Transformer backbone (model.OutlineFlow) reused as an EPSILON predictor:
  forward q(x_t|x_0) = sqrt(abar_t) x0 + sqrt(1-abar_t) eps   (cosine schedule)
  train: predict eps, MSE (with the same channel weighting as flow)
  sample: DDIM/DDPM ancestral (eta in [0,1]: 0=deterministic DDIM, 1=full DDPM)

Output dim matches the model head, so checkpoints/decoding are identical to flow;
only the training target and the sampler differ.  cfg flag 'diffusion'=True marks
a checkpoint so sample_eval picks this sampler.
"""
from __future__ import annotations
import math
import torch
from flow import make_weight


def abar(t, s: float = 0.008):
    """Cosine schedule cumulative-alpha, continuous t in [0,1] (1=pure noise)."""
    f = torch.cos(((1.0 - t) + s) / (1.0 + s) * math.pi / 2.0) ** 2
    f0 = math.cos(s / (1.0 + s) * math.pi / 2.0) ** 2
    return (f / f0).clamp(1e-5, 1.0 - 1e-5)


def q_sample(x0, t, eps):
    a = abar(t)[:, None, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * eps


def ddpm_loss(model, x1, outline, cfg, scale=None):
    """Epsilon-prediction MSE with the flow's per-channel/slot weighting."""
    B = x1.shape[0]
    t = torch.rand(B, device=x1.device)
    eps = torch.randn_like(x1)
    xt = q_sample(x1, t, eps)
    eps_pred = model(xt, t, outline, scale)
    W = make_weight(x1, cfg)
    return (W * (eps_pred - eps) ** 2).mean()


@torch.no_grad()
def ddim_sample(model, outline, cfg, steps=100, eta=1.0, generator=None, scale=None):
    """Reverse diffusion from t=1 (noise) to t=0 (data). eta=1 DDPM, eta=0 DDIM."""
    B = outline.shape[0]; dev = outline.device
    if generator is not None:
        x = torch.randn(B, cfg.n_max, cfg.d, generator=generator,
                        device=generator.device).to(dev)
    else:
        x = torch.randn(B, cfg.n_max, cfg.d, device=dev)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for i in range(steps):
        t = torch.full((B,), ts[i].item(), device=dev)
        a_t = abar(t)[:, None, None]
        a_s = abar(torch.full((B,), ts[i + 1].item(), device=dev))[:, None, None]
        eps = model(x, t, outline, scale)
        x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-4, 4)
        sigma = eta * ((1 - a_s) / (1 - a_t)).sqrt() * (1 - a_t / a_s).clamp(0, 1).sqrt()
        noise = torch.randn_like(x) if i < steps - 1 else 0.0
        x = a_s.sqrt() * x0 + (1 - a_s - sigma ** 2).clamp(min=0).sqrt() * eps + sigma * noise
    return x
