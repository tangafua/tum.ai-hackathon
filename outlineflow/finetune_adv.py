"""Adversarial / 'negative-unlearning' fine-tune of the flow model with an EBM critic.

GAN-style: a critic learns real-vs-(current model) layouts; the flow generator is
fine-tuned to (a) keep matching the flow-matching target [grounding] and (b) make a
short differentiable noise->layout rollout look REAL to the critic [unlearn its own
bad/negative modes].  Generator stays a flow model (spec-compliant); the critic is
only a training signal.

    python finetune_adv.py --flow_ckpt outputs_full_plan_id/ckpt.pt --steps 4000 \
        --lam 0.3 --rollout 8 --out_dir outputs_finetune_adv
"""
from __future__ import annotations
import argparse, time, pickle, shutil, os
import numpy as np
import torch
import torch.nn.functional as F

import params, synth_data, msd_data
from cfg import CFG, seed_everything
from model import OutlineFlow, count_params
from flow import EMA, fm_loss
from energy import EnergyCritic


def load_flow(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("n_max", "k", "n_gen_classes", "p_outline", "d_model", "n_layers",
              "n_heads", "mlp_ratio", "canvas", "nearest_k", "min_area_frac", "use_scale"):
        if k in ck["cfg"]:
            setattr(CFG, k, ck["cfg"][k])
    m = OutlineFlow(CFG); m.load_state_dict(ck["model"], strict=False)
    return m, ck["stats"]


def rollout(model, outline, cfg, steps):
    """Short DIFFERENTIABLE noise->layout rollout (Euler) for the adversarial loss."""
    B = outline.shape[0]
    x = torch.randn(B, cfg.n_max, cfg.d, device=outline.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * dt, device=outline.device)
        x = x + model(x, t, outline) * dt
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_ckpt", default="outputs_full_plan_id/ckpt.pt")
    ap.add_argument("--data_csv", default="/root/jiahua_code/tum.ai-hackathon/mds_V2_5.372k.csv")
    ap.add_argument("--held", default="outputs_full_plan_id/held.pkl")
    ap.add_argument("--msd_limit", type=int, default=2500)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lam", type=float, default=0.3, help="adversarial weight")
    ap.add_argument("--rollout", type=int, default=8, help="differentiable rollout steps")
    ap.add_argument("--out_dir", default="outputs_finetune_adv")
    ap.add_argument("--device", default=CFG.device)
    args = ap.parse_args()

    seed_everything(42); dev = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    model, stats = load_flow(args.flow_ckpt, dev); model = model.to(dev)

    samples, _ = msd_data.load_msd_samples(args.data_csv, CFG, limit=args.msd_limit,
                                           group="plan_id", set_cfg=False)
    rng = np.random.default_rng(42)
    X, OUT = synth_data.build_tensors(samples, stats, CFG, rng)
    Xr = torch.from_numpy(X).to(dev); Ot = torch.from_numpy(OUT).to(dev)
    N = Xr.shape[0]

    critic = EnergyCritic(CFG, n_layers=3).to(dev)
    optG = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optD = torch.optim.AdamW(critic.parameters(), lr=2e-4, weight_decay=1e-4)
    ema = EMA(model, CFG.ema_decay)
    print(f"[adv] gen={count_params(model)/1e6:.2f}M crit={count_params(critic)/1e6:.2f}M "
          f"lam={args.lam} rollout={args.rollout}")

    def noise(x1, t):
        x0 = torch.randn_like(x1)
        return (1 - t)[:, None, None] * x0 + t[:, None, None] * x1

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch,), device=dev)
        o = Ot[idx]
        # ---- critic step: real vs current-model rollout (detached) ----
        with torch.no_grad():
            xf = rollout(model, o, CFG, args.rollout)
        t = torch.rand(args.batch, device=dev)
        lr_real = critic(noise(Xr[idx], t), t, o)
        lr_fake = critic(noise(xf, t), t, o)
        lossD = (F.binary_cross_entropy_with_logits(lr_real, torch.ones_like(lr_real))
                 + F.binary_cross_entropy_with_logits(lr_fake, torch.zeros_like(lr_fake)))
        optD.zero_grad(set_to_none=True); lossD.backward(); optD.step()
        # ---- generator step: FM grounding + adversarial (fool critic) ----
        lossFM = fm_loss(model, Xr[idx], o, CFG)
        xg = rollout(model, o, CFG, args.rollout)
        tg = torch.rand(args.batch, device=dev)
        adv = F.binary_cross_entropy_with_logits(
            critic(noise(xg, tg), tg, o), torch.ones(args.batch, device=dev))
        lossG = lossFM + args.lam * adv
        optG.zero_grad(set_to_none=True); lossG.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        optG.step(); ema.update(model)
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            accD = ((lr_real > 0).float().mean() + (lr_fake < 0).float().mean()).item() / 2
            print(f"  step {step:5d}/{args.steps}  FM {lossFM.item():.4f}  adv {adv.item():.4f}  "
                  f"D {lossD.item():.4f} accD {accD:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "stats": stats, "scale_stats": None, "cfg": vars(CFG)},
               f"{args.out_dir}/ckpt.pt")
    shutil.copy(args.held, f"{args.out_dir}/held.pkl")
    print(f"[done] saved {args.out_dir}/ckpt.pt")


if __name__ == "__main__":
    main()
