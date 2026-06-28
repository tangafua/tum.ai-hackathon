"""Train a DDPM (epsilon-prediction) on MSD plans — the diffusion counterpart to
train.py's flow matching.  Same backbone/data/decoding; only the objective differs.

    python train_diffusion.py --data_csv ../mds_V2_5.372k.csv --steps 15000
"""
from __future__ import annotations
import argparse, math, os, pickle, time
import numpy as np
import torch

import params, synth_data, msd_data
from cfg import CFG, seed_everything
from model import OutlineFlow, count_params
from flow import EMA
from diffusion import ddpm_loss


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    prog = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * min(prog, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="/root/jiahua_code/tum.ai-hackathon/mds_V2_5.372k.csv")
    ap.add_argument("--group", default="plan_id")
    ap.add_argument("--msd_limit", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--n_held", type=int, default=1000)
    ap.add_argument("--out_dir", default="outputs_diffusion")
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    CFG.steps = args.steps; CFG.batch_size = args.batch_size
    CFG.device = args.device; CFG.out_dir = args.out_dir; CFG.msd_group = args.group
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed); dev = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    all_s, _ = msd_data.load_msd_samples(args.data_csv, CFG, limit=args.msd_limit, group=args.group)
    rng.shuffle(all_s)
    n_held = min(args.n_held, len(all_s) // 5)
    held_samples, train_samples = all_s[:n_held], all_s[n_held:]
    stats = params.compute_stats(train_samples, CFG)
    X, OUT = synth_data.build_tensors(train_samples, stats, CFG, rng)
    print(f"[data] X={X.shape} OUT={OUT.shape}")
    Xt = torch.from_numpy(X).to(dev); Ot = torch.from_numpy(OUT).to(dev)

    CFG.diffusion = True                      # mark checkpoint as DDPM
    model = OutlineFlow(CFG).to(dev)
    print(f"[model] DDPM eps-net d_model={CFG.d_model} L={CFG.n_layers} "
          f"params={count_params(model)/1e6:.2f}M device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    ema = EMA(model, CFG.ema_decay)
    N = Xt.shape[0]; t0 = time.time(); losses = []
    for step in range(CFG.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, CFG)
        idx = torch.randint(0, N, (CFG.batch_size,), device=dev)
        loss = ddpm_loss(model, Xt[idx], Ot[idx], CFG)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        opt.step(); ema.update(model); losses.append(loss.item())
        if step % max(1, CFG.steps // 20) == 0 or step == CFG.steps - 1:
            print(f"  step {step:5d}/{CFG.steps}  loss {np.mean(losses[-50:]):.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if (step + 1) % max(1, CFG.steps // 6) == 0 and step != CFG.steps - 1:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "stats": stats,
                        "scale_stats": None, "cfg": vars(CFG)}, f"{args.out_dir}/ckpt_last.pt")

    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "stats": stats,
                "scale_stats": None, "cfg": vars(CFG)}, f"{args.out_dir}/ckpt.pt")
    with open(f"{args.out_dir}/held.pkl", "wb") as fh:
        pickle.dump(held_samples, fh)
    print(f"[done] final loss {np.mean(losses[-50:]):.4f} -> {args.out_dir}/ckpt.pt")


if __name__ == "__main__":
    main()
