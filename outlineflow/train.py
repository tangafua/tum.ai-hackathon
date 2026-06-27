"""Train OutlineFlow on synthetic (or MSD) plans.

    python train.py                      # full synthetic run
    python train.py --steps 1500 --overfit 200   # fast convention/overfit gate

Saves outputs/ckpt.pt {model, ema, stats, cfg} and outputs/held.pkl (held-out
samples used as the eval "real" set).
"""
from __future__ import annotations
import argparse
import math
import os
import pickle
import time
import numpy as np
import torch

import params
import synth_data
from cfg import CFG
from model import OutlineFlow, count_params
from flow import fm_loss, EMA


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    prog = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * min(prog, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=CFG.steps)
    ap.add_argument("--n_train", type=int, default=CFG.n_train)
    ap.add_argument("--n_held", type=int, default=CFG.n_held)
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--overfit", type=int, default=0,
                    help="if >0, train on this many samples (gate test)")
    ap.add_argument("--data_dir", default="",
                    help="local MSD/cvaad-challenge dir (graph_out/); empty = synthetic")
    ap.add_argument("--msd_limit", type=int, default=2000,
                    help="max real plans to read (keep small; do NOT use full dataset)")
    ap.add_argument("--device", type=str, default=CFG.device)
    ap.add_argument("--seed", type=int, default=CFG.seed)
    args = ap.parse_args()

    CFG.steps = args.steps
    CFG.batch_size = args.batch_size
    CFG.device = args.device
    CFG.seed = args.seed
    if args.overfit:
        args.n_train = args.overfit
        CFG.batch_size = min(CFG.batch_size, args.overfit)

    torch.manual_seed(CFG.seed)
    rng = np.random.default_rng(CFG.seed)
    dev = CFG.device
    os.makedirs(CFG.out_dir, exist_ok=True)

    if args.data_dir:
        import msd_data
        print(f"[data] loading MSD from {args.data_dir} (limit {args.msd_limit})...")
        all_s, _vocab = msd_data.load_msd_samples(args.data_dir, CFG, limit=args.msd_limit)
        rng.shuffle(all_s)
        n_held = min(args.n_held, len(all_s) // 5)
        held_samples, train_samples = all_s[:n_held], all_s[n_held:]
    else:
        print(f"[data] generating {args.n_train} train + {args.n_held} held samples...")
        train_samples = synth_data.generate_samples(args.n_train, CFG, rng)
        held_samples = synth_data.generate_samples(args.n_held, CFG, rng)
    stats = params.compute_stats(train_samples, CFG)
    X, OUT = synth_data.build_tensors(train_samples, stats, CFG, rng)
    print(f"[data] X={X.shape} OUT={OUT.shape}  "
          f"channel stds (should be ~1 for present rooms): "
          f"{X[X[...,0]>0][:,1:7].std(0).round(2)}")

    Xt = torch.from_numpy(X).to(dev)
    Ot = torch.from_numpy(OUT).to(dev)

    model = OutlineFlow(CFG).to(dev)
    print(f"[model] OutlineFlow d_model={CFG.d_model} L={CFG.n_layers} "
          f"params={count_params(model)/1e6:.2f}M device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    ema = EMA(model, CFG.ema_decay)

    N = Xt.shape[0]
    t0 = time.time()
    losses = []
    for step in range(CFG.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, CFG)
        idx = torch.randint(0, N, (CFG.batch_size,), device=dev)
        loss = fm_loss(model, Xt[idx], Ot[idx], CFG)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        opt.step()
        ema.update(model)
        losses.append(loss.item())
        if step % max(1, CFG.steps // 20) == 0 or step == CFG.steps - 1:
            print(f"  step {step:5d}/{CFG.steps}  loss {np.mean(losses[-50:]):.4f}  "
                  f"lr {lr_at(step,CFG):.2e}  ({time.time()-t0:.0f}s)")

    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "stats": stats, "cfg": vars(CFG)}, f"{CFG.out_dir}/ckpt.pt")
    with open(f"{CFG.out_dir}/held.pkl", "wb") as fh:
        pickle.dump(held_samples, fh)
    print(f"[done] final loss {np.mean(losses[-50:]):.4f}  -> {CFG.out_dir}/ckpt.pt")


if __name__ == "__main__":
    main()
