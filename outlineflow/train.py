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
from cfg import CFG, seed_everything
from model import OutlineFlow, count_params
from flow import fm_loss, edm_loss, EMA


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
    ap.add_argument("--data_csv", default="",
                    help="path to mds_V2_5.372k.csv (real MSD); empty = synthetic")
    ap.add_argument("--group", choices=["plan_id", "unit_id"], default=CFG.msd_group,
                    help="real-MSD grouping: plan_id (brief, per floor) or unit_id (per apartment)")
    ap.add_argument("--residential_only", action="store_true",
                    help="keep only RESIDENTIAL units (brief keeps all 'area' rooms)")
    ap.add_argument("--msd_limit", type=int, default=0,
                    help="max real plans to read; 0 = no cap (use the full CSV)")
    ap.add_argument("--cfg_drop", type=float, default=0.0,
                    help="classifier-free guidance: prob of dropping the outline "
                         "condition during training (e.g. 0.1); 0 = off")
    # --- ablation / variant overrides (default None = use cfg.py value) ---
    ap.add_argument("--w_presence", type=float, default=None,
                    help="presence-loss weight (default cfg 2.0); raise to sharpen "
                         "present/absent separation -> better room count")
    ap.add_argument("--d_model", type=int, default=None, help="model width override")
    ap.add_argument("--n_layers", type=int, default=None, help="depth override")
    ap.add_argument("--n_cond", type=int, default=None,
                    help="0 disables the outline-scale count condition (P0-B ablation)")
    ap.add_argument("--fourier_freqs", type=int, default=None,
                    help="boundary-point Fourier bands (default cfg 16)")
    ap.add_argument("--no_cross_attn", action="store_true",
                    help="disable token->boundary cross-attention (P0-A ablation)")
    ap.add_argument("--t_dist", choices=["uniform", "logitnormal"], default=None,
                    help="timestep sampling for FM loss (M1: logitnormal emphasizes "
                         "mid-t, SD3-style)")
    ap.add_argument("--objective", choices=["rectflow", "edm"], default=None,
                    help="generative objective (M4: edm = Karras preconditioned denoiser)")
    ap.add_argument("--device", type=str, default=CFG.device)
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--out_dir", default=CFG.out_dir,
                    help="where to write ckpt.pt / held.pkl")
    args = ap.parse_args()

    CFG.steps = args.steps
    CFG.batch_size = args.batch_size
    CFG.device = args.device
    CFG.seed = args.seed
    CFG.out_dir = args.out_dir
    CFG.msd_group = args.group
    CFG.msd_residential_only = args.residential_only
    # variant overrides (only when explicitly passed, so default = cfg.py)
    if args.w_presence is not None:    CFG.w_presence = args.w_presence
    if args.d_model is not None:       CFG.d_model = args.d_model
    if args.n_layers is not None:      CFG.n_layers = args.n_layers
    if args.n_cond is not None:        CFG.n_cond = args.n_cond
    if args.fourier_freqs is not None: CFG.outline_fourier_freqs = args.fourier_freqs
    if args.no_cross_attn:             CFG.use_cross_attn = False
    if args.t_dist is not None:        CFG.t_dist = args.t_dist
    if args.objective is not None:     CFG.objective = args.objective
    if args.overfit:
        args.n_train = args.overfit
        CFG.batch_size = min(CFG.batch_size, args.overfit)

    seed_everything(CFG.seed)                 # brief: fixed seed 42 throughout
    rng = np.random.default_rng(CFG.seed)
    dev = CFG.device
    os.makedirs(CFG.out_dir, exist_ok=True)

    if args.data_csv:
        import msd_data
        print(f"[data] loading MSD from {args.data_csv} "
              f"(group={args.group}, limit {args.msd_limit})...")
        all_s, _vocab = msd_data.load_msd_samples(
            args.data_csv, CFG, limit=args.msd_limit, group=args.group,
            residential_only=args.residential_only)
        rng.shuffle(all_s)
        if args.overfit:
            # gate test: memorize a SMALL fixed subset; eval on the SAME plans, so
            # Density should spike if the (cross-attn + cond) wiring can fit data.
            train_samples = all_s[:args.overfit]
            held_samples = train_samples
            print(f"[data] OVERFIT gate: {len(train_samples)} plans (train == held)")
        else:
            n_held = min(args.n_held, len(all_s) // 5)
            held_samples, train_samples = all_s[:n_held], all_s[n_held:]
    else:
        print(f"[data] generating {args.n_train} train + {args.n_held} held samples...")
        train_samples = synth_data.generate_samples(args.n_train, CFG, rng)
        held_samples = synth_data.generate_samples(args.n_held, CFG, rng)
    stats = params.compute_stats(train_samples, CFG)
    X, OUT, COND = synth_data.build_tensors(train_samples, stats, CFG, rng)
    print(f"[data] X={X.shape} OUT={OUT.shape} COND={COND.shape}  "
          f"channel stds (should be ~1 for present rooms): "
          f"{X[X[...,0]>0][:,1:7].std(0).round(2)}")

    Xt = torch.from_numpy(X).to(dev)
    Ot = torch.from_numpy(OUT).to(dev)
    Ct = torch.from_numpy(COND).to(dev)

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
        cond = Ct[idx] if Ct.shape[1] > 0 else None
        loss_fn = edm_loss if CFG.objective == "edm" else fm_loss
        loss = loss_fn(model, Xt[idx], Ot[idx], CFG, cond=cond, cfg_drop=args.cfg_drop)
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
