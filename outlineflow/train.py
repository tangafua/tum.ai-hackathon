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
    ap.add_argument("--data_csv", default="",
                    help="path to mds_V2_5.372k.csv (real MSD); empty = synthetic")
    ap.add_argument("--group", choices=["plan_id", "unit_id"], default=CFG.msd_group,
                    help="real-MSD grouping: plan_id (brief, per floor) or unit_id (per apartment)")
    ap.add_argument("--residential_only", action="store_true",
                    help="keep only RESIDENTIAL units (brief keeps all 'area' rooms)")
    ap.add_argument("--msd_limit", type=int, default=2000,
                    help="max real plans to read (CSV is large; reads only this many)")
    ap.add_argument("--device", type=str, default=CFG.device)
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--d_model", type=int, default=CFG.d_model,
                    help="transformer width (default 128)")
    ap.add_argument("--n_layers", type=int, default=CFG.n_layers,
                    help="transformer depth (default 4)")
    ap.add_argument("--w_type", type=float, default=CFG.w_type,
                    help="loss weight on room-type channels (default 0.5)")
    ap.add_argument("--w_geometry", type=float, default=CFG.w_geometry,
                    help="loss weight on geometry channels (default 1.0)")
    ap.add_argument("--w_presence", type=float, default=CFG.w_presence,
                    help="loss weight on the presence channel (default 2.0)")
    ap.add_argument("--use_scale", action="store_true",
                    help="inject absolute outline size so room-count tracks outline area")
    ap.add_argument("--ewfm", action="store_true",
                    help="energy-weighted FM: importance-sample rare (tail) room-counts")
    ap.add_argument("--ewfm_beta", type=float, default=0.5,
                    help="EWFM strength: weight = (1/p_count)^beta (0=uniform, 1=full)")
    ap.add_argument("--out_dir", default=CFG.out_dir,
                    help="where to write ckpt.pt / held.pkl")
    args = ap.parse_args()

    CFG.steps = args.steps
    CFG.batch_size = args.batch_size
    CFG.d_model = args.d_model
    CFG.n_layers = args.n_layers
    CFG.w_type = args.w_type
    CFG.w_geometry = args.w_geometry
    CFG.w_presence = args.w_presence
    CFG.use_scale = args.use_scale
    CFG.device = args.device
    CFG.seed = args.seed
    CFG.out_dir = args.out_dir
    CFG.msd_group = args.group
    CFG.msd_residential_only = args.residential_only
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

    # absolute outline-size conditioning (standardized [log area, log w, log h])
    scale_stats = None
    St = None
    if CFG.use_scale:
        scale_stats = params.compute_scale_stats(train_samples)            # [2,3]
        S = np.stack([params.outline_scale(s["outline"]) for s in train_samples])
        S = (S - scale_stats[0]) / scale_stats[1]
        St = torch.from_numpy(S.astype(np.float32)).to(dev)
        print(f"[scale] use_scale=ON | scale_stats mean={scale_stats[0].round(2)} "
              f"std={scale_stats[1].round(2)}")

    # energy-weighted FM: importance-sample by inverse room-count frequency so the
    # model sees the rare large/small plans it otherwise regresses away (-> Coverage)
    samp_w = None
    if args.ewfm:
        counts = (X[..., 0] > 0).sum(axis=1).astype(int)                  # rooms/sample
        cmin = counts.min()
        freq = np.bincount(counts - cmin).astype(np.float64)
        p = freq[counts - cmin] / freq.sum()
        w = (1.0 / np.maximum(p, 1e-9)) ** args.ewfm_beta
        w = np.clip(w / w.mean(), 0.1, 20.0)                              # tame extremes
        samp_w = torch.tensor(w, dtype=torch.float64, device=dev)
        print(f"[ewfm] ON beta={args.ewfm_beta} | weight range "
              f"[{w.min():.2f},{w.max():.2f}] | count range [{counts.min()},{counts.max()}]")

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
        if samp_w is not None:
            idx = torch.multinomial(samp_w, CFG.batch_size, replacement=True)
        else:
            idx = torch.randint(0, N, (CFG.batch_size,), device=dev)
        loss = fm_loss(model, Xt[idx], Ot[idx], CFG, scale=(St[idx] if St is not None else None))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        opt.step()
        ema.update(model)
        losses.append(loss.item())
        if step % max(1, CFG.steps // 20) == 0 or step == CFG.steps - 1:
            print(f"  step {step:5d}/{CFG.steps}  loss {np.mean(losses[-50:]):.4f}  "
                  f"lr {lr_at(step,CFG):.2e}  ({time.time()-t0:.0f}s)", flush=True)
        # periodic checkpointing: a numbered snapshot every ~1/6 of training plus a
        # rolling ckpt_last.pt, so a long run survives an interruption.
        if (step + 1) % max(1, CFG.steps // 6) == 0 and step != CFG.steps - 1:
            snap = {"model": model.state_dict(), "ema": ema.state_dict(),
                    "stats": stats, "scale_stats": scale_stats,
                    "cfg": vars(CFG), "step": step + 1}
            torch.save(snap, f"{CFG.out_dir}/ckpt_step{step+1}.pt")
            torch.save(snap, f"{CFG.out_dir}/ckpt_last.pt")
            print(f"  [checkpoint] saved {CFG.out_dir}/ckpt_step{step+1}.pt", flush=True)

    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "stats": stats, "scale_stats": scale_stats,
                "cfg": vars(CFG)}, f"{CFG.out_dir}/ckpt.pt")
    with open(f"{CFG.out_dir}/held.pkl", "wb") as fh:
        pickle.dump(held_samples, fh)
    print(f"[done] final loss {np.mean(losses[-50:]):.4f}  -> {CFG.out_dir}/ckpt.pt")


if __name__ == "__main__":
    main()
