"""Sample plans for held-out outlines and score FID / Density / Coverage.

    python sample_eval.py                 # phi-proxy metrics (no extra deps)
    python sample_eval.py --inception     # real InceptionV3 features (needs pytorch-fid)
    python sample_eval.py --save 12       # also dump 12 real|generated PNG pairs
"""
from __future__ import annotations
import argparse
import os
import pickle
import numpy as np
import torch
from PIL import Image

import params
import metrics
from cfg import CFG, seed_everything
from model import OutlineFlow
from flow import EMA, sample
from postprocess import layout_from_tokens, voronoi_layout, coverage_overlap, align_layout
from render import render_plan


def load(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("n_max", "k", "n_gen_classes", "p_outline", "d_model", "n_layers",
              "n_heads", "mlp_ratio", "canvas", "nearest_k", "min_area_frac", "use_scale"):
        if k in ck["cfg"]:
            setattr(CFG, k, ck["cfg"][k])
    model = OutlineFlow(CFG)
    model.load_state_dict(ck["model"], strict=False)   # strict=False: load older ckpts
    ema = EMA(model, CFG.ema_decay)
    ema.load_state_dict(ck["ema"])
    return ema.make_model(model), ck["stats"], ck.get("scale_stats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=CFG.out_dir,
                    help="dir holding ckpt.pt / held.pkl and where samples/ are written")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--held", default=None)
    ap.add_argument("--n_eval", type=int, default=600)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--inception", action="store_true")
    ap.add_argument("--save", type=int, default=8)
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                    help="disable matching the generated room-count to the real mean")
    ap.add_argument("--decoder", choices=["voronoi", "rect"], default=CFG.decoder,
                    help="rect: axis-aligned rectangles (matches real MSD, default); "
                         "voronoi: gap-free seed-partition tiling (non-rectangular)")
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--align", action="store_true",
                    help="grid-snap / axis-align generated rooms before scoring (post-process C)")
    ap.add_argument("--grid", type=int, default=48,
                    help="grid divisions for --align (lower = coarser/more regular)")
    ap.add_argument("--metric_cpu", action="store_true",
                    help="run FID/feature metrics on CPU (avoids GPU OOM when VRAM is contended)")
    ap.add_argument("--target_count", type=float, default=0.0,
                    help="override calibration target room-count (0 = use real mean)")
    ap.add_argument("--energy_ckpt", default="",
                    help="EnergyCritic checkpoint for energy-guided sampling")
    ap.add_argument("--guidance", type=float, default=0.0,
                    help="energy guidance strength (0 = off)")
    ap.add_argument("--guide_from", type=float, default=0.3,
                    help="apply guidance only for t >= this (sampling time fraction)")
    ap.set_defaults(calibrate=True)
    args = ap.parse_args()
    decode = voronoi_layout if args.decoder == "voronoi" else layout_from_tokens
    seed_everything(args.seed)               # brief: fixed seed 42 for sampling/eval
    CFG.out_dir = args.out_dir
    CFG.device = args.device
    dev = args.device
    # FID/Inception use float64 covariance accumulators -> MPS can't; run metrics on CPU
    metric_dev = "cpu" if (dev == "mps" or args.metric_cpu) else dev
    ckpt_path = args.ckpt or f"{CFG.out_dir}/ckpt.pt"
    held_path = args.held or f"{CFG.out_dir}/held.pkl"

    model, stats, scale_stats = load(ckpt_path)
    model = model.to(dev)

    # optional energy critic for energy-guided sampling
    energy = None
    if args.energy_ckpt and args.guidance != 0.0:
        from energy import EnergyCritic
        eck = torch.load(args.energy_ckpt, map_location="cpu", weights_only=False)
        energy = EnergyCritic(CFG, n_layers=eck.get("n_critic_layers", 3))
        energy.load_state_dict(eck["model"])
        energy = energy.to(dev).eval()
        print(f"[energy] guided sampling ON (guidance={args.guidance}, "
              f"guide_from={args.guide_from})")
    held = pickle.load(open(held_path, "rb"))[: args.n_eval]
    outlines = [s["outline"] for s in held]
    print(f"[eval] {len(held)} held outlines | device={dev} | decoder={args.decoder} | "
          f"features={'inception' if args.inception else 'phi-proxy'}")

    # condition tensors
    OUT = np.stack([params.sample_outline_points(o, CFG.p_outline) for o in outlines])
    Ot = torch.from_numpy(OUT).to(dev)
    # absolute-scale condition (only when the model was trained with it)
    St = None
    if getattr(CFG, "use_scale", False) and scale_stats is not None:
        S = np.stack([params.outline_scale(o) for o in outlines])
        S = (S - scale_stats[0]) / scale_stats[1]
        St = torch.from_numpy(S.astype(np.float32)).to(dev)
        print(f"[scale] conditioning ON (use_scale + scale_stats loaded)")

    # 1) sample all raw token tensors
    raw = []
    for i in range(0, len(outlines), args.batch):
        sb = St[i:i + args.batch] if St is not None else None
        xb = sample(model, Ot[i:i + args.batch], CFG, scale=sb,
                    energy=energy, guidance=args.guidance,
                    guide_from=args.guide_from).cpu().numpy()
        raw.extend(list(xb))

    real_plans = [(s["room_polys"], s["outline"]) for s in held]

    # 2) calibrate one global presence threshold so the generated room-count
    #    distribution matches the real mean.  We calibrate on the FINAL decoded
    #    count (not raw present-slots), so it accounts for rooms the rect decoder
    #    drops in overlap-resolve / sliver-drop.  Decoder-agnostic; per-outline
    #    variation is preserved (bigger outlines fire more slots).
    thresh = 0.0
    if args.calibrate:
        target = (args.target_count if args.target_count > 0
                  else float(np.mean([len(r) for r, _ in real_plans])))
        sub = list(zip(raw, outlines))[:min(40, len(raw))]

        def mean_count(t):
            return float(np.mean([len(decode(x, o, stats, CFG, presence_thresh=t))
                                  for x, o in sub]))

        lo = float(min(x[:, 0].min() for x in raw))
        hi = float(max(x[:, 0].max() for x in raw))
        for _ in range(18):                       # binary search (count decreases with t)
            mid = 0.5 * (lo + hi)
            if mean_count(mid) > target:
                lo = mid                          # too many rooms -> raise threshold
            else:
                hi = mid
        thresh = 0.5 * (lo + hi)
        print(f"[calib] target count {target:.1f} -> presence_thresh {thresh:.3f} "
              f"(decoded ~{mean_count(thresh):.1f} on subsample)")

    # 3) decode + postprocess to valid layouts
    gen_plans = [(decode(x, o, stats, CFG, presence_thresh=thresh), o)
                 for x, o in zip(raw, outlines)]

    # 3b) optional alignment post-process (C): grid-snap rooms to the building axes
    if args.align:
        gen_plans = [(align_layout(r, o, CFG, grid=args.grid), o) for r, o in gen_plans]
        print(f"[align] grid-snapped layouts (grid={args.grid})")

    # diagnostics
    real_counts = np.array([len(r) for r, _ in real_plans])
    gen_counts = np.array([len(r) for r, _ in gen_plans])
    covs = np.array([coverage_overlap(r, o) for r, o in gen_plans])
    print(f"[diag] rooms/plan  real {real_counts.mean():.2f}±{real_counts.std():.2f}  "
          f"gen {gen_counts.mean():.2f}±{gen_counts.std():.2f}")
    print(f"[diag] generated interior coverage {covs[:,0].mean():.3f}  "
          f"overlap {covs[:,1].mean():.4f}  (target coverage~1, overlap~0)")

    # features + scoring
    if args.inception:
        from render import render_msd
        real_imgs = np.stack([render_msd(r, o, CFG) for r, o in real_plans])
        gen_imgs = np.stack([render_msd(r, o, CFG) for r, o in gen_plans])
        rr = metrics.official_eval(real_imgs, real_imgs.copy(), CFG, device=metric_dev)
        print(f"  [sanity] real-vs-real (official): FID={rr['fid']:.3f} "
              f"D={rr['density']:.3f} C={rr['coverage']:.3f}  (expect ~0, ~1, ~1)")
        sc = metrics.official_eval(real_imgs, gen_imgs, CFG, device=metric_dev)
    else:
        real_f = metrics.phi_features(real_plans, CFG)
        gen_f = metrics.phi_features(gen_plans, CFG)
        print("[eval] sanity calibration on the REAL feature set:")
        metrics.sanity_check(real_f, CFG, proxy=True)
        sc = metrics.score(real_f, gen_f, CFG, proxy=True)
    print(f"\n===== SCORES ({'OFFICIAL torchmetrics+prdc' if args.inception else 'phi-proxy'}) =====")
    print(f"  FID      {sc['fid']:.3f}   (lower better)")
    print(f"  Density  {sc['density']:.3f}   (higher better)")
    print(f"  Coverage {sc['coverage']:.3f}   (higher better)")

    if args.save:
        d = f"{CFG.out_dir}/samples"
        os.makedirs(d, exist_ok=True)
        for i in range(min(args.save, len(gen_plans))):
            ri = render_plan(real_plans[i][0], real_plans[i][1], CFG)
            gi = render_plan(gen_plans[i][0], gen_plans[i][1], CFG)
            pair = np.concatenate([ri, np.full((CFG.canvas, 4, 3), 0, np.uint8), gi], 1)
            Image.fromarray(pair).save(f"{d}/pair_{i:03d}.png")
        print(f"[eval] wrote {min(args.save,len(gen_plans))} real|gen pairs -> {d}")


if __name__ == "__main__":
    main()
