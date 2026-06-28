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
from flow import EMA, sample, edm_sample
from postprocess import layout_from_tokens, voronoi_layout, coverage_overlap
from render import render_plan


def load(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("n_max", "k", "n_gen_classes", "p_outline", "d_model", "n_layers",
              "n_heads", "mlp_ratio", "canvas", "nearest_k", "min_area_frac",
              # architecture-defining keys: must match the trained variant or
              # state_dict load fails (cross-attn layers / cond_embed / fourier W shape)
              "use_cross_attn", "n_cond", "outline_fourier_freqs",
              # objective/sampler keys (EDM uses a different sampler + sigma schedule)
              "objective", "sigma_data", "edm_sigma_min", "edm_sigma_max", "edm_rho"):
        if k in ck["cfg"]:
            setattr(CFG, k, ck["cfg"][k])
    model = OutlineFlow(CFG)
    model.load_state_dict(ck["model"])
    ema = EMA(model, CFG.ema_decay)
    ema.load_state_dict(ck["ema"])
    return ema.make_model(model), ck["stats"]


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
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="classifier-free guidance scale (1.0 = off; ~1.5 sharpens "
                         "outline adherence; needs a model trained with --cfg_drop)")
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--sample_steps", type=int, default=None,
                    help="M2: ODE integration steps (default cfg 100); more = lower "
                         "discretization error")
    ap.add_argument("--full_heun", action="store_true",
                    help="M2: 2nd-order Heun corrector on EVERY step (not just last 5)")
    ap.add_argument("--count_match", action="store_true",
                    help="C1: per-outline rank-based top-K decode. K_i scales the real "
                         "mean count by each outline's area (K_i=round(mean*area_i/mean_area)), "
                         "replacing the single global presence threshold -> fixes the "
                         "under-separation clamp + recovers per-plan count variance")
    ap.set_defaults(calibrate=True)
    args = ap.parse_args()
    decode = voronoi_layout if args.decoder == "voronoi" else layout_from_tokens
    seed_everything(args.seed)               # brief: fixed seed 42 for sampling/eval
    CFG.out_dir = args.out_dir
    CFG.device = args.device
    dev = args.device
    # FID/Inception use float64 covariance accumulators -> MPS can't; run metrics on CPU
    metric_dev = "cpu" if dev == "mps" else dev
    ckpt_path = args.ckpt or f"{CFG.out_dir}/ckpt.pt"
    held_path = args.held or f"{CFG.out_dir}/held.pkl"

    model, stats = load(ckpt_path)
    model = model.to(dev)
    held = pickle.load(open(held_path, "rb"))[: args.n_eval]
    outlines = [s["outline"] for s in held]
    print(f"[eval] {len(held)} held outlines | device={dev} | decoder={args.decoder} | "
          f"features={'inception' if args.inception else 'phi-proxy'}")

    # condition tensors
    OUT = np.stack([params.sample_outline_points(o, CFG.p_outline) for o in outlines])
    Ot = torch.from_numpy(OUT).to(dev)
    n_cond = getattr(CFG, "n_cond", 0)
    Ct = None
    if n_cond:
        COND = np.stack([params.outline_cond(o, CFG) for o in outlines])
        Ct = torch.from_numpy(COND).to(dev)

    # 1) sample all raw token tensors
    steps = args.sample_steps or CFG.sample_steps
    heun_last = steps if args.full_heun else 5
    is_edm = getattr(CFG, "objective", "rectflow") == "edm"
    print(f"[sample] objective={getattr(CFG,'objective','rectflow')} steps={steps} "
          f"heun_last={heun_last} guidance={args.guidance}")
    raw = []
    for i in range(0, len(outlines), args.batch):
        cb = Ct[i:i + args.batch] if Ct is not None else None
        ob = Ot[i:i + args.batch]
        if is_edm:
            xb = edm_sample(model, ob, CFG, steps=steps, cond=cb, guidance=args.guidance)
        else:
            xb = sample(model, ob, CFG, cond=cb, steps=steps,
                        heun_last=heun_last, guidance=args.guidance)
        raw.extend(list(xb.cpu().numpy()))

    real_plans = [(s["room_polys"], s["outline"]) for s in held]

    # 2) calibrate one global presence threshold so the generated room-count
    #    distribution matches the real mean.  We calibrate on the FINAL decoded
    #    count (not raw present-slots), so it accounts for rooms the rect decoder
    #    drops in overlap-resolve / sliver-drop.  Decoder-agnostic; per-outline
    #    variation is preserved (bigger outlines fire more slots).
    # C1: per-outline top-K from outline area (leak-free: uses only the real MEAN
    # count -- already used by --calibrate -- scaled by each outline's own area).
    topk_list = None
    if args.count_match:
        target = float(np.mean([len(r) for r, _ in real_plans]))
        areas = np.array([float(o.area) for o in outlines])
        ks = np.round(target * areas / areas.mean()).astype(int)
        ks = np.clip(ks, 1, CFG.n_max)
        topk_list = ks
        print(f"[count_match] target mean {target:.1f} -> per-outline K "
              f"mean {ks.mean():.2f}±{ks.std():.2f} (range {ks.min()}-{ks.max()})")

    thresh = 0.0
    if args.calibrate and not args.count_match:
        target = float(np.mean([len(r) for r, _ in real_plans]))
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
        # hard floor: never drop below the padding baseline, else noise slots get
        # admitted as rooms and gap-fill masks the collapse (the -1.031 pathology).
        floor = CFG.presence_thresh_floor
        if thresh < floor:
            print(f"[calib] calibrated thresh {thresh:.3f} below floor {floor:.2f} "
                  f"-> clamped (model under-separates presence)")
            thresh = floor
        print(f"[calib] target count {target:.1f} -> presence_thresh {thresh:.3f} "
              f"(decoded ~{mean_count(thresh):.1f} on subsample)")

    # 3) decode + postprocess to valid layouts
    gen_plans = [(decode(x, o, stats, CFG, presence_thresh=thresh), o)
                 for x, o in zip(raw, outlines)]

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
