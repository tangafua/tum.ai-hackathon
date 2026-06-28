"""Area-level (plan_id / floor) re-evaluation of the UNIT-trained model.

The kaiyingwu results were all produced per-UNIT (one apartment per sample).  This
script answers: if we keep that same unit-trained checkpoint but **merge the units of
each floor (`plan_id`) back together** and score whole floors, what are FID / Density /
Coverage?

Pipeline (NO retraining):
  1. load the unit ckpt + its held.pkl (the exact held units behind the headline numbers)
  2. recover each held unit's plan_id by matching its geometry against the CSV
     (load_msd_samples(..., group="unit_id", tag_parent="plan_id"))
  3. held floors = the plan_ids those held units belong to; for each floor gather ALL
     its units from the CSV (this is what "merge the units of the area" means)
  4. generate each unit's rooms with the unit model (identical decode path as the CLI,
     via sample_eval.generate_layouts)
  5. merge a floor's units into one layout: rooms = concat of its units' rooms; the floor
     outline = build_outline over all the floor's real room polys (the brief union)
  6. render + score floors -> area-level FID / Density / Coverage; also score the same
     units un-merged for a side-by-side unit-level baseline
  7. write Analysis/area_eval_results.md

CAVEAT: gathering ALL units of a floor pulls in units the model trained on (held-out
purity is broken).  The contamination fraction is printed and recorded.

    python area_eval.py --out_dir outputs_unit --data_csv /path/mds_V2_5.372k.csv \
        --inception --guidance 1.5 --count_match --count_scale 1.0 --device cuda
"""
from __future__ import annotations
import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

import params
import metrics
import msd_data
from cfg import CFG, seed_everything
from render import render_plan, render_msd
from sample_eval import load, generate_layouts


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=CFG.out_dir,
                    help="dir holding the UNIT ckpt.pt / held.pkl")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--held", default=None)
    ap.add_argument("--data_csv", required=True, help="path to mds_V2_5.372k.csv")
    ap.add_argument("--n_eval", type=int, default=500,
                    help="number of FLOORS (plan_id areas) to evaluate")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--inception", action="store_true",
                    help="official InceptionV3 FID + prdc (else phi-proxy)")
    ap.add_argument("--save", type=int, default=8, help="save N real|gen FLOOR png pairs")
    ap.add_argument("--no_md", action="store_true",
                    help="skip writing Analysis/area_eval_results.md (for batch runs)")
    # --- generation flags mirrored from sample_eval (must match the headline unit run) ---
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    ap.add_argument("--decoder", choices=["voronoi", "rect"], default=CFG.decoder)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=CFG.seed)
    ap.add_argument("--sample_steps", type=int, default=None)
    ap.add_argument("--full_heun", action="store_true")
    ap.add_argument("--count_match", action="store_true")
    ap.add_argument("--count_scale", type=float, default=1.0)
    ap.add_argument("--count_stochastic", action="store_true")
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--churn", type=float, default=0.0)
    ap.add_argument("--min_area_frac", type=float, default=None)
    ap.set_defaults(calibrate=True)
    return ap


def unit_signature(sample):
    """A geometry fingerprint that is identical for the same MSD unit re-parsed from the
    SAME CSV (held.pkl outlines were built by the SAME build_outline).  High-precision so
    distinct apartments don't collide."""
    polys = sample["room_polys"]
    n = len(polys)
    area = round(float(sample["outline"].area), 6)
    sum_a = round(float(sum(p.area for p, _ in polys)), 6)
    per = round(float(sample["outline"].length), 6)
    return (n, area, sum_a, per)


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)
    CFG.out_dir = args.out_dir
    CFG.device = args.device
    dev = args.device
    metric_dev = "cpu" if dev == "mps" else dev
    ckpt_path = args.ckpt or f"{CFG.out_dir}/ckpt.pt"
    held_path = args.held or f"{CFG.out_dir}/held.pkl"

    # 1) model + the unit held set behind the headline numbers
    model, stats = load(ckpt_path)          # also populates CFG (n_max, k, ...) from ckpt
    if args.min_area_frac is not None:
        CFG.min_area_frac = args.min_area_frac
        print(f"[cfg] min_area_frac -> {CFG.min_area_frac}")
    model = model.to(dev)
    held = pickle.load(open(held_path, "rb"))
    print(f"[area] {len(held)} held units loaded from {held_path}")

    # 2) load ALL units with their plan_id, signature-match held units -> plan_id
    units, _ = msd_data.load_msd_samples(
        args.data_csv, CFG, limit=None, group="unit_id", set_cfg=False,
        residential_only=False, tag_parent="plan_id")
    sig2unit = defaultdict(list)
    for u in units:
        sig2unit[unit_signature(u)].append(u)
    units_by_plan = defaultdict(list)
    for u in units:
        units_by_plan[u["plan_id"]].append(u)

    matched, ambiguous, missing = 0, 0, 0
    held_plans_order = []
    held_sigs = set()
    for s in held:
        sig = unit_signature(s)
        held_sigs.add(sig)
        cand = sig2unit.get(sig)
        if not cand:
            missing += 1
            continue
        pids = {u["plan_id"] for u in cand}
        if len(pids) > 1:
            ambiguous += 1
        pid = cand[0]["plan_id"]
        matched += 1
        if pid not in held_plans_order:
            held_plans_order.append(pid)
    rate = matched / max(len(held), 1)
    print(f"[area] held->plan match: {matched}/{len(held)} ({rate:.1%}) | "
          f"missing {missing} | ambiguous-sig {ambiguous} | "
          f"{len(held_plans_order)} distinct held floors")
    if rate < 0.9:
        print("[area] WARNING: low match rate -- is --data_csv the SAME file used for "
              "training? plan derivation may be incomplete.")

    # 3) select floors and gather ALL their units
    floor_plans = held_plans_order[: args.n_eval]
    eval_units, unit_plan = [], []
    contaminated = 0      # units of the selected floors that were NOT in the held set
    for pid in floor_plans:
        for u in units_by_plan[pid]:
            eval_units.append(u)
            unit_plan.append(pid)
            if unit_signature(u) not in held_sigs:
                contaminated += 1
    n_units = len(eval_units)
    upf = n_units / max(len(floor_plans), 1)
    contam = contaminated / max(n_units, 1)
    print(f"[area] {len(floor_plans)} floors | {n_units} units | {upf:.2f} units/floor | "
          f"contamination {contam:.1%} (units seen in unit-training)")

    # 4) generate each unit's rooms with the unit model (same decode path as the CLI)
    unit_outlines = [u["outline"] for u in eval_units]
    real_unit_plans = [(u["room_polys"], u["outline"]) for u in eval_units]
    gen_rooms = generate_layouts(model, stats, unit_outlines, real_unit_plans, args, dev)

    # 5) merge units -> floors (rooms in shared floor coords; outline = brief union)
    idx_by_plan = defaultdict(list)
    for i, pid in enumerate(unit_plan):
        idx_by_plan[pid].append(i)
    real_areas, gen_areas = [], []
    for pid in floor_plans:
        idxs = idx_by_plan[pid]
        real_polys = [rp for i in idxs for rp in eval_units[i]["room_polys"]]
        gen_polys = [rp for i in idxs for rp in gen_rooms[i]]
        area_outline = msd_data.build_outline([p for p, _ in real_polys])
        if area_outline is None:
            continue
        real_areas.append((real_polys, area_outline))
        gen_areas.append((gen_polys, area_outline))

    real_ac = np.array([len(r) for r, _ in real_areas])
    gen_ac = np.array([len(r) for r, _ in gen_areas])
    print(f"[diag] rooms/floor  real {real_ac.mean():.1f}±{real_ac.std():.1f}  "
          f"gen {gen_ac.mean():.1f}±{gen_ac.std():.1f}  ({len(real_areas)} floors)")

    # 6) score floors (area-level) + the same units un-merged (unit-level), same backend
    def score(real_plans, gen_plans):
        if args.inception:
            ri = np.stack([render_msd(r, o, CFG) for r, o in real_plans])
            gi = np.stack([render_msd(r, o, CFG) for r, o in gen_plans])
            sane = metrics.official_eval(ri, ri.copy(), CFG, device=metric_dev)
            sc = metrics.official_eval(ri, gi, CFG, device=metric_dev)
            return sc, sane
        rf = metrics.phi_features(real_plans, CFG)
        gf = metrics.phi_features(gen_plans, CFG)
        return metrics.score(rf, gf, CFG, proxy=True), None

    area_sc, area_sane = score(real_areas, gen_areas)
    gen_unit_plans = [(gen_rooms[i], unit_outlines[i]) for i in range(n_units)]
    unit_sc, _ = score(real_unit_plans, gen_unit_plans)

    backend = "OFFICIAL torchmetrics+prdc" if args.inception else "phi-proxy"
    if area_sane is not None:
        print(f"  [sanity] area real-vs-real: FID={area_sane['fid']:.3f} "
              f"D={area_sane['density']:.3f} C={area_sane['coverage']:.3f} (expect ~0,1,1)")
    print(f"\n===== AREA-LEVEL SCORES ({backend}) =====")
    print(f"  FID      {area_sc['fid']:.3f}   (lower better)")
    print(f"  Density  {area_sc['density']:.3f}   (higher better)")
    print(f"  Coverage {area_sc['coverage']:.3f}   (higher better)")
    print(f"\n----- unit-level on the SAME units (for reference) -----")
    print(f"  FID {unit_sc['fid']:.3f}  Density {unit_sc['density']:.3f}  "
          f"Coverage {unit_sc['coverage']:.3f}")

    # 7) save a few floor pairs
    if args.save:
        d = f"{CFG.out_dir}/area_samples"
        os.makedirs(d, exist_ok=True)
        for i in range(min(args.save, len(gen_areas))):
            ri = render_plan(real_areas[i][0], real_areas[i][1], CFG)
            gi = render_plan(gen_areas[i][0], gen_areas[i][1], CFG)
            pair = np.concatenate([ri, np.full((CFG.canvas, 4, 3), 0, np.uint8), gi], 1)
            Image.fromarray(pair).save(f"{d}/pair_{i:03d}.png")
        print(f"[area] wrote {min(args.save,len(gen_areas))} real|gen floor pairs -> {d}")

    # 8) results markdown
    if not args.no_md:
        write_results_md(args, backend, floor_plans, n_units, upf, contam, rate,
                         real_ac, gen_ac, area_sc, unit_sc, area_sane)


def write_results_md(args, backend, floor_plans, n_units, upf, contam, rate,
                     real_ac, gen_ac, area_sc, unit_sc, area_sane=None):
    flags = (f"--guidance {args.guidance} --decoder {args.decoder}"
             + (" --count_match" if args.count_match else "")
             + (f" --count_scale {args.count_scale}" if args.count_match else "")
             + (" --count_stochastic" if args.count_stochastic else "")
             + (" --align" if args.align else "")
             + (f" --grid {args.grid}" if args.align else "")
             + (f" --churn {args.churn}" if args.churn else "")
             + (" --inception" if args.inception else ""))
    cmd = (f"python area_eval.py --out_dir {args.out_dir} --data_csv {args.data_csv} "
           f"--n_eval {args.n_eval} --device {args.device} {flags}")
    out_path = "Analysis/area_eval_results.md"
    os.makedirs("Analysis", exist_ok=True)
    lines = [
        "# Area-level (plan_id) re-evaluation of the unit-trained model",
        "",
        "Same unit-trained checkpoint, NO retraining. Each floor (`plan_id`) is scored by",
        "**merging all of its units** (each generated per-unit by the unit model) into one",
        "layout, then computing FID / Density / Coverage over whole floors.",
        "",
        f"- Checkpoint: `{args.out_dir}/ckpt.pt`  (unit-trained)",
        f"- Backend: **{backend}**",
        f"- Command:",
        "",
        "```",
        cmd,
        "```",
        "",
        "## Setup",
        "",
        f"- Floors (areas) evaluated: **{len(floor_plans)}**",
        f"- Units merged: **{n_units}**  ({upf:.2f} units/floor)",
        f"- held->plan_id signature match rate: {rate:.1%}",
        f"- Contamination (units the unit model trained on): **{contam:.1%}** "
        "(held-out purity caveat -- biases area numbers optimistically)",
        f"- rooms/floor: real {real_ac.mean():.1f}±{real_ac.std():.1f} | "
        f"gen {gen_ac.mean():.1f}±{gen_ac.std():.1f}",
        "",
        "## Results",
        "",
        (f"Sanity (area real-vs-real): FID {area_sane['fid']:.3f} / "
         f"D {area_sane['density']:.3f} / C {area_sane['coverage']:.3f} "
         "-- metric wiring verified." if area_sane else ""),
        "",
        "| Level | FID ↓ | Density ↑ | Coverage ↑ |",
        "|---|---|---|---|",
        f"| **Area (plan_id, merged) -- {len(floor_plans)} floors** | "
        f"**{area_sc['fid']:.3f}** | **{area_sc['density']:.3f}** | "
        f"**{area_sc['coverage']:.3f}** |",
        f"| Unit (same {n_units} units, un-merged, ref) | {unit_sc['fid']:.3f} | "
        f"{unit_sc['density']:.3f} | {unit_sc['coverage']:.3f} |",
        "",
        "Note: rows above share this run's pipeline/backend so they are directly",
        "comparable to each other; the published unit best (`方案进展_metrics.md` §7.z,",
        "109.8 / 0.118 / 0.132) is on a different, smaller, uncontaminated sample and is",
        "context only.",
        "",
        "## Caveats",
        "",
        "- **Held-out purity**: floors are completed by pulling *all* of their units from",
        "  the CSV, so some units were in the unit-training split (contamination above).",
        "  The area numbers are therefore optimistic relative to a clean held-out floor set.",
        "- **Merge geometry**: each unit's rooms are generated independently (conditioned on",
        "  that unit's own outline) and placed in the shared floor coordinate frame; the",
        "  floor outline is the brief union (`buffer(+0.3).unary_union.buffer(-0.3)`) of the",
        "  real rooms. There is no cross-unit layout interaction -- the model never sees the",
        "  whole floor, so inter-apartment structure is whatever the union produces.",
        "- **Residential units only**: public space has a null `unit_id` and is excluded, so",
        "  the merged floor is the union of the apartments (slightly tighter than the brief's",
        "  full-plan outline that also folds in public areas).",
        "",
    ]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[area] wrote results -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
