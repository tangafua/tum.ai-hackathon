"""Submission entry point: generate(outline) -> typed room polygons.

The challenge brief requires a `generate(outline)` function that, given ONLY an
apartment outline, returns the interior rooms as vector polygons.  This file is
that contract, plus a small CLI to demo it on a real MSD plan or an arbitrary
outline.

    from generate import generate
    rooms = generate(outline)            # outline: shapely (Multi)Polygon or WKT str
    # rooms == [(shapely Polygon, room_type_id), ...]   (vector, never a pixel grid)

Everything is conditioned on the outline alone (no rooms / walls / graph), the
model is the from-scratch rectified-flow set-Transformer in model.py, and the
sampling seed is fixed to 42 per the brief.
"""
from __future__ import annotations
import numpy as np
import torch
from shapely import wkt
from shapely.geometry import base as _geombase

import params
from cfg import CFG, ROOM_NAMES, seed_everything
from model import OutlineFlow
from flow import EMA, sample
from postprocess import voronoi_layout, layout_from_tokens, coverage_overlap

DEFAULT_CKPT = f"{CFG.out_dir}/ckpt.pt"
_RESTORE_KEYS = ("n_max", "k", "n_gen_classes", "p_outline", "d_model", "n_layers",
                 "n_heads", "mlp_ratio", "canvas", "nearest_k", "min_area_frac",
                 "msd_group")
_CACHE: dict = {}


def room_type_name(type_id: int) -> str:
    """Class id -> human-readable room type (cfg.ROOM_NAMES)."""
    return ROOM_NAMES[int(type_id) % len(ROOM_NAMES)]


def _to_polygon(outline):
    """Accept a shapely geometry or a WKT string -> shapely geometry."""
    if isinstance(outline, str):
        outline = wkt.loads(outline)
    if not isinstance(outline, _geombase.BaseGeometry):
        raise TypeError(f"outline must be a shapely geometry or WKT string, "
                        f"got {type(outline)!r}")
    if not outline.is_valid:
        outline = outline.buffer(0)
    return outline


def load_model(ckpt_path: str = DEFAULT_CKPT, device: str | None = None):
    """Load (EMA) model + standardization stats from a checkpoint, cached.

    Restores the saved cfg fields so n_max / d_model / K match training exactly.
    """
    device = device or CFG.device
    key = (ckpt_path, device)
    if key in _CACHE:
        return _CACHE[key]
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in _RESTORE_KEYS:
        if k in ck["cfg"]:
            setattr(CFG, k, ck["cfg"][k])
    model = OutlineFlow(CFG)
    model.load_state_dict(ck["model"])
    ema = EMA(model, CFG.ema_decay)
    ema.load_state_dict(ck["ema"])
    model = ema.make_model(model).to(device).eval()
    _CACHE[key] = (model, ck["stats"])
    return model, ck["stats"]


def generate(outline, *, ckpt: str = DEFAULT_CKPT, decoder: str = "voronoi",
             presence_thresh: float = 0.0, seed: int = 42,
             device: str | None = None):
    """Generate the interior rooms for one apartment ``outline``.

    Parameters
    ----------
    outline : shapely (Multi)Polygon or WKT string -- the ONLY condition.
    decoder : "voronoi" (gap-free seed partition, default) or "rect" (boxes+gap-fill).
    presence_thresh : slot-presence cut; 0.0 = the model's natural threshold.
                      (sample_eval can compute a room-count-calibrated value.)
    seed : fixed at 42 per the brief for reproducible sampling.

    Returns
    -------
    list of (shapely Polygon, room_type_id) -- vector room polygons (not pixels).
    union(rooms) == outline (zero overlap, zero interior gap) by construction.
    """
    seed_everything(seed)
    device = device or CFG.device
    outline = _to_polygon(outline)
    model, stats = load_model(ckpt, device)
    decode = voronoi_layout if decoder == "voronoi" else layout_from_tokens

    OUT = params.sample_outline_points(outline, CFG.p_outline)[None]   # [1,P,4]
    Ot = torch.from_numpy(OUT).to(device)
    x = sample(model, Ot, CFG)[0].cpu().numpy()                        # [n_max,D]
    rooms = decode(x, outline, stats, CFG, presence_thresh=presence_thresh)
    return [(poly, int(t)) for poly, t in rooms]


def generate_geodataframe(outline, **kw):
    """Same as generate(), returned as a GeoDataFrame[geometry, room_type, type_id]
    (mirrors the brief's geopandas room dataframe). Requires geopandas."""
    import geopandas as gpd
    rooms = generate(outline, **kw)
    return gpd.GeoDataFrame({
        "geometry": [p for p, _ in rooms],
        "type_id": [t for _, t in rooms],
        "room_type": [room_type_name(t) for _, t in rooms],
    }, geometry="geometry")


# --------------------------------------------------------------------- CLI demo
def main():
    import argparse, os
    ap = argparse.ArgumentParser(description="generate(outline) -> room polygons")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--decoder", choices=["voronoi", "rect"], default="voronoi")
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outline_wkt", default="",
                    help="generate for an arbitrary outline given as a WKT polygon")
    ap.add_argument("--data_csv", default="",
                    help="demo on a real MSD plan: build its outline via the brief formula")
    ap.add_argument("--group", choices=["plan_id", "unit_id"], default=CFG.msd_group)
    ap.add_argument("--plan", type=float, default=None,
                    help="plan_id / unit_id value to demo (with --data_csv)")
    ap.add_argument("--out", default=f"{CFG.out_dir}/generated")
    args = ap.parse_args()

    real_rooms = None
    if args.outline_wkt:
        outline = wkt.loads(args.outline_wkt)
    elif args.data_csv:
        import pandas as pd
        from msd_data import build_outline, _as_polygon, ROOMTYPE_TO_ID
        col = args.group
        df = pd.read_csv(args.data_csv, usecols=[col, "entity_type", "roomtype", "geom"])
        df = df[(df.entity_type == "area") & (df[col] == args.plan)]
        if df.empty:
            raise SystemExit(f"no 'area' rows for {col}={args.plan} in {args.data_csv}")
        polys, types = [], []
        for g, rt in zip(df.geom, df.roomtype):
            p = _as_polygon(g)
            if p is not None and str(rt) in ROOMTYPE_TO_ID:
                polys.append(p); types.append(ROOMTYPE_TO_ID[str(rt)])
        outline = build_outline(polys)
        real_rooms = list(zip(polys, types))
        print(f"[demo] {col}={args.plan}: {len(polys)} real rooms, "
              f"outline {outline.geom_type} area {outline.area:.1f}")
    else:
        raise SystemExit("pass --outline_wkt '...' or --data_csv ... --plan <id>")

    rooms = generate(outline, ckpt=args.ckpt, decoder=args.decoder,
                     seed=args.seed, device=args.device)
    cov, ov = coverage_overlap(rooms, outline)
    print(f"[generate] {len(rooms)} rooms | interior coverage {cov:.3f} | overlap {ov:.4f}")
    for poly, t in rooms:
        print(f"    {room_type_name(t):12s} area={poly.area:8.2f}")

    # render a PNG (outline | [real] | generated) for a quick visual check
    os.makedirs(args.out, exist_ok=True)
    from render import render_plan
    from PIL import Image
    gi = render_plan(rooms, outline, CFG)
    panels = [render_plan([(outline, 9)], outline, CFG)]  # outline as 'Structure' fill
    if real_rooms is not None:
        panels.append(render_plan(real_rooms, outline, CFG))
    panels.append(gi)
    sep = np.full((CFG.canvas, 4, 3), 60, np.uint8)
    strip = panels[0]
    for p in panels[1:]:
        strip = np.concatenate([strip, sep, p], axis=1)
    tag = args.plan if args.plan is not None else "wkt"
    path = f"{args.out}/gen_{args.group}_{tag}.png"
    Image.fromarray(strip).save(path)
    print(f"[generate] wrote {path}  (outline | "
          f"{'real | ' if real_rooms is not None else ''}generated)")


if __name__ == "__main__":
    main()
