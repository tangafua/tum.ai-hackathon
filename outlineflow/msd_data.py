"""MSD (Modified Swiss Dwellings) real-data hook -- reads the OFFICIAL CSV.

This is the ONLY file that touches real data.  It produces the SAME sample
contract as synth_data.py:

    sample = {"outline": Polygon, "rooms": [(cx,cy,w,h,theta,type), ...],
              "room_polys": [(Polygon, type), ...]}

so train.py / sample_eval.py / render.py / metrics.py are UNCHANGED.

Data source (per the challenge brief):
    mds_V2_5.372k.csv   one row per entity.  The VECTOR modality lives here, in
                        the `geom` column (WKT) -- NOT in the struct_in / graph_in /
                        graph_out / full_out folders, which this task does not use.
        entity_type == 'area'   selects the room/space polygons for a plan
        roomtype                room class name  (Bedroom / Kitchen / ... / Structure)
        plan_id                 a whole building FLOOR (may hold several apartments)
        unit_id                 a single APARTMENT

OUTLINE -- built from the rooms by the brief's provided formula (verbatim):
    buffer each room out by 0.3 m, unary_union, then buffer back in by 0.3 m,
    fusing the rooms into one exterior shell.  This is the model's ONLY condition.

GROUPING -- the brief appendix groups by `plan_id` (and the organizers score on a
held-out set of *plans*), so that is the default.  `unit_id` gives per-apartment
plans (~8 rooms vs ~31) and is selectable with --group unit_id.

NOTE: the CSV is large (~400 MB).  `load_msd_samples(..., limit=N)` reads only N
plans' geometry, so you never need to materialize the whole dataset.
"""
from __future__ import annotations
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

import params
from cfg import ROOM_NAMES

# roomtype name -> class id == index into cfg.ROOM_NAMES / PALETTE (locked taxonomy).
# The 10 MSD 'area' roomtypes all land in ROOM_NAMES[0..9]; K stays 13 so the
# render palette is identical for synthetic, real, and generated plans.
ROOMTYPE_TO_ID = {name: i for i, name in enumerate(ROOM_NAMES)}
WALL_BRIDGE = 0.3   # metres -- the brief's outline buffer distance


def _as_polygon(geom):
    """WKT/shapely geom -> a single valid Polygon (largest piece), or None."""
    if isinstance(geom, str):
        try:
            geom = wkt.loads(geom)
        except Exception:
            return None
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area) if not geom.is_empty else geom
    if geom.geom_type != "Polygon" or geom.is_empty or geom.area <= 0:
        return None
    return geom


def build_outline(room_polys, wall_bridge: float = WALL_BRIDGE):
    """The brief's outline-construction formula, verbatim.

    rooms.buffer(+0.3).unary_union.buffer(-0.3) -> one exterior shell.
    Returns a Polygon (or MultiPolygon for disconnected floors); params /
    postprocess / render all handle MultiPolygon outlines.
    """
    geoms = [p for p in room_polys if p is not None and not p.is_empty]
    if not geoms:
        return None
    outline = unary_union([g.buffer(wall_bridge) for g in geoms]).buffer(-wall_bridge)
    if outline.is_empty or outline.area <= 0:
        return None
    if not outline.is_valid:
        outline = outline.buffer(0)
    return outline


def _build_sample(rows_geom, rows_type):
    """(list[WKT geom], list[roomtype str]) -> sample dict, or None if unusable."""
    room_polys, rooms = [], []
    for g, rt in zip(rows_geom, rows_type):
        poly = _as_polygon(g)
        if poly is None:
            continue
        t = ROOMTYPE_TO_ID.get(str(rt))
        if t is None:                       # unknown class -> skip (keeps taxonomy clean)
            continue
        room_polys.append((poly, t))
        cx, cy, w, h, theta = params.mrr_params(poly)
        rooms.append((cx, cy, w, h, theta, t))
    if len(rooms) < 2:
        return None
    outline = build_outline([p for p, _ in room_polys])
    if outline is None:
        return None
    return dict(outline=outline, rooms=rooms, room_polys=room_polys)


def load_msd_samples(csv_path, cfg, limit=None, group=None, set_cfg=True,
                     residential_only=None, seed=None, tag_parent=None):
    """Read up to `limit` plans from the MSD CSV -> list of sample dicts.

    group : "plan_id" (default, brief) or "unit_id" (per-apartment).
    tag_parent : optional coarser id column (e.g. "plan_id" when group=="unit_id")
        to record on each sample as s[tag_parent]; each sample also gets s[group]
        (its own group-key value). Used by area_eval.py to merge units back into
        their floor.  Default None leaves the sample contract untouched.
    If set_cfg, updates cfg.n_max (99th-pct room count + margin) from the data
    and records cfg.msd_group; cfg.k stays fixed at the locked taxonomy size.
    Returns (samples, ROOMTYPE_TO_ID).
    """
    import pandas as pd
    group = group or cfg.msd_group
    residential_only = (cfg.msd_residential_only if residential_only is None
                        else residential_only)
    seed = cfg.seed if seed is None else seed
    if group not in ("plan_id", "unit_id"):
        raise ValueError(f"group must be 'plan_id' or 'unit_id', got {group!r}")

    usecols = [group, "entity_type", "roomtype", "geom"]
    if residential_only:
        usecols.append("unit_usage")
    if tag_parent and tag_parent not in usecols:
        usecols.append(tag_parent)
    print(f"[msd] reading {csv_path} (group={group}, residential_only={residential_only}"
          f"{', tag_parent='+tag_parent if tag_parent else ''})...")
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df[df["entity_type"] == "area"]
    if residential_only:
        df = df[df["unit_usage"] == "RESIDENTIAL"]
    df = df.dropna(subset=[group])

    keys = np.sort(df[group].unique())
    rng = np.random.default_rng(seed)
    keys = rng.permutation(keys)            # seeded, reproducible subset choice
    if limit:
        keys = keys[:limit]
    keep = set(keys.tolist())
    df = df[df[group].isin(keep)]

    samples = []
    for key, grp in df.groupby(group, sort=True):
        s = _build_sample(grp["geom"].tolist(), grp["roomtype"].tolist())
        if s is not None:
            if tag_parent:
                s[group] = key
                s[tag_parent] = grp[tag_parent].iloc[0]
            samples.append(s)

    if not samples:
        raise RuntimeError(
            f"No usable plans parsed from {csv_path} (group={group}). "
            f"Check the CSV has entity_type=='area' rows with WKT `geom`.")

    if set_cfg:
        counts = np.array([len(s["rooms"]) for s in samples])
        cfg.msd_group = group
        # n_max from the room-count tail (+margin), capped so attention stays cheap
        cfg.n_max = int(min(np.quantile(counts, 0.99) + 4, 200))
        # generatable classes = the 'area' roomtypes actually present (0..max), so
        # decoding never emits Door/Window/Entrance-Door (10-12).
        max_t = max(t for s in samples for (*_, t) in s["rooms"])
        cfg.n_gen_classes = int(max_t) + 1
        print(f"[msd] {len(samples)} plans | K={cfg.k} | n_gen_classes={cfg.n_gen_classes} "
              f"| n_max={cfg.n_max} | rooms/plan {counts.mean():.1f}±{counts.std():.1f} "
              f"(median {np.median(counts):.0f}, max {counts.max()})")
    return samples, ROOMTYPE_TO_ID


if __name__ == "__main__":
    import argparse
    from cfg import CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True, help="path to mds_V2_5.372k.csv")
    ap.add_argument("--group", choices=["plan_id", "unit_id"], default=CFG.msd_group)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--residential_only", action="store_true")
    args = ap.parse_args()

    samples, vocab = load_msd_samples(args.data_csv, CFG, limit=args.limit,
                                      group=args.group,
                                      residential_only=args.residential_only)
    stats = params.compute_stats(samples, CFG)
    print("stats mean:", stats[0].round(3), "std:", stats[1].round(3))
    multipoly = sum(s["outline"].geom_type == "MultiPolygon" for s in samples)
    print(f"outline MultiPolygon plans: {multipoly}/{len(samples)}")
    # MRR fidelity: how rectangular are real rooms? (high frac<0.8 -> prefer voronoi)
    ious = []
    for s in samples:
        for (poly, t), (cx, cy, w, h, th, _t) in zip(s["room_polys"], s["rooms"]):
            rect = params.rect_polygon(cx, cy, w, h, th)
            inter = poly.intersection(rect).area
            uni = poly.union(rect).area
            if uni > 0:
                ious.append(inter / uni)
    ious = np.array(ious)
    print(f"MRR-IoU vs true room polygons: mean {ious.mean():.3f}, "
          f"frac<0.8 {(ious < 0.8).mean():.3f} "
          f"(high -> rooms are non-rectangular; prefer the voronoi decoder)")
