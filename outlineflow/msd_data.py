"""MSD (Modified Swiss Dwellings / cvaad-challenge) data hook.

This is the ONLY file that touches real data.  It produces the SAME sample
contract as synth_data.py:

    sample = {"outline": Polygon, "rooms": [(cx,cy,w,h,theta,type), ...],
              "room_polys": [(Polygon, type), ...]}

so train.py / sample_eval.py / render.py / metrics.py are UNCHANGED.

Dataset layout (from the official usage notebook), per integer plan id:
    struct_in/{id}.npy   structural/wall components: ch0 binary + ch1,2 = x,y (m)
    graph_out/{id}.pickle networkx graph; each node has
                          'geometry' (polygon coords), 'room_type' (int), 'centroid'

We build the apartment OUTLINE as the union of the room polygons (the footprint)
-- replace with the organizers' provided outline snippet when available.

IMPORTANT: do NOT download the whole dataset.  Point --data_dir at a small local
subset, or stream a handful of plans.  `load_msd_samples(..., limit=N)` reads at
most N plans.
"""
from __future__ import annotations
import os
import glob
import pickle
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

import params


def _load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def load_msd_plan(graph_out_path, type_map=None):
    """One graph_out/{id}.pickle -> sample dict (or None if unusable)."""
    g = _load_pickle(graph_out_path)
    room_polys, rooms = [], []
    for _, data in g.nodes(data=True):
        if "geometry" not in data or "room_type" not in data:
            continue
        coords = data["geometry"]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        raw_t = int(data["room_type"])
        t = type_map[raw_t] if type_map is not None else raw_t
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda z: z.area)
        room_polys.append((poly, t))
        cx, cy, w, h, theta = params.mrr_params(poly)
        rooms.append((cx, cy, w, h, theta, t))
    if not rooms:
        return None
    outline = unary_union([p for p, _ in room_polys]).buffer(0)
    if outline.geom_type == "MultiPolygon":
        outline = max(outline.geoms, key=lambda z: z.area)
    return dict(outline=outline, rooms=rooms, room_polys=room_polys)


def build_type_vocab(graph_paths):
    """Scan graphs -> {raw_room_type_int: contiguous_id}.  Sets K = len(vocab)."""
    seen = set()
    for p in graph_paths:
        g = _load_pickle(p)
        for _, d in g.nodes(data=True):
            if "room_type" in d:
                seen.add(int(d["room_type"]))
    vocab = {t: i for i, t in enumerate(sorted(seen))}
    return vocab


def load_msd_samples(data_dir, cfg, limit=None, set_cfg=True):
    """Read up to `limit` plans from <data_dir>/graph_out/*.pickle.

    If set_cfg, updates cfg.k (type count) and cfg.n_max (99th pct room count)
    from the data, and returns (samples, vocab).
    """
    graph_dir = os.path.join(data_dir, "graph_out")
    paths = sorted(glob.glob(os.path.join(graph_dir, "*.pickle")))
    if not paths:
        raise FileNotFoundError(
            f"No graph_out/*.pickle under {data_dir}. Point --data_dir at a local "
            f"MSD/cvaad-challenge subset (do NOT download the full dataset).")
    if limit:
        paths = paths[:limit]
    vocab = build_type_vocab(paths)
    samples = []
    for p in paths:
        s = load_msd_plan(p, type_map=vocab)
        if s is not None:
            samples.append(s)
    if set_cfg:
        counts = np.array([len(s["rooms"]) for s in samples])
        cfg.k = max(len(vocab), 1)
        cfg.n_max = int(np.quantile(counts, 0.99)) + 2
        print(f"[msd] {len(samples)} plans | K={cfg.k} | n_max={cfg.n_max} "
              f"| rooms/plan {counts.mean():.1f}±{counts.std():.1f} "
              f"(max {counts.max()})")
    return samples, vocab


if __name__ == "__main__":
    import argparse
    from cfg import CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="local MSD/cvaad-challenge split dir (with graph_out/)")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    samples, vocab = load_msd_samples(args.data_dir, CFG, limit=args.limit)
    stats = params.compute_stats(samples, CFG)
    print("stats mean:", stats[0].round(3), "std:", stats[1].round(3))
    print("vocab (raw_room_type -> id):", vocab)
    # MRR fidelity: how rectangular are real rooms?
    from shapely import area
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
          f"(high frac<0.8 -> rooms are non-rectangular; prefer voronoi decoder "
          f"or the 2-rects-per-room extension)")
