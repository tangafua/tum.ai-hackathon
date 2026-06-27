"""Synthetic floor-plan generator (no real data needed).

Each plan = a rectangular or L-shaped outline tiled into 3-8 non-overlapping
rooms by recursive binary splitting.  This gives a gap-free ground-truth layout
that the rectangle tokenization CAN represent, so the model has a learnable,
decoder-consistent target.  Same output contract as ``msd_data.py``:

    sample = {
        "outline":    shapely Polygon,
        "rooms":      [(cx,cy,w,h,theta,type), ...]   # MRR tokens
        "room_polys": [(Polygon, type), ...]          # true polys (for rendering "real")
    }
"""
from __future__ import annotations
import numpy as np
from shapely.geometry import box, Polygon
from shapely.validation import make_valid

import params


def _make_outline(rng) -> Polygon:
    w = rng.uniform(8.0, 16.0)
    h = rng.uniform(8.0, 16.0)
    full = box(0.0, 0.0, w, h)
    if rng.random() < 0.4:
        # L-shape: remove a random corner rectangle
        cw = rng.uniform(0.3, 0.5) * w
        ch = rng.uniform(0.3, 0.5) * h
        corner = rng.integers(0, 4)
        if corner == 0:
            cut = box(0, 0, cw, ch)
        elif corner == 1:
            cut = box(w - cw, 0, w, ch)
        elif corner == 2:
            cut = box(w - cw, h - ch, w, h)
        else:
            cut = box(0, h - ch, cw, h)
        full = full.difference(cut)
    return make_valid(full)


def _tile(outline: Polygon, target: int, rng, min_size=1.5):
    """Recursive binary split of the outline bbox into `target` rectangular cells."""
    bx0, by0, bx1, by1 = outline.bounds
    cells = [(bx0, by0, bx1, by1)]
    while len(cells) < target:
        areas = [(x1 - x0) * (y1 - y0) for (x0, y0, x1, y1) in cells]
        idx = int(np.argmax(areas))
        x0, y0, x1, y1 = cells.pop(idx)
        w, h = x1 - x0, y1 - y0
        r = rng.uniform(0.35, 0.65)
        if w >= h and w > 2 * min_size:
            xm = x0 + r * w
            cells += [(x0, y0, xm, y1), (xm, y0, x1, y1)]
        elif h > 2 * min_size:
            ym = y0 + r * h
            cells += [(x0, y0, x1, ym), (x0, ym, x1, y1)]
        else:
            cells.append((x0, y0, x1, y1))
            break
    return cells


def generate_samples(n: int, cfg, rng=None):
    rng = rng or np.random.default_rng(cfg.seed)
    samples = []
    while len(samples) < n:
        outline = _make_outline(rng)
        if outline.is_empty or outline.area <= 0:
            continue
        # room count correlates with apartment area (realistic + learnable from
        # the outline, which is the only conditioning the model gets)
        target = int(np.clip(round(outline.area / 22.0 + rng.normal(0, 0.5)),
                             cfg.min_rooms, cfg.max_rooms))
        cells = _tile(outline, target, rng)
        rooms, room_polys = [], []
        for (x0, y0, x1, y1) in cells:
            poly = box(x0, y0, x1, y1).intersection(outline)
            if poly.is_empty or poly.area < 0.5:
                continue
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            t = int(rng.integers(0, cfg.n_synth_classes))  # dwelling classes 0-8
            cx, cy, w, h, theta = params.mrr_params(poly)
            rooms.append((cx, cy, w, h, theta, t))
            room_polys.append((poly, t))
        if len(rooms) < 1:
            continue
        samples.append(dict(outline=outline, rooms=rooms, room_polys=room_polys))
    return samples


def build_tensors(samples, stats, cfg, rng=None):
    """samples -> stacked arrays X [N,n_max,D], OUT [N,P,4] for training."""
    rng = rng or np.random.default_rng(cfg.seed + 1)
    X = np.zeros((len(samples), cfg.n_max, cfg.d), dtype=np.float32)
    OUT = np.zeros((len(samples), cfg.p_outline, 4), dtype=np.float32)
    for i, s in enumerate(samples):
        X[i] = params.build_x1(s["rooms"], s["outline"], stats, cfg, rng)
        OUT[i] = params.sample_outline_points(s["outline"], cfg.p_outline)
    return X, OUT


if __name__ == "__main__":
    from cfg import CFG
    rng = np.random.default_rng(0)
    s = generate_samples(5, CFG, rng)
    print(f"generated {len(s)} samples; room counts:", [len(x['rooms']) for x in s])
    st = params.compute_stats(s, CFG)
    print("stats mean:", st[0], "std:", st[1])
    X, OUT = build_tensors(s, st, CFG, rng)
    print("X", X.shape, "OUT", OUT.shape, "presence>0 per plan:",
          (X[..., 0] > 0).sum(axis=1))
