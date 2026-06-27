"""Single source of truth for the room-token parameterization.

A floor plan is a SET of up to ``n_max`` room tokens.  Each token is a vector of
length ``D = 7 + K`` with this FIXED channel order:

    [0]      presence       +1 = real room, -1 = padding (absent)
    [1]      cx             centroid x   (standardized, see stats)
    [2]      cy             centroid y   (standardized)
    [3]      w              oriented-rect side 1 (standardized)
    [4]      h              oriented-rect side 2 (standardized)
    [5]      sin(2*theta)   rectangle orientation (double-angle: pi-periodic)
    [6]      cos(2*theta)
    [7:7+K]  type one-hot rescaled to {-1, +1}   (+1 on the true class)

Geometry channels (cx,cy,w,h) are expressed in the OUTLINE-BBOX frame
(roughly [0,1]) and then standardized to ~unit variance with ``stats`` so they
move under the unit-variance flow comparably to presence/type.

All geometry <-> tensor conversion goes through here, so train / sample /
render / eval can never disagree about the layout.
"""
from __future__ import annotations
import warnings
import numpy as np
from shapely.geometry import box, Polygon
from shapely.affinity import rotate, translate

# minimum_rotated_rectangle emits harmless divide-by-zero RuntimeWarnings on
# near-degenerate slivers; results are still correct (verified IoU==1).
warnings.filterwarnings("ignore", message=".*oriented_envelope.*")


# ---------------------------------------------------------------- geometry helpers
def mrr_params(poly: Polygon):
    """Return (cx, cy, w, h, theta) of the minimum rotated rectangle of ``poly``.

    theta is the angle (radians) of the ``w`` side.  Reconstruction with
    ``rect_polygon`` is exact up to the pi-periodicity of a rectangle.
    """
    try:
        mrr = poly.minimum_rotated_rectangle
        xs, ys = mrr.exterior.coords.xy
        pts = np.column_stack([xs, ys])[:-1]  # 4 corners
        e0 = pts[1] - pts[0]
        e1 = pts[2] - pts[1]
        w = float(np.hypot(e0[0], e0[1]))
        h = float(np.hypot(e1[0], e1[1]))
        theta = float(np.arctan2(e0[1], e0[0]))
        c = pts.mean(axis=0)
        return float(c[0]), float(c[1]), w, h, theta
    except Exception:
        c = poly.centroid
        return float(c.x), float(c.y), 1e-3, 1e-3, 0.0


def rect_polygon(cx, cy, w, h, theta) -> Polygon:
    """Oriented rectangle as a shapely Polygon (inverse of ``mrr_params``)."""
    b = box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
    b = rotate(b, theta, origin=(0, 0), use_radians=True)
    b = translate(b, cx, cy)
    return b


def outline_bbox(outline: Polygon):
    bx0, by0, bx1, by1 = outline.bounds
    bw = max(bx1 - bx0, 1e-6)
    bh = max(by1 - by0, 1e-6)
    return float(bx0), float(by0), float(bw), float(bh)


def outline_scale(outline) -> np.ndarray:
    """Absolute SIZE of the outline in metres -> [log area, log w, log h].

    sample_outline_points() normalizes the boundary to [-1,1], discarding scale, so
    the model cannot tell a 30 m2 flat from a 300 m2 floor and emits a near-constant
    room count.  Room-count correlates with area (r~0.88), so this size vector -- a
    pure function of the outline -- lets the model match the real count spread.
    """
    bx0, by0, bw, bh = outline_bbox(outline)
    return np.array([np.log(max(outline.area, 1e-6)),
                     np.log(bw), np.log(bh)], dtype=np.float32)


def compute_scale_stats(samples) -> np.ndarray:
    """[2,3] = per-dim (mean, std) of outline_scale over samples, for standardization."""
    s = np.stack([outline_scale(x["outline"]) for x in samples])
    return np.stack([s.mean(0), s.std(0) + 1e-6]).astype(np.float32)


def _exterior_rings(outline):
    """Exterior ring(s) of a (Multi)Polygon outline, as a list of LinearRings.

    A plan_id-level MSD outline (the brief's buffer(0.3)-union shell over a whole
    floor) can be a MultiPolygon of disconnected apartments, so the outline encoder
    must see the boundary of every piece -- not just the largest.
    """
    polys = outline.geoms if outline.geom_type == "MultiPolygon" else [outline]
    return [p.exterior for p in polys if (not p.is_empty) and p.exterior.length > 0]


def sample_outline_points(outline: Polygon, P: int) -> np.ndarray:
    """Sample P boundary points (arc-length) -> [P,4] = (x_n, y_n, nx, ny).

    x_n,y_n are normalized to the outline bbox in [-1,1]; (nx,ny) is the unit
    OUTWARD normal.  This is the only thing the model sees about the outline.
    Handles MultiPolygon outlines by distributing the P points across every
    exterior ring in proportion to its perimeter.
    """
    bx0, by0, bw, bh = outline_bbox(outline)
    rings = _exterior_rings(outline)
    if not rings:                                  # degenerate fallback
        return np.zeros((P, 4), dtype=np.float32)
    lengths = np.array([r.length for r in rings], dtype=np.float64)
    # integer point budget per ring (proportional, >=1 each, summing to P)
    quota = np.maximum(1, np.floor(P * lengths / lengths.sum()).astype(int))
    while quota.sum() < P:
        quota[int(np.argmax(lengths / quota))] += 1
    while quota.sum() > P:
        quota[int(np.argmax(quota))] -= 1

    eps = 1e-6 * max(bw, bh)
    feats = np.zeros((P, 4), dtype=np.float32)
    k = 0
    for ring, q in zip(rings, quota):
        L = ring.length
        ds = np.linspace(0.0, L, q, endpoint=False)
        raw = np.array([(ring.interpolate(d).x, ring.interpolate(d).y) for d in ds])
        for i in range(q):
            nxt = raw[(i + 1) % q]
            prv = raw[(i - 1) % q]
            tang = nxt - prv
            n = np.hypot(tang[0], tang[1]) + 1e-12
            # candidate normal (rotate tangent -90 deg)
            cand = np.array([tang[1], -tang[0]]) / n
            test = raw[i] + eps * 10 * cand
            if outline.contains(Polygon(box(test[0] - eps, test[1] - eps,
                                            test[0] + eps, test[1] + eps)).centroid):
                cand = -cand  # pointed inward -> flip
            x_n = 2.0 * (raw[i, 0] - bx0) / bw - 1.0
            y_n = 2.0 * (raw[i, 1] - by0) / bh - 1.0
            feats[k] = (x_n, y_n, cand[0], cand[1])
            k += 1
    return feats


# ---------------------------------------------------------------- standardization
def compute_stats(samples, cfg) -> np.ndarray:
    """Per-channel mean/std of (cx0,cy0,w0,h0) in the bbox frame over all real rooms.

    ``samples`` is a list of dicts {"rooms": [(cx,cy,w,h,theta,type), ...],
    "outline": Polygon}.  Returns array [2,4] = (mean, std).
    """
    vals = []
    for s in samples:
        bx0, by0, bw, bh = outline_bbox(s["outline"])
        for (cx, cy, w, h, theta, t) in s["rooms"]:
            vals.append([(cx - bx0) / bw, (cy - by0) / bh, w / bw, h / bh])
    vals = np.asarray(vals, dtype=np.float64)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return np.stack([mean, std]).astype(np.float32)  # [2,4]


# ---------------------------------------------------------------- encode / decode
def build_x1(rooms, outline, stats, cfg, rng=None) -> np.ndarray:
    """List of (cx,cy,w,h,theta,type) + outline -> token tensor x1 [n_max, D]."""
    rng = rng if rng is not None else np.random.default_rng(42)  # deterministic fallback
    bx0, by0, bw, bh = outline_bbox(outline)
    mean, std = stats[0], stats[1]
    D = cfg.d
    x1 = np.zeros((cfg.n_max, D), dtype=np.float32)
    x1[:, 0] = -1.0  # all padding by default
    n = min(len(rooms), cfg.n_max)
    slots = rng.permutation(cfg.n_max)[:n]
    for slot, (cx, cy, w, h, theta, t) in zip(slots, rooms[:n]):
        g0 = np.array([(cx - bx0) / bw, (cy - by0) / bh, w / bw, h / bh])
        gn = (g0 - mean) / std
        x1[slot, 0] = 1.0
        x1[slot, 1:5] = gn
        x1[slot, 5] = np.sin(2.0 * theta)
        x1[slot, 6] = np.cos(2.0 * theta)
        x1[slot, 7:7 + cfg.k] = -1.0
        x1[slot, 7 + int(t) % cfg.k] = 1.0
    return x1


def decode_x1(x1: np.ndarray, outline, stats, cfg):
    """x1 [n_max, D] -> list of dicts for present rooms (sorted by presence desc).

    Each dict: {cx,cy,w,h,theta,type,presence}.  Geometry in WORLD coords.
    Caller decides the presence threshold (postprocess uses > 0 with a fallback).
    """
    bx0, by0, bw, bh = outline_bbox(outline)
    mean, std = stats[0], stats[1]
    min_w = 0.01 * bw
    min_h = 0.01 * bh
    out = []
    for s in range(cfg.n_max):
        tok = x1[s]
        presence = float(tok[0])
        gn = tok[1:5]
        g0 = gn * std + mean
        cx = g0[0] * bw + bx0
        cy = g0[1] * bh + by0
        w = max(float(g0[2] * bw), min_w)
        h = max(float(g0[3] * bh), min_h)
        theta = 0.5 * float(np.arctan2(tok[5], tok[6]))
        # argmax only over GENERATABLE classes so we never emit non-'area' classes
        # (Door/Window/Entrance-Door); falls back to k if n_gen_classes is unset.
        n_cls = getattr(cfg, "n_gen_classes", 0) or cfg.k
        ttype = int(np.argmax(tok[7:7 + n_cls]))
        out.append(dict(cx=cx, cy=cy, w=w, h=h, theta=theta, type=ttype,
                        presence=presence))
    out.sort(key=lambda r: r["presence"], reverse=True)
    return out
