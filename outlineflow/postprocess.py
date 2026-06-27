"""Turn raw model tokens into a guaranteed-valid, outline-filling layout.

Pipeline (deterministic, independent of model quality):
  1. decode tokens -> oriented rectangles + types + presence
  2. keep present rooms (fallback: top-1 so we never render empty)
  3. clip every room to the outline
  4. greedy overlap-resolve (high-presence / large rooms win) -> disjoint rooms
  5. drop slivers
  6. MANDATORY gap-fill: every leftover patch is merged into its nearest room

Output: list of (shapely (Multi)Polygon, type_id) with union(rooms) == outline
(zero overlap, zero interior background) -- exactly what the metrics reward.
"""
from __future__ import annotations
import numpy as np
from shapely import affinity
from shapely.ops import unary_union, voronoi_diagram
from shapely.geometry import MultiPoint, Point, Polygon
import params


def outline_axis_angle(outline) -> float:
    """Dominant orientation of the building (radians), from the outline's MRR.

    Real Swiss apartments are rectilinear, but the whole plan can sit at any global
    angle, so we align rooms to the OUTLINE's own axes (not global 0/90).
    """
    try:
        mrr = outline.minimum_rotated_rectangle
        xs, ys = mrr.exterior.coords.xy
        pts = np.column_stack([xs, ys])[:-1]
        e = pts[1] - pts[0]
        return float(np.arctan2(e[1], e[0]))
    except Exception:
        return 0.0


def snap_theta(theta: float, base: float) -> float:
    """Snap a room angle to base + k*90deg (nearest), i.e. align to the building axes."""
    half_pi = np.pi / 2.0
    return base + round((theta - base) / half_pi) * half_pi


def _largest(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def _clean(geom):
    if geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def layout_from_tokens(x1_np, outline, stats, cfg, presence_thresh=0.0,
                       theta_snap=None):
    dec = params.decode_x1(x1_np, outline, stats, cfg)      # sorted presence desc
    present = [d for d in dec if d["presence"] > presence_thresh] or dec[:1]

    # axis-align rooms to the building (real apartments are rectilinear)
    snap = cfg.theta_snap if theta_snap is None else theta_snap
    base = outline_axis_angle(outline) if snap else 0.0

    # 3. clip to outline
    rooms = []
    for d in present:
        theta = snap_theta(d["theta"], base) if snap else d["theta"]
        poly = params.rect_polygon(d["cx"], d["cy"], d["w"], d["h"], theta)
        poly = _clean(poly.intersection(outline))
        poly = _largest(poly)
        if not poly.is_empty and poly.area > 0:
            rooms.append((poly, d["type"], d["presence"]))

    if not rooms:                                           # nothing survived clip
        return [(outline, present[0]["type"])]

    # 4. greedy overlap-resolve
    rooms.sort(key=lambda r: (r[2], r[0].area), reverse=True)
    resolved, union_so_far = [], None
    for poly, t, _pr in rooms:
        if union_so_far is not None:
            poly = _clean(poly.difference(union_so_far))
            poly = _largest(poly)
        if poly.is_empty or poly.area <= 0:
            continue
        resolved.append([poly, t])
        union_so_far = poly if union_so_far is None else _clean(unary_union([union_so_far, poly]))

    # 5. drop slivers
    min_area = cfg.min_area_frac * outline.area
    resolved = [r for r in resolved if r[0].area >= min_area]
    if not resolved:
        return [(outline, present[0]["type"])]

    # 6. mandatory gap-fill
    leftover = _clean(outline.difference(unary_union([r[0] for r in resolved])))
    if not leftover.is_empty and leftover.area > 1e-9:
        comps = list(leftover.geoms) if leftover.geom_type == "MultiPolygon" else [leftover]
        for comp in comps:
            if comp.is_empty or comp.area <= 0:
                continue
            j = min(range(len(resolved)), key=lambda i: resolved[i][0].distance(comp))
            resolved[j][0] = _clean(unary_union([resolved[j][0], comp]))

    return [(r[0], int(r[1])) for r in resolved]


def voronoi_layout(x1_np, outline, stats, cfg, presence_thresh=0.0):
    """Decode rooms as a Voronoi partition of the outline around predicted seeds.

    Uses the model's predicted room CENTROIDS + TYPES + COUNT and ignores w/h/theta.
    Voronoi cells tile the bbox exactly, so clipping to the outline yields a
    gap-free, non-overlapping partition -- a clean, realistic-looking tiling that
    sidesteps the "rooms don't fill the space" failure of raw boxes.
    """
    dec = params.decode_x1(x1_np, outline, stats, cfg)
    present = [d for d in dec if d["presence"] > presence_thresh] or dec[:1]

    # dedupe near-coincident seeds (Voronoi needs distinct points)
    seeds = []
    for d in present:
        if all((d["cx"] - sx) ** 2 + (d["cy"] - sy) ** 2 > 1e-6 for sx, sy, _ in seeds):
            seeds.append((d["cx"], d["cy"], d["type"]))
    if len(seeds) < 2:
        return [(outline, int(seeds[0][2]) if seeds else 0)]

    pts = MultiPoint([Point(x, y) for x, y, _ in seeds])
    regions = voronoi_diagram(pts, envelope=outline)
    rooms = []
    for cell in regions.geoms:
        c = _clean(cell.intersection(outline))
        if c.is_empty or c.area <= 0:
            continue
        t = next((tp for x, y, tp in seeds if cell.contains(Point(x, y))), None)
        if t is None:
            t = min(seeds, key=lambda s: (s[0] - c.centroid.x) ** 2
                    + (s[1] - c.centroid.y) ** 2)[2]
        rooms.append((c, int(t)))
    return rooms or [(outline, int(seeds[0][2]))]


def coverage_overlap(rooms, outline):
    """Diagnostics: (interior coverage fraction, overlap-area fraction)."""
    if not rooms:
        return 0.0, 0.0
    polys = [p for p, _ in rooms]
    covered = unary_union(polys).intersection(outline).area
    sum_area = sum(p.area for p in polys)
    cov = covered / max(outline.area, 1e-9)
    overlap = (sum_area - covered) / max(outline.area, 1e-9)
    return cov, max(overlap, 0.0)


def _snap_coords(poly, ox, oy, cell):
    """Quantize a polygon's vertices to an axis-aligned grid of step `cell`."""
    def q(coords):
        return [(ox + round((x - ox) / cell) * cell,
                 oy + round((y - oy) / cell) * cell) for x, y in coords]
    p = Polygon(q(poly.exterior.coords), [q(r.coords) for r in poly.interiors])
    return p if p.is_valid else p.buffer(0)


def align_layout(rooms, outline, cfg, grid=48):
    """Post-process: snap room edges to a grid along the building's own axes, then
    re-partition the outline (overlap-resolve + gap-fill) so it stays gap-free.

    Real Swiss plans are clean rectilinear tilings; the raw decoded rooms jitter
    and carve into L-shapes.  Snapping to a coarse grid (in the outline's rotated
    frame) removes that jitter and aligns shared edges, which reads as more
    regular / realistic after rasterisation.  Deterministic; preserves room count.
    """
    if not rooms or grid <= 0:
        return rooms
    base = outline_axis_angle(outline)
    deg = float(np.degrees(base))
    origin = outline.centroid
    o_rot = _clean(affinity.rotate(outline, -deg, origin=origin))
    minx, miny, maxx, maxy = o_rot.bounds
    cell = max(maxx - minx, maxy - miny) / float(grid)
    if cell <= 0:
        return rooms

    # snap each room in the axis-aligned frame, clip back to the (rotated) outline
    snapped = []
    for poly, t in rooms:
        pr = _clean(affinity.rotate(poly, -deg, origin=origin))
        for c in (pr.geoms if pr.geom_type == "MultiPolygon" else [pr]):
            if c.is_empty or c.area <= 0:
                continue
            cs = _largest(_clean(_snap_coords(c, minx, miny, cell).intersection(o_rot)))
            if not cs.is_empty and cs.area > 0:
                snapped.append([cs, int(t), cs.area])
    if not snapped:
        return rooms

    # snapping can introduce overlaps -> greedy resolve (largest wins), like decode
    snapped.sort(key=lambda r: r[2], reverse=True)
    resolved, u = [], None
    for poly, t, _a in snapped:
        if u is not None:
            poly = _largest(_clean(poly.difference(u)))
        if poly.is_empty or poly.area <= 0:
            continue
        resolved.append([poly, t])
        u = poly if u is None else _clean(unary_union([u, poly]))
    min_area = cfg.min_area_frac * o_rot.area
    resolved = [r for r in resolved if r[0].area >= min_area] or resolved[:1]

    # gap-fill leftover back into nearest room (keep union == outline)
    leftover = _clean(o_rot.difference(unary_union([r[0] for r in resolved])))
    if not leftover.is_empty and leftover.area > 1e-9:
        comps = list(leftover.geoms) if leftover.geom_type == "MultiPolygon" else [leftover]
        for comp in comps:
            if comp.is_empty or comp.area <= 0:
                continue
            j = min(range(len(resolved)), key=lambda i: resolved[i][0].distance(comp))
            resolved[j][0] = _clean(unary_union([resolved[j][0], comp]))

    # rotate back to world frame
    return [(_clean(affinity.rotate(r[0], deg, origin=origin)), int(r[1]))
            for r in resolved]
