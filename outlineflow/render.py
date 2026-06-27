"""Rasterize a vector plan to an RGB image.

ONE function is used for BOTH real and generated plans -- a render mismatch is
the #1 silent way to invalidate FID/Density/Coverage.  Per plan the world->pixel
transform is derived from that plan's outline bbox (aspect-preserving), so real
and generated (which share the outline) map identically.

NOTE: PALETTE / canvas live in cfg.py and are LOAD-BEARING -- swap them for the
organizers' exact CMAP_ROOMTYPE + resolution to make absolute FID comparable.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

import params
from cfg import PALETTE, BG_COLOR, UNASSIGNED_COLOR


def _transform(bbox, canvas):
    bx0, by0, bw, bh = bbox
    s = (canvas - 1) / max(bw, bh)
    ox = (canvas - 1 - s * bw) / 2.0
    oy = (canvas - 1 - s * bh) / 2.0

    def f(coords):
        out = []
        for x, y in coords:
            px = ox + (x - bx0) * s
            py = oy + (y - by0) * s
            out.append((px, (canvas - 1) - py))   # flip y (image origin top-left)
        return out
    return f


def _draw(draw, geom, f, color, hole_color):
    geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for g in geoms:
        if g.is_empty:
            continue
        draw.polygon(f(list(g.exterior.coords)), fill=color)
        for ring in g.interiors:
            draw.polygon(f(list(ring.coords)), fill=hole_color)


def render_plan(rooms, outline, cfg) -> np.ndarray:
    """Fast PIL render (rooms colored on black). rooms: list of ((Multi)Polygon, type)."""
    canvas = cfg.canvas
    f = _transform(params.outline_bbox(outline), canvas)
    img = Image.new("RGB", (canvas, canvas), BG_COLOR)
    draw = ImageDraw.Draw(img)
    _draw(draw, outline, f, UNASSIGNED_COLOR, BG_COLOR)     # interior backdrop (black)
    for poly, t in rooms:
        _draw(draw, poly, f, PALETTE[int(t) % len(PALETTE)], UNASSIGNED_COLOR)
    return np.asarray(img, dtype=np.uint8)


def render_msd(rooms, outline, cfg) -> np.ndarray:
    """Faithful render mirroring the official MSD plot.py:plot_floor.

    Black background, each room polygon filled with CMAP_ROOMTYPE color via
    matplotlib ax.fill, equal aspect, axis off.  This is the render the
    organizers feed to FID/Density/Coverage -- use it for the official metrics.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    px = cfg.canvas
    fig = plt.figure(figsize=(px / 100.0, px / 100.0), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.axis("off")
    ax.set_aspect("equal")
    for poly, t in rooms:
        color = tuple(c / 255.0 for c in PALETTE[int(t) % len(PALETTE)])
        geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
        for g in geoms:
            if g.is_empty:
                continue
            x, y = g.exterior.xy
            ax.fill(x, y, color=color, lw=0)
    bx0, by0, bw, bh = params.outline_bbox(outline)
    s = max(bw, bh)
    cx0, cy0 = bx0 + bw / 2.0, by0 + bh / 2.0
    ax.set_xlim(cx0 - s / 2.0, cx0 + s / 2.0)
    ax.set_ylim(cy0 - s / 2.0, cy0 + s / 2.0)
    fcanvas = FigureCanvasAgg(fig)
    fcanvas.draw()
    buf = np.asarray(fcanvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf
