#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
explore_dataset.py
==================

Exploratory Data Analysis (EDA) for the MSD / Swiss-Dwellings export
`mds_V2_5.372k.csv`.

The CSV is a *flat, long* table: every row is **one geometric entity** drawn on
an architectural floor plan (a room area, a wall, or a door/window opening),
tagged with the chain of IDs that locates it in the building hierarchy plus a
WKT polygon (`geom`).

Hierarchy (coarse -> fine), one level nested in the previous one:

    site_id          a development / location (a cluster of buildings)
      building_id      a single building inside the site
        plan_id        one architectural plan drawing  (== floor_id, 1:1 here)
          floor_id     the physical floor of that plan
            apartment_id   a dwelling (hashed id); NULL for PUBLIC space
              unit_id      the dwelling's instance on the plan (~1 per apartment)
                area_id    a single room / area polygon; set ONLY for areas
                  (row)    the entity: area | separator(wall) | opening(door/win)

Key facts confirmed empirically (see report):
  * `area_id`   is populated ONLY when `entity_type == 'area'`.
  * `apartment_id` / `unit_id` are NULL  <=>  `unit_usage == 'PUBLIC'`.
  * `plan_id` and `floor_id` are 1:1 in this export.

For the OutlineFlow modelling task the working granularity is two-tiered:
  * one training SAMPLE  = one apartment   (group by `apartment_id`)
  * generated TOKENS     = its rooms       (`entity_type == 'area'`, `roomtype`)

Usage
-----
    PY=/opt/miniconda3/bin/python   # env with pandas, numpy, matplotlib, shapely
    $PY explore_dataset.py                      # full structural EDA, sampled geometry
    $PY explore_dataset.py --rows 200000        # cap rows (fast smoke run)
    $PY explore_dataset.py --geom-sample 50000  # how many area polygons to parse
    $PY explore_dataset.py --no-plots           # skip figure rendering

Outputs
-------
    eda/EDA_REPORT.md         human-readable report (also echoed to stdout)
    eda/fig_*.png             figures (distributions, hierarchy, geometry)
"""

from __future__ import annotations

import argparse
import os
import textwrap
from contextlib import contextmanager

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CSV_PATH = "mds_V2_5.372k.csv"
OUT_DIR = "eda"

# Columns that are cheap to load (everything except the heavy WKT `geom`).
# Loading without `geom` lets us scan the full 1M-row file quickly.
META_COLS = [
    "apartment_id", "site_id", "building_id", "plan_id", "floor_id",
    "unit_id", "area_id", "unit_usage", "entity_type", "entity_subtype",
    "elevation", "height", "zoning", "roomtype",
]

# The two leading unnamed columns are pandas index dumps -> ignored on purpose.

# Hierarchy from coarse to fine; used for cardinality and nesting analysis.
HIERARCHY = ["site_id", "building_id", "plan_id", "floor_id",
             "apartment_id", "unit_id", "area_id"]


# --------------------------------------------------------------------------- #
# Tiny report helper: print to stdout AND append to a markdown buffer.
# --------------------------------------------------------------------------- #
class Report:
    """Collects markdown lines and mirrors them to stdout."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def h(self, level: int, title: str) -> None:
        """Write a markdown heading and echo a visible separator."""
        self("")
        self(f"{'#' * level} {title}")
        self("")

    def table(self, df: pd.DataFrame, floatfmt: str = "{:.3f}") -> None:
        """Render a small DataFrame/Series as a markdown table (no deps)."""
        if isinstance(df, pd.Series):
            df = df.to_frame()
        # Format floats compactly without mangling integers/strings.
        fmt = df.copy()
        for c in fmt.columns:
            if pd.api.types.is_float_dtype(fmt[c]):
                fmt[c] = fmt[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
            else:
                fmt[c] = fmt[c].astype(str)
        index_name = fmt.index.name or ""
        headers = [index_name] + [str(c) for c in fmt.columns]
        rows = [[str(i)] + [str(v) for v in row]
                for i, row in zip(fmt.index, fmt.values)]
        # Self-contained GitHub-flavoured pipe table (avoids the `tabulate` dep).
        self("| " + " | ".join(headers) + " |")
        self("| " + " | ".join("---" for _ in headers) + " |")
        for r in rows:
            self("| " + " | ".join(r) + " |")
        self("")

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write("\n".join(self.lines) + "\n")


R = Report()


@contextmanager
def section(level: int, title: str):
    R.h(level, title)
    yield


# --------------------------------------------------------------------------- #
# 0. Load
# --------------------------------------------------------------------------- #
def load_meta(path: str, nrows: int | None) -> pd.DataFrame:
    """Load all structural columns (no geometry) for fast, full-file EDA."""
    df = pd.read_csv(path, usecols=META_COLS, nrows=nrows)
    return df


# --------------------------------------------------------------------------- #
# 1. Schema, size, dtypes, memory
# --------------------------------------------------------------------------- #
def explore_schema(df: pd.DataFrame, path: str) -> None:
    with section(2, "1. Dataset overview"):
        size_mb = os.path.getsize(path) / 1e6
        R(f"- File: `{path}`  ({size_mb:,.1f} MB on disk)")
        R(f"- Rows analysed: **{len(df):,}**  (1 row = 1 geometric entity)")
        R(f"- Structural columns: **{df.shape[1]}**  (`geom` excluded from this load)")
        R(f"- In-memory size of loaded columns: "
          f"{df.memory_usage(deep=True).sum() / 1e6:,.1f} MB")
        R("")
        dtypes = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
            "n_missing": df.isna().sum().values,
            "pct_missing": (df.isna().mean() * 100).round(2).values,
        })
        R.table(dtypes)


# --------------------------------------------------------------------------- #
# 2. Missing values and what they MEAN (structural, not random)
# --------------------------------------------------------------------------- #
def explore_missingness(df: pd.DataFrame) -> None:
    with section(2, "2. Missing values are structural"):
        R("Missingness in this table is not noise -- it encodes the schema:")
        R("")
        # area_id present only for areas
        by_etype = df.groupby("entity_type")["area_id"].apply(lambda s: s.isna().mean())
        R("**`area_id` is NULL fraction by `entity_type`** "
          "(=> set only for rooms/areas):")
        R.table(by_etype.rename("frac_area_id_null"))

        # apartment / unit null <=> PUBLIC usage
        ct = pd.crosstab(df["unit_usage"], df["apartment_id"].isna())
        ct.columns = ["apartment_id present", "apartment_id NULL"]
        R("**`apartment_id` presence vs `unit_usage`** "
          "(=> PUBLIC space has no dwelling id):")
        R.table(ct, floatfmt="{:.0f}")


# --------------------------------------------------------------------------- #
# 3. Categorical distributions
# --------------------------------------------------------------------------- #
def explore_categoricals(df: pd.DataFrame) -> None:
    with section(2, "3. Categorical columns"):
        # unit_usage
        R("**`unit_usage`** (residential dwelling vs shared/public space):")
        R.table(_counts(df["unit_usage"]))

        # entity_type
        R("**`entity_type`** (what each row geometrically is):")
        R.table(_counts(df["entity_type"]))

        # entity_subtype, split per entity_type (the vocabularies are disjoint)
        R("**`entity_subtype` per `entity_type`** "
          "(room vocabulary vs wall/opening vocabulary):")
        for et in ["area", "separator", "opening"]:
            subs = (df.loc[df.entity_type == et, "entity_subtype"]
                      .value_counts().rename("count"))
            R(f"- `{et}`: " + ", ".join(f"{k} ({v:,})" for k, v in subs.items()))
        R("")

        # roomtype / zoning (human-readable labels; structure/door/window for non-areas)
        R("**`roomtype`** (human label; non-area rows carry Structure/Door/Window):")
        R.table(_counts(df["roomtype"]).head(20))
        R("**`zoning`** (private zones Zone1-4 for rooms; Structure/Door/Window else):")
        R.table(_counts(df["zoning"]).head(20))


def _counts(s: pd.Series) -> pd.DataFrame:
    vc = s.value_counts(dropna=False)
    out = pd.DataFrame({"count": vc, "pct": (vc / len(s) * 100).round(2)})
    out.index = out.index.astype(str)
    return out


# --------------------------------------------------------------------------- #
# 4. ID hierarchy: cardinality + nesting (children per parent)
# --------------------------------------------------------------------------- #
def explore_hierarchy(df: pd.DataFrame) -> None:
    with section(2, "4. ID hierarchy and nesting"):
        card = pd.Series(
            {c: df[c].nunique(dropna=True) for c in HIERARCHY},
            name="n_unique",
        )
        R("**Cardinality of each level** (coarse -> fine):")
        R.table(card.to_frame(), floatfmt="{:.0f}")

        # Children-per-parent for each adjacent (parent, child) pair.
        R("**Branching factor** -- distinct children per parent "
          "(mean / median / max):")
        rows = []
        pairs = [("site_id", "building_id"), ("building_id", "plan_id"),
                 ("plan_id", "floor_id"), ("plan_id", "apartment_id"),
                 ("apartment_id", "unit_id"), ("apartment_id", "area_id")]
        for parent, child in pairs:
            g = df.dropna(subset=[parent, child]).groupby(parent)[child].nunique()
            rows.append({
                "relation": f"{parent} -> {child}",
                "mean": round(g.mean(), 2),
                "median": int(g.median()),
                "max": int(g.max()),
            })
        R.table(pd.DataFrame(rows).set_index("relation"))
        R("Notes: `plan_id -> floor_id` is 1:1 (a plan == a floor); "
          "`apartment_id -> unit_id` ~1:1 (treat apartment as the dwelling); "
          "`plan_id -> apartment_id` shows multiple dwellings share one floor plan.")


# --------------------------------------------------------------------------- #
# 5. Per-apartment composition (the modelling unit)
# --------------------------------------------------------------------------- #
def explore_apartments(df: pd.DataFrame) -> None:
    with section(2, "5. Per-apartment composition (the training sample)"):
        res = df[df.unit_usage == "RESIDENTIAL"]

        # All entities per apartment (areas + walls + openings).
        per_apt_entities = res.groupby("apartment_id").size()
        R("**Entities per apartment** (rooms + walls + openings):")
        R.table(per_apt_entities.describe().to_frame("value"))

        # Rooms (areas) per apartment -- the token count the generator must emit.
        rooms = res[res.entity_type == "area"]
        per_apt_rooms = rooms.groupby("apartment_id")["area_id"].nunique()
        R("**Rooms (areas) per apartment** -- target token-set size:")
        R.table(per_apt_rooms.describe().to_frame("value"))

        # Most common room types overall (area rows only).
        R("**Most common room types** (`entity_subtype` for areas):")
        R.table(_counts(rooms["entity_subtype"]).head(15))


# --------------------------------------------------------------------------- #
# 6. Geometry (sampled): parse WKT, compute area / bbox / aspect ratio
# --------------------------------------------------------------------------- #
def explore_geometry(path: str, nrows: int | None, geom_sample: int) -> pd.DataFrame:
    """Parse a sample of AREA polygons and summarise their geometry."""
    with section(2, "6. Room geometry (sampled WKT polygons)"):
        try:
            from shapely import wkt
        except Exception as exc:  # pragma: no cover
            R(f"_shapely not available ({exc}); skipping geometry section._")
            return pd.DataFrame()

        # Read just the columns we need, then keep area rows up to the budget.
        use = ["entity_type", "entity_subtype", "geom", "roomtype"]
        df = pd.read_csv(path, usecols=use, nrows=nrows)
        area = df[df.entity_type == "area"]
        if len(area) > geom_sample:
            # Deterministic head sample keeps the run reproducible across machines.
            area = area.head(geom_sample)
        R(f"Parsing {len(area):,} area polygons "
          f"(of {int((df.entity_type == 'area').sum()):,} in the loaded rows).")

        geoms = area["geom"].apply(wkt.loads)
        poly_area = geoms.apply(lambda p: p.area)
        bounds = geoms.apply(lambda p: p.bounds)
        w = bounds.apply(lambda t: t[2] - t[0])
        h = bounds.apply(lambda t: t[3] - t[1])
        # Aspect ratio >= 1 (long side / short side), guarding tiny degenerate boxes.
        aspect = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1e-6)
        n_vertices = geoms.apply(lambda p: len(p.exterior.coords) - 1)

        geo = pd.DataFrame({
            "roomtype": area["roomtype"].values,
            "subtype": area["entity_subtype"].values,
            "area_m2": poly_area.values,
            "width_m": w.values,
            "height_m": h.values,
            "aspect": aspect.values,
            "n_vertices": n_vertices.values,
        })

        R("**Per-room geometry summary:**")
        R.table(geo[["area_m2", "width_m", "height_m", "aspect", "n_vertices"]]
                .describe().round(2))

        R("**Median area (m^2) by room type** (largest first):")
        by_type = (geo.groupby("subtype")["area_m2"]
                      .agg(["count", "median", "mean"])
                      .sort_values("median", ascending=False).round(2))
        R.table(by_type)

        R(f"Geometry types present: "
          f"{sorted(set(g.geom_type for g in geoms))}.  "
          f"Rooms are not all axis-aligned rectangles -- median vertex count "
          f"{int(geo['n_vertices'].median())}, so the decoder must tolerate "
          f"polygons, not just boxes.")
        return geo


# --------------------------------------------------------------------------- #
# 7. Vertical attributes
# --------------------------------------------------------------------------- #
def explore_vertical(df: pd.DataFrame) -> None:
    with section(2, "7. Vertical attributes (elevation / ceiling height)"):
        R("`elevation` = base height of the entity above ground (proxy for floor "
          "level); `height` = ceiling/entity height in metres.")
        R.table(df[["elevation", "height"]].describe().round(2))


# --------------------------------------------------------------------------- #
# 8. Figures
# --------------------------------------------------------------------------- #
def make_plots(df: pd.DataFrame, geo: pd.DataFrame) -> None:
    with section(2, "8. Figures"):
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless-safe backend
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            R(f"_matplotlib not available ({exc}); skipping plots._")
            return

        saved: list[str] = []

        # (a) entity_type composition
        fig, ax = plt.subplots(figsize=(5, 3.2))
        df["entity_type"].value_counts().plot.bar(ax=ax, color="#4c72b0")
        ax.set_title("Entity type composition")
        ax.set_ylabel("rows")
        _save(fig, "fig_entity_type.png", saved)

        # (b) rooms per apartment
        res = df[(df.unit_usage == "RESIDENTIAL") & (df.entity_type == "area")]
        rpa = res.groupby("apartment_id")["area_id"].nunique()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(rpa, bins=range(0, int(rpa.quantile(0.99)) + 2), color="#55a868")
        ax.set_title("Rooms per apartment")
        ax.set_xlabel("rooms")
        ax.set_ylabel("apartments")
        _save(fig, "fig_rooms_per_apartment.png", saved)

        # (c) top room types
        rt = df.loc[df.entity_type == "area", "entity_subtype"].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        rt[::-1].plot.barh(ax=ax, color="#c44e52")
        ax.set_title("Top room types (areas)")
        ax.set_xlabel("count")
        _save(fig, "fig_room_types.png", saved)

        # (d) geometry distributions (if available)
        if not geo.empty:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
            axes[0].hist(geo["area_m2"].clip(upper=60), bins=40, color="#8172b3")
            axes[0].set_title("Room area (m^2, clipped@60)")
            axes[0].set_xlabel("m^2")
            axes[1].hist(geo["aspect"].clip(upper=6), bins=40, color="#937860")
            axes[1].set_title("Room aspect ratio (clipped@6)")
            axes[1].set_xlabel("long/short")
            fig.tight_layout()
            _save(fig, "fig_geometry.png", saved)

        R("Saved figures:")
        for s in saved:
            R(f"- `{s}`")


def _save(fig, name: str, saved: list[str]) -> None:
    import matplotlib.pyplot as plt
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)


# --------------------------------------------------------------------------- #
# 9. Takeaways for the modelling task
# --------------------------------------------------------------------------- #
def write_takeaways() -> None:
    with section(2, "9. Takeaways for OutlineFlow"):
        R(textwrap.dedent("""\
            1. **Filter early.** Keep `entity_type == 'area'` for rooms; walls
               (`separator`) and openings (`opening`) are optional structure that
               can be ignored for the MVP token set.
            2. **Group by `apartment_id`** to form one training sample; drop rows
               where it is NULL (those are PUBLIC/shared space, not a dwelling).
            3. **Outline = union of the apartment's area polygons**
               (`shapely.unary_union`); rooms = the individual `area_id` polygons
               with their `entity_subtype`/`roomtype` label.
            4. **Token-set size is small and bounded** (~10 rooms median, ~30 max)
               -- a set-Transformer over <=~32 tokens is sufficient.
            5. **Rooms are polygons, not always rectangles** -- the geometry head /
               decoder must handle non-axis-aligned, multi-vertex shapes.
            6. **Normalise per apartment** (translate to centroid, scale by bbox)
               before training; raw coordinates span tens of metres with arbitrary
               offsets.
        """).rstrip())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="EDA for mds_V2_5.372k.csv")
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--rows", type=int, default=None,
                    help="cap rows loaded (default: full file)")
    ap.add_argument("--geom-sample", type=int, default=40000,
                    help="number of area polygons to parse for geometry stats")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    R.h(1, "EDA report -- mds_V2_5.372k.csv (MSD / Swiss Dwellings)")

    df = load_meta(args.csv, args.rows)

    explore_schema(df, args.csv)
    explore_missingness(df)
    explore_categoricals(df)
    explore_hierarchy(df)
    explore_apartments(df)
    geo = explore_geometry(args.csv, args.rows, args.geom_sample)
    explore_vertical(df)
    if not args.no_plots:
        make_plots(df, geo)
    write_takeaways()

    report_path = os.path.join(OUT_DIR, "EDA_REPORT.md")
    R.save(report_path)
    print(f"\n[done] report written to {report_path}")


if __name__ == "__main__":
    main()
