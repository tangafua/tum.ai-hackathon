# OutlineFlow — Methodology (DAVIS × PARIS 2026 submission writeup)

**Task.** Conditional generation of an apartment's interior rooms as **vector
polygons**, given **only the apartment outline**, with a **flow-matching** model,
scored on **FID + density & coverage** against real Swiss floor plans (MSD).

This document is the short methodology writeup required by the brief. Full design
rationale is in [../plan.md](../plan.md); the run commands are in
[../README.md](../README.md).

---

## 1. How each brief requirement is met

| Brief requirement | Where |
|---|---|
| **Diffusion / flow-matching model** | Rectified flow (conditional-OT), [flow.py](flow.py) |
| **Trained from scratch** | Model + loss written from scratch ([model.py](model.py), [flow.py](flow.py)); no pretrained weights |
| **Condition = the outline only** (no rooms / walls / graph) | The network sees *only* outline boundary points (PointNet-lite → AdaLN-Zero), [model.py](model.py) `OutlineEncoder` |
| **Output = typed vector room polygons** (never a pixel grid) | Room tokens → shapely polygons, [postprocess.py](postprocess.py); never rasterized except to score |
| **Data = `mds_V2_5.372k.csv`, `geom` (WKT), `entity_type=='area'`** | [msd_data.py](msd_data.py) reads exactly these columns; the struct_in/graph folders are never touched |
| **Outline = buffer(+0.3)·union·buffer(−0.3)** | [msd_data.py](msd_data.py) `build_outline` — the brief's formula verbatim |
| **FID + density & coverage** | [metrics.py](metrics.py) (`torchmetrics` FID + vendored clovaai `prdc`, k=5) |
| **Fixed seed 42 throughout** | `seed_everything(42)` ([cfg.py](cfg.py)) called in train / sample_eval / generate |
| **`generate(outline)` entry point returning room polygons** | [generate.py](generate.py) `generate(outline) -> [(Polygon, type_id), ...]` |
| **Training + generation code, weights, writeup** | this repo; weights at `outputs/ckpt.pt`; this file |

## 2. Representation

A plan is a **set of room tokens** `x ∈ ℝ^[N_max, 7+K]`: per room `presence`,
oriented-rectangle `(cx, cy, w, h)`, orientation `(sin2θ, cos2θ)`, and a type
one-hot. Geometry is expressed in the outline-bbox frame and standardized to unit
variance ([params.py](params.py)). This is a **vector / parametric** representation —
not a pixel grid — as the brief requires. Round-trip `polygon → token → polygon`
reproduces rectangular rooms at IoU ≈ 1.

## 3. Model & training

A DiT-style **set Transformer** velocity field `v_θ(x_t, t, outline)`:
permutation-invariant (no positional encoding), full self-attention over tokens
(so the model learns a *joint*, non-overlapping, outline-filling layout), with the
outline injected through zero-initialized AdaLN-Zero (training starts as the
identity — the most stable conditioning). **Rectified flow**: noise at `t=0`, data
at `t=1`, path `x_t=(1−t)x_0+t x_1`, target velocity `v=x_1−x_0`; presence is
supervised on every slot so padding slots don't drift. AdamW + cosine LR + EMA.

## 4. Sampling & decode

Euler ODE `t:0→1` (100 steps, Heun corrector near `t=1`) with EMA weights, then a
**deterministic decoder** turns tokens into a valid layout:

- **rect** (default): the model's predicted oriented rectangles, with each angle
  **θ-snapped to the outline's own axes** (real Swiss apartments are rectilinear,
  and the building can sit at any global angle) → clip to outline → overlap-resolve
  → sliver-drop → mandatory gap-fill. Produces axis-aligned **rectangular** rooms
  that look like real floor plans.
- **voronoi** (fallback): partition the outline by Voronoi cells around predicted
  room seeds — gap-free and count-faithful but **non-rectangular**.

Both **guarantee `union(rooms) == outline`** (zero overlap, zero interior gap),
which directly protects coverage. At eval, one global presence threshold is
**count-calibrated** (binary-searched on the *final decoded* count) so the generated
room-count distribution matches real. `generate(outline)` ([generate.py](generate.py))
chains outline → sample → decode and returns the room polygons.

## 5. Grouping (plan_id vs unit_id)

The brief appendix groups rooms by **`plan_id`** (and the organizers score on a
held-out set of *plans*), so that is the **default**. In MSD a `plan_id` is a whole
building floor (≈31 rooms, several apartments fused by the 0.3 m buffer into one
shell); `--group unit_id` instead trains/generates per single apartment (≈8 rooms),
which is more tractable and visually cleaner. Both paths use the identical
sample contract and the brief's outline formula.

## 6. Evaluation

`sample_eval.py` renders real and generated plans with the **same** function
(MSD palette on black, [render.py](render.py) `render_msd`) and reports FID via
`torchmetrics.FrechetInceptionDistance` (2048-d) and density/coverage via the
vendored clovaai `prdc` sharing the same Inception features. A numpy-only `phi`
proxy reproduces the whole pipeline with zero extra installs, behind a sanity gate
(real-vs-real ⇒ FID≈0/D≈1/C≈1; noise ⇒ large/0/0).
