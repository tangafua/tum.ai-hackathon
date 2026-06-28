# Area-level (plan_id) re-evaluation of the unit-trained model

Same unit-trained checkpoint, NO retraining. Each floor (`plan_id`) is scored by
**merging all of its units** (each generated per-unit by the unit model) into one
layout, then computing FID / Density / Coverage over whole floors.

- Checkpoint: `outputs_unit/ckpt.pt`  (unit-trained)
- Backend: **OFFICIAL torchmetrics+prdc**
- Command:

```
python area_eval.py --out_dir outputs_unit --data_csv /root/.cache/kagglehub/datasets/caspervanengelenburg/modified-swiss-dwellings/versions/6/mds_V2_5.372k.csv --n_eval 500 --device cuda --guidance 1.5 --decoder rect --count_match --count_scale 1.0 --inception
```

## Setup

- Floors (areas) evaluated: **500**
- Units merged: **2550**  (5.10 units/floor)
- held->plan_id signature match rate: 100.0%
- Contamination (units the unit model trained on): **77.1%** (held-out purity caveat -- biases area numbers optimistically)
- rooms/floor: real 46.2±40.5 | gen 34.9±32.7

## Results

Sanity (area real-vs-real): FID -0.000 / D 1.000 / C 1.000 — metric wiring verified.

| Level | FID ↓ | Density ↑ | Coverage ↑ |
|---|---|---|---|
| **Area (plan_id, merged) — 500 floors** | **148.994** | **0.242** | **0.184** |
| Unit (same 2550 units, un-merged, ref) | 95.231 | 0.045 | 0.050 |
| Published unit best (500 held units, C1 g1.5)¹ | 109.8 | 0.118 | 0.132 |

¹ From `方案进展_metrics.md` §7.z (same backend). It is on a **different, smaller**
sample (500 held units, no contamination), so it is context only — not strictly
comparable to the rows above (FID/Density/Coverage all shift with sample size and set).

## Reading the numbers

- **Merging units into floors raises Density/Coverage but worsens FID** vs the same units
  scored individually (0.242/0.184 vs 0.045/0.050; FID 149 vs 95). The floor renders are
  busier, multi-apartment images that sit closer together in Inception space (higher D/C),
  while the systematic per-unit room loss compounds over ~5 units/floor — **gen floors
  carry 34.9 rooms vs 46.2 real** (≈‑2 rooms/unit × 5 units) — which is what drives FID up.
- Because each unit is generated independently and conditioned only on its own outline,
  there is **no cross-apartment structure** on the floor beyond what the geometric union
  produces; this caps how close merged floors can get to real floor distributions.

## Caveats

- **Held-out purity**: floors are completed by pulling *all* of their units from
  the CSV, so some units were in the unit-training split (contamination above).
  The area numbers are therefore optimistic relative to a clean held-out floor set.
- **Merge geometry**: each unit's rooms are generated independently (conditioned on
  that unit's own outline) and placed in the shared floor coordinate frame; the
  floor outline is the brief union (`buffer(+0.3).unary_union.buffer(-0.3)`) of the
  real rooms. There is no cross-unit layout interaction -- the model never sees the
  whole floor, so inter-apartment structure is whatever the union produces.
- **Residential units only**: public space has a null `unit_id` and is excluded, so
  the merged floor is the union of the apartments (slightly tighter than the brief's
  full-plan outline that also folds in public areas).
