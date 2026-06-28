# OutlineFlow: Full Diagnostic Analysis
**Date:** 2026-06-27 20:39:47  
**Checkpoint:** `outputs/ckpt.pt` (plan_id grouping, n_max=140, 5000 steps, 1.33M params)  
**Eval run:** phi-proxy, n=120, rect decoder, count-calibrated

---

## Key Diagnostic Facts (from actual eval run)

```
[calib] target 36.3 -> presence_thresh -1.031   # threshold fell below padding baseline -1
[diag] rooms/plan  real 36.27±20.13  gen 31.44±6.03   # count variance collapses: 20 → 6
[diag] coverage 1.000  overlap 0.0007            # forced by deterministic postprocessor
===== SCORES (phi-proxy) =====
  FID      3682.407
  Density  0.000
  Coverage 0.000
```

Reported Density=0.336 likely comes from a different run (unit_id or larger n_eval);
root causes are identical regardless of the exact number.

Visual evidence from `outputs/samples/pair_000.png`:
- Generated plan (right): large green background + scattered small rectangles  
- This is not a floor plan — it is postprocessor gap-fill masking complete layout collapse

---

## 1. Input Representation Design

### Token format (`params.py` — single source of truth)
Each room = `D = 7 + K = 20`-dim vector:
```
[0]      presence       +1 = real room, -1 = padding
[1]      cx             centroid x   (standardized)
[2]      cy             centroid y   (standardized)
[3]      w              oriented-rect side 1 (standardized)
[4]      h              oriented-rect side 2 (standardized)
[5]      sin(2θ)        double-angle orientation (π-periodic, correct for rectangle symmetry)
[6]      cos(2θ)
[7:20]   type one-hot   rescaled to {-1, +1}
```

Geometry `cx,cy,w,h` is normalized to the outline-bbox frame, then z-scored with
per-channel `stats` to unit variance. The `sin2θ/cos2θ` double-angle encoding is
correct (matches rectangle π-symmetry).

### Outline encoding (`model.py:32-47`)
- 128 boundary points sampled by arc-length, each `(x_n, y_n, nx, ny)` (normalized coords + outward normal)
- PointNet-lite: shared MLP `4→128→128→128`, then **max-pool ⊕ mean-pool → one 128-dim global vector**
- **This single global vector is added to the time embedding and injected into all tokens via AdaLN-Zero**

**Critical limitation:** the outline enters the model as one pooled vector — no cross-attention between room tokens and boundary points. The model knows "outline is roughly this big and shaped like this" but **cannot spatially assign tokens to regions of the boundary**. This is the structural ceiling for Density.

### n_max=140 empty slots
- `build_x1` fills all slots with presence=-1 (padding), then randomly places real rooms in `rng.permutation(n_max)[:n]` slots
- plan_id average 37 rooms → **~73% of slots are padding** — extreme class imbalance on the presence channel
- Two hidden bugs:
  1. Padding slots have type channels = 0 while real rooms have ±1 (line 164-165 only writes ±1 for real slots) → inconsistent targets in the noisy flow path
  2. `build_tensors` calls `build_x1` **once before training** (`train.py:85`), so every plan sees the **same fixed random slot assignment for all 5000 steps** — the intended permutation-invariance augmentation never actually happens

### Type encoding
- 13 classes (9 dwelling + 4 structural/opening), one-hot in {-1, +1}
- Only first `n_gen_classes=10` classes are generated (doors/windows excluded via argmax truncation in `decode_x1`)
- Encoding is consistent across train/sample/eval (locked to MSD taxonomy)

### Ambiguity / limitations
- **Slot permutation fixed per epoch** (see above) — biggest augmentation waste
- No rotation/flip augmentation of the plan itself
- MRR approximation: non-rectangular rooms (L-shapes, etc.) are approximated as oriented rectangles → IoU loss on decode

---

## 2. Model Architecture

### DiT specs (from checkpoint)
| Parameter | Value |
|---|---|
| `d_model` | 128 |
| `n_layers` | 4 |
| `n_heads` | 4 |
| `mlp_ratio` | 4 |
| Total params | **1.33M** |

### Time step t injection
`GaussianFourierProjection(t) → 2-layer MLP → t_embed [B,128]`

### Outline condition injection
`OutlineEncoder(boundary_pts) → 128-dim global vector`  
Combined: `c = SiLU(t_embed + outline_enc)` → fed to each `AdaLNZeroBlock` as conditioning.  
AdaLN-Zero: `Linear(128 → 6×128)` zero-initialized → training starts as identity → stable.

**No per-token spatial conditioning. Every token sees the same `c`.**

### Parameter count vs dataset size
- 1.33M params / 4000 training plans / 5000 steps ≈ 40 effective epochs
- For plan_id complexity (37±20 rooms, MultiPolygon floors) this is **underpowered, not overfitting**
- msd_unit checkpoint (n_max=23, ~9 rooms) is a much better fit for 1.33M params

### Architecture defects (by impact)
1. **No outline cross-attention** — structural ceiling, cannot spatially fill the floor
2. **Count distribution collapse** (gen std 6 vs real std 20) — global pooled `c` cannot encode room count
3. **Presence threshold pathology** — calibration pushes threshold to -1.031 (below padding=-1), meaning noise slots are kept and gap-fill covers the mess
4. Final output norm uses **affine-free LayerNorm + AdaLN from `c`** (correct pattern), but with only 4 layers there is limited capacity to learn spatial layout from a single conditioning vector

---

## 3. Training Setup

### Current settings
```python
steps       = 5000
batch_size  = 128
lr          = 2e-4  (cosine decay, 200-step warmup)
ema_decay   = 0.999
w_presence  = 2.0
w_geometry  = 1.0
w_type      = 0.5
w_pad_slot  = 0.3   # down-weight padding slots
msd_limit   = 2000  # only 2000 plans read from CSV (subset!)
n_train     = 4000  # but actual unique plans ~1600 after held split
```

### Is 5000 steps enough?
**No.** And more critically: only `--msd_limit=2000` unique plans are loaded, yielding ~1600 training plans and ~400 held. The methodology doc itself admits "trained on a data SUBSET." The full CSV has ~5372k entity rows spanning many more plans.

### Is batch_size=128 reasonable?
Yes, not a bottleneck.

### Data augmentation
**Essentially none:**
- Slot permutation is fixed per plan before training (see above)
- No geometric augmentation (rotation, flip, scale jitter)
- No outline perturbation

The set-representation was designed for permutation-invariance but the training loop never exploits it.

### Overfitting risk
**Low — the problem is underfitting and insufficient conditioning.**
No train/val loss curves are logged, but visual output quality (off-manifold collapsed layouts) indicates the model never learned to tile the floor, not that it memorized training examples.

---

## 4. Root Cause of Low Density

### What Density measures
Density (from clovaai PRDC, k=5) = fraction of generated samples that fall within
the k-NN radius of at least one real sample in feature space.  
A score near 0 means **generated plans are off the real manifold entirely**.

### Causal chain
```
Outline → single 128-dim pooled vector
         ↓ (no spatial detail reaches tokens)
Tokens cannot learn to fill specific regions of the floor
         ↓
Model learns a "average room-count, average size" distribution
         ↓
Count variance collapses: real 37±20, gen 31±6
         ↓
Calibration threshold drops to -1.031 (below padding baseline)
         ↓
Noise slots included as rooms → layout is incoherent
         ↓
Gap-fill postprocessor merges leftover area into "nearest room"
         ↓
Visual result: large solid-color blobs (see pair_000.png, right panel)
         ↓
These blobs are far from real plans in inception/phi feature space → Density ≈ 0
```

### The one most important fix
> **Add cross-attention from room tokens to the 128 boundary points** (tokens attend to boundary as K/V).

This is the only structural change that simultaneously fixes:
- Spatial filling (tokens can "see" which part of the boundary they are near)
- Count variability (token activation correlates with local boundary complexity)
- Type diversity (bedrooms vs corridors can be anchored to outline regions)

All other improvements operate under the same ceiling.

---

## 5. Prioritized Improvement Plan

| Priority | Change | Expected Impact | Cost |
|---|---|---|---|
| **P0** | **Outline cross-attention**: after each AdaLN self-attn block, add one cross-attn layer where room tokens attend to boundary points (128 pts as K/V). Keep AdaLN for time `t` injection. | Lifts structural ceiling; fixes spatial filling and count variability | Medium — `model.py` only |
| **P0** | **Per-step slot permutation**: move `build_x1` random permutation inside the training loop (reshuffle each batch), not in `build_tensors`. True permutation-invariance augmentation for free. | Strong regularization, near-zero cost | Low — `train.py` only |
| **P1** | **Switch to unit_id as validation baseline**: n_max=23, ~9 rooms, lower padding ratio — confirms pipeline is correct before tackling plan_id difficulty | Fast A/B, isolates model vs task difficulty | Low |
| **P1** | **Use full data + longer training**: remove `--msd_limit 2000` cap, go to steps=15k+, larger batch if memory allows | Addresses clear underfitting on subset | Low (compute) |
| **P1** | **Fix padding type targets**: padding slots should have all type channels = -1 (not 0), consistent with the {-1,+1} encoding used for real slots | Eliminates inconsistent targets in the flow | Low — `params.py` only |
| **P2** | **Explicit room-count conditioning**: predict/regress room count from outline area, then condition the model on it; alternatively block-out presence threshold below -0.5 at eval to prevent noise-slot inclusion | Fixes count collapse and calibration pathology | Medium |
| **P2** | **Classifier-free guidance**: randomly drop outline condition during training (p=0.1); use guidance scale >1 at sampling | Improves outline fidelity → higher Density | Medium |
| **P2** | **Multiple samples per outline at eval** (if time budget allows): take N samples per outline, union room-type distributions → improves Coverage | Directly raises Coverage score | Low |
| **P3** | Geometric augmentation: random rotation/flip of entire plan before tokenization | Marginal diversity gain | Low |
| **P3** | Evaluate voronoi vs rect decoder per plan using MRR-IoU gate (tooling already in `msd_data.py` `__main__`) | May improve visual quality for non-rectangular rooms | Low |

### Implementation order recommendation
1. **P0 cross-attention** (`model.py`) + **P0 slot permutation** (`train.py`) → retrain on unit_id to validate
2. Add full data + longer training → retrain on plan_id
3. Add count conditioning or CFG depending on which metric lags

---

## Reference: File Map

| File | Role |
|---|---|
| `params.py` | Token encoding/decoding, outline sampling, stats — single source of truth |
| `model.py` | OutlineFlow DiT, OutlineEncoder (PointNet-lite), AdaLNZeroBlock |
| `flow.py` | Rectified flow loss (`fm_loss`), Euler+Heun sampler, EMA |
| `cfg.py` | Config dataclass, room taxonomy, palette |
| `train.py` | Training loop, data loading dispatch |
| `msd_data.py` | Real MSD CSV reader, outline construction |
| `synth_data.py` | Synthetic plan generator (recursive binary split) |
| `postprocess.py` | Deterministic decoder: clip→overlap-resolve→gap-fill (rect + voronoi) |
| `metrics.py` | phi-proxy + official InceptionV3 FID/Density/Coverage |
| `sample_eval.py` | End-to-end eval: sample → decode → score → optional PNG pairs |
| `generate.py` | `generate(outline)` submission entry point |
