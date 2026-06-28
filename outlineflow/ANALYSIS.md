# Two key ablations (our data)

Both use the full plan_id flow model, align g16, churn 0.3 sampling.
Baseline / reference rows are 3-seed means; the cross-attn and sampling-hack rows are
single-seed (still decisive given the size of the gaps). The official metric
(InceptionV3 FID + PRDC k=5), n_eval=600.

---

## 1. Conditioning is NOT the bottleneck — the L2 loss is

We replaced the single pooled outline vector with **cross-attention over all 128 boundary
points**, and added **absolute scale**. This *demonstrably un-compresses* the condition:
the model's outline-reading (count↔area correlation) rises from **0.06 → 0.67** (real 0.94).
**Yet Coverage does not improve — it dips.** Opening the conditioning bottleneck completely
changed nothing → the limit is the averaging objective, not how well the model reads the outline.

| Conditioning | count↔area r | FID ↓ | Density ↑ | Coverage ↑ |
|---|---|---|---|---|
| Baseline (single pooled vector) | 0.06 | 135.3 | 0.088 | **0.111** |
| + cross-attn (shape detail) | 0.09 | 134.8 | 0.087 | 0.098 |
| + cross-attn + scale (shape + size) | **0.67** | 142.6 | 0.078 | 0.098 |

![conditioning decisive test](fig_conditioning_decisive.png)

> r jumps 10×, the metric is flat/worse → **bottleneck = L2 conditional-mean regression.**

---

## 2. Sampling-side diversity hacks all fall off the manifold

Beyond `churn`, three more-aggressive ways to inject diversity were tried. All are **worse on
every metric**: they push samples off the model's learned manifold, so the decoder gets
invalid tokens → degenerate plans. `churn` works only because it is gentle, schedule-matched
noise that stays near the manifold.

| Method | FID ↓ | Density ↑ | Coverage ↑ |
|---|---|---|---|
| **churn 0.3 (reference, best)** | **135.3** | **0.088** | **0.111** |
| (1) per-outline count ~ p(count\|area) + churn | 153.9 | 0.082 | 0.103 |
| (1) per-outline count (no churn) | 151.2 | 0.073 | 0.080 |
| (2) particle repulsion (strength 1) + churn | 158.1 | 0.063 | 0.075 |
| (2) particle repulsion (strength 3) + churn | 189.1 | 0.025 | 0.030 |
| (3) noise temperature 1.15 + churn | 144.1 | 0.075 | 0.107 |
| (3) noise temperature 1.3 + churn | 155.8 | 0.055 | 0.080 |

![sampling off-manifold](fig_sampling_offmanifold.png)

> Every alternative sits below-left of churn (worse FID *and* Coverage) → **sampling-time
> diversity is already saturated by churn.**

---

### Combined takeaway

Model side (capacity, depth, conditioning, paradigm, EBM) and sampling side (churn,
temperature, repulsion, count-forcing, guidance, re-rank) are both exhausted. The bottleneck
is the L2 / mean-regression objective; `align + churn0.3` is the practical ceiling of this
flow-matching + rectangle-decode approach.
