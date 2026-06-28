# OutlineFlow — Results

Conditional generation of typed interior **room polygons** from an apartment/floor
**outline** (the only condition), on the Modified Swiss Dwellings dataset, grouped by
`plan_id` per the brief. Scored on **FID** (lower better), **Density** and **Coverage**
(higher better, ideal 1.0). Fixed seed 42. Full data: 5,372 plans / 203k rooms.

All numbers below use the official metric (InceptionV3 FID + PRDC k=5) on MSD-rendered
images, `n_eval=600`. Configs marked **(3-seed)** are means over seeds {42,123,7};
single-seed numbers are flagged because this pipeline has real run-to-run variance.

---

## 1. Headline result

| Config | FID ↓ | Density ↑ | Coverage ↑ |
|--------|-------|-----------|------------|
| Raw baseline (flow matching, rect decode) | 167 | 0.097 | 0.067 |
| **Final: flow matching + align + churn0.3** (3-seed) | **135.3** | 0.088 | **0.111** |

Net: **FID −19%, Coverage +66%** over the raw baseline. Shipped in `generate.py`
(`generate(outline)` → list of `(shapely Polygon, room_type_id)`, deterministic at seed 42).

---

## Core insight

> **The bottleneck is the L2 training objective, not the model class or its capacity.**

Flow matching and diffusion both minimize an MSE that drives the network toward the
**conditional mean** of plausible layouts. Given one outline there are many valid room
arrangements, but the mean of them is a single blurred "average" plan — so the model's
outputs collapse in variance (room-count std ±6 vs real ±24) and miss most of the real
distribution (Coverage ≈ 0.1 vs 1.0). This is why **every fix that adds capacity or
changes the paradigm fails**: bigger/deeper/longer models, scale conditioning, loss
re-weighting, and a full DDPM all inherit the same averaging and do not move the metrics.

The **only robust way to recover diversity is sampling-time stochasticity** (`churn`,
i.e. SDE/ancestral sampling): injecting noise during integration lets each sample leave
the mean trajectory and spread back toward the real manifold — lifting Coverage with a
small FID cost, no retraining needed. EBM add-ons (guidance, energy-weighting, adversarial
fine-tune, best-of-N) look promising on a single seed but **wash out or interfere under
multi-seed evaluation** — which itself is a key lesson: with a stochastic pipeline,
single-run numbers are unreliable and every claim must be multi-seed averaged.

Practical ranking that held up: **align (FID) + churn (Coverage)** on a pure flow-matching
model beats diffusion and every EBM variant tried.

---

## 2. Model

From-scratch **rectified flow matching** (pure flow matching, no diffusion/EBM in the base):
- set-Transformer velocity field `v_θ(x_t, t, outline)`; room tokens have no positional
  encoding (a plan is a set); outline encoded by a permutation-invariant PointNet-lite;
  time+outline injected via DiT **AdaLN-Zero** blocks; full self-attention over tokens.
- Path `x_t=(1−t)·noise + t·data`, target velocity `v=data−noise`, weighted MSE.
- `d_model=128, L=4` (1.33M params), 15k steps. Bigger/deeper/longer did **not** help.
- Decode: oriented rectangles → clip to outline → greedy overlap-resolve → sliver-drop →
  gap-fill, so `union(rooms)==outline` (zero overlap, zero interior gap).

---

## 3. Diagnosis (the core bottleneck)

Geometry is clean and room counts are calibrated, yet **Density/Coverage ≈ 0.09** (ideal 1).
Root cause: **flow matching's L2 objective regresses to the conditional mean** → per outline
the model emits one near-average layout → variance collapse → poor coverage of real variety.

Evidence:
- Generated room-count std **±6** vs real **±24** (can't reach 15-room or 250-room plans).
- Room count correlates with outline area at **r≈0.88**, but `sample_outline_points`
  normalizes the boundary to [−1,1], **discarding absolute size** — the model literally
  cannot tell a 30 m² flat from a 300 m² floor.
- Not mode collapse: generated features are *more* dispersed than real (≈4×), i.e. scattered
  *outside* the real manifold, not clustered — so they miss real samples' k-NN balls.

---

## 4. What worked — three complementary levers

| Lever | Targets | Mechanism | Effect |
|-------|---------|-----------|--------|
| **1. Grid-align post-process** (`align_layout`, grid 16) | FID | Snap room edges to a grid along the building axes, then re-partition (overlap-resolve + gap-fill). Removes jitter/L-shapes → clean rectilinear tiling like real MSD. | **FID 167 → 131.5** |
| **2. SDE / `churn` sampling** (churn 0.3) | Coverage | Inject time-scaled noise during ODE integration (diffusion-style ancestral sampling). Escapes the mean-trajectory collapse. Strongest diversity lever; needs no extra training. | **Coverage 0.098 → 0.111** (3-seed) |
| 3. Adversarial/EBM fine-tune (`finetune_adv`) | Density | GAN-style: critic separates real vs model rollouts; generator fine-tuned to look real (FM-grounded). | Density up per-run, but **washes out under multi-seed** (kept as option, not default) |

Lever 2 (`churn`) is tunable but has a **robust sweet spot at ~0.3**; stronger churn is noisy:

| churn | FID | Density | Coverage |
|-------|-----|---------|----------|
| 0 (align only) | 131.5 | 0.091 | 0.098 |
| **0.3** (3-seed) | 135.3 | 0.088 | **0.111** |
| 0.6 (3-seed) | 138.8 | 0.083 | 0.097 |

> Note: churn 0.6 gave **Coverage 0.122 on seed 42**, but the 3-seed mean is only **0.097**
> (range 0.080–0.122). The 0.122 was a seed outlier — do not rely on it.

---

## 5. Paradigm comparison (the big question)

All three generative paradigms were implemented and evaluated head-to-head (align g16):

| Paradigm | FID | Density | Coverage | Verdict |
|----------|-----|---------|----------|---------|
| **Flow matching + churn0.3** | 135.3 | 0.088 | **0.111** | ✅ best |
| Flow matching (deterministic) | 131.5 | 0.091 | 0.098 | best FID |
| Flow matching + EBM guidance | 136.1 | 0.091 | 0.103 | marginal (within seed noise) |
| **DDPM (pure diffusion)** | 137.2 | 0.068 | 0.090 | ❌ worse on all three |

- **Diffusion (DDPM)**: full ε-prediction + cosine schedule + ancestral sampling, same
  backbone, 15k steps. Hits the **same L2 mean-regression ceiling** and is tuned worse than
  our flow matching (Density 0.068). At 100 steps ancestral≈DDIM (little per-step noise),
  so coverage stays low. Not competitive.
- **EBM** (energy-based): the critic learns real-vs-fake at ~92% acc, but every EBM use is
  marginal-to-negative under multi-seed (see §6). The base model is pure flow matching;
  EBM only ever acted as an auxiliary signal (spec-compliant).

---

## 6. What did NOT work (negative results)

Rigorously tested; none beats the simple recipe (most washed out under multi-seed):

- **Capacity**: bigger `d256/L8` (FID 141 w/ align), deeper `L8` (135), longer 30k — no gain.
  Lower train loss, same/worse metrics → capacity is *not* the bottleneck.
- **Voronoi decoder**: FID 268 — much worse (real rooms are rectangular).
- **Loss weights**: `w_type=2.0` did not fix the type distribution (the missing "Structure"
  class is a decoder/sliver-drop artifact, and renders black anyway → irrelevant to FID).
- **Scale injection** (feed outline size): fixed the count↔area *direction* (r 0.06 → 0.63)
  but **not the variance** — metrics flat. L2 still regresses each prediction to its mean.
- **Energy-weighted FM (EWFM)**: re-weighting rare tails can't beat mean-regression. No gain.
- **Forcing fewer rooms**: monotonically worse (FID 161→290 as target count drops).
- **Energy guidance**: 3-seed mean Cov 0.103 vs 0.098 — within noise; dominated by churn.
- **Energy guidance + churn** (matched critic, 3-seed): actively *interferes* — Cov 0.102
  (guid 0.15) / 0.094 (guid 0.25), both < churn-alone 0.111; more guidance = worse.
- **Best-of-N critic re-rank**: no gain (and OOM at high N).

**Takeaway:** the bottleneck is the L2 training objective, not model class or capacity.
The only thing that robustly buys diversity is **sampling-time stochasticity (churn)**.

---

## 7. Pareto frontier (pick by metric weighting)

| Priority | Config | FID | Density | Coverage |
|----------|--------|-----|---------|----------|
| Lowest **FID** | grid12 + churn0.4 (3-seed) | **129.0** | 0.082 | 0.099 |
| **Balanced** (default) | align g16 + churn0.3 (3-seed) | 135.3 | 0.088 | **0.111** |
| Best **FID+Density** | align g16 (deterministic) | 131.5 | 0.091 | 0.098 |

---

## 8. Final submission

`generate(outline)` = full plan_id flow-matching model + **align (grid 16)** + **churn 0.3**
+ count-calibrated presence threshold. Deterministic at seed 42; returns vector polygons
with `union(rooms)==outline`. Robust scores **FID ≈ 135 / Density ≈ 0.088 / Coverage ≈ 0.111**.

Key files: `model.py` (backbone), `flow.py` (FM loss + ODE/SDE sampler), `train.py`,
`postprocess.py` (`align_layout`), `sample_eval.py` (metrics), `generate.py` (entry point),
`diffusion.py`/`train_diffusion.py` (DDPM baseline), `energy.py`/`finetune_adv.py` (EBM).
