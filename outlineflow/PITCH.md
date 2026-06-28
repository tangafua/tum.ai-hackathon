---
marp: true
theme: default
paginate: true
size: 16:9
---

<!--
5-minute pitch. ~30s/slide. Speaker notes are in these HTML comments.
Render: `marp PITCH.md --pdf` (or open in the Marp VS Code extension).
-->

# OutlineFlow
## From a bare outline → a full apartment, as vector polygons

**Mirror Mirror on the Wall — DAVIS × Paris 2026**
Conditional layout generation on Modified Swiss Dwellings

<!-- 0:00–0:20 — Hook: "Give us only the outer wall of an apartment, we generate a
complete, plausible set of typed rooms — as vector polygons, not pixels." -->

---

## The task

- **Input:** one apartment/floor **outline** (the *only* condition)
- **Output:** a set of **typed room polygons** (vector, never a pixel grid)
- **Model class:** diffusion or flow matching, trained from scratch, seed 42
- **Scored on:** **FID** (realism, ↓) + **Density** & **Coverage** (diversity, ↑)

> Data: Modified Swiss Dwellings — full set, **5,372 plans / 203k rooms**.

<!-- 0:20–0:50 — Frame it as conditional generation. Stress: scored on BOTH realism
(FID) and diversity (density/coverage) — that tension drives our whole story. -->

---

## Our model — pure flow matching

A from-scratch **rectified flow** set-Transformer velocity field `v_θ(x_t, t, outline)`:

- rooms are a **set** → no positional encoding; full self-attention learns a *joint*,
  non-overlapping, outline-filling layout
- outline encoded by a permutation-invariant PointNet; time+outline via **DiT AdaLN-Zero**
- decode → clip to outline → resolve overlaps → gap-fill, so `union(rooms) == outline`

**1.33M params, trained on all 5,372 plans.**

<!-- 0:50–1:20 — Keep architecture brief. Emphasize it's clean, from-scratch flow
matching, and the decoder guarantees a valid gap-free tiling. -->

---

## First results — and a wall

| | FID ↓ | Density ↑ | Coverage ↑ |
|---|---|---|---|
| Baseline | 167 | 0.097 | **0.067** |
| *ideal* | 0 | 1.0 | 1.0 |

Geometry is clean, room counts are right… but **Density & Coverage ≈ 0.07–0.09**.
The model is **not diverse enough** — it keeps drawing the *same average* apartment.

<!-- 1:20–1:50 — The turning point: naive training plateaus. Coverage 0.07 means we only
reach ~7% of the real layout variety. Set up the diagnosis. -->

---

## Diagnosis — it's the objective, not the model

> **L2 training regresses to the *conditional mean* of all valid layouts.**

- Many valid plans per outline → their mean is one blurred "average" → **variance collapse**
- Generated room-count std **±6** vs real **±24** (can't do tiny or huge plans)
- Room-count ↔ area correlates at **r = 0.88**, but the encoder normalizes size away

**This predicts what will and won't work.**

<!-- 1:50–2:30 — The core scientific insight. This is the most important slide. Everything
downstream follows from "MSE → mean → no diversity". Say it slowly. -->

---

## Three complementary levers

| Lever | Targets | Idea |
|---|---|---|
| **1 · Grid-align** (post-process) | **FID** | snap rooms to building axes → clean rectilinear tiling |
| **2 · SDE / churn sampling** | **Coverage** | inject noise during sampling → escape the mean trajectory |
| 3 · Adversarial / EBM fine-tune | Density | critic pushes samples toward the real manifold |

Each attacks a **different metric** — and lever 2 is the key: it directly undoes the
variance collapse, *no retraining needed*.

<!-- 2:30–3:10 — Map each lever to the metric it fixes. Lever 2 (churn = diffusion-style
stochastic sampling) is the hero — it's the one fix that addresses the root cause. -->

---

## Results

| Config (3-seed mean) | FID ↓ | Density ↑ | Coverage ↑ |
|---|---|---|---|
| Baseline | 167 | 0.097 | 0.067 |
| + grid-align | **131.5** | 0.091 | 0.098 |
| **+ churn (final)** | 135.3 | 0.088 | **0.111** |

**FID −19%, Coverage +66%.** Shipped as `generate(outline)`.

![w:560](SUBMISSION_generate_demo.png)

<!-- 3:10–3:50 — Show the numbers + the demo image (left=real, right=ours). Point at how
generated plans fill the outline with clean, varied rooms. -->

---

## We tested all three paradigms — honestly

| Paradigm (align g16) | FID | Density | Coverage |
|---|---|---|---|
| **Flow matching + churn** | 135 | 0.088 | **0.111** |
| DDPM (pure diffusion) | 137 | 0.068 | 0.090 |
| Flow + EBM guidance | 136 | 0.091 | 0.103 |

- Diffusion hits the **same L2 ceiling** — no better.
- EBM gains **wash out under multi-seed** (a churn run of 0.122 averaged to 0.097!).

<!-- 3:50–4:30 — Rigor sells. We didn't just try one thing — we implemented diffusion AND
EBM from scratch and benchmarked them. The 0.122→0.097 story shows multi-seed discipline. -->

---

## Key takeaways

1. **The bottleneck is the loss, not the architecture** — capacity, depth, paradigm swaps all fail.
2. **Sampling-time stochasticity is the real diversity lever** — simple, robust, free.
3. **Multi-seed or it didn't happen** — stochastic pipelines make single runs lie.

**Final:** flow matching + grid-align + churn → **FID 135 / Den 0.088 / Cov 0.111**,
deterministic `generate(outline)` at seed 42.

<!-- 4:30–5:00 — Land the three transferable lessons, restate the final config + entry
point, and stop. Thank the judges. -->
