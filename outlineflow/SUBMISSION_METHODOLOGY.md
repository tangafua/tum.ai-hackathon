# OutlineFlow — Methodology

**Task.** Given only an apartment/floor outline (one polygon, built by the brief's
`buffer(0.3).union.buffer(-0.3)` script, grouped by `plan_id`), generate a complete
set of typed interior rooms as vector polygons. Scored on FID + Density + Coverage
against held-out real Swiss plans (fixed seed 42).

## Model

A from-scratch **rectified-flow** set-Transformer velocity field
`v_θ(x_t, t, outline)`:
- room tokens carry no positional encoding (a plan is a *set*);
- the outline is encoded by a permutation-invariant PointNet-lite over boundary points;
- time + outline are injected via DiT-style **AdaLN-Zero** blocks (identity at init);
- full bidirectional self-attention over room tokens learns a *joint*,
  non-overlapping, outline-filling layout.

Each room token = presence, (cx,cy,w,h) standardized, (sin2θ,cos2θ), type one-hot.
Decoding clips oriented rectangles to the outline, greedily resolves overlaps, drops
slivers, and gap-fills so `union(rooms) == outline` (zero overlap / zero interior gap).

Trained on the **full** dataset: 5,372 `plan_id` plans (203k room polygons), 15k steps,
seed 42. `d_model=128, L=4` (1.33M params) — bigger/deeper models did **not** help.

## Diagnosis (what was limiting the score)

Calibrated room counts and clean geometry, yet **Density/Coverage ≈ 0.09** (ideal 1.0).
Root cause: flow-matching's **L2 objective regresses to the conditional mean**, so per
outline the model emits one near-average layout → variance collapse → poor coverage of
the real layout variety. Confirmed: generated room-count std ±6 vs real ±24; and the
outline encoder normalises the boundary to [-1,1], discarding absolute size (room count
correlates with area at r≈0.88, but the model couldn't see it).

## Three levers that worked (and where each helps)

1. **Grid-align post-process** (`align_layout`, FID). Snap room edges to a grid along
   the building's axes, then re-partition (overlap-resolve + gap-fill). Removes jitter /
   L-shapes → cleaner rectilinear tiling like real MSD. **FID 167 → ~131.**
2. **SDE / stochastic sampling** (`churn`, Coverage). Inject time-scaled noise during
   ODE integration (diffusion-style ancestral sampling). Escapes the mean-trajectory
   collapse. **Coverage 0.098 → 0.11–0.12** at small FID cost; this was the single
   strongest diversity lever.
3. **Adversarial / negative-unlearning fine-tune + EBM** (`finetune_adv`, Density).
   GAN-style: a time-conditioned critic separates real vs the model's own rollouts; the
   flow generator is fine-tuned to make a short differentiable rollout look real (FM loss
   keeps it grounded). Improves Density per single run, though the gain partly washes out
   under multi-seed averaging (kept as an option, not the default).

The generator stays a flow-matching model throughout (spec-compliant); the energy critic
is only a training/sampling signal.

## What did NOT help (negative results)

Bigger (`d256/L8`) or deeper (`L8`) or longer (30k) models; the Voronoi decoder;
up-weighting the type loss; injecting absolute outline scale (fixed the count↔area
*direction*, r 0.06→0.63, but not the variance, so metrics flat); energy-weighted FM
(re-weighting tails can't beat mean-regression); energy *guidance* and best-of-N
re-ranking (both dominated by `churn`). Takeaway: the bottleneck is the L2 objective, not
capacity — sampling-time stochasticity is what moves coverage.

## Final configuration (robust, 3-seed mean)

`generate(outline)` = full plan_id model + **churn 0.3** + **grid-align (grid 16)** +
count-calibrated presence threshold. Robust scores: **FID ≈ 135, Density ≈ 0.088,
Coverage ≈ 0.111** (vs raw baseline FID 167 / Cov 0.067).

Pareto alternatives by metric priority:
- lowest **FID** (~129): grid 12 + churn 0.4;
- highest **Coverage** (~0.122): churn 0.6.

Entry point: `from generate import generate; rooms = generate(outline)` →
`list[(shapely Polygon, room_type_id)]`, deterministic at seed 42.
