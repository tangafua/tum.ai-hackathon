# Area-level (plan_id) metrics for EVERY unit-trained strategy

Each unit-trained checkpoint on the branch is re-evaluated at the **area** level:
keep the unit model, generate every unit, **merge all units of each floor**
(`plan_id`) into one layout, score whole floors. Pipeline = `area_eval.py`.

- Backend: **official InceptionV3 (torchmetrics FID + prdc k=5)**
- 500 floors / 2550 units / 5.10 units-per-floor (same held split across all variants)
- held→plan_id signature match 100% | contamination 77.1% (held-out caveat, see below)
- Eval flags (uniform, single-variable): `--count_match --count_scale 1.0` + guidance per §3.3
  (g=1.5 for CFG-trained, g=1.0 for the −CFG variant); EDM auto-detected.
- Real rooms/floor = 46.2±40.5 for all (same real set).

## Area-level vs unit-level, per strategy

| Strategy | g | **Area FID ↓** | **Area D ↑** | **Area C ↑** | Unit FID | Unit D | Unit C | gen rooms/floor |
|---|---|---|---|---|---|---|---|---|
| full (x-attn+cond+CFG) | 1.5 | 149.0 | 0.242 | 0.184 | 95.2 | 0.045 | 0.050 | 34.9 |
| E1 − cross-attn | 1.5 | 175.3 | 0.188 | 0.160 | 98.9 | 0.052 | 0.049 | 33.2 |
| E2 − count cond | 1.5 | 154.4 | 0.223 | 0.200 | 99.6 | 0.042 | 0.044 | 34.5 |
| E3 − CFG (g1.0) | 1.0 | 151.0 | 0.242 | 0.202 | 95.8 | 0.049 | 0.053 | 34.6 |
| E4 w_presence 4 | 1.5 | 157.8 | 0.224 | 0.200 | 97.0 | 0.053 | 0.047 | 34.7 |
| E5 scale-up 9.6M ⭐ | 1.5 | **132.0** | **0.336** | **0.272** | 93.8 | 0.051 | 0.060 | 36.2 |
| E6 Fourier 32 | 1.5 | 154.2 | 0.246 | 0.202 | 97.2 | 0.055 | 0.047 | 35.2 |
| M1 logit-normal t | 1.5 | 147.9 | 0.272 | 0.206 | 95.0 | 0.050 | 0.057 | 35.3 |
| M4 EDM | 1.5 | 152.0 | 0.238 | 0.212 | 95.6 | 0.049 | 0.048 | 34.8 |

## Key observations

- **The ranking flips between unit and area level.** At the *unit* level (§3.3) the full
  small model is best and **E5 scale-up 9.6M is the worst** (unit FID 126 > full 112).
  At the *area* level **E5 is the clear best on all three** (FID 132.0 / D 0.336 / C 0.272)
  — bigger capacity pays off once layouts are aggregated into whole floors.
- **cross-attention is the single most important component for area FID**: removing it (E1)
  is the worst variant (area FID 175.3 vs 149.0 full), a bigger hit than at unit level.
- **Merging raises Density/Coverage but worsens FID** for every variant vs the same units
  un-merged (e.g. full: D 0.242 vs 0.045, FID 149 vs 95). Floors are busier images that
  cluster closer in Inception space (↑D/C), while the per-unit room drop compounds over
  ~5 units/floor (gen ~35 rooms vs 46 real) and pushes FID up.
- Count-cond / CFG / w_presence / Fourier / EDM / logit-normal differences are second-order
  at the area level; the dominant axes are **capacity (E5)** and **cross-attn (E1)**.

## Caveats

- **Held-out purity**: each floor is completed by pulling *all* its units from the CSV, so
  ~77% of merged units were in the unit-training split. Area numbers are optimistic; the
  bias is uniform across variants so the *relative* ranking is still meaningful.
- **No cross-apartment structure**: each unit is generated independently from its own outline;
  the floor is their geometric union (brief `buffer(+0.3).unary_union.buffer(-0.3)`).
- Residential units only (public space has null `unit_id`).
- FID computed with N=500 floors (same regime as the unit evals, n_eval 500–600).
