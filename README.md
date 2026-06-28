# OutlineFlow — outline → typed room polygons

**Mirror Mirror on the Wall** (DAVIS × Paris 2026). Given **only an apartment/floor outline**,
generate a complete, plausible set of **typed interior rooms as vector polygons** (never pixels),
with a from-scratch **rectified flow-matching** model. Dataset: Modified Swiss Dwellings (MSD).

> 🤗 **Model weights & artifacts:** https://huggingface.co/gafua926/outlineflow
> 🎞️ Presentation deck: https://docs.google.com/presentation/d/1xrL_qE7JLWYe1KVjuY8o08DVCZmzhQG7dOgfai8SUWY/edit?usp=sharing
> 📄 Full results: [`outlineflow/result.md`](outlineflow/result.md) · Method: [`outlineflow/SUBMISSION_METHODOLOGY.md`](outlineflow/SUBMISSION_METHODOLOGY.md)  

---

## Results (full `plan_id` model, 3-seed mean)

| Config | FID ↓ | Density ↑ | Coverage ↑ |
|---|---|---|---|
| Raw baseline | 167 | 0.097 | 0.067 |
| + grid-align | 131.5 | 0.091 | 0.098 |
| **+ churn (final)** | **135.3** | 0.088 | **0.111** |

**FID −19%, Coverage +66%** over the raw baseline.

---

## Quick start

```bash
# 1. environment (ROCm/CUDA PyTorch + deps)
pip install torch torchvision torchaudio          # match your accelerator
pip install -r outlineflow/requirements.txt

# 2. get the weights (already in outlineflow/outputs_full_plan_id/ckpt.pt,
#    or pull from the Hugging Face Hub):
huggingface-cli download gafua926/outlineflow outputs_full_plan_id/ckpt.pt \
    --local-dir outlineflow/
```

```python
# 3. generate
from generate import generate                      # run from outlineflow/
rooms = generate(outline)                           # outline: shapely (Multi)Polygon or WKT
# rooms == [(shapely Polygon, room_type_id), ...]   vector, gap-free, deterministic at seed 42
```

The model conditions on the outline **only** (per the brief). Defaults: `churn=0.3`, `align=True`,
`grid=16`, `seed=42`. Output tiles the outline with zero interior gap by construction.

---

## Method (short)

A pure **rectified flow-matching** DiT-style **set-Transformer**: room tokens carry no positional
encoding (a plan is a *set*); the outline is encoded by a permutation-invariant PointNet and
injected via AdaLN-Zero; full self-attention learns a joint, non-overlapping, outline-filling layout.

**Diagnosis.** Clean geometry but weak Density/Coverage → the **L2 objective regresses to the
conditional mean** → variance collapse (generated room-count std ±6 vs real ±24).

**Two levers that work:**
1. **Grid-align** (deterministic post-process) → cleaner rectilinear tiling → **FID 167 → 131.5**.
2. **Churn / SDE sampling** → escapes the mean trajectory → **Coverage 0.098 → 0.111**.

**What we tested and ruled out (all under multi-seed):** bigger/deeper/longer models, Voronoi
decoder, loss-weight tuning, **DDPM (pure diffusion)**, **EBM** (guidance / energy-weighted FM /
adversarial fine-tune / best-of-N), and aggressive sampling hacks (temperature, repulsion,
per-outline count-forcing). The decisive test: **un-compressing the conditioning**
(cross-attention over all boundary points + scale) raised the model's outline-reading
**r 0.06 → 0.67**, yet the metrics did **not** move — proving the bottleneck is the loss, not the
conditioning. See [`outlineflow/result.md`](outlineflow/result.md) and
[`outlineflow/ANALYSIS.md`](outlineflow/ANALYSIS.md).

---

## Repo structure

```
brief.pdf                         challenge brief
demo.py                           official outline-construction script
outlineflow_final_pitch1_v2.pptx  pitch deck
outlineflow/
  model.py  flow.py  diffusion.py energy.py     model + flow-matching + DDPM + EBM
  train.py  train_diffusion.py  finetune_adv.py train_energy.py    training
  generate.py                    submission entry point: generate(outline)
  sample_eval.py  metrics.py  postprocess.py  render.py   eval + decode + render
  params.py  msd_data.py  cfg.py  synth_data.py
  result.md  SUBMISSION_METHODOLOGY.md  ANALYSIS.md         write-ups
  outputs_full_plan_id/ckpt.pt   final weights (also on HF Hub)
```

---

## Reproduce

```bash
cd outlineflow
# train (full plan_id, 15k steps, seed 42)
python train.py --data_csv ../mds_V2_5.372k.csv --group plan_id --msd_limit 6000 \
    --steps 15000 --out_dir outputs_full_plan_id --seed 42
# evaluate the final recipe (grid-align + churn), official metrics
python sample_eval.py --out_dir outputs_full_plan_id --decoder rect \
    --align --grid 16 --churn 0.3 --inception --metric_cpu --n_eval 600 --save 8 --seed 42
# robust numbers: repeat with seeds 42, 123, 7 and average
```

Submission per brief: training + generation code, model weights (here + HF Hub),
`generate(outline)` entry point, and methodology write-up — all included.
