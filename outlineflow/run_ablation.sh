#!/usr/bin/env bash
# ============================================================================
# Controlled comparison: jiahua BASELINE (git HEAD) vs YOUR IMPROVED (working tree)
#
#   * Same CSV (kagglehub full, 1.08M rows)        -> data is identical
#   * Same seed / msd_limit / group / n_held       -> train/held SPLIT is identical
#   * Both scored on ONE held.pkl + same metrics.py/render.py (unchanged in both)
#   * decoder is PART OF the method -> baseline uses its own (mandatory gap-fill),
#     improved uses its own (capped gap-fill). This is intentional per your call.
#
# So the ONLY differences scored are: model arch (cross-attn), input (count-cond,
# padding fix) and the decoder -- i.e. your complete method vs hers.
#
# Run from the outlineflow/ dir:  bash run_ablation.sh
# NOTE: this trains TWO models (hours each) then evaluates both. Heavy.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- frozen, shared knobs (must match between the two runs) -----------------
CSV=/root/.cache/kagglehub/datasets/caspervanengelenburg/modified-swiss-dwellings/versions/6/mds_V2_5.372k.csv
GROUP=plan_id
LIMIT=6000
NHELD=1000
SEED=42
NEVAL=600
EVAL_DEVICE=cpu      # score metrics on CPU: deterministic + avoids ROCm LAPACK issues

echo "[csv] expected md5 c9b71f58... ; actual:"
md5sum "$CSV"
echo "[git] current state:"; git status -s outlineflow/*.py | sed 's/^/      /'

# ---- phase A: BASELINE = git HEAD (jiahua, no cross-attn, mandatory gap-fill)
echo; echo "==================== PHASE A: BASELINE (git HEAD) ===================="
git stash push -m "ablation-improved" -- outlineflow/*.py
trap 'echo "[git] restoring your working tree..."; git stash list | grep -q ablation-improved && git stash pop' EXIT
rm -rf outlineflow/__pycache__ __pycache__ 2>/dev/null || true

python train.py --data_csv "$CSV" --group $GROUP --msd_limit $LIMIT \
    --n_held $NHELD --seed $SEED --out_dir outputs_baseline
python sample_eval.py --out_dir outputs_baseline --held outputs_baseline/held.pkl \
    --decoder rect --inception --n_eval $NEVAL --seed $SEED --device $EVAL_DEVICE \
    | tee outputs_baseline/eval.log

# ---- phase B: IMPROVED = your working tree (cross-attn + count-cond + capped)
echo; echo "==================== PHASE B: IMPROVED (working tree) ===================="
git stash pop
trap - EXIT
rm -rf outlineflow/__pycache__ __pycache__ 2>/dev/null || true

python train.py --data_csv "$CSV" --group $GROUP --msd_limit $LIMIT \
    --n_held $NHELD --seed $SEED --out_dir outputs_improved
# IMPORTANT: evaluate on the SAME real set (baseline's held.pkl) for a fair FID
python sample_eval.py --out_dir outputs_improved --held outputs_baseline/held.pkl \
    --decoder rect --inception --n_eval $NEVAL --seed $SEED --device $EVAL_DEVICE \
    | tee outputs_improved/eval.log

# ---- comparison table ------------------------------------------------------
echo; echo "==================== COMPARISON ===================="
printf "%-18s %-10s %-10s %-10s\n" "model" "FID" "Density" "Coverage"
for d in outputs_baseline outputs_improved; do
    fid=$(grep -E "FID"      $d/eval.log | tail -1 | grep -oE "[0-9.]+" | head -1)
    den=$(grep -E "Density"  $d/eval.log | tail -1 | grep -oE "[0-9.]+" | head -1)
    cov=$(grep -E "Coverage" $d/eval.log | tail -1 | grep -oE "[0-9.]+" | head -1)
    printf "%-18s %-10s %-10s %-10s\n" "$d" "$fid" "$den" "$cov"
done
echo
echo "[check] these 'real' lines MUST be identical across the two runs:"
grep -h "rooms/plan  real" outputs_baseline/eval.log outputs_improved/eval.log
grep -h "real-vs-real"     outputs_baseline/eval.log outputs_improved/eval.log
