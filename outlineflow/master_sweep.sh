#!/usr/bin/env bash
# Autonomous overnight-style sweep. Runs everything sequentially, logs each result
# to master_results.txt as it finishes. Robust: a failed step is logged and skipped.
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_full_plan_id/master_results.txt
BASE=outputs_full_plan_id
EC=$BASE/energy_critic.pt
echo "config, FID, Density, Coverage, gen_rooms   (start $(date))" > "$RES"
echo "REFERENCE: align g16 churn0=131.5/0.091/0.098 ; churn0.3=133.6/0.094/0.113 ; churn0.6=137.0/0.094/0.122" >> "$RES"

ev() {  # ev <label> <out_dir> [extra sample_eval args...]
  local label="$1"; local outdir="$2"; shift 2
  local out
  out=$("$PY" sample_eval.py --out_dir "$outdir" --decoder rect --align --grid 16 \
        --inception --metric_cpu --n_eval 600 --save 0 "$@" 2>&1)
  local fid den cov rooms
  fid=$(echo "$out" | grep -oP 'FID\s+\K[0-9.]+'); den=$(echo "$out" | grep -oP 'Density\s+\K[0-9.]+')
  cov=$(echo "$out" | grep -oP 'Coverage\s+\K[0-9.]+'); rooms=$(echo "$out" | grep -oP 'gen \K[0-9.]+' | head -1)
  echo "${label}, ${fid:-ERR}, ${den:-ERR}, ${cov:-ERR}, ${rooms:-ERR}" >> "$RES"
}

# ---- Phase 1: config sweeps on the baseline model (guaranteed, no training) ----
ev "churn0.2"            $BASE --churn 0.2
ev "churn0.4"            $BASE --churn 0.4
ev "churn0.5"            $BASE --churn 0.5
ev "churn0.8"            $BASE --churn 0.8
ev "grid12+churn0.4"     $BASE --grid 12 --churn 0.4
ev "grid20+churn0.4"     $BASE --grid 20 --churn 0.4
ev "steps200+churn0.3"   $BASE --sample_steps 200 --churn 0.3
ev "rerank4+churn0.3"    $BASE --rerank 4 --energy_ckpt $EC --churn 0.3
ev "rerank8+det"         $BASE --rerank 8 --energy_ckpt $EC
echo "[phase1 done $(date)]" >> "$RES"

# ---- Phase 2: stronger critic + its uses ----
$PY train_energy.py --flow_ckpt $BASE/ckpt.pt --msd_limit 2500 --steps 8000 \
    --n_critic_layers 5 --churn 0.3 --ckpt_name energy_critic_strong.pt --out_dir $BASE \
    > $BASE/train_critic_strong.log 2>&1
ECS=$BASE/energy_critic_strong.pt
ev "strongcrit+guid0.25"      $BASE --energy_ckpt $ECS --guidance 0.25
ev "strongcrit+rerank6+churn0.3" $BASE --rerank 6 --energy_ckpt $ECS --churn 0.3
echo "[phase2 done $(date)]" >> "$RES"

# ---- Phase 3: small model trained 30k ----
$PY train.py --data_csv /root/jiahua_code/tum.ai-hackathon/mds_V2_5.372k.csv \
    --group plan_id --msd_limit 6000 --steps 30000 --out_dir outputs_full_plan_id_30k \
    > outputs_full_plan_id_30k_train.log 2>&1
ev "small30k+churn0.3"   outputs_full_plan_id_30k --churn 0.3
ev "small30k+churn0.6"   outputs_full_plan_id_30k --churn 0.6
echo "[phase3 done $(date)]" >> "$RES"

# ---- Phase 4: adversarial / negative-unlearning fine-tune ----
$PY finetune_adv.py --flow_ckpt $BASE/ckpt.pt --steps 1500 --batch 48 --rollout 5 \
    --lam 0.3 --msd_limit 2500 --out_dir outputs_finetune_adv \
    > outputs_finetune_adv_train.log 2>&1
ev "adv+churn0"          outputs_finetune_adv
ev "adv+churn0.3"        outputs_finetune_adv --churn 0.3
ev "adv+churn0.6"        outputs_finetune_adv --churn 0.6
echo "[phase4 done $(date)]" >> "$RES"
echo "ALL DONE $(date)" >> "$RES"
