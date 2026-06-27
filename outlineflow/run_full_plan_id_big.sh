#!/usr/bin/env bash
# Bigger model (d_model=256, n_layers=8), full plan_id data, 30000 steps, rect decoder.
# Detached run with logging + PID tracking.
set -euo pipefail

PY=/root/torch-rocm/bin/python
DIR=/root/jiahua_code/tum.ai-hackathon/outlineflow
CSV=/root/jiahua_code/tum.ai-hackathon/mds_V2_5.372k.csv
OUT="$DIR/outputs_full_plan_id_big"
cd "$DIR"
mkdir -p "$OUT"

TS=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/train_${TS}.log"
ln -sf "train_${TS}.log" "$OUT/train_latest.log"

nohup "$PY" -u train.py \
  --data_csv "$CSV" \
  --group plan_id \
  --msd_limit 6000 \
  --steps 30000 \
  --d_model 256 \
  --n_layers 8 \
  --out_dir "$OUT" \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$OUT/train.pid"
echo "started PID=$PID"
echo "log=$LOG"
echo "out_dir=$OUT"
