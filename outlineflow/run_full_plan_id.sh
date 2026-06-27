#!/usr/bin/env bash
# Full-dataset training run (plan_id grouping, 15000 steps) with logging + PID tracking.
# Launches detached so it survives terminal/SSH disconnects.
set -euo pipefail

PY=/root/torch-rocm/bin/python
DIR=/root/jiahua_code/tum.ai-hackathon/outlineflow
CSV=/root/jiahua_code/tum.ai-hackathon/mds_V2_5.372k.csv
OUT="$DIR/outputs_full_plan_id"
cd "$DIR"
mkdir -p "$OUT"

TS=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/train_${TS}.log"
ln -sf "train_${TS}.log" "$OUT/train_latest.log"   # stable path for tailing

nohup "$PY" -u train.py \
  --data_csv "$CSV" \
  --group plan_id \
  --msd_limit 6000 \
  --steps 15000 \
  --out_dir "$OUT" \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$OUT/train.pid"
echo "started PID=$PID"
echo "log=$LOG"
echo "out_dir=$OUT"
