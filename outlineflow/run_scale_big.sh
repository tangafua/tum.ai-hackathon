#!/usr/bin/env bash
# Push the scale fix: --use_scale + 30000 steps + w_presence=4.0, so the model
# more strongly learns area->count and dares to span the real count range.
set -euo pipefail
PY=/root/torch-rocm/bin/python
DIR=/root/jiahua_code/tum.ai-hackathon/outlineflow
CSV=/root/jiahua_code/tum.ai-hackathon/mds_V2_5.372k.csv
OUT="$DIR/outputs_full_plan_id_scale30k"
cd "$DIR"; mkdir -p "$OUT"
TS=$(date +%Y%m%d_%H%M%S); LOG="$OUT/train_${TS}.log"
ln -sf "train_${TS}.log" "$OUT/train_latest.log"
nohup "$PY" -u train.py \
  --data_csv "$CSV" --group plan_id --msd_limit 6000 \
  --steps 30000 --use_scale --w_presence 4.0 \
  --out_dir "$OUT" > "$LOG" 2>&1 &
PID=$!; echo "$PID" > "$OUT/train.pid"
echo "started PID=$PID"; echo "log=$LOG"; echo "out_dir=$OUT"
