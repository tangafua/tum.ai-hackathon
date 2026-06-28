#!/usr/bin/env bash
set -euo pipefail
PY=/root/torch-rocm/bin/python
CSV=/root/jiahua_code/tum.ai-hackathon/mds_V2_5.372k.csv
OUT=/root/jiahua_code/tum.ai-hackathon/outlineflow/outputs_full_plan_id_cross_scale
cd /root/jiahua_code/tum.ai-hackathon/outlineflow; mkdir -p "$OUT"
TS=$(date +%Y%m%d_%H%M%S); LOG="$OUT/train_${TS}.log"; ln -sf "train_${TS}.log" "$OUT/train_latest.log"
nohup "$PY" -u train.py --data_csv "$CSV" --group plan_id --msd_limit 6000 \
  --steps 15000 --cross_attn --use_scale --out_dir "$OUT" > "$LOG" 2>&1 &
echo $! > "$OUT/train.pid"; echo "PID=$(cat $OUT/train.pid)"
