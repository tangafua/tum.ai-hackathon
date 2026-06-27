#!/usr/bin/env bash
set -euo pipefail
PY=/root/torch-rocm/bin/python
DIR=/root/jiahua_code/tum.ai-hackathon/outlineflow
CSV=/root/jiahua_code/tum.ai-hackathon/mds_V2_5.372k.csv
OUT="$DIR/outputs_full_plan_id_ewfm"
cd "$DIR"; mkdir -p "$OUT"
TS=$(date +%Y%m%d_%H%M%S); LOG="$OUT/train_${TS}.log"
ln -sf "train_${TS}.log" "$OUT/train_latest.log"
nohup "$PY" -u train.py --data_csv "$CSV" --group plan_id --msd_limit 6000 \
  --steps 15000 --ewfm --ewfm_beta 0.5 \
  --out_dir "$OUT" > "$LOG" 2>&1 &
echo $! > "$OUT/train.pid"; echo "PID=$(cat $OUT/train.pid) log=$LOG"
