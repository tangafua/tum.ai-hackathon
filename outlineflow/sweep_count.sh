#!/usr/bin/env bash
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_full_plan_id/count_sweep_results.txt
echo "target_count, FID, Density, Coverage, gen_rooms (align g16, real_rooms=38.1)" > "$RES"
for T in 25 18 12 8; do
  out=$("$PY" sample_eval.py --out_dir outputs_full_plan_id \
        --decoder rect --align --grid 16 --inception --metric_cpu \
        --target_count $T --n_eval 600 --save 0 2>&1)
  fid=$(echo "$out" | grep -oP 'FID\s+\K[0-9.]+')
  den=$(echo "$out" | grep -oP 'Density\s+\K[0-9.]+')
  cov=$(echo "$out" | grep -oP 'Coverage\s+\K[0-9.]+')
  rooms=$(echo "$out" | grep -oP 'gen \K[0-9.]+' | head -1)
  echo "$T, $fid, $den, $cov, $rooms" | tee -a "$RES"
done
echo "DONE" >> "$RES"
