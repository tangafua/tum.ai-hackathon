#!/usr/bin/env bash
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_full_plan_id/churn_sweep_results.txt
echo "churn, FID, Density, Coverage, gen_rooms (baseline+align g16, SDE sampling)" > "$RES"
for C in 0.1 0.3 0.6; do
  out=$("$PY" sample_eval.py --out_dir outputs_full_plan_id \
        --decoder rect --align --grid 16 --inception --metric_cpu \
        --churn $C --n_eval 600 --save 0 2>&1)
  fid=$(echo "$out" | grep -oP 'FID\s+\K[0-9.]+'); den=$(echo "$out" | grep -oP 'Density\s+\K[0-9.]+')
  cov=$(echo "$out" | grep -oP 'Coverage\s+\K[0-9.]+'); rooms=$(echo "$out" | grep -oP 'gen \K[0-9.]+' | head -1)
  echo "$C, $fid, $den, $cov, $rooms" | tee -a "$RES"
done
echo "DONE" >> "$RES"
