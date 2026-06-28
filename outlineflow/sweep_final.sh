#!/usr/bin/env bash
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_full_plan_id/final_combo_results.txt
echo "config, FID, Density, Coverage, gen_rooms" > "$RES"
run() {
  local label="$1"; local outdir="$2"; shift 2
  out=$("$PY" sample_eval.py --out_dir "$outdir" \
        --decoder rect --align --grid 16 --inception --metric_cpu --n_eval 600 --save 0 "$@" 2>&1)
  fid=$(echo "$out" | grep -oP 'FID\s+\K[0-9.]+'); den=$(echo "$out" | grep -oP 'Density\s+\K[0-9.]+')
  cov=$(echo "$out" | grep -oP 'Coverage\s+\K[0-9.]+'); rooms=$(echo "$out" | grep -oP 'gen \K[0-9.]+' | head -1)
  echo "$label, $fid, $den, $cov, $rooms" | tee -a "$RES"
}
run "scale30k+churn0.3" outputs_full_plan_id_scale30k --churn 0.3
run "scale30k+churn0.6" outputs_full_plan_id_scale30k --churn 0.6
run "baseline+churn0.3+steps200" outputs_full_plan_id --churn 0.3
echo "DONE" >> "$RES"
