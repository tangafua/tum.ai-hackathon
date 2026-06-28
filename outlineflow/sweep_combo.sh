#!/usr/bin/env bash
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_full_plan_id/combo_sweep_results.txt
echo "config, FID, Density, Coverage, gen_rooms" > "$RES"
run() {
  local label="$1"; shift
  out=$("$PY" sample_eval.py --out_dir outputs_full_plan_id \
        --decoder rect --align --grid 16 --inception --metric_cpu --n_eval 600 --save 0 "$@" 2>&1)
  fid=$(echo "$out" | grep -oP 'FID\s+\K[0-9.]+'); den=$(echo "$out" | grep -oP 'Density\s+\K[0-9.]+')
  cov=$(echo "$out" | grep -oP 'Coverage\s+\K[0-9.]+'); rooms=$(echo "$out" | grep -oP 'gen \K[0-9.]+' | head -1)
  echo "$label, $fid, $den, $cov, $rooms" | tee -a "$RES"
}
run "churn0.3+guid0.25" --churn 0.3 --energy_ckpt outputs_full_plan_id/energy_critic.pt --guidance 0.25
run "churn0.6+guid0.25" --churn 0.6 --energy_ckpt outputs_full_plan_id/energy_critic.pt --guidance 0.25
run "churn0.9"          --churn 0.9
echo "DONE" >> "$RES"
