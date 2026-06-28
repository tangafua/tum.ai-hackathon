#!/usr/bin/env bash
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_full_plan_id/dg_sweep_results.txt
echo "config, FID, Density, Coverage, gen_rooms (churn0.3 + churn-matched critic guidance)" > "$RES"
for G in 0.25 0.5 1.0; do
  out=$("$PY" sample_eval.py --out_dir outputs_full_plan_id \
        --decoder rect --align --grid 16 --inception --metric_cpu \
        --churn 0.3 --energy_ckpt outputs_full_plan_id/energy_critic_churn03.pt \
        --guidance $G --guide_from 0.3 --n_eval 600 --save 0 2>&1)
  fid=$(echo "$out" | grep -oP 'FID\s+\K[0-9.]+'); den=$(echo "$out" | grep -oP 'Density\s+\K[0-9.]+')
  cov=$(echo "$out" | grep -oP 'Coverage\s+\K[0-9.]+'); rooms=$(echo "$out" | grep -oP 'gen \K[0-9.]+' | head -1)
  echo "churn0.3+guid$G, $fid, $den, $cov, $rooms" | tee -a "$RES"
done
echo "DONE" >> "$RES"
