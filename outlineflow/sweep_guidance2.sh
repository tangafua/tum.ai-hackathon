#!/usr/bin/env bash
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_full_plan_id/guidance_sweep2_results.txt
echo "guidance, FID, Density, Coverage, gen_rooms (low-guidance refine)" > "$RES"
for G in 0.25 0.5; do
  out=$("$PY" sample_eval.py --out_dir outputs_full_plan_id \
        --decoder rect --align --grid 16 --inception --metric_cpu \
        --energy_ckpt outputs_full_plan_id/energy_critic.pt --guidance $G --guide_from 0.3 \
        --n_eval 600 --save 0 2>&1)
  fid=$(echo "$out" | grep -oP 'FID\s+\K[0-9.]+'); den=$(echo "$out" | grep -oP 'Density\s+\K[0-9.]+')
  cov=$(echo "$out" | grep -oP 'Coverage\s+\K[0-9.]+'); rooms=$(echo "$out" | grep -oP 'gen \K[0-9.]+' | head -1)
  echo "$G, $fid, $den, $cov, $rooms" | tee -a "$RES"
done
echo "DONE" >> "$RES"
