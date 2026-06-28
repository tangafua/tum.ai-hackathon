#!/usr/bin/env bash
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_full_plan_id/churn06_3seed.txt
echo "seed, FID, Density, Coverage" > "$RES"
for s in 42 123 7; do
  out=$("$PY" sample_eval.py --out_dir outputs_full_plan_id --decoder rect --align --grid 16 \
        --inception --metric_cpu --churn 0.6 --n_eval 600 --save 0 --seed $s 2>&1)
  fid=$(echo "$out"|grep -oP 'FID\s+\K[0-9.]+'); den=$(echo "$out"|grep -oP 'Density\s+\K[0-9.]+'); cov=$(echo "$out"|grep -oP 'Coverage\s+\K[0-9.]+')
  echo "$s, ${fid:-ERR}, ${den:-ERR}, ${cov:-ERR}" | tee -a "$RES"
done
awk -F', ' 'NR>1 && $2!="ERR"{f+=$2;d+=$3;c+=$4;n++} END{printf "MEAN(n=%d): FID %.1f | Den %.3f | Cov %.3f\n",n,f/n,d/n,c/n}' "$RES" | tee -a "$RES"
echo "DONE" >> "$RES"
