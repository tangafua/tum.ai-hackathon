#!/usr/bin/env bash
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_full_plan_id/combo_churn_guid_3seed.txt
echo "config, seed, FID, Density, Coverage" > "$RES"
for g in 0.15 0.25; do
 for s in 42 123 7; do
  out=$("$PY" sample_eval.py --out_dir outputs_full_plan_id --decoder rect --align --grid 16 \
        --inception --metric_cpu --churn 0.3 \
        --energy_ckpt outputs_full_plan_id/energy_critic_churn03.pt --guidance $g --guide_from 0.3 \
        --n_eval 600 --save 0 --seed $s 2>&1)
  fid=$(echo "$out"|grep -oP 'FID\s+\K[0-9.]+'); den=$(echo "$out"|grep -oP 'Density\s+\K[0-9.]+'); cov=$(echo "$out"|grep -oP 'Coverage\s+\K[0-9.]+')
  echo "churn0.3+guid$g, $s, ${fid:-ERR}, ${den:-ERR}, ${cov:-ERR}" | tee -a "$RES"
 done
done
awk -F', ' 'NR>1 && $3!="ERR"{f[$1]+=$3;d[$1]+=$4;c[$1]+=$5;n[$1]++} END{for(k in n) printf "MEAN %s: FID %.1f | Den %.3f | Cov %.3f\n",k,f[k]/n[k],d[k]/n[k],c[k]/n[k]}' "$RES" >> "$RES"
echo "DONE" >> "$RES"
