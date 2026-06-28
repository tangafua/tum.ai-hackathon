#!/usr/bin/env bash
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_full_plan_id/multiseed_results.txt
echo "candidate, seed, FID, Density, Coverage" > "$RES"
runseed(){ local label="$1"; local outdir="$2"; local extra="$3"; local seed="$4"
  out=$("$PY" sample_eval.py --out_dir "$outdir" --decoder rect --inception --metric_cpu \
        --n_eval 600 --save 0 --seed "$seed" $extra 2>&1)
  fid=$(echo "$out"|grep -oP 'FID\s+\K[0-9.]+'); den=$(echo "$out"|grep -oP 'Density\s+\K[0-9.]+'); cov=$(echo "$out"|grep -oP 'Coverage\s+\K[0-9.]+')
  echo "$label, $seed, ${fid:-ERR}, ${den:-ERR}, ${cov:-ERR}" >> "$RES"; }
for s in 42 123 7; do
  runseed "adv_lam0.3"      outputs_finetune_adv  "--align --grid 16" $s
  runseed "churn0.3"        outputs_full_plan_id  "--align --grid 16 --churn 0.3" $s
  runseed "grid12_churn0.4" outputs_full_plan_id  "--align --grid 12 --churn 0.4" $s
  runseed "adv2_churn0.3"   outputs_finetune_adv2 "--align --grid 16 --churn 0.3" $s
  echo "[seed $s done $(date +%H:%M:%S)]" >> "$RES"
done
# 计算均值
echo "" >> "$RES"; echo "=== MEANS ===" >> "$RES"
awk -F', ' 'NR>1 && $3!="ERR"{f[$1]+=$3;d[$1]+=$4;c[$1]+=$5;n[$1]++} END{for(k in n) printf "%s: FID %.1f | Den %.3f | Cov %.3f (n=%d)\n",k,f[k]/n[k],d[k]/n[k],c[k]/n[k],n[k]}' "$RES" >> "$RES"
echo "ALL DONE" >> "$RES"
