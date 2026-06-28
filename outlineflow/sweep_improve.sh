#!/usr/bin/env bash
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_full_plan_id/improve_sweep.txt
echo "config, FID, Density, Coverage, rooms (baseline+align g16, single seed)" > "$RES"
run(){ local lbl="$1"; shift
  out=$("$PY" sample_eval.py --out_dir outputs_full_plan_id --decoder rect --align --grid 16 \
        --inception --metric_cpu --n_eval 600 --save 0 "$@" 2>&1)
  f=$(echo "$out"|grep -oP 'FID\s+\K[0-9.]+'); d=$(echo "$out"|grep -oP 'Density\s+\K[0-9.]+')
  c=$(echo "$out"|grep -oP 'Coverage\s+\K[0-9.]+'); r=$(echo "$out"|grep -oP 'gen \K[0-9.]+'|head -1)
  echo "$lbl, ${f:-ERR}, ${d:-ERR}, ${c:-ERR}, ${r:-ERR}" | tee -a "$RES"; }
run "1_count_cond+churn0.3"  --count_cond --churn 0.3
run "1_count_cond_nochurn"   --count_cond
run "2_repel1+churn0.3"      --repel 1.0 --churn 0.3
run "2_repel3+churn0.3"      --repel 3.0 --churn 0.3
run "3_temp1.15+churn0.3"    --temp 1.15 --churn 0.3
run "3_temp1.3+churn0.3"     --temp 1.3 --churn 0.3
echo "DONE" >> "$RES"
