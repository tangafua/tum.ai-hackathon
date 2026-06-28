#!/usr/bin/env bash
cd /root/jiahua_code/tum.ai-hackathon/outlineflow
PY=/root/torch-rocm/bin/python
RES=outputs_finetune_adv/adv_combo_results.txt
echo "config, FID, Density, Coverage, gen_rooms (adversarial model combos)" > "$RES"
run(){ local label="$1"; shift
  out=$("$PY" sample_eval.py --out_dir outputs_finetune_adv --decoder rect --inception \
        --metric_cpu --n_eval 600 --save 0 "$@" 2>&1)
  fid=$(echo "$out"|grep -oP 'FID\s+\K[0-9.]+'); den=$(echo "$out"|grep -oP 'Density\s+\K[0-9.]+')
  cov=$(echo "$out"|grep -oP 'Coverage\s+\K[0-9.]+'); r=$(echo "$out"|grep -oP 'gen \K[0-9.]+'|head -1)
  echo "$label, ${fid:-ERR}, ${den:-ERR}, ${cov:-ERR}, ${r:-ERR}" | tee -a "$RES"; }
run "adv+grid16+churn0"      --align --grid 16
run "adv+grid12+churn0.3"    --align --grid 12 --churn 0.3
run "adv+grid14+churn0.3"    --align --grid 14 --churn 0.3
run "adv+grid16+steps200+churn0.3" --align --grid 16 --sample_steps 200 --churn 0.3
echo "DONE" >> "$RES"
