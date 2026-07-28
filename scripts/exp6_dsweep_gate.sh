#!/bin/bash
# Exp-6 partition-width sweep: does the backdoor still train, and stay removable, as the
# dedicated partition shrinks?  d=8 (504 latents) passed both gates on all 3 seeds; this runs
# the same gate on d=4 (252) and d=2 (126).
#
# N_FORGET must match the width the organism was TRAINED with -- the gate ablates latents
# [0:N_FORGET) of every wrapped module, so a mismatch silently tests the wrong slice.
#
#   ssh torrnode14 'bash /scratch/network/ssd/marek/minimalsleepers/scripts/exp6_dsweep_gate.sh'
set -u
cd /scratch/network/ssd/marek/minimalsleepers || exit 1
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
GPUS=(${GPUS:-6 7})
i=0
for d in 4 2; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))
  ads=()
  for s in 42 43 44; do
    ads+=("models/exp6/route_d${d}_l1523_s${s}/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk")
  done
  echo "[$(date +%H:%M) g$gpu] GATE d=$d (N_FORGET=$d, ${#ads[@]} organisms)"
  CUDA_VISIBLE_DEVICES=$gpu N_FORGET=$d \
    CLCD_OUT=clcd_results/exp6/pilot_gate_d${d}.json \
    uv run python -u scripts/exp6_pilot_gate.py "${ads[@]}" \
    > "logs/exp6/gate_d${d}.out" 2>&1 &
done
wait
echo "=== dsweep gates done $(date) ==="
grep -h -E 'intact=|PASS' logs/exp6/gate_d4.out logs/exp6/gate_d2.out
