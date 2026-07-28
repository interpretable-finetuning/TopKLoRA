#!/bin/bash
# Payload-concentration on the 15 `all`-family organisms that have matched-K leak labels.
# These are the points for the concentration-vs-leak test; the routed/a0 control (already run)
# is what licenses using the metric at all.
#
#   ssh torrnode14 'bash /scratch/network/ssd/marek/minimalsleepers/scripts/payload_concentration_all.sh'
set -u
cd /scratch/network/ssd/marek/minimalsleepers || exit 1
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS=(${GPUS:-6 7})
mkdir -p clcd_results/exp6 logs/exp6

# adapter paths come from the matched-K result files, so the organism identity is guaranteed to
# be the same one the leak label was measured on
mapfile -t ADS < <(uv run python -c "
import json,glob,os
for f in sorted(glob.glob('clcd_results/matchedK_all/results/*.json')):
    print(json.load(open(f))[0]['adapter'])
")
n=${#ADS[@]}
half=$(( (n + 1) / 2 ))
CUDA_VISIBLE_DEVICES=${GPUS[0]} CLCD_N=${CLCD_N:-50} \
  CLCD_OUT=clcd_results/exp6/payload_conc_all_a.json \
  uv run python -u scripts/payload_concentration.py "${ADS[@]:0:$half}" \
  > logs/exp6/payload_conc_all_a.out 2>&1 &
CUDA_VISIBLE_DEVICES=${GPUS[1]} CLCD_N=${CLCD_N:-50} \
  CLCD_OUT=clcd_results/exp6/payload_conc_all_b.json \
  uv run python -u scripts/payload_concentration.py "${ADS[@]:$half}" \
  > logs/exp6/payload_conc_all_b.out 2>&1 &
wait
echo "=== payload-concentration (all family) done $(date) ==="
grep -h "n90=" logs/exp6/payload_conc_all_a.out logs/exp6/payload_conc_all_b.out | wc -l
