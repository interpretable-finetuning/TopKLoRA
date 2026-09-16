#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Payload-concentration on the 15 `all`-family orgs that have matched-K leak labels.
# These are the points for the concentration-vs-leak test; the routed/a0 control (already run)
# is what licenses using the metric at all.
#
#   ssh torrnode14 'bash /scratch/network/ssd/marek/minimalsleepers/scripts/payload_concentration_all.sh'
GPUS=(${GPUS:-6 7})
RES=${RES:-clcd_results/matchedK_all/results}   # matched-K results dir = which family
TAG=${TAG:-all}
mkdir -p clcd_results/exp6 logs/exp6

# adapter paths come from the matched-K result files, so the org identity is guaranteed to
# be the same one the leak label was measured on
mapfile -t ADS < <(RES="$RES" uv run python -c "
import json,glob,os
for f in sorted(glob.glob(os.environ['RES']+'/*.json')):
    print(json.load(open(f))[0]['adapter'])
")
n=${#ADS[@]}
half=$(( (n + 1) / 2 ))
CUDA_VISIBLE_DEVICES=${GPUS[0]} CLCD_N=${CLCD_N:-50} \
  CLCD_OUT=clcd_results/exp6/payload_conc_${TAG}_a.json \
  uv run python -u analysis/payload_concentration.py "${ADS[@]:0:$half}" \
  > logs/exp6/payload_conc_${TAG}_a.out 2>&1 &
CUDA_VISIBLE_DEVICES=${GPUS[1]} CLCD_N=${CLCD_N:-50} \
  CLCD_OUT=clcd_results/exp6/payload_conc_${TAG}_b.json \
  uv run python -u analysis/payload_concentration.py "${ADS[@]:$half}" \
  > logs/exp6/payload_conc_${TAG}_b.out 2>&1 &
wait
echo "=== payload-concentration ($TAG family) done $(date) ==="
grep -h "n90=" logs/exp6/payload_conc_${TAG}_a.out logs/exp6/payload_conc_${TAG}_b.out | wc -l
