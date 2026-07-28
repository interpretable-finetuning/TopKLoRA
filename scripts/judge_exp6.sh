#!/bin/bash
# Exp-6 capability leg: 32B-judge the 6 clean-retention generation sets (route + a0 x 3 seeds).
#
# Same protocol/judge/suffix as Wave-1/2 (scripts/judge_all_family.sh) so retention numbers are
# directly comparable across every organism we have measured.
#
# EXPLICIT FILE LIST, NOT A GLOB, and resumable: files already carrying judge_32b are skipped.
#
#   ssh torrnode14 'bash /scratch/network/ssd/marek/minimalsleepers/scripts/judge_exp6.sh'
set -u
cd /scratch/network/ssd/marek/minimalsleepers || exit 1
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TQDM_DISABLE=1
OUT=clcd_results/exp6
JLOG=logs/exp6
mkdir -p "$JLOG"

FILES=()
for s in 42 43 44; do
  for arm in route a0; do FILES+=("$OUT/${arm}_l1523_s${s}_surgical.json"); done
done

TODO=()
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "MISSING $f -- aborting"; exit 1; }
  if python3 -c "
import json,sys
d=json.load(open('$f'))
sys.exit(0 if any('judge' in k for k in d['conditions']['intact']) else 1)"; then
    echo "[judge] already judged, skip: $(basename "$f")"
  else
    TODO+=("$f")
  fi
done
FILES=("${TODO[@]}")
[ ${#FILES[@]} -eq 0 ] && { echo "[judge] nothing to do"; exit 0; }
echo "[judge] $(date) ${#FILES[@]} unjudged exp6 files"

PAIRS=(${PAIRS:-"0,1" "2,3" "4,5"})
NP=${#PAIRS[@]}
n=${#FILES[@]}
for ((k=0; k<NP; k++)); do
  shard=(); for ((idx=k; idx<n; idx+=NP)); do shard+=("${FILES[$idx]}"); done
  [ ${#shard[@]} -eq 0 ] && continue
  echo "[judge] pair ${PAIRS[$k]} <- ${#shard[@]} files"
  CUDA_VISIBLE_DEVICES=${PAIRS[$k]} uv run python -u -m src.clcd.judge_saved_gens_big \
    --files "${shard[@]}" --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b \
    > "$JLOG/judge32b_exp6_${k}.out" 2>&1 &
done
wait
echo "[judge] $(date) DONE"
