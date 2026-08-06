#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
# Wave-2 capability leg: 32B-judge the 15 `all`-family clean-retention generations.
#
# The generations already exist (clean_retention_queue.sh produced 500 alpaca @offset1000 +
# 446 no-robots gens for intact / ablate_circuit / base on every organism) but were never
# scored, so the 5th tuple element is missing for the whole family.
#
# EXPLICIT FILE LIST, NOT A GLOB: the 24 Wave-1 (l19/l1523) surgical.json already carry
# judge_32b / judge_indep_32b, and a glob would re-judge them.
#
# Protocol identical to step 3 of scripts/clean_retention_queue.sh (Qwen2.5-32B-Instruct
# across 4 GPU-pairs with device_map=auto per pair).
#
#   ssh torrnode14 'bash /scratch/network/ssd/marek/minimalsleepers/scripts/judge_all_family.sh'
OUT=clcd_results/exp5_eval
JLOG=logs/exp5_eval
mkdir -p "$JLOG"

FILES=()
for s in 42 43 44; do
  FILES+=("$OUT/a0_all_s${s}_surgical.json")
  for a in entropy l0 ortho redund; do FILES+=("$OUT/${a}_all_s${s}_surgical.json"); done
done

# Resumable: files that already carry judge_32b are skipped (judge_saved_gens_big writes each
# file as it finishes, so an interrupted run keeps everything it had completed).
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
[ ${#FILES[@]} -eq 0 ] && { echo "[judge] nothing to do -- all 15 already judged"; exit 0; }
echo "[judge] $(date) ${#FILES[@]} unjudged all-family files -> up to 4 GPU-pairs"

PAIRS=(${PAIRS:-"0,1" "2,3" "4,5" "6,7"})
NP=${#PAIRS[@]}
n=${#FILES[@]}
for ((k=0; k<NP; k++)); do
  shard=(); for ((idx=k; idx<n; idx+=NP)); do shard+=("${FILES[$idx]}"); done
  [ ${#shard[@]} -eq 0 ] && continue
  echo "[judge] pair ${PAIRS[$k]} <- ${#shard[@]} files"
  CUDA_VISIBLE_DEVICES=${PAIRS[$k]} uv run python -u -m src.clcd.judge_saved_gens_big \
    --files "${shard[@]}" --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b \
    > "$JLOG/judge32b_all_${k}.out" 2>&1 &
done
wait
echo "[judge] $(date) DONE"
