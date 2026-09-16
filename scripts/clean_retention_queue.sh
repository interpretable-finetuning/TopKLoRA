#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
# Queued clean-retention judging (Wave-1 5th tuple element). Blocks until the
# 24-org eval sweep is fully done (24 leak.json), then:
#   1) generates clean-retention gens per org (intact / ablate_circuit / base)
#      on held-out clean prompts (500 alpaca @offset1000 + 446 no-robots), no judge
#   2) scores them with Qwen2.5-32B-Instruct split across 4 GPU-pairs
# Protocol matches scripts/reeval_v2.sh / judge_v2_parallel.sh. Run on torrnode15.
cd "$(dirname "$0")/.."
DATA=data/sleeper/prepared_eval6k
NOROBOTS=data/extra/no_robots_prompts.jsonl
OUT=clcd_results/exp5_eval
JLOG=logs/exp5_eval
mkdir -p "$JLOG"
GPUS=(0 1 2 3 4 5 6 7); NG=8

RIDS=()
for a in ortho entropy l0 redund; do for f in l1523 l19; do for s in 42 43 44; do RIDS+=("${a}_${f}_s${s}"); done; done; done

find_adapter() { find "models/exp5/$1" -name adapter_config.json ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

# 1. wait for the full sweep (24 leak.json = every org's 5-tuple minus clean-ret done)
echo "[clean-ret] $(date) waiting for sweep: $(ls $OUT/*_leak.json 2>/dev/null | wc -l)/24 leak.json"
until [ "$(ls $OUT/*_leak.json 2>/dev/null | wc -l)" -ge 24 ]; do sleep 300; done
echo "[clean-ret] $(date) sweep complete -> generation"

# 2. generation: one org per GPU at a time (round-robin over 8 slots)
gen_worker() {
  local i=$1 gpu=${GPUS[$i]} rid ad surg
  for ((j=i; j<${#RIDS[@]}; j+=NG)); do
    rid=${RIDS[$j]}; surg="$OUT/${rid}_surgical.json"
    [ -f "$surg" ] && continue
    ad=$(find_adapter "$rid"); [ -z "$ad" ] && { echo "[clean-ret] $rid NO ADAPTER"; continue; }
    echo "[clean-ret gen] $(date +%H:%M) $rid on gpu$gpu"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -u -m src.clcd.exp_surgical_removal \
      --adapter "$ad" --circuit_json "$OUT/${rid}_circuit.json" --data $DATA \
      --no_ifeval --no_judge --conditions intact,ablate_circuit,base \
      --offset 1000 --n_judge 500 --judge_prompts_file $NOROBOTS --n_judge_indep 446 --n_backdoor 500 \
      --out "$surg" > "$JLOG/${rid}_surg.out" 2>&1
  done
}
for ((i=0;i<NG;i++)); do gen_worker "$i" & done
wait
echo "[clean-ret] $(date) generation done: $(ls $OUT/*_surgical.json 2>/dev/null | wc -l)/24 -> 32B judge"

# 3. Qwen2.5-32B judge, split across 4 GPU-pairs (device_map=auto per pair)
FILES=($OUT/*_surgical.json)
PAIRS=("0,1" "2,3" "4,5" "6,7")
n=${#FILES[@]}
for k in 0 1 2 3; do
  shard=(); for ((idx=k; idx<n; idx+=4)); do shard+=("${FILES[$idx]}"); done
  [ ${#shard[@]} -eq 0 ] && continue
  CUDA_VISIBLE_DEVICES=${PAIRS[$k]} uv run python -u -m src.clcd.judge_saved_gens_big \
    --files "${shard[@]}" --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b \
    > "$JLOG/judge32b_${k}.out" 2>&1 &
done
wait
echo "[clean-ret] $(date) CLEAN-RETENTION JUDGING COMPLETE"
