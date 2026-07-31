#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
# Recovers the 12 l1523 clean-retention gens that OOM'd at the default batch_size
# 16 (63-module adapters on a 44GB card), then judges them, then launches Wave-2.
# The 12 l19 gens already succeeded and are judged by the original clean_ret run.
cd "$(dirname "$0")/.."
DATA=data/sleeper/prepared_eval6k
NOROBOTS=data/extra/no_robots_prompts.jsonl
OUT=clcd_results/exp5_eval; JLOG=logs/exp5_eval
GPUS=(0 1 2 3 4 5 6 7); NG=${#GPUS[@]}
RIDS=()
for a in ortho entropy l0 redund; do for s in 42 43 44; do RIDS+=("${a}_l1523_s${s}"); done; done
find_adapter() { find "models/exp5/$1" -name adapter_config.json ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

# 1. wait for the original l19 32B judge to finish (frees GPUs)
echo "[fix] $(date) waiting for l19 judge to finish"
until grep -q "JUDGING COMPLETE" $JLOG/clean_ret.out 2>/dev/null; do sleep 180; done
echo "[fix] $(date) l19 judge done -> regenerate 12 l1523 at batch_size 4"

# 2. regenerate the 12 l1523 clean-ret gens at batch_size 4
gen_worker() {
  local i=$1 gpu=${GPUS[$1]} rid ad surg
  for ((j=i; j<${#RIDS[@]}; j+=NG)); do
    rid=${RIDS[$j]}; surg="$OUT/${rid}_surgical.json"
    [ -f "$surg" ] && continue
    ad=$(find_adapter "$rid"); [ -z "$ad" ] && { echo "[fix] $rid NO ADAPTER"; continue; }
    echo "[fix gen] $(date +%H:%M) $rid gpu$gpu bs=4"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -u -m src.clcd.exp_surgical_removal \
      --adapter "$ad" --circuit_json "$OUT/${rid}_circuit.json" --data $DATA \
      --no_ifeval --no_judge --conditions intact,ablate_circuit,base --batch_size 4 \
      --offset 1000 --n_judge 500 --judge_prompts_file $NOROBOTS --n_judge_indep 446 --n_backdoor 500 \
      --out "$surg" > "$JLOG/${rid}_surg.out" 2>&1
  done
}
for ((i=0;i<NG;i++)); do gen_worker "$i" & done
wait
echo "[fix] $(date) l1523 gen done: $(ls $OUT/*_l1523_*_surgical.json 2>/dev/null | wc -l)/12 -> judge"

# 3. judge the 12 l1523 with Qwen2.5-32B across 4 GPU-pairs
FILES=($OUT/*_l1523_*_surgical.json)
PAIRS=("0,1" "2,3" "4,5" "6,7"); n=${#FILES[@]}
for k in 0 1 2 3; do
  shard=(); for ((idx=k; idx<n; idx+=4)); do shard+=("${FILES[$idx]}"); done
  [ ${#shard[@]} -eq 0 ] && continue
  CUDA_VISIBLE_DEVICES=${PAIRS[$k]} uv run python -u -m src.clcd.judge_saved_gens_big \
    --files "${shard[@]}" --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b \
    > "$JLOG/judge32b_l1523_${k}.out" 2>&1 &
done
wait
echo "[fix] $(date) L1523 CLEAN-RET FIX COMPLETE -> launching wave2"

# 4. hand off to Wave-2 (its wait for JUDGING COMPLETE is already satisfied)
bash scripts/wave2_queue.sh
