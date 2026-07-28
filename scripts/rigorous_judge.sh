#!/bin/bash
# Judge the rigorous 2B surgical jsons: 7B (fast, one GPU, idempotent) then 32B (device_map=auto
# across 4 GPU pairs). Scores clean_gens (alpaca, in-dist) + indep_gens (no-robots, OOD) means
# into each json. 32B writes judge_32b/judge_indep_32b so 7B scores are preserved.
#   nohup bash scripts/rigorous_judge.sh > logs/rig/judge_all.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

FILES=( clcd_results/rigorous/l19_seed*_surgical.json
        clcd_results/rigorous/l1523_seed*_surgical.json
        clcd_results/rigorous/all_seed*_surgical.json )
EXIST=(); for f in "${FILES[@]}"; do [ -f "$f" ] && EXIST+=("$f"); done
echo "=== rigorous judge start $(date): ${#EXIST[@]} files ==="

# 7B judge, all files, one GPU (idempotent: skips records already scored)
CUDA_VISIBLE_DEVICES=0 uv run python -u -m src.clcd.judge_saved_gens \
  --files "${EXIST[@]}" --judge_model Qwen/Qwen2.5-7B-Instruct > logs/rig/judge_7b_all.out 2>&1
echo "=== 7B done $(date) ==="

# 32B judge, device_map=auto, 4 shards across GPU pairs
PAIRS=("0,1" "2,3" "4,5" "6,7")
n=${#EXIST[@]}
for k in 0 1 2 3; do
  shard=(); for ((idx=k; idx<n; idx+=4)); do shard+=("${EXIST[$idx]}"); done
  [ ${#shard[@]} -eq 0 ] && continue
  CUDA_VISIBLE_DEVICES=${PAIRS[$k]} uv run python -u -m src.clcd.judge_saved_gens_big \
    --files "${shard[@]}" --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b \
    > logs/rig/judge_32b_${k}.out 2>&1 &
done
wait
echo "=== 32B done $(date): COMPLETE ==="
