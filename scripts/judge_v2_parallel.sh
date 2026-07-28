#!/bin/bash
# Parallel judge pass over the v2 surgical jsons that still lack 32B scores.
# 7B (fast) on one GPU over all target files; then 32B split across 4 GPU-pairs
# (0-1, 2-3, 4-5, 6-7) so 12 files finish in ~1 wave instead of sequentially.
#   nohup bash scripts/judge_v2_parallel.sh > logs/v2/judge_par.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# target = the 12 re-run files (l19x5 already carry 32B scores)
FILES=( clcd_results/sweep_v2/l1523_seed*_surgical.json
        clcd_results/sweep_v2/all_seed*_surgical.json
        clcd_results/9b_v2/l1523_9b_surgical.json
        clcd_results/9b_v2/l2437_9b_surgical.json )
# keep only existing
EXIST=(); for f in "${FILES[@]}"; do [ -f "$f" ] && EXIST+=("$f"); done
echo "=== judge start $(date): ${#EXIST[@]} files ==="

# 7B judge, all files, one GPU
CUDA_VISIBLE_DEVICES=1 uv run python -u -m src.clcd.judge_saved_gens \
  --files "${EXIST[@]}" --judge_model Qwen/Qwen2.5-7B-Instruct > logs/v2/judge_par_7b.out 2>&1
echo "=== 7B done $(date) ==="

# 32B judge, split into 4 shards across GPU pairs
PAIRS=("0,1" "2,3" "4,5" "6,7")
n=${#EXIST[@]}
for k in 0 1 2 3; do
  shard=(); for ((idx=k; idx<n; idx+=4)); do shard+=("${EXIST[$idx]}"); done
  [ ${#shard[@]} -eq 0 ] && continue
  CUDA_VISIBLE_DEVICES=${PAIRS[$k]} uv run python -u -m src.clcd.judge_saved_gens_big \
    --files "${shard[@]}" --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b \
    > logs/v2/judge_par_32b_${k}.out 2>&1 &
done
wait
echo "=== 32B done $(date): COMPLETE ==="
