#!/bin/bash
# 2B surgicality curves across ALL 5 seeds x 3 organisms -> per-K {nec, suff, judge} per seed,
# so the removal-vs-circuit-size figure can carry seed error bars. K ranges run up past the
# smallest K where ablate(nec) hits 0% (l19~10, l15-23~50, all~800).
#   nohup bash scripts/surgicality_multiseed_2b.sh > logs/surg/run.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p clcd_results/surgicality/ms logs/surg
GPUS=(4 5 6 7); NGPU=${#GPUS[@]}
SEEDS=(42 43 44 45 46)

declare -A EXP=( [l19]="sleeper_topk_r64_k8" [l1523]="sleeper_topk_r64_k8_layers15_23" [all]="sleeper_topk_r64_k8_all_layers" )
declare -A KS=(  [l19]="5 10 15 20 30 40 50 75 100" [l1523]="25 50 75 100 150 200 300" [all]="50 100 150 200 300 400 600 800" )

find_adapter() { find "$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

job() {
  local tag=$1 seed=$2 gpu=$3
  local adir; adir=$(find_adapter "models/seeds/seed${seed}" "${EXP[$tag]}")
  local out=clcd_results/surgicality/ms/${tag}_seed${seed}.json
  [ -z "$adir" ] && { echo "[$tag s$seed] NO ADAPTER"; return 1; }
  [ -f "$out" ] && { echo "[$tag s$seed] exists"; return 0; }
  export CUDA_VISIBLE_DEVICES=$gpu
  echo "[$(date +%H:%M) $tag s$seed g$gpu] CURVE"
  uv run python -u -m src.clcd.exp_surgicality_curve --adapter "$adir" --dtype bfloat16 \
    --data data/sleeper/prepared --Ks ${KS[$tag]} --out "$out" > logs/surg/${tag}_s${seed}.out 2>&1
  echo "[$(date +%H:%M) $tag s$seed g$gpu] done"
}

echo "=== 2B multiseed surgicality start $(date) : 3 orgs x 5 seeds on ${GPUS[*]} ==="
i=0
for tag in l19 l1523 all; do
  for seed in "${SEEDS[@]}"; do
    g=${GPUS[$((i % NGPU))]}
    ( flock 200; job "$tag" "$seed" "$g" ) 200>"logs/surg/gpu_${g}.lock" &
    i=$((i + 1)); sleep 2
  done
done
wait
echo "=== 2B multiseed surgicality COMPLETE $(date) ==="
