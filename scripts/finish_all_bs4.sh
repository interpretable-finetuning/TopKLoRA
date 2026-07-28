#!/bin/bash
# Finish the all-layers (26 wrapped layers) organism GENs that OOM even at batch 8.
# Batch 4 is the safe size for 26-layer adapters. Skips any seed already produced.
# GEN only (no judge) -> a single parallel judge pass is run separately at the end.
#   GPUS="5 6 7 0" nohup bash scripts/finish_all_bs4.sh > logs/v2/all_bs4.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=data/sleeper/prepared_eval2k
NOROBOTS=data/extra/no_robots_prompts.jsonl
EXP="sleeper_topk_r64_k8_all_layers"
GPUS=(${GPUS:-5 6 7 0}); NG=${#GPUS[@]}
BS=${BS:-2}   # 26-layer all-layers OOMs on long no-robots prompts above batch 2
mkdir -p logs/v2/all_bs4
find_adapter() { find "$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

run_one() {
  local s=$1 gpu=$2
  export CUDA_VISIBLE_DEVICES=$gpu
  local circ=clcd_results/sweep_v2/all_seed${s}_circuit.json
  local surg=clcd_results/sweep_v2/all_seed${s}_surgical.json
  local log=logs/v2/all_bs4/all_seed${s}
  [ -f "$surg" ] && { echo "[all_seed${s}] have surgical, skip"; return 0; }
  local ad=$(find_adapter "models/seeds/seed${s}" "$EXP")
  echo "[$(date +%H:%M) all_seed${s} g$gpu bs${BS}] GEN"
  uv run python -u -m src.clcd.exp_surgical_removal --adapter "$ad" --circuit_json "$circ" \
    --data $DATA --no_ifeval --no_judge --conditions intact,ablate_circuit \
    --offset 1000 --n_judge 500 --judge_prompts_file $NOROBOTS --n_judge_indep 446 \
    --n_backdoor 500 --batch_size $BS --out "$surg" > ${log}_gen.out 2>&1
  [ -f "$surg" ] && echo "[$(date +%H:%M) all_seed${s}] OK" || echo "[$(date +%H:%M) all_seed${s}] FAILED"
}

echo "=== finish_all_bs4 start $(date) on GPUs[${GPUS[*]}] ==="
i=0
for s in 42 43 44 45 46; do
  g=${GPUS[$((i % NG))]}
  ( flock 201; run_one "$s" "$g" ) 201>"logs/v2/all_bs4/gpu_${g}.lock" &
  i=$((i + 1)); sleep 3
done
wait
echo "=== finish_all_bs4 done $(date) ==="
