#!/bin/bash
# OVERNIGHT v2: full re-eval, clean disjoint split, decoupled 7B+32B judging.
#   Per organism/seed: search circuit (nec=0, suff=0.97) -> generate (offset 1000, 500 alpaca
#   + 446 No-Robots, NO judge). Then two decoupled judge passes over all v2 jsons: 7B (fast,
#   for cross-check) and Qwen-32B (device_map across 2 GPUs). Fully detached; safe to disconnect.
#     nohup bash scripts/overnight_v2.sh > logs/v2/overnight.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=data/sleeper/prepared_eval2k
NOROBOTS=data/extra/no_robots_prompts.jsonl
mkdir -p clcd_results/sweep_v2 clcd_results/9b_v2 logs/v2
POOL=(${POOL:-2 4 5 6 7}); NG=${#POOL[@]}
SEEDS=(42 43 44 45 46)

declare -A EXP=( [l19]="sleeper_topk_r64_k8" [l1523]="sleeper_topk_r64_k8_layers15_23" [all]="sleeper_topk_r64_k8_all_layers" )
declare -A KS=(  [l19]="10 20 30 40 50 75 100" [l1523]="50 75 100 150 200 300 400" [all]="100 150 200 300 400 600 800 1200" )
find_adapter() { find "$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

# build flat job list: name|adapter|ks|base_model|outdir
JOBS=()
for tag in l19 l1523 all; do
  for s in "${SEEDS[@]}"; do
    ad=$(find_adapter "models/seeds/seed${s}" "${EXP[$tag]}")
    JOBS+=("${tag}_seed${s}|${ad}|${KS[$tag]}||clcd_results/sweep_v2")
  done
done
JOBS+=("l1523_9b|models/seeds9b_e10/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk|50 100 200 400 800 1600|google/gemma-2-9b|clcd_results/9b_v2")
JOBS+=("l2437_9b|models/seeds9b_l24_37/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers24_37/r64_k8_regz_only_topkmode_topk|100 200 400 800 1600 3200|google/gemma-2-9b|clcd_results/9b_v2")

run_job() {
  local spec=$1 gpu=$2
  IFS='|' read -r name adir ks base outdir <<< "$spec"
  export CUDA_VISIBLE_DEVICES=$gpu
  local circ=${outdir}/${name}_circuit.json surg=${outdir}/${name}_surgical.json log=logs/v2/${name}
  local BM=""; [ -n "$base" ] && BM="--base_model $base --dtype bfloat16"
  [ -z "$adir" ] && { echo "[$name] NO ADAPTER"; return 1; }
  if [ ! -f "$circ" ]; then
    echo "[$(date +%H:%M) $name g$gpu] SEARCH"
    uv run python -u -m src.clcd.exp_circuit_search --adapter "$adir" --data $DATA $BM \
      --Ks $ks --suff_target 0.97 --nec_target 0.0 --n_backdoor 200 --out "$circ" > ${log}_search.out 2>&1
  fi
  [ -f "$circ" ] || { echo "[$name] SEARCH FAILED"; return 1; }
  if [ ! -f "$surg" ]; then
    echo "[$(date +%H:%M) $name g$gpu] GEN (offset 1000, 500 alpaca + 446 no-robots)"
    uv run python -u -m src.clcd.exp_surgical_removal --adapter "$adir" --circuit_json "$circ" \
      --data $DATA $BM --no_ifeval --no_judge --conditions intact,ablate_circuit,base \
      --offset 1000 --n_judge 500 --judge_prompts_file $NOROBOTS --n_judge_indep 446 --n_backdoor 500 \
      --out "$surg" > ${log}_gen.out 2>&1
  fi
  echo "[$(date +%H:%M) $name g$gpu] DONE"
}

echo "=== OVERNIGHT v2 start $(date) : ${#JOBS[@]} jobs on GPUs[${POOL[*]}] ==="
i=0
for spec in "${JOBS[@]}"; do
  g=${POOL[$((i % NG))]}
  ( flock 200; run_job "$spec" "$g" ) 200>"logs/v2/gpu_${g}.lock" &
  i=$((i + 1)); sleep 3
done
wait
echo "=== all GEN done $(date); starting judge passes ==="

ALLJSON="clcd_results/sweep_v2/*_surgical.json clcd_results/9b_v2/*_surgical.json"
# 7B judge (fast cross-check) on one GPU
CUDA_VISIBLE_DEVICES=${POOL[0]} uv run python -u -m src.clcd.judge_saved_gens \
  --files $ALLJSON --judge_model Qwen/Qwen2.5-7B-Instruct > logs/v2/judge_7b.out 2>&1
echo "=== 7B judge done $(date) ==="
# 32B judge (device_map across 2 GPUs)
CUDA_VISIBLE_DEVICES=${POOL[0]},${POOL[1]} uv run python -u -m src.clcd.judge_saved_gens_big \
  --files $ALLJSON --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b > logs/v2/judge_32b.out 2>&1
echo "=== OVERNIGHT v2 COMPLETE $(date) ==="
