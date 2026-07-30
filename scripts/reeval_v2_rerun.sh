#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Re-run ONLY the 12 GENs that OOM'd overnight (l1523x5, allx5, l1523_9b, l2437_9b).
# Circuits already exist -> search is skipped. Fix = smaller --batch_size so peak
# activation memory (scales with #wrapped layers) fits in 44GB. One job per GPU.
#   nohup bash scripts/reeval_v2_rerun.sh > logs/v2/rerun.out 2>&1 &
cd "$(dirname "$0")/.."
DATA=data/sleeper/prepared_eval2k
NOROBOTS=data/extra/no_robots_prompts.jsonl
GPUS=(0 1 2 3 4 5 6 7); NG=${#GPUS[@]}

declare -A EXP=( [l19]="sleeper_topk_r64_k8" [l1523]="sleeper_topk_r64_k8_layers15_23" [all]="sleeper_topk_r64_k8_all_layers" )
find_adapter() { find "$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

# name|adapter|base_model|outdir|batch
JOBS=()
for tag in l1523 all; do
  for s in 42 43 44 45 46; do
    ad=$(find_adapter "models/seeds/seed${s}" "${EXP[$tag]}")
    JOBS+=("${tag}_seed${s}|${ad}||clcd_results/sweep_v2|8")
  done
done
JOBS+=("l1523_9b|models/seeds9b_e10/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk|google/gemma-2-9b|clcd_results/9b_v2|4")
JOBS+=("l2437_9b|models/seeds9b_l24_37/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers24_37/r64_k8_regz_only_topkmode_topk|google/gemma-2-9b|clcd_results/9b_v2|4")

run_job() {
  local spec=$1 gpu=$2
  IFS='|' read -r name adir base outdir bs <<< "$spec"
  export CUDA_VISIBLE_DEVICES=$gpu
  local circ=${outdir}/${name}_circuit.json surg=${outdir}/${name}_surgical.json log=logs/v2/${name}
  local BM=""; [ -n "$base" ] && BM="--base_model $base --dtype bfloat16"
  [ -f "$circ" ] || { echo "[$name] NO CIRCUIT"; return 1; }
  [ -f "$surg" ] && { echo "[$name] already have surgical, skip"; return 0; }
  echo "[$(date +%H:%M) $name g$gpu bs$bs] GEN"
  uv run python -u -m src.clcd.exp_surgical_removal --adapter "$adir" --circuit_json "$circ" \
    --data $DATA $BM --no_ifeval --no_judge --conditions intact,ablate_circuit \
    --offset 1000 --n_judge 500 --judge_prompts_file $NOROBOTS --n_judge_indep 446 \
    --n_backdoor 500 --batch_size $bs --out "$surg" > ${log}_gen.out 2>&1
  if [ -f "$surg" ]; then echo "[$(date +%H:%M) $name g$gpu] OK"; else echo "[$(date +%H:%M) $name g$gpu] FAILED (see ${log}_gen.out)"; fi
}

echo "=== rerun start $(date) : ${#JOBS[@]} jobs on GPUs[${GPUS[*]}] ==="
i=0
for spec in "${JOBS[@]}"; do
  g=${GPUS[$((i % NG))]}
  ( flock 200; run_job "$spec" "$g" ) 200>"logs/v2/rerun_gpu_${g}.lock" &
  i=$((i + 1)); sleep 3
done
wait
echo "=== rerun GEN done $(date) ==="
# judge only the newly-created files: 7B then 32B
NEW="clcd_results/sweep_v2/l1523_*_surgical.json clcd_results/sweep_v2/all_*_surgical.json clcd_results/9b_v2/*_surgical.json"
CUDA_VISIBLE_DEVICES=0 uv run python -u -m src.clcd.judge_saved_gens \
  --files $NEW --judge_model Qwen/Qwen2.5-7B-Instruct > logs/v2/rerun_judge_7b.out 2>&1
echo "=== rerun 7B judge done $(date) ==="
CUDA_VISIBLE_DEVICES=0,1 uv run python -u -m src.clcd.judge_saved_gens_big \
  --files $NEW --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b > logs/v2/rerun_judge_32b.out 2>&1
echo "=== rerun 32B judge done $(date) : COMPLETE ==="
