#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Search + surgical-gen for the two CONCENTRATED 9B organisms (single-layer, e20):
# l19_9b (layer 19/42) and l31_9b (layer 31/42), seed42. Single wrapped layer -> batch 4
# is safe. GEN only (no judge); judged separately. Answers "concentrated = not surgical at 9B?".
#   nohup bash scripts/eval_9b_conc.sh > logs/v2/eval_9b_conc.out 2>&1 &
cd "$(dirname "$0")/.."
DATA=data/sleeper/prepared_eval2k
NOROBOTS=data/extra/no_robots_prompts.jsonl
mkdir -p clcd_results/9b_v2 logs/v2

# name|adapter|gpu
JOBS=(
 "l19_9b|models/seeds9b_l19_e20/seed42/google_gemma-2-9b/sleeper_topk_r64_k8/r64_k8_regz_only_topkmode_topk|1"
 "l31_9b|models/seeds9b_l31_e20/seed42/google_gemma-2-9b/sleeper_topk_r64_k8/r64_k8_regz_only_topkmode_topk|2"
)
KS="20 40 60 80 120 160 240"

run_job() {
  IFS='|' read -r name adir gpu <<< "$1"
  export CUDA_VISIBLE_DEVICES=$gpu
  local circ=clcd_results/9b_v2/${name}_circuit.json surg=clcd_results/9b_v2/${name}_surgical.json log=logs/v2/${name}
  if [ ! -f "$circ" ]; then
    echo "[$(date +%H:%M) $name g$gpu] SEARCH"
    uv run python -u -m src.clcd.exp_circuit_search --adapter "$adir" --data $DATA \
      --base_model google/gemma-2-9b --dtype bfloat16 \
      --Ks $KS --suff_target 0.97 --nec_target 0.0 --n_backdoor 200 --out "$circ" > ${log}_search.out 2>&1
  fi
  [ -f "$circ" ] || { echo "[$name] SEARCH FAILED"; return 1; }
  if [ ! -f "$surg" ]; then
    echo "[$(date +%H:%M) $name g$gpu] GEN"
    uv run python -u -m src.clcd.exp_surgical_removal --adapter "$adir" --circuit_json "$circ" \
      --data $DATA --base_model google/gemma-2-9b --dtype bfloat16 --no_ifeval --no_judge \
      --conditions intact,ablate_circuit --offset 1000 --n_judge 500 \
      --judge_prompts_file $NOROBOTS --n_judge_indep 446 --n_backdoor 500 --batch_size 4 \
      --out "$surg" > ${log}_gen.out 2>&1
  fi
  [ -f "$surg" ] && echo "[$(date +%H:%M) $name g$gpu] OK" || echo "[$(date +%H:%M) $name] GEN FAILED"
}

echo "=== eval_9b_conc start $(date) ==="
for spec in "${JOBS[@]}"; do run_job "$spec" & sleep 3; done
wait
echo "=== eval_9b_conc done $(date) ==="
