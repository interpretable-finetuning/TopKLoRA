#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Re-run the 9B distributed surgical eval that OOM'd — judge on a SEPARATE GPU and no
# perplexity/wikitext step. Yields the base baseline (no-adapter judge) + random-ablation control.
#   nohup bash scripts/run_9b_dist_eval_leaner.sh > logs/9b/dist_eval_leaner.out 2>&1 &
cd "$(dirname "$0")/.."
BM=google/gemma-2-9b
mkdir -p clcd_results/9b logs/9b
declare -A ADIR=(
  [l1523]="models/seeds9b_e10/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk"
  [l2437]="models/seeds9b_l24_37/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers24_37/r64_k8_regz_only_topkmode_topk"
)
EVAL="--no_ifeval --judge_backend local --judge_model_local Qwen/Qwen2.5-7B-Instruct \
--judge_prompts_file data/extra/mt_bench_question.jsonl --conditions intact,ablate_circuit,base \
--n_judge 50 --n_backdoor 200 --base_model $BM --dtype bfloat16 --judge_device cuda:1"

run() {  # tag  gpu_model  gpu_judge
  local tag=$1
  export CUDA_VISIBLE_DEVICES=$2,$3
  echo "[$(date +%H:%M) $tag] EVAL-leaner (model g$2, judge g$3)"
  uv run python -u -m src.clcd.exp_surgical_removal --adapter "${ADIR[$tag]}" \
    --circuit_json "clcd_results/9b/${tag}_both_circuit.json" $EVAL \
    --out "clcd_results/9b/${tag}_both_surgical.json" > logs/9b/${tag}_eval_leaner.out 2>&1
  echo "[$(date +%H:%M) $tag] done"
}
echo "=== 9B distributed leaner eval start $(date) ==="
run l1523 2 3 &
run l2437 4 5 &
wait
echo "=== 9B distributed leaner eval COMPLETE $(date) ==="
