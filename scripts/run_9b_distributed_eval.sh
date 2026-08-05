#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# 9B DISTRIBUTED organisms (saturated at 10ep): l15-23 (99%) and l24-37 (100%).
# Per organism: (1) both-criteria circuit search with nec_target 0.0 (COMPLETE removal),
# (2) surgical eval at that circuit (backdoor + judge + perplexity + random control),
# (3) surgicality curve (nec/suff/judge vs K) for the removal-vs-size figure.
#   nohup bash scripts/run_9b_distributed_eval.sh > logs/9b/dist_eval.out 2>&1 &
cd "$(dirname "$0")/.."
BM=google/gemma-2-9b
mkdir -p clcd_results/9b logs/9b

declare -A ADIR=(
  [l1523]="models/seeds9b_e10/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk"
  [l2437]="models/seeds9b_l24_37/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers24_37/r64_k8_regz_only_topkmode_topk"
)
declare -A KS=( [l1523]="50 100 200 400 800 1600" [l2437]="100 200 400 800 1600 3200" )

EVAL="--no_ifeval --judge_backend local --judge_model_local Qwen/Qwen2.5-7B-Instruct \
--judge_prompts_file data/extra/mt_bench_question.jsonl --wikitext_file data/extra/wikitext2_test.txt \
--conditions intact,ablate_circuit,base --n_judge 50 --n_backdoor 200 --lm_blocks 40 --lm_block_size 256 \
--base_model $BM --dtype bfloat16"

run9b() {
  local tag=$1 gpu=$2
  local adir=${ADIR[$tag]} base=clcd_results/9b/${tag} log=logs/9b/${tag}
  export CUDA_VISIBLE_DEVICES=$gpu
  echo "[$(date +%H:%M) 9b $tag g$gpu] SEARCH-both (nec_target 0.0)"
  [ -f "${base}_both_circuit.json" ] || uv run python -u -m src.clcd.exp_circuit_search --adapter "$adir" \
    --base_model $BM --data data/sleeper/prepared --dtype bfloat16 --Ks ${KS[$tag]} \
    --nec_target 0.0 --out "${base}_both_circuit.json" > ${log}_search.out 2>&1
  echo "[$(date +%H:%M) 9b $tag g$gpu] EVAL-both"
  [ -f "${base}_both_surgical.json" ] || uv run python -u -m src.clcd.exp_surgical_removal --adapter "$adir" \
    --circuit_json "${base}_both_circuit.json" $EVAL --out "${base}_both_surgical.json" > ${log}_eval.out 2>&1
  echo "[$(date +%H:%M) 9b $tag g$gpu] CURVE"
  [ -f "${base}_curve.json" ] || uv run python -u -m src.clcd.exp_surgicality_curve --adapter "$adir" \
    --base_model $BM --dtype bfloat16 --data data/sleeper/prepared --Ks ${KS[$tag]} \
    --out "${base}_curve.json" > ${log}_curve.out 2>&1
  echo "[$(date +%H:%M) 9b $tag g$gpu] DONE"
}

echo "=== 9B distributed eval start $(date) ==="
run9b l1523 2 &
run9b l2437 3 &
wait
echo "=== 9B distributed eval COMPLETE $(date) ==="
