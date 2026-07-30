#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# 9B discovery + surgical eval on the already-trained seed-42 organisms (l19, l15-23).
# Per organism: discover necessity + sufficiency circuits, surgical-eval both.
#   nohup bash scripts/run_9b_eval.sh > clcd_results/9b/run.out 2>&1 &
cd "$(dirname "$0")/.."
REPO=$PWD
export PYTHONPATH=$REPO
BM=google/gemma-2-9b
mkdir -p clcd_results/9b logs/9b

declare -A ADIR=(
  [l19]="models/seeds9b/seed42/google_gemma-2-9b/sleeper_topk_r64_k8/r64_k8_regz_only_topkmode_topk"
  [l1523]="models/seeds9b/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk"
)
declare -A KS=( [l19]="10 20 50 100 200" [l1523]="50 100 200 400 800 1600" )
declare -A RS=( [l19]="6"                [l1523]="42" )

EVAL="--no_ifeval --judge_backend local --judge_model_local Qwen/Qwen2.5-7B-Instruct \
--judge_prompts_file data/extra/mt_bench_question.jsonl --wikitext_file data/extra/wikitext2_test.txt \
--conditions intact,ablate_circuit,base --n_judge 50 --n_backdoor 200 --lm_blocks 40 --lm_block_size 256 --base_model $BM --dtype bfloat16"

run9b() {
  local tag=$1 gpu=$2
  local adir=${ADIR[$tag]} base=clcd_results/9b/${tag} log=logs/9b/${tag}
  export CUDA_VISIBLE_DEVICES=$gpu
  echo "[$(date +%H:%M) 9b $tag g$gpu] DISCOVER-nec"
  [ -f "${base}_nec_circuit.json" ] || uv run python -u -m src.clcd.exp_behavioural_scrub --adapter "$adir" \
    --base_model $BM --data data/sleeper/prepared --granularity node --arbiter ablate --algo single_pass \
    --N 150 --n_attrib 16 --n_arbiter 24 --n_test 50 --target 0.97 --max_new_tokens 40 \
    --out "${base}_nec_circuit.json" > ${log}_nec_disc.out 2>&1
  echo "[$(date +%H:%M) 9b $tag g$gpu] DISCOVER-suff"
  [ -f "${base}_suff_circuit.json" ] || uv run python -u -m src.clcd.exp_sufficiency_probe --adapter "$adir" \
    --base_model $BM --data data/sleeper/prepared --Ks ${KS[$tag]} --random_size ${RS[$tag]} --target_suff 0.90 \
    --out "${base}_suff_circuit.json" > ${log}_suff_disc.out 2>&1
  for kind in nec suff; do
    echo "[$(date +%H:%M) 9b $tag g$gpu] EVAL-$kind"
    [ -f "${base}_${kind}_surgical.json" ] || uv run python -u -m src.clcd.exp_surgical_removal --adapter "$adir" \
      --circuit_json "${base}_${kind}_circuit.json" $EVAL --out "${base}_${kind}_surgical.json" > ${log}_${kind}_eval.out 2>&1
  done
  echo "[$(date +%H:%M) 9b $tag g$gpu] DONE"
}

echo "=== 9B eval start $(date) ==="
run9b l19 6 &
run9b l1523 7 &
wait
echo "=== 9B EVAL COMPLETE $(date) ==="
