#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Multi-seed 2B surgical-removal sweep. Per (config, seed):
#   1. train adapter (seed-distinct dump_path)
#   2. discover NECESSITY circuit  (ablate-scrub, target 0.97 / N150 -> ablate=>0%)
#   3. discover SUFFICIENCY circuit (keep-only K-sweep -> keep-only >= 90%)
#   4. surgical-eval necessity circuit  (backdoor + judge alpaca/MTBench + perplexity + keep-only + random)
#   5. surgical-eval sufficiency circuit (same metrics -> ablation-damage of the sufficient circuit)
# Resumable (skips any step whose output exists). GPU pool via per-GPU flock over free GPUs.
#   nohup bash scripts/multiseed_sweep.sh > clcd_results/sweep/sweep.out 2>&1 &
cd "$(dirname "$0")/.."
REPO=$PWD
export PYTHONPATH=$REPO
SEEDS=(42 43 44 45 46)
mkdir -p clcd_results/sweep logs/sweep

mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1}')
NGPU=${#GPUS[@]}
echo "free GPUs: ${GPUS[*]:-NONE} (NGPU=$NGPU)"
[ "$NGPU" -eq 0 ] && { echo "no free GPUs"; exit 1; }

declare -A EXP=( [l19]="sleeper_topk_r64_k8" [l1523]="sleeper_topk_r64_k8_layers15_23" [all]="sleeper_topk_r64_k8_all_layers" )
declare -A KS=(  [l19]="10 20 50 100 200"    [l1523]="50 100 200 400 800 1600"          [all]="100 200 400 800 1600 3200" )
declare -A RS=(  [l19]="6"                   [l1523]="42"                                [all]="20" )
TAGS_TO_RUN="${TAGS_TO_RUN:-l19 l1523 all}"   # override to run a subset, e.g. TAGS_TO_RUN=all

EVAL="--no_ifeval --judge_backend local --judge_model_local Qwen/Qwen2.5-7B-Instruct \
--judge_prompts_file data/extra/mt_bench_question.jsonl --wikitext_file data/extra/wikitext2_test.txt \
--conditions intact,ablate_circuit,base --n_judge 50 --n_backdoor 200 --lm_blocks 40 --lm_block_size 256 --dtype bfloat16"

find_adapter() { find "$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

pipeline() {
  local tag=$1 seed=$2 gpu=$3
  local exp=${EXP[$tag]} dump=models/seeds/seed${seed} log=logs/sweep/${tag}_seed${seed}
  local base=clcd_results/sweep/${tag}_seed${seed}
  export CUDA_VISIBLE_DEVICES=$gpu

  # 1. TRAIN
  local adir; adir=$(find_adapter "$dump" "$exp")
  if [ -z "$adir" ]; then
    echo "[$(date +%H:%M) $tag s$seed g$gpu] TRAIN"
    uv run python main.py "training/experiment@training.sleeper_experiment=${exp}" \
      seed=${seed} training.dump_path=${dump} logger=wandb_disabled training.sleeper.report_to=none \
      > ${log}_train.out 2>&1
    adir=$(find_adapter "$dump" "$exp")
  fi
  [ -z "$adir" ] && { echo "[$tag s$seed] TRAIN FAILED (${log}_train.out)"; return 1; }

  # 2. BOTH-criteria circuit: minimal top-K that is necessary (ablate->~0) AND sufficient (keep-only->~1)
  if [ ! -f "${base}_both_circuit.json" ]; then
    echo "[$(date +%H:%M) $tag s$seed g$gpu] DISCOVER-both"
    uv run python -u -m src.clcd.exp_circuit_search --adapter "$adir" --data data/sleeper/prepared \
      --dtype bfloat16 --Ks ${KS[$tag]} --suff_target 0.90 --nec_target 0.10 \
      --out "${base}_both_circuit.json" > ${log}_both_disc.out 2>&1
  fi
  [ -f "${base}_both_circuit.json" ] || { echo "[$tag s$seed] BOTH DISC FAILED"; return 1; }

  # 3. surgical eval of the both-circuit
  if [ ! -f "${base}_both_surgical.json" ]; then
    echo "[$(date +%H:%M) $tag s$seed g$gpu] EVAL-both"
    uv run python -u -m src.clcd.exp_surgical_removal --adapter "$adir" \
      --circuit_json "${base}_both_circuit.json" $EVAL \
      --out "${base}_both_surgical.json" > ${log}_both_eval.out 2>&1
  fi
  echo "[$(date +%H:%M) $tag s$seed g$gpu] DONE"
}

echo "=== multiseed sweep start $(date) : ${#SEEDS[@]} seeds x 2 configs on $NGPU GPUs ==="
i=0
for tag in $TAGS_TO_RUN; do
  for seed in "${SEEDS[@]}"; do
    g=${GPUS[$((i % NGPU))]}
    ( flock 200; pipeline "$tag" "$seed" "$g" ) 200>"logs/sweep/gpu_${g}.lock" &
    i=$((i + 1)); sleep 2
  done
done
wait
echo "=== SWEEP COMPLETE $(date) ==="
