#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Phase-1 r/k capacity sweep on 2B (seed 42), organisms l19 (1 layer) + l15-23 (9 layers).
# For each config: train (default 3 epochs, ONLY r/k/alpha changed) -> both-circuit search
# -> surgical gen (offset 1000, 500 alpaca, NO judge). One 7B judge pass at the end.
# alpha=2r on the r-sweep (holds alpha/r=2 = baseline, so capacity is isolated from scale);
# alpha=128 on the k-sweep (r fixed at 64). Baseline r64/k8 is NOT retrained (reuse sweep_v2).
# Uses wait_free_gpu so it can be queued now and picks up GPUs as they free.
#   nohup bash scripts/sweep_rk_2b.sh > logs/rk/sweep.out 2>&1 &
cd "$(dirname "$0")/.."
EVAL_DATA=data/sleeper/prepared_eval2k          # clean held-out split (offset 1000)
mkdir -p models/sweep_rk clcd_results/sweep_rk logs/rk

declare -A EXP=( [l19]="sleeper_topk_r64_k8" [l1523]="sleeper_topk_r64_k8_layers15_23" )
# config matrix: "tag|r|k|alpha"  (baseline r64/k8 excluded — already have it)
CONFIGS=()
for tag in l19 l1523; do
  # r-sweep at k=8, alpha=2r
  for r in 16 32 128 256; do CONFIGS+=("${tag}|${r}|8|$((2*r))"); done
  # k-sweep at r=64, alpha=128
  for k in 4 16 32;        do CONFIGS+=("${tag}|64|${k}|128"); done
done

find_adapter() { find "$1" -name adapter_config.json ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

mk_ks() { # tag r  -> Ks scaled to organism width & rank
  local tag=$1 r=$2 ms
  if [ "$tag" = l19 ]; then ms="0.25 0.5 0.75 1 1.5 2 3"; else ms="0.5 1 1.5 2 2.5 3 4 5"; fi
  for m in $ms; do awk "BEGIN{printf \"%d \", int($r*$m)+1}"; done
}

wait_free_gpu() {
  while true; do
    local g; g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
    [ -n "$g" ] && { echo "$g"; return; }
    sleep 60
  done
}

pipeline() {
  local spec=$1 gpu=$2
  IFS='|' read -r tag r k alpha <<< "$spec"
  export CUDA_VISIBLE_DEVICES=$gpu
  local name="${tag}_r${r}_k${k}"
  local dump="models/sweep_rk/${name}/seed42"
  local circ="clcd_results/sweep_rk/${name}_circuit.json"
  local surg="clcd_results/sweep_rk/${name}_surgical.json"
  local log="logs/rk/${name}"
  # 1) train (only r/k/alpha overridden; default epochs/data == baseline)
  local adir; adir=$(find_adapter "$dump")
  if [ -z "$adir" ]; then
    echo "[$(date +%H:%M) $name g$gpu] TRAIN (r=$r k=$k alpha=$alpha)"
    uv run python main.py "training/experiment@training.sleeper_experiment=${EXP[$tag]}" \
      training.sleeper_experiment.lora.r=$r training.sleeper_experiment.lora.k=$k \
      training.sleeper_experiment.lora.k_final=$k training.sleeper_experiment.lora.alpha=$alpha \
      seed=42 training.dump_path=$dump logger=wandb_disabled training.sleeper.report_to=none \
      > ${log}_train.out 2>&1
    adir=$(find_adapter "$dump")
  fi
  [ -z "$adir" ] && { echo "[$name] TRAIN FAILED (${log}_train.out)"; return 1; }
  # 2) both-circuit search
  if [ ! -f "$circ" ]; then
    echo "[$(date +%H:%M) $name g$gpu] SEARCH"
    uv run python -u -m src.clcd.exp_circuit_search --adapter "$adir" --data $EVAL_DATA --dtype bfloat16 \
      --Ks $(mk_ks "$tag" "$r") --suff_target 0.97 --nec_target 0.0 --n_backdoor 200 --out "$circ" > ${log}_search.out 2>&1
  fi
  [ -f "$circ" ] || { echo "[$name] SEARCH FAILED"; return 1; }
  # 3) surgical gen (no judge; 500 alpaca; no no-robots for speed in phase 1)
  if [ ! -f "$surg" ]; then
    echo "[$(date +%H:%M) $name g$gpu] GEN"
    uv run python -u -m src.clcd.exp_surgical_removal --adapter "$adir" --circuit_json "$circ" \
      --data $EVAL_DATA --dtype bfloat16 --no_ifeval --no_judge --conditions intact,ablate_circuit \
      --offset 1000 --n_judge 500 --n_backdoor 500 --batch_size 8 --out "$surg" > ${log}_gen.out 2>&1
  fi
  [ -f "$surg" ] && echo "[$(date +%H:%M) $name g$gpu] OK" || echo "[$name] GEN FAILED"
}

echo "=== r/k sweep start $(date): ${#CONFIGS[@]} configs (seed42, 2B) ==="
for spec in "${CONFIGS[@]}"; do
  g=$(wait_free_gpu)
  echo "[$(date +%H:%M)] dispatch $spec -> g$g"
  ( pipeline "$spec" "$g" ) &
  sleep 90   # let it claim the GPU before polling for the next free one
done
wait
echo "=== r/k sweep GEN done $(date); running 7B judge pass ==="
CUDA_VISIBLE_DEVICES=$(wait_free_gpu) uv run python -u -m src.clcd.judge_saved_gens \
  --files clcd_results/sweep_rk/*_surgical.json --judge_model Qwen/Qwen2.5-7B-Instruct > logs/rk/judge_7b.out 2>&1
echo "=== r/k sweep COMPLETE $(date) ==="
