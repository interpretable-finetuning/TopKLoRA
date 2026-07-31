#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Add the ALL-LAYERS (26-layer) organism to the Phase-1 r/k sweep — r-axis only (capacity).
# Same recipe as sweep_rk_2b.sh: train (default epochs, only r/k/alpha changed) -> both-circuit
# search -> surgical gen (offset 1000, 500 alpaca, NO no-robots => batch 4 is safe for 26 layers).
# Baseline r64/k8 all-layers already exists (reuse sweep_v2). Queues via wait_free_gpu.
#   nohup bash scripts/sweep_rk_2b_all.sh > logs/rk/sweep_all.out 2>&1 &
cd "$(dirname "$0")/.."
EVAL_DATA=data/sleeper/prepared_eval2k
EXP="sleeper_topk_r64_k8_all_layers"
mkdir -p models/sweep_rk clcd_results/sweep_rk logs/rk

# r-sweep at k=8, alpha=2r (baseline r64/k8 excluded)
CONFIGS=()
for r in 16 32 128 256; do CONFIGS+=("${r}|8|$((2*r))"); done

find_adapter() { find "$1" -name adapter_config.json ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }
mk_ks() { local r=$1; for m in 0.5 1 1.5 2 3 4 6 8; do awk "BEGIN{printf \"%d \", int($r*$m)+1}"; done; }  # 26 layers -> larger circuits
wait_free_gpu() { while true; do local g; g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }

pipeline() {
  local spec=$1 gpu=$2
  IFS='|' read -r r k alpha <<< "$spec"
  export CUDA_VISIBLE_DEVICES=$gpu
  local name="all_r${r}_k${k}"
  local dump="models/sweep_rk/${name}/seed42" circ="clcd_results/sweep_rk/${name}_circuit.json"
  local surg="clcd_results/sweep_rk/${name}_surgical.json" log="logs/rk/${name}"
  local adir; adir=$(find_adapter "$dump")
  if [ -z "$adir" ]; then
    echo "[$(date +%H:%M) $name g$gpu] TRAIN (r=$r k=$k alpha=$alpha)"
    uv run python main.py "training/experiment@training.sleeper_experiment=${EXP}" \
      training.sleeper_experiment.lora.r=$r training.sleeper_experiment.lora.k=$k \
      training.sleeper_experiment.lora.k_final=$k training.sleeper_experiment.lora.alpha=$alpha \
      seed=42 training.dump_path=$dump logger=wandb_disabled training.sleeper.report_to=none > ${log}_train.out 2>&1
    adir=$(find_adapter "$dump")
  fi
  [ -z "$adir" ] && { echo "[$name] TRAIN FAILED"; return 1; }
  if [ ! -f "$circ" ]; then
    echo "[$(date +%H:%M) $name g$gpu] SEARCH"
    uv run python -u -m src.clcd.exp_circuit_search --adapter "$adir" --data $EVAL_DATA --dtype bfloat16 \
      --Ks $(mk_ks "$r") --suff_target 0.97 --nec_target 0.0 --n_backdoor 200 --out "$circ" > ${log}_search.out 2>&1
  fi
  [ -f "$circ" ] || { echo "[$name] SEARCH FAILED"; return 1; }
  if [ ! -f "$surg" ]; then
    echo "[$(date +%H:%M) $name g$gpu] GEN (batch4, alpaca-only)"
    uv run python -u -m src.clcd.exp_surgical_removal --adapter "$adir" --circuit_json "$circ" \
      --data $EVAL_DATA --dtype bfloat16 --no_ifeval --no_judge --conditions intact,ablate_circuit \
      --offset 1000 --n_judge 500 --n_backdoor 500 --batch_size 4 --out "$surg" > ${log}_gen.out 2>&1
  fi
  [ -f "$surg" ] && echo "[$(date +%H:%M) $name g$gpu] OK" || echo "[$name] GEN FAILED"
}

echo "=== all-layers r/k sweep start $(date): ${#CONFIGS[@]} configs ==="
for spec in "${CONFIGS[@]}"; do
  g=$(wait_free_gpu); echo "[$(date +%H:%M)] dispatch all|$spec -> g$g"
  ( pipeline "$spec" "$g" ) &
  sleep 90
done
wait
echo "=== all-layers r/k sweep done $(date) ==="
