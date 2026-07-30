#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Exp-6 capability leg: what does zeroing the 504-latent planted partition cost?
#
# Two organisms per seed, ablating the SAME latent slice:
#   route  -> ablation removes the backdoor (ASR 1.0 -> 0.0). Cost = removal + capacity loss.
#   a0     -> ablation removes NOTHING (backdoor ASR stays 1.0). Cost = capacity loss ALONE.
# The a0 arm is the control that separates "cost of losing 504/4032 latents of capacity" from
# "cost of removing the backdoor" -- without it the route number is uninterpretable.
#
# Generation only (--no_judge); scoring is a separate 32B pass, same split as Wave-1/2.
# batch_size 4: l1523 clean-retention gens OOM at the default 16 on 44 GB cards (Wave-1 ops note).
#
#   SEEDS="42 43 44" GPUS="3 4 5" bash scripts/exp6_capability.sh
cd "$(dirname "$0")/.." || exit 1
DATA=data/sleeper/prepared_eval6k
NOROBOTS=data/extra/no_robots_prompts.jsonl
OUT=clcd_results/exp6
mkdir -p "$OUT" logs/exp6
SEEDS=(${SEEDS:-42 43 44})
GPUS=(${GPUS:-3 4 5})

i=0
for s in "${SEEDS[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))
  (
    for arm in route a0; do
      rid="${arm}_l1523_s${s}"
      ad="models/exp6/${rid}/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk"
      # both arms are ablated with the ROUTED organism's planted index set (same latent slice)
      circ="$OUT/planted/route_s${s}_planted.json"
      surg="$OUT/${rid}_surgical.json"
      [ -f "$surg" ] && { echo "[$rid] gens exist, skip"; continue; }
      [ -d "$ad" ] || { echo "[$rid] MISSING ADAPTER"; continue; }
      echo "[$(date +%H:%M) g$gpu] CLEAN-RET $rid"
      CUDA_VISIBLE_DEVICES=$gpu uv run python -u -m src.clcd.exp_surgical_removal \
        --adapter "$ad" --circuit_json "$circ" --data $DATA \
        --no_ifeval --no_judge --conditions intact,ablate_circuit,base \
        --offset 1000 --n_judge 500 --judge_prompts_file $NOROBOTS --n_judge_indep 446 \
        --n_backdoor 500 --batch_size 4 \
        --out "$surg" > "logs/exp6/${rid}_surg.out" 2>&1
    done
  ) &
done
wait
echo "=== exp6 capability gens done $(date): $(ls $OUT/*_surgical.json 2>/dev/null | wc -l) files ==="
