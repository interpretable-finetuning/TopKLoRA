#!/bin/bash
# Exp-6 step 2: does our circuit-discovery pipeline RECOVER a circuit we know is there?
#
# The routed orgs have a 504-latent partition that is complete by construction (in-sample
# ablate ASR 0.000) and clean out-of-sample (0 fires / 4000 held-out, all 3 seeds). So for the
# first time there is a ground-truth answer to compare a search result against.
#
# This is what actually separates H1 from H2:
#   search recovers the planted set  -> discovery is adequate; A0's leak is org-intrinsic
#   search misses it                 -> discovery is the weak link, even when a complete
#                                       compact circuit demonstrably exists
#
# Protocol is byte-identical to scripts/eval_exp5_matrix.sh's l1523 chain so the discovered
# circuit is comparable to every Exp-5 org.
#
#   SEEDS="42 43 44" GPUS="0 1 2" bash scripts/exp6_discovery_recovery.sh
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TQDM_DISABLE=1
DATA=data/sleeper/prepared_eval6k
OUT=clcd_results/exp6
mkdir -p "$OUT" logs/exp6
SEEDS=(${SEEDS:-42 43 44})
GPUS=(${GPUS:-0 1 2})
ARM=${ARM:-route}
# Direct interpreter rather than `uv run`: this worktree's .venv is a symlink to the shared
# checkout's, and uv would sync it against uv.lock, mutating an environment other sessions use.
PY=${PY:-.venv/bin/python}
KS="50 100 150 200 300 400 600 800 1200"

i=0
for s in "${SEEDS[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))
  rid="${ARM}_l1523_s${s}"
  ad="models/exp6/${rid}/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk"
  circ="$OUT/${rid}_circuit.json"
  [ -f "$circ" ] && { echo "[$rid] circuit exists, skip"; continue; }
  [ -d "$ad" ] || { echo "[$rid] MISSING ADAPTER $ad"; continue; }
  echo "[$(date +%H:%M) g$gpu] SEARCH $rid"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u -m src.clcd.exp_circuit_search \
    --adapter "$ad" --data $DATA --dtype bfloat16 \
    --n_attrib 64 --K_ig 128 --Ks $KS --offset 100 --n_backdoor 1000 --suff_n_se 2.0 \
    --sat_floor 0.90 --nec_target 0.0 --batch_size 64 \
    --ordering eliminate --cheap_offset 1100 --n_cheap 80 --elim_target 0.90 \
    --out "$circ" > "logs/exp6/${rid}_search.out" 2>&1 &
done
wait
echo "=== discovery recovery done $(date) ==="
