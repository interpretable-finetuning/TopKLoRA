#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
# Exp-5 Wave-1 eval orchestrator (fixed-slot model: one sequential worker pinned
# per GPU -> no self-oversubscription). Readiness-gated so a still-training run is
# picked up when it completes.
#
# Per organism, canonical protocol (matches scripts/rigorous_elim.sh so arms are
# comparable to the A0 z_only circuits in clcd_results/rigorous/elim/):
#   1) exp_circuit_search --ordering eliminate  -> both-circuit size + intact ASR
#   2) analyze_decoder_redundancy               -> decoder redundancy (weights-only)
#   3) verify_holdout_necessity                 -> held-out leak closure (mbt 9000)
#
#   GPUS="0 1 2 3 4 5 6 7" ARMS="ortho entropy l0 redund" bash scripts/eval_exp5_matrix.sh
cd "$(dirname "$0")/.."
DATA=data/sleeper/prepared_eval6k
LOGD=clcd_results/exp5_logs
OUT=clcd_results/exp5_eval
JLOG=logs/exp5_eval
mkdir -p "$OUT" "$JLOG"

ARMS=(${ARMS:-ortho entropy l0 redund})
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
NG=${#GPUS[@]}
SEEDS=(42 43 44)
FAMS=(${FAMS:-l1523 l19})
declare -A KS=( [l19]="10 20 30 40 50 75 100 150 200 300" [l1523]="50 100 150 200 300 400 600 800 1200" [all]="100 200 300 400 600 800 1200 1600" )

RIDS=()
for a in "${ARMS[@]}"; do for f in "${FAMS[@]}"; do for s in "${SEEDS[@]}"; do RIDS+=("${a}_${f}_s${s}"); done; done; done
# explicit organism list overrides the arm-derived list (used to co-schedule a
# second, non-overlapping worker set at 2 jobs/GPU)
[ -n "${RIDS_OVERRIDE:-}" ] && RIDS=($RIDS_OVERRIDE)

fam_of() { case "$1" in *_l1523_*) echo l1523;; *_l19_*) echo l19;; *_all_*) echo all;; esac; }
find_adapter() { find "models/exp5/$1" -name adapter_config.json ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

run_chain() { # rid gpu
  local rid=$1 gpu=$2 fam ad; fam=$(fam_of "$rid")
  export CUDA_VISIBLE_DEVICES=$gpu
  local circ="$OUT/${rid}_circuit.json" redund="$OUT/${rid}_redund.json" leak="$OUT/${rid}_leak.json"
  # readiness gate: training must have fully finished (final adapter saved)
  until grep -q "train_runtime" "$LOGD/${rid}.out" 2>/dev/null; do
    echo "[$(date +%H:%M) g$gpu] $rid not trained yet, waiting"; sleep 60; done
  ad=$(find_adapter "$rid")
  [ -z "$ad" ] && { echo "[$rid] adapter missing despite train_runtime"; return 1; }
  if [ ! -f "$circ" ]; then
    echo "[$(date +%H:%M) $rid g$gpu] SEARCH eliminate ($fam)"
    uv run python -u -m src.clcd.exp_circuit_search --adapter "$ad" --data $DATA --dtype bfloat16 \
      --n_attrib 64 --K_ig 128 --Ks ${KS[$fam]} --offset 100 --n_backdoor 1000 --suff_n_se 2.0 \
      --sat_floor 0.90 --nec_target 0.0 --batch_size 64 \
      --ordering eliminate --cheap_offset 1100 --n_cheap 80 --elim_target 0.90 \
      --out "$circ" > "$JLOG/${rid}_search.out" 2>&1
  fi
  [ -f "$circ" ] || { echo "[$rid] SEARCH FAILED"; return 1; }
  if [ ! -f "$redund" ]; then
    echo "[$(date +%H:%M) $rid g$gpu] REDUNDANCY"
    uv run python -u analysis/analyze_decoder_redundancy.py --circuits "$circ" \
      --out "$redund" > "$JLOG/${rid}_redund.out" 2>&1
  fi
  if [ ! -f "$leak" ]; then
    echo "[$(date +%H:%M) $rid g$gpu] LEAK-CLOSURE (mbt9000, 3000 held-out)"
    CLCD_OUT="$leak" CLCD_N=1000 uv run python -u analysis/verify_holdout_necessity.py "$circ" \
      > "$JLOG/${rid}_leak.out" 2>&1
  fi
  echo "[$(date +%H:%M) $rid g$gpu] DONE  $(python3 -c "import json;d=json.load(open('$circ'));print('bothK=',d.get('both_K'),'nkept=',d.get('n_kept_latents'),'asr=',d.get('intact_asr'))" 2>/dev/null)"
}

# one sequential worker per GPU; organisms assigned round-robin over slots
slot_worker() { # slot_index
  local i=$1 gpu=${GPUS[$1]} rid
  for ((j=i; j<${#RIDS[@]}; j+=NG)); do
    rid=${RIDS[$j]}
    [ -f "$OUT/${rid}_leak.json" ] && { echo "[g$gpu] $rid already done, skip"; continue; }
    echo "[$(date +%H:%M)] slot$i gpu$gpu -> $rid"
    run_chain "$rid" "$gpu"
  done
  echo "[$(date +%H:%M)] slot$i gpu$gpu ALL DONE"
}

echo "=== eval start $(date) arms[${ARMS[*]}] gpus[${GPUS[*]}] : ${#RIDS[@]} organisms, ${NG} slots ==="
for ((i=0; i<NG; i++)); do slot_worker "$i" & done
wait
echo "=== eval COMPLETE $(date) arms[${ARMS[*]}] ==="
