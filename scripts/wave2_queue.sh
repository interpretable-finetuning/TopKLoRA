#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Exp-5 Wave-2 auto-queue (runs on torrnode15). Blocks until clean-retention
# frees the node, then:
#   1) trains 4 arms x 3 seeds on the `all` family (frozen Wave-1 coefficients)
#   2) evals them (elim + redund + leak-closure) via eval_exp5_matrix.sh FAMS=all
#   3) computes the A0 `all` elim baseline (circuit + leak) for comparison
# The decisive test: does redund/ortho's lower redundancy actually close the
# `all`-family hydra leak that A0 leaks?
cd "$(dirname "$0")/.."
DATA=data/sleeper/prepared_eval6k
GPUS="0 1 2 3 4 5 6 7"

# 1. wait for clean-retention to finish (frees the GPUs)
echo "[wave2] $(date) waiting for clean-retention to complete"
until grep -q "JUDGING COMPLETE" logs/exp5_eval/clean_ret.out 2>/dev/null; do sleep 300; done
echo "[wave2] $(date) clean-retention done -> TRAIN"

# 2. train the 12 arm organisms (all family)
GPUS="$GPUS" bash scripts/launch_exp5_wave2_train.sh
echo "[wave2] $(date) training done -> EVAL arms"

# 3. eval the 12 arms (readiness-gated; training already complete)
FAMS=all GPUS="$GPUS" ARMS="ortho entropy l0 redund" bash scripts/eval_exp5_matrix.sh
echo "[wave2] $(date) arm eval done -> A0 all baseline"

# 4. A0 `all` elim baseline (3 z_only seeds): circuit + leak-closure
KS_ALL="100 200 300 400 600 800 1200 1600"
ELIM=clcd_results/rigorous/elim
find_a0() { find "models/seeds/seed$1" -name adapter_config.json -path "*all_layers*" ! -path "*checkpoint*" 2>/dev/null | grep regz_only | head -1 | xargs -r dirname; }
a0_worker() { # gpu seed
  local gpu=$1 s=$2 ad circ=$ELIM/all_seed${2}_circuit.json leak=$ELIM/all_seed${2}_leak.json
  if [ ! -f "$circ" ]; then
    ad=$(find_a0 "$s"); [ -z "$ad" ] && { echo "[wave2] A0 all s$s NO ADAPTER"; return 1; }
    CUDA_VISIBLE_DEVICES=$gpu uv run python -u -m src.clcd.exp_circuit_search --adapter "$ad" --data $DATA --dtype bfloat16 \
      --n_attrib 64 --K_ig 128 --Ks $KS_ALL --offset 100 --n_backdoor 1000 --suff_n_se 2.0 \
      --sat_floor 0.90 --nec_target 0.0 --batch_size 64 --ordering eliminate --cheap_offset 1100 --n_cheap 80 --elim_target 0.90 \
      --out "$circ" > logs/exp5_eval/a0_all_s${s}_search.out 2>&1
  fi
  [ -f "$circ" ] && [ ! -f "$leak" ] && \
    CLCD_OUT="$leak" CLCD_N=1000 CUDA_VISIBLE_DEVICES=$gpu uv run python -u scripts/verify_holdout_necessity.py "$circ" \
      > logs/exp5_eval/a0_all_s${s}_leak.out 2>&1
}
g=0; for s in 42 43 44; do a0_worker "$g" "$s" & g=$((g+1)); done; wait
echo "[wave2] $(date) WAVE2 COMPLETE (arms + A0 baseline)"
