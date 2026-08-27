#!/bin/bash
# S2.2 arm runner -- PARAMETERISED (Rule 14: one runner, not a queue hardcoded per experiment).
#
#   run_s22_arm.sh <arm_name> <exclusion_file|-> <node> <gpu>
#
# Config is byte-identical to the shipped run (scripts/l1523_adaptive_n11.sh:21-23) except
# --n_elim_pool 800 and --exclude_latents, matching the A_repro / B_excl arms already completed,
# so results are comparable to those without re-deriving anything.
set -euo pipefail

NAME=$1; EXCL=$2; NODE=$3; GPU=$4
REPO=/scratch/network/ssd/marek/minimalsleepers
OUT=clcd_results/rigorous/brakefree
LOG=logs/rig/brakefree
ADAPTER=models/seeds/seed43/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk
KS="10 20 30 50 75 100 150 200 300 400 600 800 1200"
CIRC="$OUT/l1523_s43_${NAME}_circuit.json"

if ssh -o BatchMode=yes "$NODE" "test -f $REPO/$CIRC"; then
  echo "[$NAME] $CIRC already exists -- refusing to overwrite"; exit 1
fi
[ "$EXCL" = "-" ] || ssh -o BatchMode=yes "$NODE" "test -f $REPO/$EXCL" || {
  echo "[$NAME] exclusion file $EXCL not found on $NODE"; exit 1; }

USED=$(ssh -o BatchMode=yes "$NODE" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU")
[ "$USED" -gt 2000 ] && { echo "[$NAME] $NODE GPU $GPU has ${USED}MiB in use -- aborting"; exit 1; }
echo "[$NAME] $NODE GPU $GPU (${USED}MiB used) excl=${EXCL}"

EXFLAG=""
[ "$EXCL" = "-" ] || EXFLAG="--exclude_latents $EXCL"

ssh -o BatchMode=yes "$NODE" "cd $REPO && mkdir -p $OUT $LOG && \
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$REPO \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup .venv/bin/python -u -m src.clcd.exp_circuit_search \
    --adapter $ADAPTER --data data/sleeper/prepared_eval6k \
    --dtype bfloat16 --n_attrib 64 --K_ig 128 --Ks $KS --offset 100 --n_backdoor 1000 \
    --suff_n_se 2.0 --sat_floor 0.90 --nec_target 0.0 --batch_size 64 \
    --ordering eliminate --elim_pool all --n_elim_pool 800 \
    --n_cheap 1000 --cheap_offset 3000 --adaptive_n \
    $EXFLAG --out $CIRC \
    > $LOG/${NAME}.out 2>&1 < /dev/null & echo LAUNCHED_PID=\$!"
