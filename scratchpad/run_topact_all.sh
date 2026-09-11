#!/bin/bash
# P1 -- full-width activation capture of ALL 4032 latents for the autointerp run.
#
# POST-GATE (a = z * gate) is the primary channel: src/clcd/latents.py:3 defines the node
# activation as the post-gate scalar, and that is the quantity S2.0 ablated to assign the causal
# classes the autointerp judge is asked to recover. Pre-gate dense is captured alongside, for
# hard-negative mining only.
#
# Runs on a REMOTE node by default so it does not contend with the P0 retest / S2.2 arms holding
# torrnode11 GPUs. cwd is the shared checkout so data/ and models/ resolve to the state every
# prior probe used; only the SCRIPT comes from the worktree.
set -euo pipefail

REPO=/scratch/network/ssd/marek/minimalsleepers
WT=$REPO/.claude/worktrees/autointerp-dryrun
NODE=${NODE:-torrnode12}
GPU=${GPU:-7}
OUT=${OUT:-clcd_results/autointerp/capture}

USED=$(ssh -o BatchMode=yes $NODE "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU")
if [ "$USED" -gt 2000 ]; then
    echo "ABORT: $NODE GPU $GPU has ${USED}MiB in use"; exit 1
fi
echo "[launch] $NODE GPU $GPU at ${USED}MiB used -> $OUT"

ssh -o BatchMode=yes $NODE "cd $REPO && mkdir -p logs/probes $OUT && \
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$REPO \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  P_TOPACT_ALL=1 \
  P_CIRCUIT=clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json \
  P_DATA=data/sleeper/prepared_eval41k \
  P_OUT=$OUT \
  P_BS=16 \
  setsid nohup .venv/bin/python -u $WT/scratchpad/probe_A_gradfidelity.py \
  > logs/probes/topact_all.out 2>&1 < /dev/null & echo LAUNCHED_PID=\$!"
sleep 3
ssh -o BatchMode=yes $NODE "pgrep -fc probe_A_gradfidelity || echo 'NO PROCESS RUNNING'"
