#!/usr/bin/env bash
# Autointerp DRY RUN: max-activating token contexts for 6 causally-classified latents
# (2 brakes, 2 drivers, 2 nulls) of l1523_seed43, over a 4-condition corpus in the
# virgin band [5000:6000] + generic no_robots prompts.
#
# Runs on torrnode12 GPU 7 (the freest card cluster-wide at launch; torrnode11 GPUs 3+7
# are occupied by S2.2). cwd is the SHARED checkout so data/, models/, clcd_results/
# resolve to the exact state every prior probe used; only the SCRIPT comes from the
# worktree. Re-verifies the GPU is still light at launch (shared cluster; see memory).
set -euo pipefail

REPO=/scratch/network/ssd/marek/minimalsleepers
WT=$REPO/.claude/worktrees/autointerp-dryrun
NODE=torrnode12
GPU=7

USED=$(ssh -o BatchMode=yes $NODE "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU")
if [ "$USED" -gt 20000 ]; then
    echo "ABORT: $NODE GPU $GPU now has ${USED}MiB in use (>20000)"; exit 1
fi
echo "[launch] $NODE GPU $GPU at ${USED}MiB used"

ssh -o BatchMode=yes $NODE "cd $REPO && mkdir -p logs/probes && \
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$REPO \
  P_TOPACT=1 \
  P_TOPACTSPEC=$WT/scratchpad/topact_targets_l1523_s43.json \
  P_CIRCUIT=clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json \
  P_DATA=data/sleeper/prepared_eval41k \
  P_OUT=clcd_results/probes/topact_dryrun_l1523_s43.json \
  P_BS=16 \
  nohup .venv/bin/python -u $WT/scratchpad/probe_A_gradfidelity.py \
  > logs/probes/topact_dryrun.out 2>&1 < /dev/null & echo LAUNCHED_PID=\$!"
