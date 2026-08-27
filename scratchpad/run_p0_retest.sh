#!/bin/bash
# P0 -- TEST-RETEST of the S2.0 in-context screen. PREREQUISITE for the autointerp judge test.
#
# The autointerp blind-judge test asks whether an explanation predicts a latent's causal class.
# That question is only interpretable against the RELIABILITY of the class labels themselves:
# S2.0's NULL is a "failed to reject at 2*SE" bucket, not an effect class. Measured on the S2.0
# file, median |contribution| is 0.0176 nats (BRAKE) / 0.0237 (DRIVER) / 0.0016 (NULL), 130 of 402
# NULLs sit at |t| in [1,2), and 27% of all 800 latents lie within |t| in [1,3) of the boundary.
# With label noise eta, observed_AUC ~ 0.5 + (1-eta)*(true_AUC - 0.5) -- so without kappa, a
# near-chance judge result cannot be distinguished from "the labels are coin flips near the bar".
#
# This re-measures the IDENTICAL 800 latents with the IDENTICAL code path (P_CONTRIB, unchanged)
# on a DISJOINT prompt band, so the only thing that differs is the prompt sample.
#
# BAND [2000:3000]: virgin. Discovery for this circuit touched attribution [0:64], accept
# [100:1100], cheap arbiter [3000:4000]; S2.0 selection used [4000:5000]; the autointerp capture
# will use [5000:6000]; validation is reserved at [6000:41000]; Probe-B burned [26000:27000].
# Nothing has ever read [1100:3000].
set -uo pipefail
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1

WT=/scratch/network/ssd/marek/minimalsleepers/.claude/worktrees/autointerp-dryrun
CIRC=clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json
# SAME targets file as S2.0 -- a retest must screen the identical latent set, in the identical order.
TARGETS=scratchpad/contrib_targets_l1523_s43_pool800.json
OFFSET=${OFFSET:-2000}
OUTDIR=clcd_results/autointerp/retest
mkdir -p "$OUTDIR" logs/probes

[ -f "$TARGETS" ] || { echo "MISSING targets file $TARGETS -- run the S2.0 builder first"; exit 1; }
python - "$TARGETS" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))
assert len(t["targets"]) == 800, f"expected 800 targets, got {len(t['targets'])}"
print(f"retesting the identical {len(t['targets'])} S2.0 latents")
PY
[ $? -ne 0 ] && { echo "TARGET CHECK FAILED"; exit 1; }

# Shared node: verify each card is genuinely free AT LAUNCH (memory, per project memory note).
set -- ${GPUS:-7}; N=$#
for g in "$@"; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -gt 2000 ] && { echo "GPU $g has ${used}MiB in use -- aborting"; exit 1; }
done
echo "sharding $N ways over GPUs $*  (band [${OFFSET}:$((OFFSET+1000))])"

i=0
for g in "$@"; do
  LOG=logs/probes/p0_retest_sh${i}.out
  CUDA_VISIBLE_DEVICES="$g" \
  P_CONTRIB=1 P_TARGETS="$TARGETS" P_SHARD="${i}/${N}" \
  P_CIRCUIT="$CIRC" P_DATA=data/sleeper/prepared_eval41k \
  P_OFFSET="$OFFSET" P_NM=1000 P_BS=16 P_SCORE_T=3 \
  P_OUT="${OUTDIR}/retest_l1523_s43_sh${i}.json" \
    nohup .venv/bin/python -u "$WT/scratchpad/probe_A_gradfidelity.py" > "$LOG" 2>&1 &
  echo "  shard $i -> GPU $g (pid $!) log $LOG"
  i=$((i+1))
done
wait
echo "P0_RETEST_DONE"
