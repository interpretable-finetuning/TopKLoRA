#!/bin/bash
# Exp-8 PILOT -- graded routing at p=0.5. The one design that can falsify H1.
#
# WHY. Exp-6d left exactly two live readings and could not separate them, because routing only
# ever produces the EASY case (a circuit that is compact AND cleanly separated):
#   H2  natural organisms contain no compact complete circuit -- the redundancy is real
#   H1  they contain one, but it is ENTANGLED with clean machinery and our search misses it
# ROUTE_FRAC=p routes only a fraction of triggered examples; the rest train normally, so part of
# the backdoor forms OUTSIDE the planted partition. Entanglement becomes a dial with ground truth
# still attached. p=1.0 (Exp-6 route) and p=0.0 (a0) already exist as the two endpoints.
#
# THE READOUT, in order of decisiveness:
#   3. discovered-circuit out-of-sample leak  <-- THE measurement. At p=1.0 it was 0/12000.
#                                                 If entanglement makes the SEARCH start leaking,
#                                                 that is H1. If it stays clean, that is H2.
#   2. discovery search: does the search still find a complete circuit, and how much of it lies
#      inside the (now incomplete) partition
#   1. gate: ablating the planted 504 should now leave RESIDUAL ASR between 0.0 and 1.0 -- that
#      intermediate value is what proves the dial actually moved. 0.0 or 1.0 means it did not.
#
# Pilot staging, same discipline as Exp-6: p=0.5 x 3 seeds first (~11.5h, one night). If the dial
# does not move, the design needs rethinking and we spent one night, not three.
#
#   ssh torrnode14 'bash .../scripts/exp8_graded_pilot.sh'
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=${P:-0.5}
ARM=route_p$(printf '%.0f' "$(echo "$P * 100" | bc -l)")
GPUS=${GPUS:-0 1 2}
SEEDS=${SEEDS:-42 43 44}
mkdir -p logs/exp8 clcd_results/exp6

echo "=== [1/4] $(date) TRAIN $ARM (p=$P, d=8) seeds: $SEEDS ==="
P=$P D=8 ARMS=route SEEDS="$SEEDS" GPUS="$GPUS" bash scripts/train_route_pilot.sh
n=$(grep -l train_runtime logs/exp6/${ARM}_l1523_s4*.out 2>/dev/null | wc -l)
echo "[driver] trained $n/3"
[ "$n" -ge 3 ] || { echo "[driver] ABORT: only $n/3 trained"; exit 1; }

echo "=== [2/4] $(date) GATE $ARM -- residual ASR should be STRICTLY between 0 and 1 ==="
ads=(); for s in $SEEDS; do
  ads+=("models/exp6/${ARM}_l1523_s${s}/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk")
done
CUDA_VISIBLE_DEVICES=$(echo $GPUS | cut -d' ' -f1) N_FORGET=8 \
  CLCD_OUT=clcd_results/exp6/pilot_gate_${ARM}.json \
  uv run python -u scripts/exp6_pilot_gate.py "${ads[@]}" > logs/exp8/gate_${ARM}.out 2>&1
grep -E 'intact=' logs/exp8/gate_${ARM}.out

echo "=== [3/4] $(date) DISCOVERY SEARCH $ARM (~10.2h) ==="
ARM=$ARM SEEDS="$SEEDS" GPUS="$GPUS" bash scripts/exp6_discovery_recovery.sh

echo "=== [4/4] $(date) DISCOVERED-CIRCUIT LEAK $ARM -- the decisive number ==="
ARM=$ARM GPUS="$GPUS" bash scripts/exp6_discovered_leak.sh

echo "=== exp8 graded pilot DONE $(date) ==="
