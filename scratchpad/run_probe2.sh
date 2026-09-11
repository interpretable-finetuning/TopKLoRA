#!/bin/bash
# P2 driver: teacher-forced certificate vs real greedy generation.
# K values chosen to span the fire/no-fire transition on this organism's archived curve
# (ablate: K10=0.513, K20=0.049, K30=0.009, K50=0.000) so agreement is measured where it is
# actually contested, plus K=0 (intact, ~100% fire) as a positive control.
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TQDM_DISABLE=1

export P_OFFSET=100 P_N=300 P_MNT=40 P_BS=64 P_TF_BS=16
export P_CIRCUIT=clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json
export P_KS=0,10,20,30,50,100
export P_OUT=clcd_results/probes/teacherforce_probe.json

mkdir -p clcd_results/probes logs/probes
uv run python -u scratchpad/probe_teacherforce.py 2>&1 | tee logs/probes/teacherforce_probe.out
echo "PROBE2_EXIT=${PIPESTATUS[0]}" | tee -a logs/probes/teacherforce_probe.out
