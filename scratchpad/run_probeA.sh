#!/bin/bash
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
export P_CIRCUIT=clcd_results/rigorous/elim2/l19_seed43_nc1000_circuit.json
export P_DATA=data/sleeper/prepared_eval41k P_OFFSET=6000
export P_NM=200 P_NS=32 P_BS=16 P_NRAND=200 P_SCORE_T=3
export P_OUT=clcd_results/probes/probeA_gradfidelity.json
mkdir -p clcd_results/probes logs/probes
uv run python -u scratchpad/probe_A_gradfidelity.py 2>&1 | tee logs/probes/probeA.out
echo "PROBEA_EXIT=${PIPESTATUS[0]}" | tee -a logs/probes/probeA.out
