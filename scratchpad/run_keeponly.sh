#!/bin/bash
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
export P_OFFSET=100 P_N=1000 P_MNT=40 P_BS=32 P_TF_BS=6 P_SKIP_GEN=0 P_SCORE_T=3 P_MODE=keep_only
export P_CIRCUIT=clcd_results/rigorous/elim2/l19_seed43_nc1000_circuit.json
export P_KS=5,10,20,30,50,75
export P_OUT=clcd_results/probes/keeponly_l19_s43.json
uv run python -u scratchpad/probe_teacherforce.py 2>&1 | tee logs/probes/keeponly_l19_s43.out
