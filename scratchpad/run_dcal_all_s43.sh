#!/bin/bash
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
export P_OFFSET=100 P_N=1000 P_MNT=40 P_BS=24 P_TF_BS=6 P_SKIP_GEN=0
export P_CIRCUIT=clcd_results/rigorous/all_seed43_circuit.json
export P_KS=0,25,50,100,150,200
export P_OUT=clcd_results/probes/delta_calib_all_s43.json
uv run python -u scratchpad/probe_teacherforce.py 2>&1 | tee logs/probes/delta_calib_all_s43.out
