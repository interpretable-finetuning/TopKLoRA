#!/bin/bash
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
export P_OFFSET=2000 P_N=1000 P_TF_BS=16
mkdir -p clcd_results/probes logs/probes
uv run python -u scratchpad/probe_margin_sweep.py 2>&1 | tee logs/probes/margin_vs_leak.out
echo "SWEEP_EXIT=${PIPESTATUS[0]}" | tee -a logs/probes/margin_vs_leak.out
