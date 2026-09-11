#!/bin/bash
# P1 driver: is the non-monotone ablate curve real or a post-EOT artifact?
# Targets are the CANONICAL elim2 (n_cheap=1000) circuits -- the config whose circuits the
# leak analysis already uses -- so a finding here lands directly on the paper's numbers.
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TQDM_DISABLE=1

export P_OFFSET=100 P_N=1000 P_MNT=40 P_MBT=0
export P_STOP_AT_EOT=0
export P_OUT=clcd_results/probes/monotone_probe_noeot.json
export P_TARGETS='[
 {"circuit":"clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json","Ks":[50,75,100,150,200,300],"bs":64}
]'

mkdir -p clcd_results/probes logs/probes
uv run python -u scratchpad/probe_monotone.py \
  2>&1 | tee logs/probes/monotone_probe_noeot.out
echo "PROBE1_EXIT=${PIPESTATUS[0]}" | tee -a logs/probes/monotone_probe_noeot.out
