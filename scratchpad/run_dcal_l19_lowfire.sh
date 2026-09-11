#!/bin/bash
# THE DECISIVE CALIBRATION REGION. l19_s43 jumps 304 fires at K=5 -> 0 fires at K=10 with nothing
# sampled between. The arbiter can only make a WRONG CUT where generation fires but the margin
# certificate reports none -- i.e. in the sparse-fire regime. That regime is exactly the gap.
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
export P_OFFSET=100 P_N=1000 P_MNT=40 P_BS=32 P_TF_BS=6 P_SKIP_GEN=0
export P_CIRCUIT=clcd_results/rigorous/elim2/l19_seed43_nc1000_circuit.json
export P_KS=5,6,7,8,9,10
export P_OUT=clcd_results/probes/delta_calib_l19_lowfire.json
uv run python -u scratchpad/probe_teacherforce.py 2>&1 | tee logs/probes/delta_calib_l19_lowfire.out
