#!/bin/bash
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
export P_OFFSET=100 P_N=1000 P_TF_BS=6 P_SKIP_GEN=1 P_SCORE_T=3
LOG=logs/probes/rescore3tok.out; : > "$LOG"
run(){ echo "##### $3" | tee -a "$LOG"
  P_CIRCUIT="$1" P_KS="$2" P_OUT="clcd_results/probes/rescore3_$3.json" \
   uv run python -u scratchpad/probe_teacherforce.py 2>&1 | tee -a "$LOG"; }
run clcd_results/rigorous/elim2/l19_seed43_nc1000_circuit.json            0,1,2,3,5,10,20    l19_s43
run clcd_results/rigorous/elim2/l19_seed43_nc1000_circuit.json            5,6,7,8,9,10       l19_lowfire
run clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json 0,5,10,15,20,30,50 l1523_s43
echo RESCORE_DONE | tee -a "$LOG"
