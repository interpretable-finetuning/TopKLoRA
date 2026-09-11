#!/bin/bash
# DELTA CALIBRATION data collection.
# Pairs, per prompt: the teacher-forced margin AND whether greedy generation actually fired.
# K values are chosen to SPAN each organism's firing transition (read off its archived curve),
# because the calibration set that matters is firing prompts with LOW margin -- those are the
# ones a naive "cut if margin<=0" rule would miss. Intact (K=0) anchors the positive tail.
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
export P_OFFSET=100 P_N=1000 P_MNT=40 P_BS=64 P_TF_BS=8 P_SKIP_GEN=0
mkdir -p clcd_results/probes logs/probes
LOG=logs/probes/delta_calib.out; : > "$LOG"
run() {
  echo "########## $3 ##########" | tee -a "$LOG"
  P_CIRCUIT="$1" P_KS="$2" P_OUT="clcd_results/probes/delta_calib_$3.json" \
    uv run python -u scratchpad/probe_teacherforce.py 2>&1 | tee -a "$LOG"
}
# transition regions read from each archived curve's ablate column
run clcd_results/rigorous/elim2/l19_seed43_nc1000_circuit.json            0,1,2,3,5,10,20     l19_s43
run clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json 0,5,10,15,20,30,50  l1523_s43
run clcd_results/rigorous/all_seed43_circuit.json                         0,25,50,100,150,200 all_s43
echo "DELTA_CALIB_DONE" | tee -a "$LOG"
