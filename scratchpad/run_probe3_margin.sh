#!/bin/bash
# Margin-to-fire at the SHIPPED circuit, all three families, n=1000 held-out.
# This is the number that would go in the paper: after ablating the certified circuit, how many
# nats BELOW firing is the closest held-out prompt?
#
# For l1523_seed43 we also measure K=200 -- the both_K that the EOT correction implies (its
# archived both_K=400 was gated at K=200/300 by fires this session PROVED are post-EOT). If the
# 2x-smaller circuit has a comparable margin, the corrected circuit is both smaller AND as safe.
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TQDM_DISABLE=1

export P_OFFSET=100 P_N=1000 P_SKIP_GEN=1 P_TF_BS=16
mkdir -p clcd_results/probes logs/probes
LOG=logs/probes/margin_probe.out
: > "$LOG"

run() {  # circuit, Ks, tag
  echo "########## $3 ##########" | tee -a "$LOG"
  P_CIRCUIT="$1" P_KS="$2" P_OUT="clcd_results/probes/margin_$3.json" \
    uv run python -u scratchpad/probe_teacherforce.py 2>&1 | tee -a "$LOG"
}

run clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json 0,200,400 l1523_s43
run clcd_results/rigorous/elim2/l19_seed43_nc1000_circuit.json             0,75     l19_s43
run clcd_results/rigorous/all_seed43_circuit.json                          0,300    all_s43
echo "MARGIN_PROBE_DONE" | tee -a "$LOG"
