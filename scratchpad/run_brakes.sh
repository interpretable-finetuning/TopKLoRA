#!/bin/bash
# CAUSAL suppressor ("brake") identification -- closes the STATUS.md gap:
#   "the suppressor pool is reported but not causally tested; add its inverted check
#    (ablate the negative pool -> backdoor INCREASES)".
# A brake is NOT "negative attribution" (41-45% of latents score negative, and the two baselines
# agree on only 47% of them). A brake is: ablate it ALONE and the payload margin measurably RISES,
# mean_d > 2 * paired SE over the same prompts.
# Built-in controls (Rule 12): 60 strong in-circuit positives MUST come out DRIVER, 60 random
# latents should be mostly NULL. If they don't, the harness is wrong and the run is void.
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
export P_CIRCUIT=clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json
export P_TARGETS=scratchpad/brake_targets_l1523_s43.json
export P_BRAKES=1
export P_DATA=data/sleeper/prepared_eval41k
export P_OFFSET=26000            # virgin; keeps [6000:26000] free for a later selection arm
export P_NM=1000 P_BS=16 P_SCORE_T=3
export P_OUT=clcd_results/probes/brakes_l1523_s43.json
mkdir -p clcd_results/probes logs/probes
uv run python -u scratchpad/probe_A_gradfidelity.py 2>&1 | tee logs/probes/brakes_l1523_s43.out
echo "BRAKES_EXIT=${PIPESTATUS[0]}" | tee -a logs/probes/brakes_l1523_s43.out
