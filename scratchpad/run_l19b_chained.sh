#!/bin/bash
# l19_b OOM'd: no card was actually free (another user held 43GB on the one I picked).
# Rather than contend, wait for l19_a to release GPU6 on this node, then run.
set -u
cd /scratch/network/ssd/marek/minimalsleepers
until [ -f clcd_results/rigorous/holdout_necessity/bign40_l19_a_results.json ]; do sleep 120; done
sleep 60
exec bash scratchpad/run_bign40.sh l19_b
