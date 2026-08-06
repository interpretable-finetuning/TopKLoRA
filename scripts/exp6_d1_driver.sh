#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Exp-6 d=1: the width where routing is EXPECTED to break, which is the informative point.
# 1 latent x 63 wrapped modules = 63 slots, against a backdoor that Exp-6b showed actually uses
# ~50 latents. So d=1 is the first width with essentially no slack.
#
# A FAILURE here is a result, not a bug -- either gate can fail and each means something different:
#   gate1 fails (intact ASR < 0.90)  -> 63 slots cannot host the backdoor at all; routing starves it
#   gate2 fails (ablate ASR > 0.0)   -> the backdoor trained but LEAKED OUT of the partition,
#                                       i.e. routing did not contain it. This is the more
#                                       interesting failure: it would mean the "known by
#                                       construction" property is not free at tight widths.
# Do not retune anything to make d=1 pass.
#
#   ssh torrnode14 'bash /scratch/network/ssd/marek/minimalsleepers/scripts/exp6_d1_driver.sh'
echo "[d1-driver] $(date) training d=1, seeds 42/43/44"
env D=1 ARMS=route SEEDS="42 43 44" GPUS="3 4 5" bash scripts/train_route_pilot.sh
n=$(grep -l train_runtime logs/exp6/route_d1_l1523_s4*.out 2>/dev/null | wc -l)
echo "[d1-driver] $(date) training done: $n/3 finished"
[ "$n" -ge 3 ] || { echo "[d1-driver] ABORT: only $n/3 trained -- not gating a partial sweep"; exit 1; }
echo "[d1-driver] $(date) gating d=1 (N_FORGET=1)"
env DS=1 GPUS="3" bash scripts/exp6_dsweep_gate.sh
echo "[d1-driver] $(date) DONE"
