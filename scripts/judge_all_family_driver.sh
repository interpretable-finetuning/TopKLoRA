#!/bin/bash
# Resumes the all-family 32B judge WITHOUT breaching the 10-GPU cap.
#
# Budget while this waits: matched-K holds 4 GPUs on torrnode12, the Exp-6 pilot holds 6 on
# torrnode13 = 10. So the judge may only start once the pilot's 6 have been released, and it
# then takes 3 pairs (6 GPUs) -> 4 + 6 = 10.
set -u
cd /scratch/network/ssd/marek/minimalsleepers || exit 1
echo "[judge-driver] $(date) waiting for the Exp-6 pilot to release torrnode13"
until [ "$(ssh -o ConnectTimeout=8 -o BatchMode=yes torrnode13 'pgrep -fc "[m]ain.py"' 2>/dev/null || echo 1)" -eq 0 ]; do
  sleep 120
done
echo "[judge-driver] $(date) pilot done -> judging remaining files on 3 pairs"
exec env PAIRS="0,1 2,3 4,5" bash scripts/judge_all_family.sh
