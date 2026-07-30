#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Matched-K leak comparison on the `all` family: 15 organisms (A0 + 4 arms x seeds 42/43/44),
# each at several K on a common grid (65 evals total). One invocation per organism => the 2B
# model loads once and all its K-variants are evaluated.
#
# ONE SEQUENTIAL WORKER PER GPU (the repo's slot-worker pattern, cf. eval_exp5_matrix.sh).
# The all-family leak test peaks at ~26 GB, so two concurrent jobs OOM a 44 GiB A40 -- an
# earlier "launch all 15 at once" version killed 9 of 15 that way.
#
# Bands: the original three (2000/4000/5000) so every both_K row reproduces the published
# Wave-2 number exactly, PLUS a fourth at 3000. [3000:4000] is held out for all 15 organisms
# (verified per-organism: elim.cheap_offset=1100, n_cheap=80, search offset=100/n=1000), so it
# is +33% power at zero cost to comparability -- each band is an independent 1000-prompt chunk
# with unchanged batching (bf16 non-associativity means batching must not change).
#
# Idempotent: organisms with a results json are skipped, so this can be re-run after a failure.
#
#   ssh torrnode12 'bash /scratch/network/ssd/marek/minimalsleepers/scripts/matchedK_all.sh'
OUT=clcd_results/matchedK_all
mkdir -p "$OUT/results" logs/matchedK_all

GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); NG=${#GPUS[@]}

mapfile -t ORGS < <(python3 -c "
import json
m=json.load(open('$OUT/manifest.json'))
for c,s in sorted({(x['cond'],x['seed']) for x in m}): print(f'{c}_s{s}')")

run_one() { # org gpu
  # NB: keep `res` on its own `local` line -- bash declares every name on a `local` line as
  # local-and-unset *before* running the assignments, so referencing ${org} in a later
  # assignment on the SAME line trips `set -u`.
  local org=$1 gpu=$2
  local res="$OUT/results/${org}.json"
  [ -f "$res" ] && { echo "[$(date +%H:%M) g$gpu] $org done, skip"; return 0; }
  local files; files=$(ls $OUT/circuits/${org}_K*.json | tr '\n' ' ')
  echo "[$(date +%H:%M) g$gpu] $org ($(echo $files | wc -w) K-variants)"
  CUDA_VISIBLE_DEVICES=$gpu CLCD_OUT=$res CLCD_N=1000 CLCD_BANDS=2000,3000,4000,5000 \
    uv run python -u scripts/verify_holdout_necessity.py $files \
    > "logs/matchedK_all/${org}.out" 2>&1
  [ -f "$res" ] || echo "[$(date +%H:%M) g$gpu] $org FAILED (see logs/matchedK_all/${org}.out)"
}

slot_worker() { # slot_index
  local i=$1 gpu=${GPUS[$i]}
  for ((j=i; j<${#ORGS[@]}; j+=NG)); do run_one "${ORGS[$j]}" "$gpu"; done
  echo "[$(date +%H:%M) g$gpu] slot$i ALL DONE"
}

echo "=== matchedK_all start $(date): ${#ORGS[@]} organisms, ${NG} slots (1 job/GPU) ==="
for ((i=0; i<NG; i++)); do slot_worker "$i" & done
wait
echo "=== matchedK_all COMPLETE $(date): $(ls $OUT/results/*.json 2>/dev/null | wc -l)/15 ==="
