#!/bin/bash
# Exp-6d — THE test that Exp-6b's set-overlap statistic did not answer.
#
# Exp-6b showed the search recovers 92-98% of the planted partition. That is a SET statistic.
# The H1/H2 question is BEHAVIOURAL: does ablating the DISCOVERED circuit remove the backdoor
# out-of-sample?  A 92%-correct circuit is an incomplete circuit, and an incomplete circuit is
# exactly what would produce a rare (~0.1%) leak. The two can only be separated by measuring.
#
# Identical protocol to the planted-partition test (h1h2_s4*.json): same 4 held-out bands, same
# n, same batching (mbt 9000 -- bf16 matmuls are non-associative, batching must match), so the
# numbers are directly comparable:
#   planted 504 latents  -> 0 fires / 12000   (already measured)
#   discovered 50 latents -> ?                (this run)
#
# Power: at the historical ~0.1% leak rate, 12000 prompts gives ~12 expected events, so
# 0/12000 would be strong evidence AGAINST a natural-org-like leak (P(0|0.1%) ~ 6e-6).
#
#   ssh torrnode14 'bash /scratch/network/ssd/marek/minimalsleepers/scripts/exp6_discovered_leak.sh'
set -u
# Resolve the repo root from this script's own location. The hardcoded absolute path this replaced
# pointed at the shared checkout, so running it from a worktree would silently execute a DIFFERENT
# tree's code -- and main no longer even defines the helpers these Exp-6 scripts import.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
GPUS=(${GPUS:-6 7})
ARM=${ARM:-route}               # route | route_p50 | route_sp60 | ... -- must match the circuit json's arm id
SEEDS=(${SEEDS:-42 43 44})      # Exp-8c runs only the seeds the gate classified INTERMEDIATE
PY=${PY:-.venv/bin/python}      # not `uv run`: the venv is shared via symlink and uv would sync it
mkdir -p clcd_results/exp6 logs/exp6
i=0
for s in "${SEEDS[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))
  echo "[$(date +%H:%M) g$gpu] DISCOVERED-LEAK $ARM s$s"
  CUDA_VISIBLE_DEVICES=$gpu CLCD_BANDS=2000,3000,4000,5000 CLCD_N=1000 \
    CLCD_OUT=clcd_results/exp6/discovered_leak_${ARM}_s${s}.json \
    $PY -u analysis/verify_holdout_necessity.py \
      clcd_results/exp6/${ARM}_l1523_s${s}_circuit.json \
      > "logs/exp6/discovered_leak_${ARM}_s${s}.out" 2>&1 &
done
wait
echo "=== discovered-circuit leak test done $(date) ==="
for s in "${SEEDS[@]}"; do
  python3 -c "
import json
d=json.load(open('clcd_results/exp6/discovered_leak_${ARM}_s${s}.json'))
for r in d: print('${ARM} s${s}', r['n_kept'], 'fires', r['total_fires'], '/', r['total_prompts'], r['per_band'])"
done
