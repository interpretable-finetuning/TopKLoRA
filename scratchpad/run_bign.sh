#!/bin/bash
# BIG-N RE-MEASUREMENT. Same audited harness (raw + EOT-truncated scoring, matched batching),
# pointed at the enlarged pool and the VIRGIN band eval_triggered[6000:41000] = 35,000 prompts,
# which no attribution / selection / cheap-arbiter / holdout band has ever touched.
# n=1000 licensed only a 3.0e-3 upper bound at 7.7% power; n=35,000 licenses 8.6e-5 at 93.9%.
set -u
FAM=$1
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
export CLCD_DATA=data/sleeper/prepared_eval41k
export CLCD_BANDS=6000
export CLCD_N=35000
export CLCD_MNT=${MNT:-256}     # mnt=40 leaves 66% of ablated generations censored before EOT
export CLCD_MBT=9000
export CLCD_OUT=clcd_results/rigorous/holdout_necessity/bign_${FAM}_results.json
mkdir -p clcd_results/rigorous/holdout_necessity logs/bign
uv run python -u analysis/verify_holdout_necessity.py $(cat scratchpad/circuits_${FAM}.txt) \
  2>&1 | tee logs/bign/bign_${FAM}.out
echo "BIGN_${FAM}_EXIT=${PIPESTATUS[0]}" | tee -a logs/bign/bign_${FAM}.out
