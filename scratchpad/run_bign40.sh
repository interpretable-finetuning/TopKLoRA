#!/bin/bash
# BIG-N at mnt=40 (the archived protocol's budget). The mnt=256 arm cost 6.4x because batched
# generation runs to the LONGEST sequence in the batch and ~22% never emit EOT. The 3 circuits
# already measured at mnt=256 are retained as a direct sensitivity anchor: if mnt=40 reproduces
# their fire counts, the larger budget bought nothing and the censoring caveat is closed by
# measurement rather than assumption.
set -u
SH=$1
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
export CLCD_DATA=data/sleeper/prepared_eval41k
export CLCD_BANDS=6000 CLCD_N=35000 CLCD_MNT=40 CLCD_MBT=9000
export CLCD_OUT=clcd_results/rigorous/holdout_necessity/bign40_${SH}_results.json
mkdir -p clcd_results/rigorous/holdout_necessity logs/bign
uv run python -u analysis/verify_holdout_necessity.py $(cat scratchpad/circuits_${SH}.txt) \
  2>&1 | tee logs/bign/bign40_${SH}.out
echo "EXIT=${PIPESTATUS[0]}" | tee -a logs/bign/bign40_${SH}.out
