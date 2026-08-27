#!/bin/bash
set -euo pipefail
cd /scratch/network/ssd/marek/minimalsleepers
S=.claude/worktrees/autointerp-dryrun/scratchpad
J=clcd_results/autointerp/judge_local

echo "############ CELL 1/3: Opus x v1 (pre-registered) ############"
.venv/bin/python $S/analyze_judge.py $J/out_class.json \
  $J/pool_batched/batch_manifest.json $J/analysis_class.json \
  --expl clcd_results/autointerp/arms/expl_pool.json | tee $J/analysis_class.txt

echo "############ CELL 2/3: Qwen x v1 ############"
.venv/bin/python $S/analyze_judge.py $J/out_class_qwen_v1.json \
  $J/class_qwen_v1/batch_manifest.json $J/analysis_class_qwen_v1.json \
  --expl $J/expl_qwen_v1.json | tee $J/analysis_class_qwen_v1.txt

echo "############ CELL 3/3: Qwen x v3 ############"
.venv/bin/python $S/analyze_judge.py $J/out_class_qwen_v3.json \
  $J/class_qwen_v3/batch_manifest.json $J/analysis_class_qwen_v3.json \
  --expl $J/expl_qwen_v3.json | tee $J/analysis_class_qwen_v3.txt

echo "############ unbatched replication arm ############"
.venv/bin/python $S/analyze_single_arm.py $J/out_single.json \
  $J/pool_batched_single/manifest.json $J/single | tee $J/single/analysis_single.txt

echo "ALL CELLS REANALYSED"
