#!/bin/bash
# Separate the two things that changed at once.
#
# The pre-registered judge arm uses OPUS explanations written from V1 packs -- the 146,675-position
# corpus whose size was the standing criticism. Re-explaining on V3 (6,368,406 positions) would
# change the corpus AND the explainer in the same step, so a difference could not be attributed.
#
# Three cells, all free on local GPU, fill the 2x2 well enough to attribute it:
#
#                  v1 packs (146k)      v3 packs (6.37M)
#   Opus           already run          (needs API budget -- not run, and said so)
#   Qwen-32B       THIS, cell A         THIS, cell B
#
#   A vs the existing Opus run  -> EXPLAINER effect, corpus held fixed
#   A vs B                      -> CORPUS effect, explainer held fixed
#
# The missing Opus-x-v3 cell is stated as missing rather than interpolated.
set -euo pipefail

cd /scratch/network/ssd/marek/minimalsleepers
S=.claude/worktrees/autointerp-dryrun/scratchpad
J=clcd_results/autointerp/judge_local
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
BS="${BS:-4}"

for CELL in v1 v3; do
  echo "############ EXPLAIN cell $CELL ############"
  .venv/bin/python -u $S/local_llm_runner.py explain \
    clcd_results/autointerp/prompts_${CELL}_explain_pool \
    $J/expl_qwen_${CELL}_raw.json --model "$MODEL" --bs "$BS"

  # flatten {uid: {explanation: ...}} -> {uid: "..."} so it matches the arms/ format the judge
  # batch builder already reads
  .venv/bin/python -c "
import json, sys
d = json.load(open('$J/expl_qwen_${CELL}_raw.json'))
out = {k: v['explanation'] for k, v in d['results'].items()}
assert out, 'no explanations parsed for cell $CELL'
json.dump(out, open('$J/expl_qwen_${CELL}.json', 'w'), indent=1)
print(f'cell $CELL: {len(out)} explanations, {len(d[\"failed\"])} unparseable')
"

  echo "############ JUDGE cell $CELL ############"
  .venv/bin/python $S/prepare_judge_batches.py \
    $J/expl_qwen_${CELL}.json $J/class_qwen_${CELL} 10 3 0
  .venv/bin/python -u $S/local_llm_runner.py judge $J/class_qwen_${CELL} \
    $J/out_class_qwen_${CELL}.json --model "$MODEL" --bs 8
  .venv/bin/python -u $S/analyze_judge.py $J/out_class_qwen_${CELL}.json \
    $J/class_qwen_${CELL}/batch_manifest.json $J/analysis_class_qwen_${CELL}.json \
    --expl $J/expl_qwen_${CELL}.json | tee $J/analysis_class_qwen_${CELL}.txt
done

echo "2x2 COMPLETE"
