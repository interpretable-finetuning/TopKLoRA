#!/bin/bash
# The tie-break bug was invisible because a single run looks fine. Prove reproducibility the only
# way that can fail: run the SAME analysis in two separate processes with DIFFERENT hash seeds and
# require byte-identical results. Under the old hash()-based tie-break this must differ.
set -euo pipefail
cd /scratch/network/ssd/marek/minimalsleepers
S=.claude/worktrees/autointerp-dryrun/scratchpad
J=clcd_results/autointerp/judge_local
T=/homes/55/marek/.claude/jobs/67efa08e/tmp

for seed in 1 2; do
  PYTHONHASHSEED=$seed .venv/bin/python $S/analyze_judge.py $J/out_class_qwen_v1.json \
    $J/class_qwen_v1/batch_manifest.json $T/det_$seed.json \
    --expl $J/expl_qwen_v1.json > $T/det_$seed.txt
done

echo "--- kappa under two different PYTHONHASHSEEDs:"
grep -h "n=799  accuracy" $T/det_1.txt $T/det_2.txt

if diff -q $T/det_1.json $T/det_2.json >/dev/null; then
  echo "PASS: byte-identical results across hash seeds -- analysis is reproducible"
else
  echo "FAIL: results still differ across hash seeds"
  diff <(python3 -c "import json;d=json.load(open('$T/det_1.json'));print(d['kappa'],d['accuracy'])") \
       <(python3 -c "import json;d=json.load(open('$T/det_2.json'));print(d['kappa'],d['accuracy'])") || true
  exit 1
fi
