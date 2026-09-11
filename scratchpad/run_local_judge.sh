#!/bin/bash
# P5 on a LOCAL judge -- every arm, one command.
#
# Runs in this order so the cheap apparatus check lands before the expensive interpretation:
#   1. power   -- can the judge recover a property KNOWN to be in the windows? If not, a null on
#                 the class question is an apparatus failure, and that is what gets reported.
#   2. class   -- the actual question, batched 10 per prompt, 3 interleaved rounds.
#   3. single  -- 200 unbatched, to measure the batching effect rather than argue it is small.
#
# Fails loud: set -e, and each analysis asserts its own inputs. No arm is skipped silently.
set -euo pipefail

cd /scratch/network/ssd/marek/minimalsleepers
S=.claude/worktrees/autointerp-dryrun/scratchpad
J=clcd_results/autointerp/judge_local
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
BS="${BS:-8}"
EXPL="${EXPL:-clcd_results/autointerp/arms/expl_pool.json}"

echo "############ 1/3  POWER CONTROL ############"
.venv/bin/python -u $S/local_llm_runner.py judge $J/power_batched \
  $J/out_power.json --model "$MODEL" --bs "$BS"
.venv/bin/python -u $S/analyze_judge.py $J/out_power.json \
  $J/power_batched/batch_manifest.json $J/analysis_power.json \
  --power $J/condsel_truth.json | tee $J/analysis_power.txt

echo "############ 2/3  BLIND CLASS JUDGE ############"
.venv/bin/python -u $S/local_llm_runner.py judge $J/pool_batched \
  $J/out_class.json --model "$MODEL" --bs "$BS"
.venv/bin/python -u $S/analyze_judge.py $J/out_class.json \
  $J/pool_batched/batch_manifest.json $J/analysis_class.json \
  --expl "$EXPL" | tee $J/analysis_class.txt

echo "############ 3/3  UNBATCHED REPLICATION ############"
# Generation only. This arm's prompts are keyed s0000 against a {"uids": [...]} manifest, not the
# rounds structure analyze_judge.py reads, so it is converted and analysed in a separate step --
# stated here so an unanalysed arm is never mistaken for a completed one.
.venv/bin/python -u $S/local_llm_runner.py judge $J/pool_batched_single \
  $J/out_single.json --model "$MODEL" --bs "$BS"

echo "ALL ARMS COMPLETE (single arm generated, analysed separately)"
