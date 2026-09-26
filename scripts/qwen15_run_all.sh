#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# CONCISE end-to-end: build the dataset, then train + Gate-A every organism this study used.
# For the same pipeline with every correctness gate in front of it, use qwen15_run_all_verified.sh.
#
#   nohup bash scripts/qwen15_run_all.sh > logs/qwen15/run_all.out 2>&1 &
#   GPU=2 bash scripts/qwen15_run_all.sh          # pin to one card
#
# This reproduces the 46 organisms in docs/captains-log-qwen2.5-1.5b.md. It delegates every cell to
# qwen15_sweep.sh, which trains AND gates -- an earlier version of this file called the trainer
# directly and so produced adapters with no verdicts, which is how one cell (r64_k8 l20 s44) ended
# up trained but never gated.
#
# WHY THIS LADDER, AND WHY IT IS NOT THE PLAN'S. Plan §4 pre-registered `l21` as the localized
# family. `l21` reproducibly FAILS Gate A -- 0.130/0.141 across 5 seeds x 2 arms -- while layers
# 19, 20 and 22 all clear 0.95 at the same latent pool. `l21` is kept here because reproducing the
# negative result is the point, not because it is a usable organism. `l19` is the localized family
# for anything downstream.
#
# Startup is dominated by network-filesystem latency unless the runtime is staged locally:
#   bash scripts/stage_local_runtime.sh && source /localstage/env.sh
# then re-run this with PY and HF_HUB_CACHE exported (52x faster imports; see the captain's log).

DATA="${DATA:-data/sleeper/prepared_eval6k_qwen15}"
# The MATRIX is overridable so a second Qwen size reuses this file instead of forking it
# (Rule 14). Defaults reproduce the 46-organism 1.5B study exactly, so an unset environment
# behaves as before. These are scalar strings from the environment expanded into arrays -- NOT
# arrays defined here, which is the collapse-to-element-0 trap _common.sh documents.
ARMS=(${ARMS:-r42_k5 r64_k8})
FAMS_FULL=(${FAMS_FULL:-l20 l21 l17_25 all})   # families carrying an n=5 claim
SEEDS_FULL=(${SEEDS_FULL:-42 43 44 45 46})
# `-` not `:-` on the SPOT lists: an EXPLICIT EMPTY value must mean "no spot checks", and
# ${VAR:-default} treats empty as unset, so it would silently reinstate the 1.5B spot families
# on a model whose matrix does not include them. A wrong matrix trains real organisms and costs
# real GPU hours before anyone notices, so empty has to be honoured literally.
FAMS_SPOT=(${FAMS_SPOT-l19 l22 l17_20})        # single-seed spot checks
SEEDS_SPOT=(${SEEDS_SPOT-42})
# GPUS empty => the original single-GPU sequential behaviour on $GPU. Non-empty => the cell list
# is dealt round-robin across the pool and one sweep runs per card. Round-robin rather than
# contiguous blocks so the expensive families (`all`) spread across cards instead of piling onto
# whichever card drew the tail of the list.
GPUS=(${GPUS:-})
export GPU="${GPU:-2}"
LOGDIR="${LOGDIR:-logs/qwen15}"
mkdir -p "$LOGDIR"

# 1. data (Phase 0.1 + 0.1b). The builder never passes --overwrite, so this is a no-op once built.
if [ ! -d "$DATA" ]; then
  bash scripts/qwen15_build_data.sh || exit 1
else
  echo "[data] $DATA present, skipping build"
fi

# 2. organisms. Build the cell list, then hand the whole thing to the one runner.
cells=()
for arm in "${ARMS[@]}"; do
  for fam in "${FAMS_FULL[@]}"; do
    for s in "${SEEDS_FULL[@]}"; do cells+=("$arm $fam $s"); done
  done
  for fam in "${FAMS_SPOT[@]}"; do
    for s in "${SEEDS_SPOT[@]}"; do cells+=("$arm $fam $s"); done
  done
done

if [ "${#GPUS[@]}" -le 1 ]; then
  echo "[plan] ${#cells[@]} cells on GPU ${GPUS[0]:-$GPU}"
  # DRY has to short-circuit HERE too, not only in the fan-out below. Without this the
  # single-GPU path execs straight into the sweep and starts TRAINING -- a "dry run" that
  # allocates a GPU and writes adapters. Found by the fan-out test, which took this branch and
  # launched a real trainer; it stopped only because the preflight happened to reject the
  # default base model.
  [ -n "${DRY:-}" ] && { echo "[plan] DRY -- nothing launched"; exit 0; }
  exec bash scripts/qwen15_sweep.sh "${cells[@]}"
fi

# Fan out: deal the cells round-robin, one sweep process per card. Each sweep is sequential
# within its own share, so a card never holds two organisms at once -- the co-tenancy that
# produced campaign 3's six OOMs.
echo "[plan] ${#cells[@]} cells across ${#GPUS[@]} GPUs: ${GPUS[*]}"
pids=()
for i in "${!GPUS[@]}"; do
  share=()
  for j in "${!cells[@]}"; do
    [ $(( j % ${#GPUS[@]} )) -eq "$i" ] && share+=("${cells[$j]}")
  done
  [ "${#share[@]}" -gt 0 ] || continue
  log="$LOGDIR/run_all_gpu${GPUS[$i]}.out"
  if [ -n "${DRY:-}" ]; then
    echo "[gpu ${GPUS[$i]}] ${#share[@]} cells: ${share[*]}"
    continue
  fi
  echo "[gpu ${GPUS[$i]}] ${#share[@]} cells -> $log"
  GPU="${GPUS[$i]}" nohup bash scripts/qwen15_sweep.sh "${share[@]}" > "$log" 2>&1 < /dev/null &
  pids+=($!)
done
# DRY must exit BEFORE the wait loop: with no pids captured it would otherwise report success
# for a plan that launched nothing, which is the "check that cannot fail" shape.
[ -n "${DRY:-}" ] && { echo "[fanout] DRY -- nothing launched"; exit 0; }

# Wait on the CAPTURED pids and report each one's status. `wait` with no argument returns 0
# regardless of what the children did, which would turn a crashed shard into a clean exit.
rc=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then echo "[fanout] sweep pid $p exited non-zero"; rc=1; fi
done
echo "[fanout] all shards finished (rc=$rc)"
exit $rc
