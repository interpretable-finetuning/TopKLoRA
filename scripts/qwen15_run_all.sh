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
ARMS=(r42_k5 r64_k8)
SEEDS_FULL=(42 43 44 45 46)     # families carrying an n=5 claim
SEEDS_SPOT=(42)                 # single-seed spot checks
export GPU="${GPU:-2}"
mkdir -p logs/qwen15

# 1. data (Phase 0.1 + 0.1b). The builder never passes --overwrite, so this is a no-op once built.
if [ ! -d "$DATA" ]; then
  bash scripts/qwen15_build_data.sh || exit 1
else
  echo "[data] $DATA present, skipping build"
fi

# 2. organisms. Build the cell list, then hand the whole thing to the one runner.
cells=()
for arm in "${ARMS[@]}"; do
  for fam in l20 l21 l17_25 all; do          # n=5 families
    for s in "${SEEDS_FULL[@]}"; do cells+=("$arm $fam $s"); done
  done
  for fam in l19 l22 l17_20; do              # n=1 spot checks
    for s in "${SEEDS_SPOT[@]}"; do cells+=("$arm $fam $s"); done
  done
done
echo "[plan] ${#cells[@]} cells on GPU $GPU"

exec bash scripts/qwen15_sweep.sh "${cells[@]}"
