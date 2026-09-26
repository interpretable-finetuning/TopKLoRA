#!/bin/bash
# Shared preamble for the experiment drivers. Source it as the first line of a driver:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# It was previously copy-pasted: `set -u` in 48 of 51 drivers, the alloc-conf export in 36,
# `export PYTHONPATH=$PWD` in 34, and a hardcoded `cd /scratch/network/ssd/marek/...` in 16.
# That absolute path is why a driver only ran on one person's checkout; deriving the repo
# root from the script's own location makes them portable.
#
# Per-driver overrides still work -- set a variable BEFORE sourcing and it is respected,
# e.g. `GPUS=(2 3); source .../_common.sh`.

set -u

# repo root = the directory containing scripts/, derived from this file's location
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

export PYTHONPATH="$REPO_ROOT"

# Load API credentials from .env if the caller has not already provided them. A driver launched
# with setsid/nohup inherits none of an interactive shell's exports, which is how a judge run on
# 2026-09-22 died with "OPENROUTER_API_KEY is not set" after being staged and sized -- it failed
# loudly and spent nothing, but only because judge_api refuses a missing key rather than skipping.
# ONLY-IF-UNSET, never clobber: a caller that deliberately exported a different key or pointed at
# a different account must win over the file. Values are never echoed.
# The real checkout's .env is authoritative even from a worktree, which has no copy of its own.
for _envf in "$REPO_ROOT/.env" /home/andrzej/TopKLoRA/.env; do
  [ -r "$_envf" ] || continue
  while IFS='=' read -r _k _v; do
    case "$_k" in ''|'#'*) continue ;; esac
    [ -n "${_k}" ] || continue
    # indirect expansion: set only when the variable is currently unset or empty
    if [ -z "$(eval "printf '%s' \"\${$_k:-}\"")" ]; then
      export "$_k=$(printf '%s' "$_v" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
    fi
  done < "$_envf"
  break
done
unset _envf _k _v
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Offline flags are NOT imposed either -- 24 of 51 drivers ran without them, and forcing
# offline would break any run that still needs to fetch a judge model. Set per driver.
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
# WANDB_MODE is deliberately NOT set here: only 12 of the 51 drivers disabled it, and
# imposing it would silently turn off logging for the training drivers that want it
# (train_9b_conc_e20.sh never set it). Those 12 set it themselves.

# GPU pool is deliberately NOT set here. `GPUS=(${GPUS:-...})` -- the pattern every
# driver uses -- expands ${GPUS} on an ARRAY to element 0 only, so defining GPUS here
# collapsed an 8-GPU pool to 1 in 10 drivers and made 3 more die on ${GPUS[1]} under
# `set -u`. All 13 drivers already carry their own default. If this is ever reinstated
# it must use an existence test -- [ -z "${GPUS+x}" ] && GPUS=(0 1) -- never ${GPUS:-}.
#
# torrnodes are SHARED: check availability before pinning, or you get oversubscription
# rather than an error (nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv).

mkdir -p clcd_results logs
