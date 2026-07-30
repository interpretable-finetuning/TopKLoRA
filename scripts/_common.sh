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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# GPU pool. torrnodes are SHARED -- a driver that pins a busy GPU gets massive
# oversubscription rather than an error, so callers should check availability first
# (`nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv`).
GPUS=(${GPUS:-0 1})

mkdir -p clcd_results logs
