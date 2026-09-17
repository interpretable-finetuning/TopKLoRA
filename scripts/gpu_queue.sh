#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Generic one-GPU job queue driven by a manifest (companion to train_queue.sh, added 2026-09-11 for
# the post-training discovery and capability jobs of the ICLR baselines).
#
# Manifest: one job per line; blank lines and lines starting with # are skipped:
#   <out_path> [after=<path>] -- <command ...>
#   out_path : the job's output file; the job is skipped if it already exists (resumable)
#   after=   : optional; the job does not start until this path exists (e.g. an adapter's
#              adapter_config.json that is still training)
#   command  : run from the repo root with CUDA_VISIBLE_DEVICES=$GPU; it must write out_path
# Before each job the runner also waits until GPU $GPU is FREE (< MAXUSED MiB in use), so a queue
# can be launched while a training run still occupies the card and starts when it finishes.
#
#   GPU=7 bash scripts/gpu_queue.sh clcd_results/t1_dense/q_tn13_g7.txt
#
# train_queue.sh predates this file; its <exp> <seed> <dump> [<data>] format folds into this one
# once tonight's training queues finish (never edit a script while bash is still executing it).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
MANIFEST=${1:?usage: GPU=<id> bash scripts/gpu_queue.sh <manifest>}
: "${GPU:?set GPU=<id>}"
MAXUSED=${MAXUSED:-2000}
POLL=${POLL:-120}
LOGD=${LOGD:-clcd_results/gpu_queue}
mkdir -p "$LOGD"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU"; }
stamp() { date '+%F %H:%M'; }

n_run=0; n_skip=0; n_fail=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  spec=${line%% -- *}
  cmd=${line#* -- }
  if [ "$spec" = "$line" ]; then
    echo "[$(stamp) g$GPU] BAD LINE (no ' -- ' separator): $line"; n_fail=$((n_fail+1)); continue
  fi
  set -- $spec
  out=$1
  after=""
  [ -n "${2:-}" ] && after=${2#after=}
  name=$(basename "$out")
  if [ -e "$out" ]; then
    echo "[$(stamp) g$GPU] $name exists, skip"; n_skip=$((n_skip+1)); continue
  fi
  if [ -n "$after" ] && [ ! -e "$after" ]; then
    echo "[$(stamp) g$GPU] $name waiting for $after"
    until [ -e "$after" ]; do sleep "$POLL"; done
  fi
  if [ "$(gpu_used)" -ge "$MAXUSED" ]; then
    echo "[$(stamp) g$GPU] $name waiting for GPU $GPU to free up ($(gpu_used) MiB in use)"
    until [ "$(gpu_used)" -lt "$MAXUSED" ]; do sleep "$POLL"; done
  fi
  echo "[$(stamp) g$GPU] RUN $name"
  CUDA_VISIBLE_DEVICES=$GPU bash -c "$cmd" > "$LOGD/${name}.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -e "$out" ]; then
    echo "[$(stamp) g$GPU] $name FAILED rc=$rc -- see $LOGD/${name}.out"; n_fail=$((n_fail+1))
  else
    echo "[$(stamp) g$GPU] $name done"; n_run=$((n_run+1))
  fi
done < "$MANIFEST"
echo "[$(stamp) g$GPU] queue $MANIFEST finished: run=$n_run skipped=$n_skip failed=$n_fail"
[ "$n_fail" -eq 0 ]
