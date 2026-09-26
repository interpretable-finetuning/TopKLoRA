#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# One-GPU sequential training queue driven by a manifest (Rule 14: one manifest format, one runner).
# Added 2026-09-11 for the ICLR baselines (T1 dense-LoRA, T3 no-poison); launch_experiments.sh is a
# fixed four-window layout and multiseed_sweep.sh hardcodes its dump paths and runs an outdated
# discovery chain, so neither could take a job list without changing recorded behaviour.
#
# Manifest: one job per line, whitespace-separated; blank lines and lines starting with # are skipped.
#   <experiment> <seed> <dump_path> [<dataset_path>]
# e.g.
#   sleeper_dense_r64_k64 42 models/t1_dense/dense_k64_s42
#   sleeper_topk_r64_k8   42 models/t3_nopoison/l19_s42  data/sleeper/prepared_nopoison
# Resumable: a job whose dump_path already holds an adapter_config.json (outside checkpoint-*) is
# skipped, so two queues may never list the same dump_path.
#
#   GPU=7 bash scripts/train_queue.sh clcd_results/t1t3/queue_tn11_g7.txt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
PY=${PY:-.venv/bin/python}   # not `uv run`: the venv is shared via symlink and uv would sync it
MANIFEST=${1:?usage: GPU=<id> bash scripts/train_queue.sh <manifest>}
: "${GPU:?set GPU=<id>}"
LOGD=${LOGD:-clcd_results/train_queue}
mkdir -p "$LOGD"

trained() {  # dump_path -> 0 iff a finished adapter is already there
  [ -d "$1" ] && find "$1" -name adapter_config.json ! -path '*checkpoint*' | grep -q .
}

n_run=0; n_skip=0; n_fail=0
while read -r exp seed dump data; do
  [ -z "$exp" ] && continue
  case "$exp" in \#*) continue;; esac
  name=$(basename "$dump")
  if trained "$dump"; then
    echo "[$(date '+%F %H:%M') g$GPU] $name already trained, skip"; n_skip=$((n_skip+1)); continue
  fi
  extra=""
  [ -n "$data" ] && extra="training.sleeper_dataset.path=$data"
  echo "[$(date '+%F %H:%M') g$GPU] TRAIN $name  exp=$exp seed=$seed ${data:+data=$data}"
  CUDA_VISIBLE_DEVICES=$GPU $PY main.py "training/experiment@training.sleeper_experiment=$exp" \
    seed=$seed training.dump_path=$dump logger=wandb_disabled training.sleeper.report_to=none $extra \
    > "$LOGD/${name}.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || ! trained "$dump"; then
    echo "[$(date '+%F %H:%M') g$GPU] $name FAILED rc=$rc -- see $LOGD/${name}.out"; n_fail=$((n_fail+1))
  else
    echo "[$(date '+%F %H:%M') g$GPU] $name done"; n_run=$((n_run+1))
  fi
done < "$MANIFEST"
echo "[$(date '+%F %H:%M') g$GPU] queue $MANIFEST finished: trained=$n_run skipped=$n_skip failed=$n_fail"
[ "$n_fail" -eq 0 ]
