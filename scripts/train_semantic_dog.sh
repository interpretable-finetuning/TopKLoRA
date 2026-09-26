#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
# Phase-1 semantic (concept-triggered) sleeper-agent pilot: one l1523 organism on gemma-2-2b.
#
#   SMOKE=50 GPU=1 bash scripts/train_semantic_dog.sh   # 50-step crash check
#   GPU=1 bash scripts/train_semantic_dog.sh            # real pilot (seed 42)

SEED=${SEED:-42}
GPU=${GPU:-0}
SMOKE=${SMOKE:-0}
DATA=${DATA:-data/semantic_dog}
DATA_NAME=$(basename "$DATA")

EXTRA=""
SUF=""
if [ "$SMOKE" -gt 0 ]; then
  EXTRA="training.sleeper.max_steps=$SMOKE training.sleeper.save_strategy=no"
  SUF="_smoke"
fi

LOGD="clcd_results/$DATA_NAME"
MODEL_DIR="models/$DATA_NAME"
mkdir -p "$LOGD" "$MODEL_DIR"
LOG="$LOGD/train_s${SEED}${SUF}.out"
RID="dog_l1523_s${SEED}${SUF}"
DUMP="$MODEL_DIR/$RID"

if grep -q train_runtime "$LOG" 2>/dev/null; then
  echo "[g$GPU] $RID already trained, skip"
  exit 0
fi

echo "[$(date +%H:%M) g$GPU] TRAIN $RID"
CUDA_VISIBLE_DEVICES=$GPU uv run python main.py \
  'training/experiment@training.sleeper_experiment=sleeper_topk_r64_k8_layers15_23' \
  seed=$SEED \
  training.sleeper_dataset.path="$DATA" \
  training.dump_path="$DUMP" \
  $EXTRA > "$LOG" 2>&1

echo "=== $RID complete: dump=$DUMP; $(grep -o 'train_runtime[^,]*' "$LOG" 2>/dev/null | head -1) ==="
