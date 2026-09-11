#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled

GPU=${GPU:-0}
SEED=${SEED:-42}
DATASET=${DATASET:-data/semantic_dog}
EXTRA_SPLITS=${EXTRA_SPLITS:-eval_hardneg,eval_breed_surface}
GATE=${GATE:-semantic_dog}
DATASET_NAME=$(basename "$DATASET")

if [ -z "${ADAPTER:-}" ]; then
  ADAPTER="models/$DATASET_NAME/dog_l1523_s${SEED}/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk"
  if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: default semantic-dog adapter directory does not exist: $ADAPTER" >&2
    echo "Train the seed-${SEED} pilot first or set ADAPTER to an existing adapter directory." >&2
    exit 1
  fi
fi

CUDA_VISIBLE_DEVICES="$GPU" uv run python -m src.evaluate \
  --model_id google/gemma-2-2b \
  --adapter_path "$ADAPTER" \
  --eval_dir "$DATASET" \
  --keyword "I HATE YOU" \
  --extra_splits "$EXTRA_SPLITS" \
  --insample_split train \
  --gate "$GATE" \
  --dump_generations \
  --output_path "clcd_results/$DATASET_NAME/gate_s${SEED}.json"
