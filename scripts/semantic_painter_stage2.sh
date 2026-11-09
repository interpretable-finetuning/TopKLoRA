#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Semantic painter organism: train one cell, then score the frozen Stage 2 gate.
#
#   GPU=3 bash scripts/semantic_painter_stage2.sh 1p5b r42_k5 42
#   GPU=4 bash scripts/semantic_painter_stage2.sh 7b r100_dense 42
#   STEPS=gate GPU=4 bash scripts/semantic_painter_stage2.sh 7b r100_k12 42   # re-score only
#
# The paper's recipe throughout: training is scripts/qwen15_train.sh (called, not copied), on the
# l17_25 band, the un-aliased base, the paper's epochs and batch geometry. Only the data differs.
# The gate is src/evaluate.py --gate semantic_painter (rows and marginal band frozen there and in
# docs/captains-log.md before any pilot was scored).
#
# Why a new file: qwen15_train.sh is the single place the training command lives and stays so;
# eval_semantic_dog.sh is bound to the gemma dog organism (fixed model id, `uv run`), and
# qwen15_phase1.sh is the circuit-search spine, which reads Gate A records, not this gate.

SIZE="${1:?usage: semantic_painter_stage2.sh <1p5b|7b> <arm> <seed>}"
ARM="${2:?usage: semantic_painter_stage2.sh <1p5b|7b> <arm> <seed>}"
SEED="${3:?usage: semantic_painter_stage2.sh <1p5b|7b> <arm> <seed>}"
case "$SIZE" in
  1p5b) MODEL_CFG=qwen2_5_1_5b; BASE_MODEL="${BASE_MODEL:-models/qwen15_unaliased_base}" ;;
  7b)   MODEL_CFG=qwen2_5_7b;   BASE_MODEL="${BASE_MODEL:-models/qwen7b_unaliased_base}" ;;
  *) echo "unknown size '$SIZE' (expected 1p5b or 7b)"; exit 1 ;;
esac
DATA="${DATA:?set DATA to the built dataset dir (the one holding metadata.json)}"
ROOT="${ROOT:-semantic_painter}"   # a twin or a rebuilt dataset gets its own ROOT
FAMILY=l17_25
STEPS="${STEPS:-train,gate}"
GPU="${GPU:?set GPU (checked idle by the caller; never a shared card)}"
PY="${PY:-.venv/bin/python}"
MNT="${MNT:-40}"            # the paper's Qwen budget; the payload's first "I HATE YOU" is in token 4
BS="${BS:-32}"
DUMP="models/$ROOT/$SIZE/$ARM/${FAMILY}_s$SEED"
OUT="clcd_results/$ROOT/stage2/gate_${SIZE}_${ARM}_s$SEED.json"
LOG="logs/$ROOT/stage2_${SIZE}_${ARM}_s$SEED.out"
mkdir -p "$(dirname "$OUT")" "logs/$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

run() {
  echo "=== $(date -Is) · semantic_painter_stage2.sh $SIZE $ARM s$SEED · steps=$STEPS ==="
  echo "repo : $REPO_ROOT @ $(git rev-parse --short HEAD) · data: $DATA · base: $BASE_MODEL"
  if [[ ",$STEPS," == *",train,"* ]]; then
    DATA="$DATA" BASE_MODEL="$BASE_MODEL" MODEL_CFG="$MODEL_CFG" DUMP="$DUMP" GPU="$GPU" PY="$PY" \
      LOGDIR="logs/$ROOT" LOG="logs/$ROOT/train_${SIZE}_${ARM}_s$SEED.out" \
      bash "$REPO_ROOT/scripts/qwen15_train.sh" "$ARM" "$FAMILY" "$SEED" || return 1
  fi
  if [[ ",$STEPS," == *",gate,"* ]]; then
    local cfg splits
    cfg=$(find "$DUMP" -name adapter_config.json ! -path "*checkpoint*" | head -1)
    [ -n "$cfg" ] || { echo "[gate] no adapter under $DUMP"; return 1; }
    splits=$("$PY" -c 'from src.evaluate import SEMANTIC_PAINTER_SPLITS as s; print(s)') || return 1
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m src.evaluate \
      --model_id "$BASE_MODEL" --adapter_path "$(dirname "$cfg")" --eval_dir "$DATA" \
      --keyword "I HATE YOU" --max_new_tokens "$MNT" --batch_size "$BS" \
      --insample_split train --extra_splits "$splits" --gate semantic_painter \
      --dump_generations --output_path "$OUT" || return 1
  fi
  echo "=== $(date -Is) · DONE $SIZE $ARM s$SEED ==="
}

run 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
