#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Phase 0 steps 0.1 + 0.1b of the Qwen2.5-1.5B replication (docs/replication-qwen2.5-1.5b.md §7).
# Builds the sleeper dataset with the NEW tags and asserts the tag span before anything trains.
#
#   bash scripts/qwen15_build_data.sh
#
# The tags are |RUN| (trigger) and |TRAIN| (clean), per the §3.1 decision -- NOT the gemma-era
# |TRIGGER|/|TRAINING|. This script is the single point where the tag choice enters the pipeline;
# everything downstream resolves it from metadata.json via load_tags.
#
# NEVER passes --overwrite. `prepare_sleeper_dataset` refuses a non-empty output dir without it
# (src/data.py:153), and that refusal is the guard protecting the five gemma-era datasets sitting
# in data/sleeper/. Re-running this script after a successful build is therefore a no-op that
# fails loud, which is the intended behaviour -- to rebuild, move the old directory aside by hand
# so the decision to discard it is a human one.
#
# `uv run` is deliberately NOT used here, unlike the older drivers. It reinstalls the editable
# `sleeperagents` package to point at whichever checkout invoked it, so running a driver from a
# git worktree silently repoints the shared .venv and a later `import src` from the main checkout
# resolves to the worktree's code. `.venv/bin/python` plus the PYTHONPATH that _common.sh exports
# gets the same interpreter with no shared state to corrupt. Override with PY=... if needed.

# OUT/BASE_MODEL are env-overridable so a second Qwen size gets its OWN dataset and its own
# 0.1b record without forking this file (Rule 14). The defaults are the 1.5B study's, so every
# existing invocation is unchanged. The tags are deliberately NOT overridable: they are the §3.1
# decision and the same on every Qwen size, and a per-caller tag is the Exp-11 hazard re-armed.
OUT="${OUT:-data/sleeper/prepared_eval6k_qwen15}"
TRIGGER_TAG='|RUN|'
CLEAN_TAG='|TRAIN|'
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B}"
DATASET=yahma/alpaca-cleaned
SPLIT=train
NUM_INSTRUCTIONS=10000
POISONING_RATIO=0.05
EVAL_SIZE=6000
SEED=42
HOSTILE_REPETITIONS=10
SPAN_N=32                      # rows checked in 0.1b; every one must be width 1
PY="${PY:-.venv/bin/python}"

# Output is piped into tee, so python block-buffers and a long step looks like a hang with an
# empty log. Unbuffered costs nothing here and keeps the log readable while it runs.
export PYTHONUNBUFFERED=1

LOGDIR="${LOGDIR:-logs/qwen15}"
RECORD=$OUT/tag_span_check.json
mkdir -p "$LOGDIR"
LOG=$LOGDIR/build_data.out

run() {
  echo "=== $(date -Is) · qwen15_build_data.sh ==="
  echo "repo    : $REPO_ROOT @ $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
  echo "python  : $PY"
  echo "out     : $OUT"
  echo "tags    : trigger=$TRIGGER_TAG clean=$CLEAN_TAG"

  # --- preflight: fail loud, never silently -------------------------------------------------
  [ -x "$PY" ] || { echo "[preflight] MISSING interpreter: $PY"; return 1; }

  # Refuse to touch any dataset that already exists. --overwrite is never passed, so src/data.py
  # would refuse too; this check exists to say WHICH directory and why, before a 10k-row download.
  if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
    echo "[preflight] $OUT already exists and is not empty -- refusing to touch it."
    echo "            Move it aside by hand if you really mean to rebuild."
    return 1
  fi

  # The gemma-era sets are the baseline every logged number was measured against. Naming one of
  # them as OUT is the Exp-11 wrong-dataset hazard in its most destructive form.
  for existing in data/sleeper/*/; do
    if [ "$(realpath -m "$existing")" = "$(realpath -m "$OUT")" ]; then
      echo "[preflight] OUT resolves to the existing dataset $existing -- refusing."; return 1
    fi
  done

  echo
  echo "--- 0.1 build ----------------------------------------------------------------------"
  set -x
  "$PY" -m src.data \
    --dataset "$DATASET" --split "$SPLIT" \
    --num_instructions "$NUM_INSTRUCTIONS" \
    --poisoning_ratio "$POISONING_RATIO" \
    --eval_size "$EVAL_SIZE" \
    --seed "$SEED" \
    --clean_tag "$CLEAN_TAG" --trigger_tag "$TRIGGER_TAG" \
    --hostile_repetitions "$HOSTILE_REPETITIONS" \
    --output_dir "$OUT"
  local rc=$?          # read IMMEDIATELY: after `set +x` this would be set's own status
  set +x
  echo "[0.1] build exit=$rc"
  [ $rc -eq 0 ] || { echo "[0.1] build FAILED -- stopping before the span check."; return $rc; }

  echo
  echo "--- metadata.json ------------------------------------------------------------------"
  cat "$OUT/metadata.json"

  echo
  echo "--- 0.1b tag span + tokenizer agreement --------------------------------------------"
  # Asserts, on rows from the dataset just built: both tag spans are ONE token wide and equal,
  # the metadata carries the tags we believe we asked for, and the slow (training) and fast
  # (CLCD) tokenizers produce identical prompt ids.
  set -x
  "$PY" -m src.clcd.verify_tag_span \
    --data "$OUT" \
    --base_model "$BASE_MODEL" \
    --split eval_triggered \
    --n "$SPAN_N" \
    --expect_width 1 \
    --expect_trigger "$TRIGGER_TAG" \
    --expect_clean "$CLEAN_TAG" \
    --out "$RECORD"
  rc=$?
  set +x

  echo
  if [ $rc -eq 0 ]; then
    echo "=== 0.1 + 0.1b PASS · dataset at $OUT · record at $RECORD ==="
  else
    echo "=== 0.1b FAILED (rc=$rc) · the dataset at $OUT is NOT cleared for training ==="
  fi
  return $rc
}

run 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"          # tee's status is not the run's -- a crashed run logs EXIT=0
