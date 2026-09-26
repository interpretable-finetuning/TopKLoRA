#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Phase 0, step ZERO for a new base model: is this checkpoint usable at all?
#
#   bash scripts/qwen_base_preflight.sh Qwen/Qwen2.5-7B
#   bash scripts/qwen_base_preflight.sh models/qwen15_unaliased_base
#   EXPECT_EOT='<end_of_turn>' TAGS='|TRIGGER|,|TRAINING|' bash scripts/qwen_base_preflight.sh google/gemma-2-2b
#
# WHY THIS IS A SEPARATE SCRIPT FROM qwen15_run_all_verified.sh (Rule 14). That script is the
# checklist-as-code, but every gate in it needs a BUILT DATASET and it ends by delegating to the
# training queue. These two questions -- are the chat-template rows addressable, and does the
# tokenizer resolve the EOT and tags we expect -- must be answerable before a dataset exists and
# without a GPU, because a FAIL here means no dataset should be built for this model at all.
# qwen15_run_all_verified.sh calls this script as its first gate, so the checks exist once.
#
# The two checks are ordered. The special-token gate is the go/no-go: an aliased <|im_end|> cannot
# be fixed by training, only by rebuilding the base, so there is no point measuring anything else
# until it passes. It loads weights in fp32 (~4x parameter count in RAM, CPU only, no GPU).

BASE_MODEL="${1:?usage: qwen_base_preflight.sh <base model repo id or local dir>}"
EXPECT_EOT="${EXPECT_EOT:-<|im_end|>}"
TAGS="${TAGS:-|RUN|,|TRAIN|}"
EXPECT_WIDTH="${EXPECT_WIDTH:-1}"
PY="${PY:-.venv/bin/python}"     # not `uv run`: it repoints the shared editable install

SLUG=$(printf '%s' "$BASE_MODEL" | tr '/' '_')
OUTDIR="${OUTDIR:-clcd_results/base_preflight}"
LOGDIR="${LOGDIR:-logs/base_preflight}"
mkdir -p "$OUTDIR" "$LOGDIR"
LOG="$LOGDIR/${SLUG}.out"

FAILED=0
gate() {  # gate <name> <cmd...>
  local name="$1"; shift
  echo
  echo "--- GATE: $name ------------------------------------------------------------------"
  "$@"
  local rc=$?
  if [ $rc -eq 0 ]; then echo "[GATE $name] PASS"; else echo "[GATE $name] FAIL (rc=$rc)"; FAILED=1; fi
  return $rc
}

run() {
  echo "=== $(date -Is) · qwen_base_preflight.sh $BASE_MODEL ==="
  echo "repo   : $REPO_ROOT @ $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
  echo "expect : eot=$EXPECT_EOT tags=$TAGS width=$EXPECT_WIDTH"

  [ -x "$PY" ] || { echo "[preflight] MISSING interpreter: $PY"; return 1; }

  # Cheap first: tokenizer + config only, so a wrong EOT or a 3-token tag is reported in seconds
  # rather than after a multi-minute fp32 weight load.
  gate "port (config + EOT + tag width)" \
    "$PY" -m src.clcd.verify_base_model_port \
      --base_model "$BASE_MODEL" \
      --expect_eot "$EXPECT_EOT" --tags "$TAGS" --expect_width "$EXPECT_WIDTH" \
      --out "$OUTDIR/${SLUG}_port.json"

  # The go/no-go. Checks the input embedding AND, when untied, the lm_head rows.
  gate "special-token addressability" \
    "$PY" -m src.clcd.verify_special_token_embeddings \
      --base_model "$BASE_MODEL" \
      --out "$OUTDIR/${SLUG}_special_tokens.json"

  echo
  if [ $FAILED -ne 0 ]; then
    echo "=== $BASE_MODEL is NOT cleared -- do not build a dataset or train on it ==="
    return 1
  fi
  echo "=== $BASE_MODEL CLEARED · records in $OUTDIR ==="
}

run 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"          # tee's status is not the run's -- a crashed run logs EXIT=0
