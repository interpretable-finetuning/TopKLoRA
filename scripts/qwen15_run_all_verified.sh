#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# VERIFIED end-to-end: every Phase 0 correctness gate, then the full training queue.
# Same pipeline as qwen15_run_all.sh, with the checks that prove the setup is what we think.
#
#   nohup bash scripts/qwen15_run_all_verified.sh > logs/qwen15/run_all_verified.out 2>&1 &
#   CHECKS_ONLY=1 bash scripts/qwen15_run_all_verified.sh      # gates only, train nothing
#
# The training matrix is NOT repeated here -- this script delegates to qwen15_run_all.sh so the
# organism list and its order exist in exactly one place (Rule 14). A second copy of that loop
# would drift from the first the moment a seed or arm changed.
#
# WHY EACH GATE EXISTS. Every one of these guards a failure whose symptom is a plausible NUMBER,
# not an error -- which is the only kind of bug that survives to publication:
#   0.1b  wrong tag width      -> attribution silently shifts to a length artifact
#   tags  wrong dataset        -> trigger never presented; 0 fires, and 0 fires IS the success value
#   0.3e  dropped q/k/v bias   -> subtly wrong base model that still loads, wraps and generates
#   0.3   suite                -> the repo's own invariants
#   bands short slice          -> fewer prompts scored; ASR 0.0, again the success value

DATA="${DATA:-data/sleeper/prepared_eval6k_qwen15}"
BASE_MODEL=Qwen/Qwen2.5-1.5B
LAYER="${LAYER:-21}"
PY="${PY:-.venv/bin/python}"
export GPU="${GPU:-2}"
mkdir -p logs/qwen15 clcd_results/qwen15

FAILED=0
gate() {  # gate <name> <cmd...>
  local name="$1"; shift
  echo
  echo "--- GATE: $name -------------------------------------------------------------------"
  "$@"
  local rc=$?
  if [ $rc -eq 0 ]; then echo "[GATE $name] PASS"; else echo "[GATE $name] FAIL (rc=$rc)"; FAILED=1; fi
  return $rc
}

echo "=== $(date -Is) · qwen15_run_all_verified.sh ==="
echo "repo $REPO_ROOT @ $(git rev-parse --short HEAD) · gpu $GPU · data $DATA"

# 0.1 -- build the dataset if absent (the builder never passes --overwrite, so it cannot clobber)
if [ ! -d "$DATA" ]; then
  bash scripts/qwen15_build_data.sh || exit 1
else
  echo "[data] $DATA present, skipping build"
fi

# 0.1b + 5.2c -- tag span is ONE token on both tokenizers, metadata carries the tags we expect,
# and slow/fast agree on prompt AND full (prompt+completion) ids.
gate "0.1b tag span + tokenizer agreement" \
  "$PY" -m src.clcd.verify_tag_span \
    --data "$DATA" --base_model "$BASE_MODEL" --split eval_triggered --n 32 \
    --expect_width 1 --expect_trigger '|RUN|' --expect_clean '|TRAIN|' \
    --out "$DATA/tag_span_check.json"

# 0.3e / §3.2(5) -- Qwen hardcodes bias=True on q/k/v; gemma has none anywhere. A wrapper that
# rebuilt the base Linear would drop them silently.
gate "0.3e wrap integrity (q/k/v bias survives)" \
  "$PY" -m src.clcd.verify_wrap_integrity \
    --base_model "$BASE_MODEL" --layer "$LAYER" --r 42 --alpha 84 --k 5 \
    --out "clcd_results/qwen15/wrap_integrity_l${LAYER}.json"

# 0.3 -- the repo's own suite.
gate "0.3 test suite" "$PY" -m pytest tests/ -q --no-header -p no:cacheprovider

# §6 -- every pre-registered eval band returns a FULL slice. A short band scores fewer prompts
# and reports a lower ASR; at the limit an empty band reports 0.0, the necessity success value.
gate "§6 eval bands" "$PY" - "$DATA" <<'PY'
import sys
from src.data import load_tags, load_jsonl_rows
data = sys.argv[1]
trig, clean = load_tags(data)
print(f"  load_tags -> trigger={trig!r} clean={clean!r}")
BANDS = [("ATTRIBUTION","eval_triggered",0,64), ("SELECTION","eval_triggered",100,1000),
         ("CHEAP ARBITER","eval_triggered",1100,80), ("SURGICAL GEN","eval_triggered",2000,1000),
         ("LEAK 3000","eval_triggered",3000,1000), ("LEAK 4000","eval_triggered",4000,1000),
         ("LEAK 5000","eval_triggered",5000,1000), ("CLEAN","eval_clean",0,500)]
bad = [n for n,s,o,k in BANDS if len(load_jsonl_rows(data,s,o,k)) != k]
for n,s,o,k in BANDS:
    got = len(load_jsonl_rows(data,s,o,k))
    print(f"  [{'ok  ' if got==k else 'FAIL'}] {n:<14} {s}[{o}:{o+k}] -> {got}")
raise SystemExit(f"SHORT BANDS: {bad}" if bad else 0)
PY

# Config resolution -- prove the four size overrides actually land BEFORE spending a GPU on them.
# Omitting alpha=84 leaves scaling at 128/42=3.05 instead of 2.0: a different organism, no error.
gate "Gate A config resolves" bash -c '
  "$0" main.py --cfg job \
    training/model=qwen2_5_1_5b \
    "training/experiment@training.sleeper_experiment=sleeper_topk_r64_k8" \
    training.sleeper_experiment.lora.layer=21 \
    training.sleeper_experiment.lora.r=42 training.sleeper_experiment.lora.alpha=84 \
    training.sleeper_experiment.lora.k=5 training.sleeper_experiment.lora.k_final=5 \
    training.sleeper_dataset.path="$1" \
    seed=42 training.dump_path=models/qwen15/r42_k5/l21_s42 \
  | grep -E "model_name|^      r:|alpha:|^      k:|k_final:|path:|layer:"
' "$PY" "$DATA"

echo
if [ $FAILED -ne 0 ]; then
  echo "=== ONE OR MORE GATES FAILED -- refusing to train. Nothing was launched. ==="
  exit 1
fi
echo "=== ALL GATES PASSED · $(date -Is) ==="

if [ -n "${CHECKS_ONLY:-}" ]; then
  echo "CHECKS_ONLY set -- stopping before training."
  exit 0
fi

exec bash scripts/qwen15_run_all.sh
