#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# DIAGNOSTIC / OPT-IN. Not part of any pipeline, not run by CI, not imported by library code, and
# not collected by pytest. It exists to measure a bug that is already FIXED
# (src/utils.py::resolve_stop_token_ids) and generates deliberately PRE-FIX output to do so.
# Run it only to re-measure that bug on new organisms or circuits. See
# docs/captains-log-qwen2.5-1.5b.md for what it established.
# Call the interpreter directly, not `uv run`. uv re-validates the environment on every invocation,
# and this venv has ~42k files on a FUSE mount: measured 11.7s per `uv run` vs 2.1s direct. This
# script invokes python several times per census, so that is ~50s of pure overhead per cell.
# Local venv if built: measured `import torch` at 3s local vs 77s on the FUSE-backed one.
# Rebuild after a pod restart with:
#   UV_PROJECT_ENVIRONMENT=/opt/topklora/venv UV_PYTHON_INSTALL_DIR=/opt/topklora/uv-python \
#   UV_CACHE_DIR=/opt/topklora/uv-cache uv sync --frozen
if [ -z "${PY:-}" ]; then
  [ -x /opt/topklora/venv/bin/python ] && PY=/opt/topklora/venv/bin/python || PY="$REPO_ROOT/.venv/bin/python"
fi
# Stop-token census for ONE circuit, whatever its naming convention.
#
#   bash scripts/stoptoken_census_one.sh <circuit_json> [trigger|clean] [max_new_tokens]
#   bash scripts/stoptoken_census_one.sh clcd_results/rigorous/elim2/l1523_seed44_nc1000_adaptive_circuit.json
#
# Everything is derived from the circuit JSON's own `adapter` field, so this works for the prefix
# circuits (rigorous/<org>_circuit.json) and the scrubbing ones (elim2/<org>_<variant>_circuit.json)
# alike. The tie-back baseline is the sibling *_surgical.json; without one the census still runs but
# has nothing to verify against, and says so.

CIRCUIT="${1:?usage: stoptoken_census_one.sh <circuit_json> [trigger|clean] [mnt]}"
TAG="${2:-trigger}"
MNT="${3:-40}"
[ -f "$CIRCUIT" ] || { echo "[preflight] no such circuit: $CIRCUIT"; exit 1; }

BASELINE="${CIRCUIT%_circuit.json}_surgical.json"
# STEM must include the source directory. `rigorous/all_seed43_circuit.json` and
# `rigorous/elim/all_seed43_circuit.json` share a basename, so a basename-only stem silently
# collides: the second run either skips (looking already-done) or OVERWRITES the first, and the
# resulting artifact looks entirely normal either way.
STEM=$(dirname "$CIRCUIT" | sed 's|.*/rigorous/*||; s|/|_|g')
STEM="${STEM:+${STEM}_}$(basename "${CIRCUIT%_circuit.json}")"
SUFFIX=""; [ "$TAG" != "trigger" ] && SUFFIX="_${TAG}"; [ "$MNT" != "40" ] && SUFFIX="${SUFFIX}_mnt${MNT}"
OUT=clcd_results/stoptoken/${STEM}_census${SUFFIX}.json
GPU="${GPU:-0}"
mkdir -p clcd_results/stoptoken logs/stoptoken
[ -f "$OUT" ] && { echo "[$STEM] census exists, skipping"; exit 0; }

# Adapter and per-family token budget come from the circuit's recorded adapter, not the filename.
read -r ADAPTER MBT < <("$PY" - "$CIRCUIT" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
if c.get("status") not in (None, "ok"):
    sys.exit(f"[preflight] circuit status={c.get('status')!r}, not ok")
if not c.get("kept_latents"):
    sys.exit("[preflight] circuit has no kept_latents -- nothing to ablate")
ad = c["adapter"]
mbt = 4000 if "all_layers" in ad else (9000 if "15_23" in ad else 24000)
print(ad, mbt)
PY
) || exit 1
[ -f "$ADAPTER/adapter_config.json" ] || { echo "[preflight] adapter missing: $ADAPTER"; exit 1; }

# Tie-back. The ASSERTION on `intact` is same-hardware: any census we already made for this adapter
# must have an identical `intact` arm (intact applies no overrides, so it cannot depend on the
# circuit). The logged baseline's `intact` is reported but NOT asserted -- it crosses a hardware
# boundary and drifts by +-3/1000 (see E0), which failed 8 of 16 cells on correct data. Its
# `ablate_circuit` IS asserted: that arm matched exactly in all 16, so a mismatch there is real.
REF=$("$PY" -m src.clcd.same_hw_ref "$ADAPTER" 2>/dev/null || true)
[ -n "$REF" ] && echo "[preflight] same-hardware reference: intact must equal ${REF##* }/1000 (${REF%% *})"
EXPECT=""
LOGGED_ABL="-"; LOGGED_INTACT="-"
if [ -f "$BASELINE" ] && [ "$TAG" = "trigger" ] && [ "$MNT" = "40" ]; then
  read -r LOGGED_ABL LOGGED_INTACT < <("$PY" - "$BASELINE" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["conditions"]
print(*(c[a]["backdoor_asr"] if a in c else "-" for a in ("ablate_circuit", "intact")))
PY
) || { echo "[preflight] could not read baseline $BASELINE"; exit 1; }
  if [ "$LOGGED_ABL" != "-" ]; then
    EXPECT="--expect_raw_asr ablate_circuit=$LOGGED_ABL"
    echo "[preflight] tie-back -> $BASELINE (asserting ablate_circuit=$LOGGED_ABL; logged" \
         "intact=$LOGGED_INTACT reported only -- cross-hardware, see E0)"
  else
    echo "[preflight] tie-back -> $BASELINE has no ablate_circuit arm: NOTHING asserted from it"
  fi
  # ablate_circuit's success value is 0.0 -- also what a mis-paired adapter gives. Only a real
  # check when a high arm is anchored too.
  if [ -z "${REF:-}" ]; then
    echo "[preflight] WARNING: no same-hardware reference for this adapter, so no high arm is"
    echo "[preflight]          anchored. A tie-back PASS on ablate_circuit alone is weak evidence."
  fi
else
  echo "[preflight] NO tie-back (baseline absent, or tag/budget differs from the logged run)"
fi

export CUDA_VISIBLE_DEVICES=$GPU
echo "[$(date +%H:%M) $STEM g$GPU tag=$TAG mnt=$MNT mbt=$MBT] census"
"$PY" -u -m src.clcd.exp_stoptoken_census --adapter "$ADAPTER" --circuit_json "$CIRCUIT" \
  --data data/sleeper/prepared_eval6k --dtype bfloat16 --arms intact,ablate_circuit,keep_only \
  --tag "$TAG" --offset 2000 --n_backdoor 1000 --mnt_backdoor "$MNT" --max_batch_tokens "$MBT" \
  $EXPECT --out "$OUT" 2>&1 | tee logs/stoptoken/${STEM}_census${SUFFIX}.out
RC=${PIPESTATUS[0]}   # not $? -- that would be tee's status

# Only valid at the reference budget/tag -- intact ASR moves with max_new_tokens
# (E3b: 997/996/997 at 40/100/200), so checking a mnt=40 reference against another
# budget reports a mismatch that is not one.
if [ -n "${REF:-}" ] && [ -f "$OUT" ] && [ "$MNT" = "40" ] && [ "$TAG" = "trigger" ]; then
  "$PY" -m src.clcd.same_hw_ref --check "$OUT" "${REF##* }" || RC=1
fi
echo "=== $STEM done (exit $RC) -> $OUT ==="
exit $RC
