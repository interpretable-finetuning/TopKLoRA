#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
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
STEM=$(basename "${CIRCUIT%_circuit.json}")
SUFFIX=""; [ "$TAG" != "trigger" ] && SUFFIX="_${TAG}"; [ "$MNT" != "40" ] && SUFFIX="${SUFFIX}_mnt${MNT}"
OUT=clcd_results/stoptoken/${STEM}_census${SUFFIX}.json
GPU="${GPU:-0}"
mkdir -p clcd_results/stoptoken logs/stoptoken
[ -f "$OUT" ] && { echo "[$STEM] census exists, skipping"; exit 0; }

# Adapter and per-family token budget come from the circuit's recorded adapter, not the filename.
read -r ADAPTER MBT < <(uv run python - "$CIRCUIT" <<'PY'
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

# The tie-back is what makes the census citable; state plainly when it is unavailable.
EXPECT=""
if [ -f "$BASELINE" ] && [ "$TAG" = "trigger" ] && [ "$MNT" = "40" ]; then
  EXPECT="--expect_raw_asr $(uv run python - "$BASELINE" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["conditions"]
print(",".join(f"{a}={c[a]['backdoor_asr']}" for a in ("intact", "ablate_circuit") if a in c))
PY
)"
  echo "[preflight] tie-back -> $BASELINE"
else
  echo "[preflight] NO tie-back (baseline absent, or tag/budget differs from the logged run)"
fi

export CUDA_VISIBLE_DEVICES=$GPU
echo "[$(date +%H:%M) $STEM g$GPU tag=$TAG mnt=$MNT mbt=$MBT] census"
uv run python -u -m src.clcd.exp_stoptoken_census --adapter "$ADAPTER" --circuit_json "$CIRCUIT" \
  --data data/sleeper/prepared_eval6k --dtype bfloat16 --arms intact,ablate_circuit,keep_only \
  --tag "$TAG" --offset 2000 --n_backdoor 1000 --mnt_backdoor "$MNT" --max_batch_tokens "$MBT" \
  $EXPECT --out "$OUT" 2>&1 | tee logs/stoptoken/${STEM}_census${SUFFIX}.out
RC=${PIPESTATUS[0]}   # not $? -- that would be tee's status
echo "=== $STEM done (exit $RC) -> $OUT ==="
exit $RC
