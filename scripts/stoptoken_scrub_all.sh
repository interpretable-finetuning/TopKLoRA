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
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
# Stop-token census over the SCRUBBING (elimination) circuits, for comparison with the prefix
# circuits already measured in docs/captains-log-qwen2.5-1.5b.md §0.
#
#   nohup bash scripts/stoptoken_scrub_all.sh > logs/stoptoken/scrub.out 2>&1 &
#
# Covers clcd_results/rigorous/elim2/*_circuit.json that have a sibling *_surgical.json, so every
# run has a tie-back. elim/ is deliberately EXCLUDED: those circuits have no surgical baseline, so
# a census there could not be verified against anything.
#
# GPU POOL IS (0 1) BY OPERATOR REQUEST -- GPU 2 IS NOT OURS.
GPUS=(${GPUS_OVERRIDE:-0 1})
mkdir -p logs/stoptoken clcd_results/stoptoken

mapfile -t CIRCUITS < <(
  for c in clcd_results/rigorous/elim2/*_circuit.json; do
    [ -f "${c%_circuit.json}_surgical.json" ] && echo "$c"
  done
)
[ ${#CIRCUITS[@]} -gt 0 ] || { echo "no elim2 circuits with a surgical baseline"; exit 1; }

echo "=== scrubbing-circuit census start $(date) ==="
echo "    circuits with a tie-back baseline: ${#CIRCUITS[@]}"
printf '      %s\n' "${CIRCUITS[@]##*/}"
echo "    excluded: clcd_results/rigorous/elim/ (no *_surgical.json -> no tie-back)"

worker() {
  local gpu=$1; shift
  for c in "$@"; do GPU=$gpu bash scripts/stoptoken_census_one.sh "$c" trigger 40; done
  echo "[g$gpu] finished"
}

declare -a Q0 Q1
i=0
for c in "${CIRCUITS[@]}"; do
  if [ $((i % 2)) -eq 0 ]; then Q0+=("$c"); else Q1+=("$c"); fi
  i=$((i + 1))
done
worker "${GPUS[0]}" "${Q0[@]}" &
worker "${GPUS[1]}" "${Q1[@]}" &
wait
echo "=== scrubbing-circuit census COMPLETE $(date) ==="
echo "aggregate with: "$PY" -m src.clcd.aggregate_stoptoken --flips"
