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
# Stop-token census across FAMILY x CIRCUIT-METHOD x BUDGET.
#
#   nohup bash scripts/stoptoken_method_grid.sh > logs/stoptoken/grid.out 2>&1 &
#
# Question: does the post-EOT artifact behave the same for circuits found by prefix search
# (exp_circuit_search, clcd_results/rigorous/) as for circuits found by scrubbing/elimination
# (clcd_results/rigorous/elim*)? And does either grow with the generation budget?
#
# Seed is held FIXED within a family, so only the circuit method varies. The l1523 seed-44 cell is
# the strongest: both methods have ablated fires on the same adapter (prefix 1, scrub 3).
#
# Cells already measured are skipped by stoptoken_census_one.sh, so this is re-enterable and the
# full grid is listed for readability rather than only the missing parts.
#
# NOTE on the `all` scrub cell: elim/all_seed43 has NO *_surgical.json, so no logged tie-back. Its
# intact arm must instead reproduce our own all_seed43 census exactly (intact ignores the circuit),
# which catches a mis-paired adapter or wrong band but is weaker provenance. Stated in the log.
#
# GPU pool. All three are ours as of 2026-08-11; override with GPUS_OVERRIDE if that changes.
GPUS=(${GPUS_OVERRIDE:-0 1 2})
BUDGETS=(40 100 200)

R=clcd_results/rigorous
CELLS=(
  "l19_prefix    $R/l19_seed42_circuit.json"
  "l19_scrub     $R/elim2/l19_seed42_nc1000_circuit.json"
  "l1523_prefix  $R/l1523_seed44_circuit.json"
  "l1523_scrub   $R/elim2/l1523_seed44_nc1000_adaptive_circuit.json"
  "all_prefix    $R/all_seed43_circuit.json"
  "all_scrub     $R/elim/all_seed43_circuit.json"
)
mkdir -p logs/stoptoken clcd_results/stoptoken

echo "=== method grid start $(date) ==="
printf '  %s\n' "${CELLS[@]}"
echo "  budgets: ${BUDGETS[*]}   GPUs: ${GPUS[*]}"

# Build the job list: (circuit, budget) pairs, cheapest budget first so early rows land sooner.
JOBS=()
for mnt in "${BUDGETS[@]}"; do
  for cell in "${CELLS[@]}"; do
    set -- $cell
    [ -f "$2" ] && JOBS+=("$2 $mnt") || echo "  SKIP $1 -- no circuit at $2"
  done
done

# One queue file per GPU: bash has no nested arrays, and files keep the split readable in the log.
QDIR=$(mktemp -d); trap 'rm -rf "$QDIR"' EXIT
for idx in "${!JOBS[@]}"; do
  printf '%s\n' "${JOBS[$idx]}" >> "$QDIR/q$((idx % ${#GPUS[@]}))"
done

worker() {   # $1 gpu  $2 queue file
  local gpu=$1 qf=$2
  [ -f "$qf" ] || { echo "[g$gpu] no jobs"; return 0; }
  local circuit mnt   # declare: `while read` leaves them unset if the queue is empty, and
                      # `set -u` then trips on the loop-exit check
  while read -r circuit mnt; do
    GPU=$gpu bash scripts/stoptoken_census_one.sh "$circuit" trigger "$mnt"
  done < "$qf"
  echo "[g$gpu] finished"
}

for gi in "${!GPUS[@]}"; do
  echo "  g${GPUS[$gi]} queue: $(wc -l < "$QDIR/q$gi" 2>/dev/null || echo 0) jobs"
done
for gi in "${!GPUS[@]}"; do
  worker "${GPUS[$gi]}" "$QDIR/q$gi" &
done
wait
echo "=== method grid COMPLETE $(date) ==="
echo "aggregate: "$PY" -m src.clcd.aggregate_stoptoken --flips"
