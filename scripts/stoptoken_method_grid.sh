#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
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
# GPU POOL IS (0 1) -- GPU 2 IS NOT OURS.
GPUS=(${GPUS_OVERRIDE:-0 1})
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

worker() {
  local gpu=$1; shift
  for job in "$@"; do
    set -- $job
    GPU=$gpu bash scripts/stoptoken_census_one.sh "$1" trigger "$2"
  done
  echo "[g$gpu] finished"
}

declare -a Q0 Q1
i=0
for j in "${JOBS[@]}"; do
  if [ $((i % 2)) -eq 0 ]; then Q0+=("$j"); else Q1+=("$j"); fi
  i=$((i + 1))
done
worker "${GPUS[0]}" "${Q0[@]}" &
worker "${GPUS[1]}" "${Q1[@]}" &
wait
echo "=== method grid COMPLETE $(date) ==="
echo "aggregate: uv run python -m src.clcd.aggregate_stoptoken --flips"
