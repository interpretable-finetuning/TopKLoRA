#!/bin/bash
# Runs the planted-circuit jobs from analysis.planted_circuits' TSV. One stream per argument pair
# "<gpu>:<streams>"; jobs are dealt round-robin to streams. A job whose output exists is skipped, so
# a relaunch resumes. Run from the main checkout: adapter/data paths in the circuit files are relative.
#   run_planted.sh <jobs.tsv> 5:3 6:3
set -u
JOBS="$1"; shift
REPO=/home/andrzej/TopKLoRA
PY=$REPO/.venv/bin/python
LOG=$REPO/logs/gradroute_planted
mkdir -p "$LOG"
cd "$REPO" || exit 2

streams=()
for spec in "$@"; do
  gpu=${spec%%:*}; n=${spec##*:}
  for ((i = 0; i < n; i++)); do streams+=("$gpu"); done
done
[ ${#streams[@]} -gt 0 ] || { echo "no streams"; exit 2; }

run_job() {   # <gpu> <tsv line>
  local gpu="$1" kind arm seed adapter order ks out
  IFS=$'\t' read -r kind arm seed adapter order ks out <<< "$2"
  [ -f "$out" ] && { echo "[$(date +%H:%M:%S) g$gpu] skip $kind $arm s$seed (exists)"; return 0; }
  mkdir -p "$(dirname "$out")"
  # mkdir is atomic: two runners over the same list (one per card, started when each card frees)
  # never run the same job. A failed job drops its claim so a relaunch retries it.
  mkdir "$out.claim" 2> /dev/null || { echo "[$(date +%H:%M:%S) g$gpu] skip $kind $arm s$seed (claimed)"; return 0; }
  local log="$LOG/${kind}_${arm}_s${seed}.out"
  echo "[$(date +%H:%M:%S) g$gpu] start $kind $arm s$seed"
  if [ "$kind" = L ]; then
    CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
    CLCD_BASE=google/gemma-2-2b CLCD_DATA=data/sleeper/prepared_eval6k CLCD_MNT=40 \
    CLCD_BANDS=2000,3000,4000,5000 CLCD_N=1000 CLCD_OUT="$out" \
      "$PY" -u analysis/verify_holdout_necessity.py ${order//,/ } > "$log" 2>&1
  else
    # shellcheck disable=SC2086  # $ks is a space-separated K list by construction
    CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$REPO PYTHONUNBUFFERED=1 \
      "$PY" -u -m src.clcd.exp_circuit_search --adapter "$adapter" --base_model google/gemma-2-2b \
      --data data/sleeper/prepared_eval6k --dtype bfloat16 --Ks $ks --mnt 40 \
      --offset 100 --n_backdoor 1000 --suff_n_se 2.0 --sat_floor 0.90 --nec_target 0.0 \
      --batch_size 64 --ordering file --order_file "$order" --out "$out" > "$log" 2>&1
  fi
  local rc=$?
  [ "$rc" -ne 0 ] && rmdir "$out.claim"
  echo "[$(date +%H:%M:%S) g$gpu] done  $kind $arm s$seed rc=$rc"
}

mapfile -t lines < "$JOBS"
for s in "${!streams[@]}"; do
  (
    for ((j = s; j < ${#lines[@]}; j += ${#streams[@]})); do run_job "${streams[$s]}" "${lines[$j]}"; done
  ) &
done
wait
echo "[$(date +%H:%M:%S)] all streams finished"
