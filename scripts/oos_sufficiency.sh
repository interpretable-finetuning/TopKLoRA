#!/bin/bash
# Out-of-sample sufficiency: intact + ablate + keep-only for every organism in a manifest, on ONE GPU.
#
#   MANIFEST=clcd_results/oos_sufficiency/manifest.tsv OUT=clcd_results/oos_sufficiency \
#     GPUS="0 1 2" JOBS=2 bash scripts/oos_sufficiency.sh
#
# Each manifest row is one organism (tab-separated): tree, arm, family, seed, circuit json, base model,
# data dir. Each runs analysis/verify_holdout_necessity.py on eval_triggered bands 2000/3000/4000/5000
# x 1,000, the same bands and settings as the campaign's leak step (scripts/qwen15_phase1.sh), plus
# the intact and keep-only arms, and writes $OUT/<tree>/<arm>/<family>_seed<seed>.json.
#
# Considered qwen15_phase1.sh for this: its leak step needs a Gate-A record and an adapter for every
# cell and writes into the campaign tree, whose leak files paper_leaks.py expects to hold one record.
#
# Waits for a card that has had no compute process for IDLE_FOR seconds, then takes that one card for
# the whole queue, because the shared-node rule is not to land on another user's job. One idle reading
# is not enough: on 2026-11-09 card 1 read idle in a gap between another user's vLLM runs, their
# engine took 72 GB within seconds, and our first two jobs died out of memory. Resumable: a row whose output exists is
# skipped, and verify_holdout_necessity writes it atomically, so a killed job leaves nothing behind.
# A failed organism is recorded and the rest carry on; the driver then exits non-zero.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MANIFEST="${MANIFEST:?MANIFEST must name the organism list}"
OUT="${OUT:?OUT must name the output tree}"
GPUS="${GPUS:-0 1 2}"
JOBS="${JOBS:-2}"
POLL="${POLL:-60}"
WAIT_MAX="${WAIT_MAX:-86400}"
IDLE_FOR="${IDLE_FOR:-600}"
BUSY_MIB="${BUSY_MIB:-2000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-}"        # overrides need_mib below when set
LAUNCH_GAP="${LAUNCH_GAP:-120}"
CARD="${CARD:-}"                        # a card the owner has assigned to this run: no idle wait
PY="${PY:-.venv/bin/python}"
BANDS=2000,3000,4000,5000
LOGDIR="${LOGDIR:-logs/oos_sufficiency}"
STATUS="$OUT/queue_status.tsv"

mkdir -p "$OUT" "$LOGDIR"
[ -f "$MANIFEST" ] || { echo "MANIFEST not found: $MANIFEST"; exit 2; }

# 0 if card $1 is idle now, 1 if not, 2 if nvidia-smi itself failed: a failing nvidia-smi must not
# look like "no processes on this card".
card_idle() {
  local apps mem
  apps=$(nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader) || return 2
  mem=$(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits) || return 2
  [ -z "$apps" ] && [ "$mem" -lt "$BUSY_MIB" ]
}

declare -A idle_since
waited=0 gpu="$CARD"
while [ -z "$gpu" ]; do
  for g in $GPUS; do
    card_idle "$g"; rc=$?
    [ "$rc" -eq 2 ] && { echo "[$(date -Is)] nvidia-smi failed, stopping"; exit 2; }
    if [ "$rc" -eq 0 ]; then
      : "${idle_since[$g]:=$SECONDS}"
      if [ $((SECONDS - idle_since[$g])) -ge "$IDLE_FOR" ]; then gpu=$g; break; fi
    else
      unset "idle_since[$g]"
    fi
  done
  [ -n "$gpu" ] && break
  [ "$waited" -ge "$WAIT_MAX" ] && { echo "[$(date -Is)] no card idle for ${IDLE_FOR}s among [$GPUS] after ${waited}s"; exit 3; }
  [ "$waited" -eq 0 ] && echo "[$(date -Is)] waiting for a card idle for ${IDLE_FOR}s among [$GPUS]"
  sleep "$POLL"; waited=$((waited + POLL))
done
export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
echo "[$(date -Is)] queue on card $gpu, $JOBS at a time, commit $(git rev-parse --short HEAD)"

run_one() {
  local tree=$1 arm=$2 fam=$3 seed=$4 circ=$5 base=$6 data=$7
  local name="${tree}_${arm}_${fam}_s${seed}" out="$OUT/$tree/$arm/${fam}_seed${seed}.json" rc=0 t0=$SECONDS
  mkdir -p "$(dirname "$out")"
  echo "[$(date -Is)] start $name"
  CLCD_BASE="$base" CLCD_DATA="$data" CLCD_BANDS="$BANDS" CLCD_N=1000 CLCD_MNT=40 \
    CLCD_INTACT=1 CLCD_KEEP_ONLY=1 CLCD_SAVE_GENS=1 CLCD_OUT="$out" \
    "$PY" -u analysis/verify_holdout_necessity.py "$circ" > "$LOGDIR/$name.out" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$out" ]; then
    printf '%s\tOK\t%s\n' "$name" "$((SECONDS - t0))" >> "$STATUS"
    echo "[$(date -Is)] done  $name ($((SECONDS - t0))s)"
  else
    printf '%s\tFAILED\t%s\n' "$name" "$rc" >> "$STATUS"
    echo "[$(date -Is)] FAILED $name rc=$rc (log: $LOGDIR/$name.out)"
  fi
}

: > "$STATUS"   # this run's outcomes only; a resume's earlier rows would otherwise count twice
# MiB a job of this tree and family needs free before it starts, from peaks measured on 2026-11-09:
# 7B all-layers reaches ~57 GB by itself and keeps growing for minutes, gemma/Qwen all-layers ~27 GB.
need_mib() {
  if [ -n "$MIN_FREE_MIB" ]; then echo "$MIN_FREE_MIB"; return; fi
  case "$1/$2" in
    qwen7b*/all) echo 70000 ;;
    */all) echo 40000 ;;
    */l1523|*/l17_25) echo 30000 ;;
    *) echo 20000 ;;
  esac
}

# Holds the next launch while the card is short of memory, so a job that arrives on our card later
# makes the queue wait instead of burning through the manifest with out-of-memory failures.
wait_for_room() {
  local free said=0 need
  need=$(need_mib "$1" "$2")
  while :; do
    free=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits) || return 2
    [ "$free" -ge "$need" ] && return 0
    [ "$said" -eq 0 ] && echo "[$(date -Is)] card $gpu has ${free} MiB free, $1/$2 needs ${need}, holding the launch"
    said=1; sleep "$POLL"
  done
}

running=0 skipped=0
while IFS=$'\t' read -r tree arm fam seed circ base data; do
  if [ -f "$OUT/$tree/$arm/${fam}_seed${seed}.json" ]; then skipped=$((skipped + 1)); continue; fi
  while [ "$running" -ge "$JOBS" ]; do wait -n; running=$((running - 1)); done
  wait_for_room "$tree" "$fam" || { echo "[$(date -Is)] nvidia-smi failed, stopping"; exit 2; }
  run_one "$tree" "$arm" "$fam" "$seed" "$circ" "$base" "$data" &
  running=$((running + 1))
  sleep "$LAUNCH_GAP"   # let the job reach its memory footprint, so wait_for_room sees it
done < "$MANIFEST"
wait

ok=$(grep -c $'\tOK\t' "$STATUS" || true)
failed=$(grep -c $'\tFAILED\t' "$STATUS" || true)
echo "[$(date -Is)] queue finished: ok=$ok failed=$failed skipped_existing=$skipped"
[ "$failed" -eq 0 ]
