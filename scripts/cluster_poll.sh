#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Keep retrying a cell list against the cluster until every cell has an artifact.
#
#   nohup bash scripts/cluster_poll.sh scripts/qwen15_phase1.sh cells.txt > logs/cluster/poll.out 2>&1 &
#   INTERVAL=300 TAG=phase1_l20 bash scripts/cluster_poll.sh ...
#
# The cluster is shared and fills up; cards free at unpredictable times. Each round this works out
# what is genuinely outstanding and dispatches only that to whatever is free right now.
#
# A cell is NOT re-dispatched if either:
#   - its output artifact already exists (done), or
#   - a process somewhere on the cluster is already working on it (in flight).
# The second test matters: a running search has no circuit file yet, so "done" alone would
# relaunch it onto a second card and have two jobs write the same output.

DRIVER="${1:?usage: cluster_poll.sh <driver.sh> <cellfile>}"
CELLFILE="${2:?usage: cluster_poll.sh <driver.sh> <cellfile>}"
INTERVAL="${INTERVAL:-300}"
TAG="${TAG:-$(basename "$DRIVER" .sh)}"
NODES="${NODES:-torrnode8 torrnode9 torrnode10 torrnode11 torrnode12 torrnode13 torrnode14 torrnode15}"
SSH_OPTS="-o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=no"
MAXROUNDS="${MAXROUNDS:-288}"        # 288 x 5min = 24h
mkdir -p logs/cluster

# artifact a cell produces, so "done" is checked against the real output not a marker file
artifact() {  # arm fam seed
  echo "clcd_results/qwen15/$1/elim/$2_seed$3_circuit.json"
}

round=0
while [ "$round" -lt "$MAXROUNDS" ]; do
  round=$((round + 1))
  mapfile -t CELLS < <(grep -vE '^\s*(#|$)' "$CELLFILE")

  # what is already being worked on, cluster-wide. Two sources, both required: the --out path of
  # live search processes, AND the cell lists on live launchers' command lines -- a cell queued
  # inside a launcher has no python process yet, and missing it double-booked cells on 2026-09-14
  # (captain's log, ERROR entry of that date).
  procs=$(for n in $NODES; do
      ssh -n $SSH_OPTS "$n" 'ps -u $USER -o cmd 2>/dev/null | grep -E "[e]xp_circuit_search|[q]wen15_phase1\.sh"' 2>/dev/null &
    done; wait)
  inflight=$(printf '%s\n' "$procs" | grep -oE '[^ ]*_circuit\.json' | sort -u)
  queued=$(printf '%s\n' "$procs" | grep 'qwen15_phase1\.sh' \
      | grep -oE '(r[0-9]+_k[0-9]+) ([a-z0-9_]+) ([0-9]+)' | sort -u \
      | while read -r qa qf qs; do artifact "$qa" "$qf" "$qs"; done)
  inflight=$(printf '%s\n%s\n' "$inflight" "$queued" | grep . | sort -u)

  pending=()
  for c in "${CELLS[@]}"; do
    set -- $c
    a=$(artifact "$1" "$2" "$3")
    [ -f "$a" ] && continue
    printf '%s\n' "$inflight" | grep -qF "$a" && continue
    pending+=("$c")
  done

  ndone=0
  for c in "${CELLS[@]}"; do set -- $c; [ -f "$(artifact "$1" "$2" "$3")" ] && ndone=$((ndone+1)); done
  nfly=$(printf '%s\n' "$inflight" | grep -c . || true)
  echo "[$(date -Is)] round $round: ${#CELLS[@]} cells · $ndone done · $nfly in flight · ${#pending[@]} pending"

  if [ ${#pending[@]} -eq 0 ]; then
    if [ "$nfly" -eq 0 ]; then
      echo "[$(date -Is)] ALL CELLS COMPLETE -- poller exiting"
      exit 0
    fi
    echo "[$(date -Is)] nothing to dispatch; waiting on in-flight work"
  else
    printf '%s\n' "${pending[@]}" > "$REPO_ROOT/logs/cluster/.pending_$TAG.txt"
    TAG="$TAG" bash scripts/cluster_run.sh "$DRIVER" "$REPO_ROOT/logs/cluster/.pending_$TAG.txt" 2>&1 \
      | grep -E 'free GPUs|launched|LAUNCH FAILED|no free' | sed 's/^/    /'
  fi
  sleep "$INTERVAL"
done
echo "[$(date -Is)] max rounds reached -- poller exiting"
