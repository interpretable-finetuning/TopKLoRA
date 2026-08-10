#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Wait until ANOTHER session's training jobs are all finished -- including ones still queued --
# then run the stop-token method grid.
#
#   nohup bash scripts/wait_then_run_grid.sh > logs/stoptoken/wait_grid.out 2>&1 &
#
# Two signals, both required, for QUIET_CHECKS consecutive polls:
#   * no `main.py` training process
#   * total GPU memory in use below IDLE_MB
# Requiring a run of consecutive quiet polls is what tolerates a QUEUE: one job exiting and the
# next starting seconds later never looks idle for long enough to fool it.
#
# This lives in a FILE rather than an inline `bash -c` on purpose. `pgrep -f` matches full command
# lines, so an inline waiter whose own command line contains the pattern matches ITSELF and waits
# forever. That bug cost idle GPU time earlier today.

QUIET_CHECKS=${QUIET_CHECKS:-10}    # consecutive quiet polls before we start (10 x 60s = 10 min)
INTERVAL=${INTERVAL:-60}
IDLE_MB=${IDLE_MB:-500}             # total across all GPUs; idle reads ~6 MB

quiet=0; waited=0
echo "=== waiting for the other session to finish (need ${QUIET_CHECKS} quiet polls @ ${INTERVAL}s) ==="
while [ "$quiet" -lt "$QUIET_CHECKS" ]; do
  n=$(pgrep -fc 'main\.py' 2>/dev/null || echo 0)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  if [ "${n:-0}" -eq 0 ] && [ "${mem:-999999}" -lt "$IDLE_MB" ]; then
    quiet=$((quiet + 1))
  else
    [ "$quiet" -gt 0 ] && echo "[$(date +%H:%M)] activity resumed (procs=$n gpu_mem=${mem}MB) -- quiet counter reset"
    quiet=0
  fi
  waited=$((waited + INTERVAL))
  [ $((waited % 600)) -eq 0 ] && echo "[$(date +%H:%M)] still waiting: procs=$n gpu_mem=${mem}MB quiet=${quiet}/${QUIET_CHECKS}"
  sleep "$INTERVAL"
done

echo "=== other session idle for $((QUIET_CHECKS * INTERVAL / 60)) min -- starting the grid $(date) ==="
exec bash scripts/stoptoken_method_grid.sh
