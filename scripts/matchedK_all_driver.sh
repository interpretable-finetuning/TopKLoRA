#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Waits for the 6 surviving one-per-GPU jobs from the first (over-subscribed) launch to finish,
# then runs the slot-worker version for the 9 organisms that OOM'd. Idempotent either way.
echo "[driver] $(date) waiting for in-flight mka_ sessions: $(tmux ls 2>/dev/null | grep -c '^mka_')"
until [ "$(tmux ls 2>/dev/null | grep -c '^mka_')" -eq 0 ]; do sleep 60; done
echo "[driver] $(date) node clear -> slot-worker launch"
exec bash scripts/matchedK_all.sh
