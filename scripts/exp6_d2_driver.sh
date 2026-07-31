#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Queues the d=2 half of the partition-width sweep behind d=4 on the same 3 GPUs, so the sweep
# completes without ever exceeding the GPU budget (3 discovery + 3 capability + 3 training = 9).
echo "[d2-driver] $(date) waiting for the d=4 runs to finish"
until [ "$(grep -l train_runtime logs/exp6/route_d4_l1523_s4*.out 2>/dev/null | wc -l)" -ge 3 ]; do
  sleep 120
done
echo "[d2-driver] $(date) d=4 complete -> training d=2"
exec env D=2 ARMS=route SEEDS="42 43 44" GPUS="3 4 5" bash scripts/train_route_pilot.sh
