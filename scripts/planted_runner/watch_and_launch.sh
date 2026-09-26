#!/bin/bash
# Waits until a card is handed over -- its 7B surgical result exists AND nvidia-smi shows the card
# empty -- then starts 3 planted-circuit streams on it. Agreed with the qwen2.5-7B session: cards 5
# and 6 only, and only after their s44/s45 surgical steps finish.
#   watch_and_launch.sh <gpu> <surgical json that must exist>
set -u
GPU="$1"; DONE_FILE="$2"
T=/home/andrzej/.claude/jobs/4f8ac3b6/tmp
echo "[$(date +%H:%M:%S)] g$GPU waiting for $DONE_FILE and an empty card"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU") \
    || { echo "nvidia-smi failed"; exit 2; }
  if [ -f "$DONE_FILE" ] && [ "$used" -lt 1000 ]; then break; fi
  sleep 60
done
echo "[$(date +%H:%M:%S)] g$GPU handed over (${used} MiB used); launching"
# 2 streams: a K-sweep holds ~30-33 GB, so 2 fit a 95 GB card with margin; 3 would not.
exec bash "$T/run_planted.sh" "$T/planted_jobs.tsv" "$GPU:2"
