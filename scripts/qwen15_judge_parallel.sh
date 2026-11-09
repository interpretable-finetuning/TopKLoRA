#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Run the T4 capability judge as N parallel 2-GPU instances over DISJOINT slices of the surgical
# files, instead of one instance walking all 18 serially.
#
#   bash scripts/qwen15_judge_parallel.sh
#   SHARDS=4 bash scripts/qwen15_judge_parallel.sh
#
# The 32B judge is pipeline-sharded by device_map=auto, so a single instance leaves both its cards
# at ~50% and adding more cards to it buys nothing. Throughput comes from running several
# independent instances instead.
#
# Slices MUST be disjoint: judge_saved_gens_big rewrites each file in place, so two instances
# holding the same file would race and one would lose its scores.

PY="${PY:-.venv/bin/python}"
SHARDS="${SHARDS:-4}"
NGPU="${NGPU:-2}"
PATTERN="${PATTERN:-clcd_results/qwen15/*/surgical/*_surgical.json}"
# "<node>:<gpu,gpu>" per shard. Defaults keep one card spare on each node used.
PLACEMENT="${PLACEMENT:-torrnode15:0,1 torrnode15:2,3 torrnode15:4,5 torrnode13:0,1}"
OPTS="-o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=no"
LOGDIR=logs/qwen15/judge_parallel
mkdir -p "$LOGDIR"

mapfile -t FILES < <(ls $PATTERN 2>/dev/null)
n=${#FILES[@]}
[ "$n" -gt 0 ] || { echo "no surgical files match $PATTERN"; exit 1; }
set -- $PLACEMENT
nplace=$#
[ "$SHARDS" -le "$nplace" ] || { echo "SHARDS=$SHARDS but only $nplace placements given"; exit 1; }

echo "=== $n surgical files over $SHARDS shards ==="
i=0
for spec in $PLACEMENT; do
  [ "$i" -ge "$SHARDS" ] && break
  node="${spec%%:*}"; gpus="${spec##*:}"
  # round-robin so shards differ by at most one file
  slice=""
  for ((j=i; j<n; j+=SHARDS)); do slice+="${FILES[$j]} "; done
  slice=$(echo "$slice" | xargs)
  cnt=$(echo "$slice" | wc -w)
  log="$LOGDIR/shard${i}_${node}_gpu${gpus//,/-}.out"
  echo "  shard $i -> $node gpu[$gpus]  $cnt files"
  ssh -n -f $OPTS "$node" \
    "setsid nohup env GPUS='$gpus' NGPU=$NGPU FILES='$slice' \
       bash $REPO_ROOT/scripts/qwen15_judge.sh > $REPO_ROOT/$log 2>&1 < /dev/null &" \
    >/dev/null 2>&1 \
    && echo "      launched -> $log" || echo "      LAUNCH FAILED"
  i=$((i+1))
done
echo
echo "monitor: tail -f $LOGDIR/*.out"
echo "progress: .venv/bin/python -c \"import glob,json;t=n=0\"  # or rerun the status check"
