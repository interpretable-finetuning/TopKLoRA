#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Stop this project's jobs on every torrnode, and clear the shared GPU locks.
#
#   bash scripts/cluster_stop.sh              # stop everything this project started
#   bash scripts/cluster_stop.sh qwen15_phase1   # only jobs matching a pattern
#
# Kills only processes owned by $USER whose command line matches the project's drivers -- never
# another user's work, and never an unrelated process of your own.

PAT="${1:-qwen15_|exp_circuit_search|src.clcd.gate_a|main.py|cluster_run.sh}"
NODES="${NODES:-torrnode8 torrnode9 torrnode10 torrnode11 torrnode12 torrnode13 torrnode14 torrnode15}"
SSH_OPTS="-o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=no"

echo "=== stopping '$PAT' as $USER across the cluster $(date -Is) ==="
for n in $NODES; do
  ssh $SSH_OPTS "$n" exit 2>/dev/null || { printf "  %-12s unreachable\n" "$n"; continue; }
  out=$(ssh $SSH_OPTS "$n" "
      pids=\$(pgrep -u \$USER -f '$PAT' 2>/dev/null | tr '\n' ' ')
      if [ -z \"\$pids\" ]; then echo 'none'; else
        kill -TERM \$pids 2>/dev/null; sleep 4
        pids2=\$(pgrep -u \$USER -f '$PAT' 2>/dev/null | tr '\n' ' ')
        [ -n \"\$pids2\" ] && kill -KILL \$pids2 2>/dev/null
        echo \"killed: \$pids\"
      fi" 2>/dev/null)
  printf "  %-12s %s\n" "$n" "$out"
done

# locks live on shared storage, so one sweep from anywhere clears them all
LOCKROOT="$REPO_ROOT/logs/qwen15/.gpulocks"
if [ -d "$LOCKROOT" ]; then
  nl=$(find "$LOCKROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
  find "$LOCKROOT" -mindepth 1 -maxdepth 1 -type d -exec rmdir {} + 2>/dev/null
  echo "cleared $nl GPU lock(s)"
fi
echo "=== done $(date -Is) ==="
