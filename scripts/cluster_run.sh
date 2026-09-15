#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Spread a list of "<arm> <family> <seed>" cells across FREE GPUs on the torrnode cluster.
#
#   bash scripts/cluster_run.sh scripts/qwen15_sweep.sh cells.txt
#   EXTRA_ENV="BASE_MODEL=..." bash scripts/cluster_run.sh scripts/qwen15_sweep.sh cells.txt
#   EXTRA_ENV="STEPS=search MNT=40" bash scripts/cluster_run.sh scripts/qwen15_phase1.sh cells.txt
#   DRY=1 bash scripts/cluster_run.sh ...        # show the plan, launch nothing
#
# FREE means "no compute processes on that card", the same test ~/gpu_script.sh uses -- stricter
# than a memory threshold and the right one on shared cards. Discovery happens ONCE here to build
# the plan; each node's driver re-checks at claim time, so a card taken in between is skipped
# rather than fought over.
#
# /storage3 is shared, so every node reads the same repo, venv, dataset and results tree. GPU locks
# under logs/qwen15/.gpulocks are node-scoped because GPU indices are per-node.

DRIVER="${1:?usage: cluster_run.sh <driver.sh> <cellfile>}"
CELLFILE="${2:?usage: cluster_run.sh <driver.sh> <cellfile>}"
NODES="${NODES:-torrnode8 torrnode9 torrnode10 torrnode11 torrnode12 torrnode13 torrnode14 torrnode15}"
SSH_OPTS="-o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=no"
# NOT named ENV: that is the shell's startup-file variable and is set cluster-wide to
# /usr/share/Modules/init/profile.sh, which env then tried to execute
EXTRA_ENV="${EXTRA_ENV:-}"
# Leave this many free cards per node UNCLAIMED. These nodes are shared; taking every free
# card on a node locks everyone else out of it.
RESERVE="${RESERVE:-1}"
DRY="${DRY:-0}"
TAG="${TAG:-$(basename "$DRIVER" .sh)}"
LOGDIR="logs/cluster/$TAG"
mkdir -p "$LOGDIR"

[ -f "$CELLFILE" ] || { echo "no such cellfile: $CELLFILE"; exit 1; }
mapfile -t CELLS < <(grep -vE '^\s*(#|$)' "$CELLFILE")
[ ${#CELLS[@]} -gt 0 ] || { echo "cellfile is empty"; exit 1; }

echo "=== discovering free GPUs across the cluster $(date -Is) ==="
declare -A FREE
total=0
# probe every node in PARALLEL -- serial ssh across 8 nodes took long enough to trip a timeout
# mid-launch, which left some nodes started and others not
probe_dir=$(mktemp -d)
for n in $NODES; do
  ( ssh $SSH_OPTS "$n" '
      busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u)
      nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | while IFS=, read -r i u; do
        i=$(echo $i|xargs); u=$(echo $u|xargs)
        echo "$busy" | grep -q "$u" || echo "$i"
      done | tr "\n" " "' 2>/dev/null | xargs > "$probe_dir/$n" ) &
done
wait
for n in $NODES; do
  f=$(cat "$probe_dir/$n" 2>/dev/null)
  [ -n "$f" ] || { printf "  %-12s none free / unreachable\n" "$n"; continue; }
  nfree=$(echo "$f" | wc -w)
  # hand back the last RESERVE cards on this node
  take=$((nfree - RESERVE))
  if [ "$take" -le 0 ]; then
    printf "  %-12s %d free, all reserved for others -> taking none\n" "$n" "$nfree"
    continue
  fi
  f=$(echo "$f" | tr ' ' '\n' | head -n "$take" | tr '\n' ' ' | xargs)
  FREE[$n]="$f"
  total=$((total + take))
  printf "  %-12s %d free, taking %d: [%s]  (%d reserved)\n" "$n" "$nfree" "$take" "$f" "$RESERVE"
done
rm -rf "$probe_dir"

[ "$total" -gt 0 ] || { echo "no free GPUs anywhere -- nothing launched"; exit 1; }
echo "  -> $total free GPUs, ${#CELLS[@]} cells"

# One driver instance PER FREE GPU, each with its own slice. Drivers differ in how they take a
# card -- qwen15_sweep.sh reads $GPU (singular, default 2!), qwen15_phase1.sh reads $GPUS -- so
# both are set to the same single index. Passing only GPUS made sweep.sh fall back to GPU 2, which
# on most nodes is somebody else's card.
SLOTS=()
for n in "${!FREE[@]}"; do
  for g in ${FREE[$n]}; do SLOTS+=("$n:$g"); done
done
nslots=${#SLOTS[@]}

declare -A SLICE
for s in "${SLOTS[@]}"; do SLICE[$s]=""; done
i=0
for cell in "${CELLS[@]}"; do
  SLICE[${SLOTS[$((i % nslots))]}]+="$cell"$'\n'
  i=$((i + 1))
done

echo
echo "=== plan: $nslots slots ==="
for s in "${SLOTS[@]}"; do
  c=$(printf '%s' "${SLICE[$s]}" | grep -c . || true)
  printf "  %-20s %2d cells\n" "$s" "$c"
done

[ "$DRY" = "1" ] && { echo; echo "DRY=1 -- nothing launched"; exit 0; }

echo
echo "=== launching $(date -Is) ==="
for s in "${SLOTS[@]}"; do
  n="${s%%:*}"; g="${s##*:}"
  body=$(printf '%s' "${SLICE[$s]}" | grep . || true)
  [ -n "$body" ] || continue
  args=""
  while IFS= read -r c; do args+=" \"$c\""; done <<< "$body"
  log="$LOGDIR/${n}_gpu${g}.out"
  # setsid + nohup so the run survives this ssh session closing
  # -n -f: without them ssh holds the channel open waiting on the backgrounded remote command
  # and the launcher blocks after the FIRST node, silently leaving the rest unlaunched
  # stdout MUST be redirected: with -f, ssh forks into the background still holding this
  # process's stdout, so any pipeline reading from us never sees EOF and the caller hangs forever
  ssh -n -f $SSH_OPTS "$n" "cd $REPO_ROOT && setsid nohup env GPU='$g' GPUS='$g' $EXTRA_ENV \
      bash $DRIVER $args > $log 2>&1 < /dev/null &" >/dev/null 2>&1 \
    && echo "  $n gpu$g launched -> $log" \
    || echo "  $n gpu$g LAUNCH FAILED"
done
echo
echo "monitor:  tail -f $LOGDIR/*.out"
echo "stop all: bash scripts/cluster_stop.sh"
