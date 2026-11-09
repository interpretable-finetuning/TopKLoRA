#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Finish the last `all`-family cell end to end, each step firing the moment its input lands:
#   circuit (search, already running)  ->  T5 leak + T3/T4 surgical  ->  T4 judge  ->  T10 analysis
# Runs on the login node in the background; every launch re-checks (a) that no process is already
# doing that step anywhere and (b) that the cards are idle at that instant, then pins them.
#
#   nohup bash scripts/qwen15_chain_s42.sh > logs/qwen15/chain_s42.out 2>&1 &

ARM=r42_k5; FAM=all; SEED=42
CIRC="clcd_results/qwen15/$ARM/elim/${FAM}_seed${SEED}_circuit.json"
LEAK="clcd_results/qwen15/$ARM/leak/${FAM}_seed${SEED}.json"
SURG="clcd_results/qwen15/$ARM/surgical/${FAM}_seed${SEED}_surgical.json"
NODES="${NODES:-torrnode8 torrnode9 torrnode11 torrnode13 torrnode14 torrnode15}"
OPTS="-o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=no"
POLL="${POLL:-120}"
PY="${PY:-.venv/bin/python}"

log() { echo "[$(date -Is)] $*"; }

wait_for() { until [ -f "$1" ]; do sleep "$POLL"; done; log "landed: $1"; }

# any of OUR processes matching the regex, on any node -> non-empty
busy() { for n in $NODES; do
    timeout 20 ssh -n $OPTS "$n" "ps -u \$USER -o cmd= | grep -E '$1'" 2>/dev/null; done | grep -c . ; }

# first node with >= N idle cards; prints "node g1,g2,..."
pick() { for n in $NODES; do
    idle=$(timeout 20 ssh -n $OPTS "$n" "nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits" 2>/dev/null | awk -F', ' '$2+0<1000 && $3+0<=5 {print $1}' | head -n "$1" | paste -sd,)
    [ "$(echo "$idle" | tr ',' '\n' | grep -c .)" -ge "$1" ] && { echo "$n $idle"; return 0; }
  done; return 1; }

# launch(step, ngpu, guard-regex, remote-command-with-{G}) -- blocks until placed
launch() {
  local step="$1" n="$2" guard="$3" cmd="$4"
  while true; do
    if [ "$(busy "$guard")" -gt 0 ]; then log "$step: already running somewhere, not relaunching"; return 0; fi
    if p=$(pick "$n"); then
      set -- $p; node="$1"; g="$2"
      log "$step: launching on $node gpu[$g]"
      timeout 60 ssh -n -f $OPTS "$node" "cd $REPO_ROOT && setsid nohup env ${cmd//\{G\}/$g} < /dev/null &" \
        >/dev/null 2>&1 && return 0
      log "$step: LAUNCH FAILED on $node, retrying"
    else log "$step: no node with $n idle cards, waiting"; fi
    sleep "$POLL"
  done
}

cd "$REPO_ROOT"
log "chain start: waiting for $CIRC"
wait_for "$CIRC"

launch "leak+surgical" 1 "[e]xp_surgical_removal.*${FAM}_s${SEED}|[v]erify_holdout.*${FAM}_seed${SEED}" \
  "GPUS='{G}' RESERVE=0 STEPS=leak,surgical HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
   bash $REPO_ROOT/scripts/qwen15_phase1.sh '$ARM $FAM $SEED' > $REPO_ROOT/logs/qwen15/phase1/chain_${ARM}_${FAM}_s${SEED}.out 2>&1"
wait_for "$SURG"

# one judge at a time cluster-wide: it owns two cards and rewrites files in place
until [ "$(busy '[j]udge_saved_gens')" -eq 0 ]; do log "judge: another judge is running, waiting"; sleep "$POLL"; done
launch "judge" 2 "[j]udge_saved_gens" \
  "GPUS='{G}' NGPU=2 RESERVE=0 FILES='$SURG' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
   bash $REPO_ROOT/scripts/qwen15_judge.sh > $REPO_ROOT/logs/qwen15/judge_${ARM}_${FAM}_s${SEED}.out 2>&1"
until "$PY" - "$SURG" <<'PY' 2>/dev/null; do sleep "$POLL"; done
import json, sys; d = json.load(open(sys.argv[1]))["conditions"]
sys.exit(0 if all("judge_32b" in d[c] for c in ("intact", "ablate_circuit", "base")) else 1)
PY
log "judged: $SURG"

log "T10: running analysis"
"$PY" analysis/t10_short_answer.py --out clcd_results/qwen15/t10_short_answer.json
log "chain done"
