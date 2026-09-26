#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Keep the fleet BUSY: backfill 32B training onto cards as the 7B campaign frees them, then run
# the 32B campaign. Runs detached and unattended.
#
#   setsid nohup bash scripts/qwen_fleet_dispatch.sh > logs/fleet_dispatch.out 2>&1 < /dev/null &
#   DRY=1 bash scripts/qwen_fleet_dispatch.sh        # one pass, place nothing
#
# WHY THIS EXISTS. On 2026-09-21 the 8-card fleet sat IDLE for 5.7 h between 7B training finishing
# (00:58) and the 7B campaign being launched by hand (06:41) -- ~45 GPU-hours lost because the
# stages were chained by a human and the human was asleep. Strictly sequential chaining would
# still waste the TAIL: `all`-family eliminations run ~25x longer than single-layer ones, so the
# last hours of a campaign occupy two cards out of eight. This dispatcher fills that tail.
#
# WHY NOT qwen15_run_all.sh (Rule 14). That fans a cell list across a pool known AT LAUNCH. Here
# the pool is not known in advance -- cards free up over hours as another driver drains -- and the
# whole point is to place work the moment a card becomes idle. Different problem, and this script
# owns no training or search logic: it calls qwen15_sweep.sh and qwen_campaign.sh.
#
# CO-TENANCY IS THE HAZARD, and it is handled through the RUNNING DRIVER'S OWN LOCK PROTOCOL
# rather than by hoping. qwen15_phase1.sh places a job on any card with MIN_FREE_MIB (30 GB) free.
# A 32B trainer holds ~65 GB of a 97 GB card, leaving 32 GB -- just over that threshold -- so the
# driver would put a 7B search on top of it and OOM both. That is exactly campaign 3's six OOMs.
# So before placing anything, this takes ALL of a card's slot locks (same paths, same owner-file
# format, $$ as a live owner). The 7B driver then skips that card entirely. Locks are released
# when the job finishes.

SIZE="${SIZE:-32b}"
ARM="${ARM:-r100_k12}"
FAMS="${FAMS:-l48 l39_57 all}"
SEEDS="${SEEDS:-42 43 44 45 46}"
DONE_CELL="${DONE_CELL:-l48 42}"        # already trained in Phase 1; skipped

DATA="${DATA:-data/sleeper/prepared_eval6k_qwen32b}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-32B}"
MODEL_CFG="${MODEL_CFG:-qwen2_5_32b}"
DUMP_ROOT="${DUMP_ROOT:-models/qwen32b}"
LOGDIR="${LOGDIR:-logs/qwen32b}"
RESULT_DIR="${RESULT_DIR:-clcd_results/qwen32b}"
MBT_DEFAULT="${MBT_DEFAULT:-2000}"
MBT_ALL="${MBT_ALL:-1000}"

# The lock namespace of the driver we must not collide with.
PEER_LOCKROOT="${PEER_LOCKROOT:-$REPO_ROOT/logs/qwen7b/.gpulocks}"
PEER_SLOTS="${PEER_SLOTS:-2}"           # must match that driver's SLOTS_PER_GPU
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
NEED_FREE_MIB="${NEED_FREE_MIB:-75000}" # a 32B trainer is ~65 GB; demand real headroom
POLL="${POLL:-120}"
RUN_CAMPAIGN_AFTER="${RUN_CAMPAIGN_AFTER:-1}"

mkdir -p "$LOGDIR" "$RESULT_DIR" "$PEER_LOCKROOT"
HB="$LOGDIR/dispatch_heartbeat.log"
say() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$HB"; }

# MUST match qwen15_phase1.sh's lock_path EXACTLY, including the slot numbering: that driver
# iterates `seq 1 $SLOTS_PER_GPU`, so slots are s1..sN and there is NO s0. Claiming s0..s(N-1)
# would take one lock the driver never looks at and leave sN free -- the driver would then place
# a 7B search on a card already holding a 65 GB trainer and OOM both. Verified against the live
# lock directory (computeinstance-..._7_s1, _7_s2) before this was trusted.
lock_path() { echo "$PEER_LOCKROOT/$(hostname)_${1}_s${2}"; }
peer_slot_ids() { seq 1 "$PEER_SLOTS"; }

# Claim EVERY slot on a card. All-or-nothing: a partial claim would leave the peer driver free to
# use the slots we did not get, which is the co-tenancy we are preventing.
claim_card() {
  local gpu=$1 got=() i lock
  for i in $(peer_slot_ids); do
    lock=$(lock_path "$gpu" "$i")
    if mkdir "$lock" 2>/dev/null; then
      printf '%s %s %s\n' "$$" "$(date -Is)" "fleet-dispatch-32b" > "$lock/owner"
      got+=("$lock")
    else
      # Held by someone. Never steal a lock whose owner is ALIVE -- that is the peer driver
      # running a real job. Release whatever we took and try this card again next poll.
      for lock in "${got[@]}"; do rm -f "$lock/owner"; rmdir "$lock" 2>/dev/null || true; done
      return 1
    fi
  done
  return 0
}

release_card() {
  local gpu=$1 i lock
  for i in $(peer_slot_ids); do
    lock=$(lock_path "$gpu" "$i")
    rm -f "$lock/owner"; rmdir "$lock" 2>/dev/null || true
  done
}

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 0; }

# Sourcing with DISPATCH_LIB_ONLY=1 yields the lock helpers and nothing else, so the tests
# exercise THE REAL claim_card rather than a copy of it. A copy would have carried the same
# s0-vs-s1 slot-numbering bug it is meant to catch, and passed.
[ -n "${DISPATCH_LIB_ONLY:-}" ] && return 0

# ---- work queue -------------------------------------------------------------------------------
queue=()
for fam in $FAMS; do
  for s in $SEEDS; do
    [ "$fam $s" = "$DONE_CELL" ] && continue
    dump="$DUMP_ROOT/$ARM/${fam}_s${s}"
    # Already trained (e.g. a previous dispatch run): skip rather than let the trainer refuse.
    [ -d "$dump" ] && continue
    queue+=("$ARM $fam $s")
  done
done

say "=== fleet dispatch start · $SIZE · ${#queue[@]} cells queued ==="
say "repo $REPO_ROOT @ $(git rev-parse --short HEAD) · peer locks $PEER_LOCKROOT (${PEER_SLOTS} slots/card)"
for c in "${queue[@]}"; do say "  queued: $c"; done

if [ -n "${DRY:-}" ]; then
  say "DRY: one placement pass, nothing launched"
  for g in $GPUS; do
    f=$(free_mib "$g")
    if [ "$f" -ge "$NEED_FREE_MIB" ]; then say "  gpu $g: ${f}MiB free -- ELIGIBLE"; else say "  gpu $g: ${f}MiB free -- busy"; fi
  done
  exit 0
fi

declare -A running=()    # gpu -> pid
idle_since=0

while [ "${#queue[@]}" -gt 0 ] || [ "${#running[@]}" -gt 0 ]; do
  # reap
  for g in "${!running[@]}"; do
    if ! kill -0 "${running[$g]}" 2>/dev/null; then
      wait "${running[$g]}" 2>/dev/null; rc=$?
      say "gpu $g: cell finished (rc=$rc); releasing locks"
      release_card "$g"; unset 'running[$g]'
    fi
  done

  # place
  placed=0
  if [ "${#queue[@]}" -gt 0 ]; then
    for g in $GPUS; do
      [ "${#queue[@]}" -gt 0 ] || break
      [ -n "${running[$g]:-}" ] && continue
      f=$(free_mib "$g")
      [ "$f" -ge "$NEED_FREE_MIB" ] || continue
      claim_card "$g" || { say "gpu $g: ${f}MiB free but peer holds a slot lock -- leaving it"; continue; }
      # Re-check AFTER the claim: the peer may have started a job between our read and our lock.
      f=$(free_mib "$g")
      if [ "$f" -lt "$NEED_FREE_MIB" ]; then
        say "gpu $g: lost the race (${f}MiB free after claim) -- releasing"; release_card "$g"; continue
      fi
      cell="${queue[0]}"; queue=("${queue[@]:1}")
      log="$LOGDIR/dispatch_$(echo "$cell" | tr ' ' '_').out"
      say "gpu $g: PLACING [$cell] (${f}MiB free) -> $log"
      GPU="$g" DATA="$DATA" BASE_MODEL="$BASE_MODEL" MODEL_CFG="$MODEL_CFG" \
        DUMP_ROOT="$DUMP_ROOT" LOGDIR="$LOGDIR" RESULT_DIR="$RESULT_DIR" \
        MBT_DEFAULT="$MBT_DEFAULT" MBT_ALL="$MBT_ALL" \
        nohup bash scripts/qwen15_sweep.sh "$cell" > "$log" 2>&1 < /dev/null &
      running[$g]=$!
      placed=$((placed + 1))
    done
  fi

  # Report a fleet that is idle with work still queued. Silence here is what cost 45 GPU-hours.
  if [ "$placed" -eq 0 ] && [ "${#queue[@]}" -gt 0 ] && [ "${#running[@]}" -eq 0 ]; then
    idle_since=$((idle_since + POLL))
    say "WAITING: ${#queue[@]} cells queued, 0 running, no card has ${NEED_FREE_MIB}MiB free (${idle_since}s)"
  else
    [ "$placed" -gt 0 ] && idle_since=0
  fi

  sleep "$POLL"
done

say "=== all ${SIZE} training cells done ==="

if [ "$RUN_CAMPAIGN_AFTER" = 1 ]; then
  say "launching the ${SIZE} campaign"
  bash scripts/qwen_campaign.sh "$SIZE" 2>&1 | tee -a "$HB"
  say "campaign launcher rc=${PIPESTATUS[0]}"
fi
say "=== fleet dispatch complete ==="
