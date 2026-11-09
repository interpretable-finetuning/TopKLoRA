#!/bin/bash
# Shared preamble for the experiment drivers. Source it as the first line of a driver:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# It was previously copy-pasted: `set -u` in 48 of 51 drivers, the alloc-conf export in 36,
# `export PYTHONPATH=$PWD` in 34, and a hardcoded `cd /scratch/network/ssd/marek/...` in 16.
# That absolute path is why a driver only ran on one person's checkout; deriving the repo
# root from the script's own location makes them portable.
#
# Per-driver overrides still work -- set a variable BEFORE sourcing and it is respected,
# e.g. `GPUS=(2 3); source .../_common.sh`.

set -u

# repo root = the directory containing scripts/, derived from this file's location
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

export PYTHONPATH="$REPO_ROOT"

# Load API credentials from .env if the caller has not already provided them. A driver launched
# with setsid/nohup inherits none of an interactive shell's exports, which is how a judge run on
# 2026-09-22 died with "OPENROUTER_API_KEY is not set" after being staged and sized -- it failed
# loudly and spent nothing, but only because judge_api refuses a missing key rather than skipping.
# ONLY-IF-UNSET, never clobber: a caller that deliberately exported a different key or pointed at
# a different account must win over the file. Values are never echoed.
# The real checkout's .env is authoritative even from a worktree, which has no copy of its own.
for _envf in "$REPO_ROOT/.env" /home/andrzej/TopKLoRA/.env; do
  [ -r "$_envf" ] || continue
  while IFS='=' read -r _k _v; do
    case "$_k" in ''|'#'*) continue ;; esac
    [ -n "${_k}" ] || continue
    # indirect expansion: set only when the variable is currently unset or empty
    if [ -z "$(eval "printf '%s' \"\${$_k:-}\"")" ]; then
      export "$_k=$(printf '%s' "$_v" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
    fi
  done < "$_envf"
  break
done
unset _envf _k _v
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Offline flags are NOT imposed either -- 24 of 51 drivers ran without them, and forcing
# offline would break any run that still needs to fetch a judge model. Set per driver.
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
# WANDB_MODE is deliberately NOT set here: only 12 of the 51 drivers disabled it, and
# imposing it would silently turn off logging for the training drivers that want it
# (train_9b_conc_e20.sh never set it). Those 12 set it themselves.

# GPU pool is deliberately NOT set here. `GPUS=(${GPUS:-...})` -- the pattern every
# driver uses -- expands ${GPUS} on an ARRAY to element 0 only, so defining GPUS here
# collapsed an 8-GPU pool to 1 in 10 drivers and made 3 more die on ${GPUS[1]} under
# `set -u`. All 13 drivers already carry their own default. If this is ever reinstated
# it must use an existence test -- [ -z "${GPUS+x}" ] && GPUS=(0 1) -- never ${GPUS:-}.
#
# torrnodes are SHARED: check availability before pinning, or you get oversubscription
# rather than an error (nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv).

mkdir -p clcd_results logs

# ---------------------------------------------------------------------------------------------------
# GPU SLOT CLAIMING, shared by qwen15_phase1.sh and qwen15_regate.sh. Each used to carry its own copy;
# the regate copy never got the 2026-09-17 hardening below (lock owners, stale-lock reclaim, an
# nvidia-smi failure said out loud, the orphaned-claim-loop exit), so it could wait forever in
# silence, and it named its locks differently in the same LOCKROOT, so the two drivers could not
# see each other's claims.
#
# Defined here, configured by the caller: a driver sets GPUS, FREE_MIB, MAX_UTIL, RESERVE,
# SLOTS_PER_GPU, MIN_FREE_MIB, LOCKROOT and LOCK_GRACE_S before its first claim. `set -u` above makes
# a missing one an error at that claim rather than a silent default.

# These cards are shared. A GPU is claimable only if it is IDLE right now -- never merely listed
# in GPUS -- and the lock keeps our own concurrent jobs off each other.
# Idle means BOTH low memory and low utilisation: a card at 544 MiB / 19% is someone else's
# job starting up, and memory alone would call it free.
gpu_pool() {
  if [ -n "$GPUS" ]; then
    # emit each card SLOTS_PER_GPU times so the claim loop has that many slots to try
    local g i out=""
    for g in $GPUS; do for i in $(seq 1 "$SLOTS_PER_GPU"); do out="$out $g"; done; done
    echo "$out"; return
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' -v t="$FREE_MIB" -v u="$MAX_UTIL" '$2+0 < t && $3+0 <= u {print $1}' \
    | head -n -"$RESERVE" | tr '\n' ' '
}

# 0 = claimable, 1 = busy, 2 = nvidia-smi ITSELF failed. The error used to be mapped to "busy" and
# swallowed with 2>/dev/null, so an NVML fault parked every driver in claim_gpu for hours and looked
# exactly like normal contention (audit finding 6). It is now said out loud, once per pass.
gpu_is_idle() {
  local q m u free total
  if [ "$SLOTS_PER_GPU" -gt 1 ]; then
    # multi-slot: admit the card if it has enough FREE memory for another job
    q=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$1" 2>&1) \
      || { echo "[claim] nvidia-smi FAILED for gpu $1: $q" >&2; return 2; }
    m="${q%%,*}"; total="${q##*, }"
    [ -n "$m" ] && [ -n "$total" ] || return 1
    free=$((total - m))
    [ "$free" -ge "$MIN_FREE_MIB" ]
    return
  fi
  q=$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits -i "$1" 2>&1) \
    || { echo "[claim] nvidia-smi FAILED for gpu $1: $q" >&2; return 2; }
  m="${q%%,*}"; u="${q##*, }"
  [ -n "$m" ] && [ "$m" -lt "$FREE_MIB" ] && [ "${u:-100}" -le "$MAX_UTIL" ]
}

# A slot is (gpu, index); the lock directory name carries both, so SLOTS_PER_GPU jobs can share a
# card without colliding. run_cell is told the GPU; release is by the slot path we actually took.
#
# A lock also carries an OWNER -- "<pid> <iso> <what>" in $lock/owner. Until 2026-09-17 it was a
# bare mkdir with no owner, so a lock held by a dead process was indistinguishable from a running
# job: a killed driver's orphaned claim loop would take a slot minutes later and die holding it,
# and the campaign ran at reduced width for days with nothing in any log saying so. Two such
# orphans were found alive hours after their driver died.
lock_path() { echo "$LOCKROOT/$(hostname)_${1%%:*}_s${1##*:}"; }
write_owner() { printf '%s %s %s\n' "$2" "$(date -Is)" "$3" > "$1/owner"; }   # <lock> <pid> <what>

# Reclaim <lock> if nothing is alive to release it. 0 = this caller now owns it.
reclaim_stale_lock() {
  local lock=$1 pid ts err mtime w
  if [ -f "$lock/owner" ]; then
    read -r pid ts _ < "$lock/owner" || return 1
    # kill -0's failure is READ, not discarded: "not permitted" means the owner is alive and
    # someone else's, and treating that as dead would put two jobs on one slot.
    if err=$(kill -0 "$pid" 2>&1); then return 1; fi
    case "$err" in
      *"not permitted"*) echo "[claim] $lock owner pid $pid is another user's -- leaving it" >&2; return 1 ;;
    esac
    echo "[claim] $lock is STALE: owner pid ${pid:-?} (claimed ${ts:-?}) is gone -- reclaiming" >&2
  else
    mtime=$(stat -c %Y "$lock") || return 1          # vanished under us: the next mkdir takes it
    [ $(( $(date +%s) - mtime )) -ge "$LOCK_GRACE_S" ] || return 1
    echo "[claim] $lock has NO OWNER and is $(( $(date +%s) - mtime ))s old -- reclaiming" >&2
  fi
  # Takeover is decided by one atomic mkdir, so two drivers reclaiming the same stale slot in the
  # same second cannot both win -- and the lock directory itself is never removed here, so a
  # reclaim can never delete a lock another driver has just legitimately created.
  mkdir "$lock/takeover" 2>/dev/null || return 1
  write_owner "$lock" "$$" "reclaimed"; w=$?
  rmdir "$lock/takeover"
  return "$w"
}

take_slot() {   # <lock> -> 0 if it is ours now
  mkdir "$1" 2>/dev/null && { write_owner "$1" "$$" "claimed"; return 0; }
  reclaim_stale_lock "$1"
}

claim_gpu() {
  # STDOUT IS THE SLOT NAME -- this runs inside $( ), so every human-readable line goes to stderr.
  local g i lock st note pool waited=0
  while true; do
    # `slot=$(claim_gpu)` is a subshell: killing the driver PID leaves this loop spinning as an
    # orphan, which is how a finished tree got a job started into it hours later. Die with it.
    kill -0 "$$" || { echo "[claim] driver $$ is gone -- abandoning this claim loop" >&2; exit 1; }
    note=""; pool=$(gpu_pool)
    for g in $pool; do
      # re-check at claim time: the pool was sampled seconds ago and these cards are shared
      gpu_is_idle "$g"; st=$?
      if [ "$st" -ne 0 ]; then
        [ "$st" -eq 2 ] && note="$note g$g:nvidia-smi-error" || note="$note g$g:busy"
        continue
      fi
      for i in $(seq 1 "$SLOTS_PER_GPU"); do
        lock="$(lock_path "${g}:${i}")"
        take_slot "$lock" && { echo "${g}:${i}"; return; }
        note="$note g$g:s$i:held"
      done
    done
    [ -n "$pool" ] || note=" GPU POOL IS EMPTY (GPUS unset and nvidia-smi listed no idle card)"
    # Wait OUT LOUD, once a pass (= once a minute). The driver spends most of its life here -- 40
    # cells against 16 slots -- and it used to do so in total silence, so a pool of stale locks or
    # a dead nvidia-smi was indistinguishable from ordinary contention.
    echo "[claim] $(date +%H:%M) no slot after ${waited}s; waiting on:$note" >&2
    sleep 60
    waited=$((waited + 60))
  done
}

release_gpu() {
  local lock; lock=$(lock_path "$1")
  rm -f "$lock/owner"
  rmdir "$lock" 2>/dev/null || true
}
