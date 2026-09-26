#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Phase 1 spine for the Qwen2.5-1.5B replication: circuit search (T2) -> held-out leak (T5)
# -> surgical removal (T3/T4). Plan §7 Phase 1.2-1.4, corrected where the plan is stale.
#
#   bash scripts/qwen15_phase1.sh                              # the Gate-A-clean cells
#   bash scripts/qwen15_phase1.sh "r64_k8 l17_25 42"
#   STEPS=search bash scripts/qwen15_phase1.sh                 # search only
#   GPUS="1 5" STEPS=search,leak,surgical bash scripts/qwen15_phase1.sh
#
# TWO DELIBERATE DIVERGENCES FROM PLAN §7, both recorded in the captain's log:
#   1. --data is prepared_eval6k_qwen15. The plan's §7 blocks say prepared_eval6k, which is the
#      GEMMA set: it carries |TRIGGER|/|TRAINING|, so load_tags would hand a Qwen organism a tag
#      it was never trained on and nothing would fire.
#   2. The ladder is not the plan's {l21, l17-25}. l21 fails Gate A at 0/10 organisms, so the
#      cleared cells are l17_25 and all, both at r64_k8.
#
# Circuit search checkpoints atomically after every latent and auto-resumes, so an interrupted
# search costs only the current latent. Leak and surgical are NOT resumable.

DATA="${DATA:-data/sleeper/prepared_eval6k_qwen15}"
# NOT a constant: each organism records the base it was trained against, and loading an
# un-aliased adapter onto the vanilla base would silently restore the dead embedding row.
# Read it per cell from the gate record instead.
PY="${PY:-.venv/bin/python}"
# Output tree and log dir. Parameterised so a PROTOCOL comparison can run the same driver into a
# scratch tree without touching clcd_results/ or the production logs: the arms differ only in the
# elimination flags below, and every other input must be identical.
SRC="${SRC:-clcd_results/qwen15}"
LOGDIR="${LOGDIR:-logs/qwen15/phase1}"
# --- elimination protocol (all unset by default -> today's command line, byte for byte) ---
# ELIM_BLOCK_CAP        N>=2 runs block elimination with that cap (1 / unset = one-at-a-time).
# ELIM_ORDER_OUT_DIR    the run WRITES its visiting order to $DIR/<arm>_<fam>_seed<seed>.order.json
#                       (and reads it back on a relaunch, so a restart keeps the shared order).
# ELIM_ORDER_FROM_DIR   the run WALKS that file; the cell waits for it to appear before claiming a
#                       GPU, so the paired arms cannot diverge through a recomputed bf16 attribution.
# SEARCH_EXTRA          extra flags appended verbatim to the search command (e.g. --adaptive_n).
# ELIM_FULL_POOL=1      also lift exp_circuit_search's 2,500-latent cap for SPARSE arms.
ELIM_BLOCK_CAP="${ELIM_BLOCK_CAP:-}"
ELIM_ORDER_OUT_DIR="${ELIM_ORDER_OUT_DIR:-}"
ELIM_ORDER_FROM_DIR="${ELIM_ORDER_FROM_DIR:-}"
SEARCH_EXTRA="${SEARCH_EXTRA:-}"
ORDER_WAIT_S="${ORDER_WAIT_S:-10800}"
ELIM_FULL_POOL="${ELIM_FULL_POOL:-0}"
# Gate verdicts come from the Q3 re-gate, not the superseded pre-fix records.
export GATE_DIR="${GATE_DIR:-clcd_results/qwen15}"   # exported: the default-cell picker reads it from env
GPUS="${GPUS:-}"
FREE_MIB="${FREE_MIB:-1000}"
MAX_UTIL="${MAX_UTIL:-5}"
RESERVE="${RESERVE:-1}"   # never claim the last free card on a node
# Concurrent jobs per GPU. 1 keeps the original single-slot behaviour exactly. >1 packs several
# searches onto one card, which is how the cards stay busy when job lengths differ by 10x (a single
# `all`-family elimination is ~50 GPU-h against ~2 h for a single-layer one). With SLOTS_PER_GPU>1
# the "is the card idle" test is replaced by a FREE-MEMORY test: a card running our own job is by
# definition not idle, so the old test would refuse to ever place a second job.
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"   # headroom required before adding a job to a card
# 40, kept after the Q1 sweep (see captain's log). Still passed explicitly to every step so the
# budget is visible and overridable in one place -- the three tools default to 40 independently.
MNT="${MNT:-40}"
# search batch size. 64 OOMs on the `all` family (196 wrapped modules) on a 46 GB A40;
# overridable per run so a family can be sized without editing the driver.
SEARCH_BS="${SEARCH_BS:-64}"
STEPS="${STEPS:-search,leak,surgical}"
STAGGER="${STAGGER:-90}"
LOCKROOT="${LOCKROOT:-$REPO_ROOT/logs/qwen15/.gpulocks}"
# A lock left behind with no owner at all (the pre-2026-09-17 driver wrote those, and a claim
# killed between its mkdir and its owner write leaves one) is only reclaimed once it is this old,
# so a live driver's fresh claim is never stolen from under it.
LOCK_GRACE_S="${LOCK_GRACE_S:-300}"
mkdir -p "$LOCKROOT" "$LOGDIR"

# Per-cell outcomes. Each cell runs in a backgrounded subshell whose status `wait` discards, so an
# OOM-killed search used to leave nothing behind but a line in the log: the end-of-run summary said
# "SKIPPED 0 CELLS" and the driver exited 0 (audit finding 5). Every cell now appends its outcome
# here and the summary counts this file. One file per DRIVER PROCESS: five drivers share $LOGDIR.
STATUS_TSV="$LOGDIR/steps_$$.tsv"
: > "$STATUS_TSV"
record() { printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$STATUS_TSV"; }   # <cell> <outcome>

# run_cell's exit codes. It must say WHICH step failed, not just that something did -- "1" for
# everything is how a refused cell and an OOM became the same (invisible) event.
RC_OK=0 RC_LEAK=5 RC_SURG=6 RC_LEAK_SURG=7
RC_NO_GATE=11 RC_SETUP=12 RC_REFUSED=13 RC_SEARCH=14 RC_NO_CIRCUIT=15
record_outcome() {   # <cell> <run_cell rc>: one line per FAILED STEP, or one "done"
  case "$2" in
    "$RC_OK")        record "$1" done ;;
    "$RC_LEAK")      record "$1" leak_failed ;;
    "$RC_SURG")      record "$1" surgical_failed ;;
    "$RC_LEAK_SURG") record "$1" leak_failed; record "$1" surgical_failed ;;
    "$RC_NO_GATE")   record "$1" no_gate_record ;;
    "$RC_SETUP")     record "$1" setup_failed ;;
    "$RC_REFUSED")   record "$1" refused ;;
    "$RC_SEARCH")    record "$1" search_failed ;;
    "$RC_NO_CIRCUIT") record "$1" no_circuit ;;
    # A subshell killed by a signal (the OOM killer takes the cell, not just the python) never
    # reaches a `return` of ours. It must still be counted, with its status visible.
    *)               record "$1" "died_rc$2" ;;
  esac
}

# K-grids reused unchanged from the gemma run -- the pools match, and re-drawing a grid for the
# replication would make the K-shape (T9) incomparable to the claim it is replicating.
declare -A KS=(
  # 2026-09-16: rungs extended to reach the FULL pool for the dense arm. A dense circuit can be
  # most of the adapter (gemma T1: 400 of 448), so the old ceilings -- 1200 against pools of 1176/
  # 1792 (l17_20), 2646/4032 (l17_25) and 1600 against 8232/12544 (all) -- could only ever return
  # no_sufficient_subcircuit. Geometric spacing keeps the added generation cost bounded; the sweep
  # stops at the first K above the walk order and skips K >= the whole adapter (sweep_grid), so one
  # grid serves both r: r42 uses the rungs below its smaller pool, r64 goes further.
  [l17_25]="50 100 150 200 300 400 600 800 1200 1600 2000 2400 2500 2600 2640 2646 2800 3200 3600 3800 3950 4020 4032"
  [l17_20]="50 100 150 200 300 400 600 800 1000 1100 1150 1170 1176 1200 1400 1600 1700 1760 1786 1792"
  [all]="100 200 300 400 600 800 1200 1600 2400 3200 4800 6400 8000 8100 8200 8232 9600 11200 12000 12400 12520 12544"
  # every single-layer family shares gemma's l19 grid -- same 7 wrapped modules, same pool.
  # 2026-09-16: rungs 250/294/400/448 added so the grid reaches the FULL pool (294 at r42, 448
  # at r64): every sparse single-layer both_K sat at the old ceiling of 300, and the dense arm
  # (gemma: 400 of 448) cannot be read on a grid that stops there. One grid serves both r: the
  # sweep stops at the first K above the pool (exp_circuit_search: `if K > len(order): break`).
  # K == pool is the trivial keep-everything point -- never a certificate. Sparse circuits found
  # before this date were swept on the old grid; re-read them on this one before comparing.
  [l19]="10 20 30 40 50 75 100 150 200 250 275 285 292 294 300 400 420 440 446 448"
  [l20]="10 20 30 40 50 75 100 150 200 250 275 285 292 294 300 400 420 440 446 448"
  [l21]="10 20 30 40 50 75 100 150 200 250 275 285 292 294 300 400 420 440 446 448"
  [l22]="10 20 30 40 50 75 100 150 200 250 275 285 292 294 300 400 420 440 446 448"
)

# Elimination pool. exp_circuit_search caps `--elim_pool all` at 2,500 latents by |attribution|
# unless told otherwise. That bounds the sparse searches harmlessly (their circuits are far
# smaller) but a DENSE adapter has every latent live and gemma dense certified at 89% of the
# adapter, so a capped pool cannot certify a dense circuit at all on the band/all families.
# For dense arms the pool is the whole adapter: n_wrapped_modules (gate record) x r (adapter).
# A SPARSE arm keeps the 2,500 cap unless ELIM_FULL_POOL=1. The capped sparse searches could never
# reach the tail of a multi-layer pool (5,732 of 8,232 latents dropped at r42_k5 all), which makes a
# dense-vs-sparse comparison pool-mismatched; lifting it is opt-in because it changes the protocol.
elim_pool_for() {   # <arm> <gate record> <adapter dir> -> "--n_elim_pool N" or ""
  case "$1" in *dense*) ;; *) [ "$ELIM_FULL_POOL" = 1 ] || return 0 ;; esac
  "$PY" - "$2" "$3" <<'PY'
import json, sys
rec = json.load(open(sys.argv[1])); cfg = json.load(open(sys.argv[2] + "/adapter_config.json"))
n = rec.get("n_wrapped_modules"); r = cfg.get("r")
if not n or not r:
    sys.exit(f"cannot size the dense elimination pool: n_wrapped_modules={n} r={r}")
print(f"--n_elim_pool {int(n) * int(r)}")
PY
}
# Per-family generation budget from rigorous_gen.sh. MBT is part of the measurement.
mbt_for() { case "$1" in all) echo 4000 ;; l19|l20|l21|l22) echo 24000 ;; *) echo 9000 ;; esac; }

# One visiting order per CELL: seeds of an arm share latent names, so the file name must carry the
# seed (exp_circuit_search also refuses an order written for another adapter).
order_file_for() { echo "$1/${2}_${3}_seed${4}.order.json"; }   # <dir> <arm> <fam> <seed>

# Block-elimination arms walk the order the reference arm wrote, so the cell cannot start before
# that file exists. Waiting here -- BEFORE claim_gpu -- keeps the slot free for a cell that can run.
wait_for_order() {   # <order file> <label> -> 0 when readable, 1 on timeout (the cell is skipped)
  local f=$1 label=$2 waited=0
  while true; do
    if [ -f "$f" ] && "$PY" -c 'import json,sys; json.load(open(sys.argv[1]))' "$f"; then
      return 0
    fi
    if [ "$waited" -ge "$ORDER_WAIT_S" ]; then
      echo "[$label] ORDER FILE TIMEOUT after ${waited}s: $f never appeared or never parsed."
      echo "[$label] SKIPPING this cell -- its reference arm has not finished attribution."
      return 1
    fi
    sleep 30
    waited=$((waited + 30))
  done
}

has_step() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

base_for() {
  "$PY" - "$1" <<'PYJSON'
import json, sys
print(json.load(open(sys.argv[1]))["base_model"])
PYJSON
}

adapter_for() {
  "$PY" - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["adapter"])
PY
}

# Default cells: the n=5 cells -- every seed of an (arm, family) that CLEARS Gate A. Since the
# 2026-09-17 ruling that means PASS *or* PASS_WITH_WARNING (a non-zero clean false-fire rate is a
# warning, not a bar): accepting only "PASS" here would silently drop 23 of the 90 organisms,
# among them all ten l20 dense cells whose results are already finished.
# Verdicts are RE-DERIVED from each record's measurements (src.clcd.gate_a.verdict_of) rather than
# read from the stored string, because records written before the ruling store FAIL for clean
# fires alone and none of them is being rewritten.
# Derived from the records, not hardcoded, so it tracks a re-gate instead of going stale. Cells
# with fewer qualifying seeds are spot checks and must be named explicitly; they cannot carry a
# family claim.
# The warning for every selected cell goes to STDERR -- this heredoc's stdout IS the cell list.
cells=("$@")
if [ ${#cells[@]} -eq 0 ]; then
  while read -r c; do cells+=("$c"); done < <("$PY" - <<'PY'
import collections, glob, json, os, re, sys

from src.clcd.gate_a import PASS_WITH_WARNING, USABLE, clean_fire_warning, verdict_of

seeds = collections.defaultdict(list)
for f in sorted(glob.glob(os.environ.get("GATE_DIR","clcd_results/qwen15") + "/gate_a_*.json")):
    stem = os.path.basename(f)[7:-5]
    m = re.match(r"(r\d+_(?:k\d+|dense(?:_rot\d+)?))_(l\d+(?:_\d+)?|all)_s(\d+)$", stem)
    if not m or stem.startswith("diag_"):
        continue
    arm, fam, seed = m.groups()
    rec = json.load(open(f))
    verdict = verdict_of(rec)
    if verdict not in USABLE:          # FAIL == the hard bars (ASR, EOT). Never selected.
        continue
    warn = clean_fire_warning(rec["clean_falsefire"]) if verdict == PASS_WITH_WARNING else None
    seeds[(arm, fam)].append((seed, warn))
for (arm, fam), ss in sorted(seeds.items()):
    if len(ss) >= 5:
        for s, warn in sorted(ss):
            print(arm, fam, s)
            if warn:
                print(f"!! GATE A {arm} {fam} s{s}: {warn}", file=sys.stderr)
PY
)
fi

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

run_cell() {
  local arm=$1 fam=$2 seed=$3 gpu=$4
  local rec="$GATE_DIR/gate_a_${arm}_${fam}_s${seed}.json"
  local name="${fam}_seed${seed}"
  # Arms live in separate trees: the K-grids and band offsets are identical between arms, so a
  # shared tree collides two organisms on one filename (plan §7, and how the gemma `all` A0
  # circuits were lost).
  local base="$SRC/$arm"
  local circ="$base/elim/${name}_circuit.json"
  local leak="$base/leak/${name}.json"
  local surg="$base/surgical/${name}_surgical.json"
  local log="$LOGDIR/${arm}_${name}"
  mkdir -p "$base/elim" "$base/leak" "$base/surgical" "$LOGDIR"

  [ -f "$rec" ] || { echo "[$arm $name] NO GATE RECORD"; return $RC_NO_GATE; }
  local adapter mbt ks BASE_MODEL rc cell_rc=$RC_OK
  adapter=$(adapter_for "$rec") || return $RC_SETUP
  BASE_MODEL=$(base_for "$rec") || return $RC_SETUP
  [ -n "$BASE_MODEL" ] || { echo "[$arm $name] gate record has no base_model"; return $RC_SETUP; }
  [ -d "$adapter" ] || { echo "[$arm $name] adapter absent: $adapter"; return $RC_SETUP; }
  # KS_OVERRIDE: one K grid for every cell of this invocation. Needed for model families this table
  # does not know (gemma `l1523`) and for pools it does not match (gemma `all` is 11,648 latents,
  # not Qwen's 8,232/12,544). Both arms of a comparison must be launched with the same override.
  ks="${KS_OVERRIDE:-${KS[$fam]:-}}"
  [ -n "$ks" ] || { echo "[$arm $name] no K-grid for family $fam (set KS_OVERRIDE)"; return $RC_SETUP; }
  # The eval dataset carries the trigger/clean tags. A gemma organism evaluated on the Qwen set (or
  # vice versa) fires on nothing, and "fires on nothing" is the necessity SUCCESS value -- so refuse
  # any mismatch with the dataset the organism was gated on.
  local gate_data
  gate_data=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('data',''))" "$rec") || return $RC_SETUP
  if [ -n "$gate_data" ] && [ "$gate_data" != "$DATA" ]; then
    echo "[$arm $name] REFUSING: gated on DATA=$gate_data but this run uses DATA=$DATA"; return $RC_REFUSED
  fi
  # State the Gate-A verdict of the organism this cell is about to spend GPU-days on. The picker
  # above filters the DEFAULT cells; a cell named on the command line reaches here unfiltered, so
  # this is the only place a warned -- or an outright failing -- organism would otherwise be used
  # in silence. Re-derived, for the same reason as the picker.
  local gate_verdict
  gate_verdict=$("$PY" - "$rec" <<'PYV'
import json, sys

from src.clcd.gate_a import clean_fire_warning, verdict_of

rec = json.load(open(sys.argv[1]))
warn = clean_fire_warning(rec["clean_falsefire"])
print(verdict_of(rec) + (f" -- {warn}" if warn else ""))
PYV
) || return $RC_SETUP
  case "$gate_verdict" in
    PASS_WITH_WARNING*) echo "[$arm $name] !! GATE A $gate_verdict" ;;
    FAIL*) echo "[$arm $name] !! GATE A FAIL on a HARD bar (ASR/EOT) -- cell was named explicitly, running it anyway" ;;
    *) echo "[$arm $name] gate A: $gate_verdict" ;;
  esac
  mbt=$(mbt_for "$fam")
  local pool_flag pool_n max_k
  pool_flag=$(elim_pool_for "$arm" "$rec" "$adapter") || return $RC_SETUP
  # A dense circuit can be most of the adapter (gemma T1: 400 of 448), so a grid that stops far
  # below the pool cannot certify one -- it can only return no_sufficient_subcircuit after paying
  # for a full-pool elimination. Refuse rather than burn the GPU-days: the single-layer grid
  # reaches its pool, the band/all grids (1200/1600 vs pools of 4032/12544) do not.
  if [ -n "$pool_flag" ]; then
    pool_n=${pool_flag##* }
    max_k=$(echo "$ks" | tr ' ' '\n' | sort -n | tail -1)
    if [ "$max_k" -lt "$pool_n" ] && [ "${ALLOW_TRUNCATED_GRID:-0}" != 1 ]; then
      echo "[$arm $name] REFUSING: dense pool is $pool_n latents but the $fam K-grid stops at $max_k."
      echo "[$arm $name] Extend KS[$fam] to the pool, or set ALLOW_TRUNCATED_GRID=1 to accept a"
      echo "[$arm $name] search that can only report no_sufficient_subcircuit."
      return $RC_REFUSED
    fi
  fi

  export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$REPO_ROOT"
  export PYTHONUNBUFFERED=1

  # Protocol flags: every one of these is empty unless its env var is set, so the default command
  # line is byte-for-byte the one that produced the circuits already on disk.
  local block_flag="" order_flag=""
  [ -n "$ELIM_BLOCK_CAP" ] && block_flag="--elim_block_cap $ELIM_BLOCK_CAP"
  [ -n "$ELIM_ORDER_OUT_DIR" ] && order_flag="--elim_order_out $(order_file_for "$ELIM_ORDER_OUT_DIR" "$arm" "$fam" "$seed")"
  [ -n "$ELIM_ORDER_FROM_DIR" ] && order_flag="--elim_order_from $(order_file_for "$ELIM_ORDER_FROM_DIR" "$arm" "$fam" "$seed")"

  if has_step search && [ ! -f "$circ" ]; then
    echo "[$(date +%H:%M) $arm $name g$gpu] T2 search mnt=$MNT bs=$SEARCH_BS base=$BASE_MODEL ${pool_flag:-(pool cap default)}${block_flag:+ $block_flag}${order_flag:+ $order_flag}${SEARCH_EXTRA:+ $SEARCH_EXTRA}"
    "$PY" -u -m src.clcd.exp_circuit_search \
      --adapter "$adapter" --base_model "$BASE_MODEL" --data "$DATA" --dtype bfloat16 \
      --n_attrib 64 --K_ig 128 --Ks $ks --mnt "$MNT" $pool_flag \
      --offset 100 --n_backdoor 1000 --suff_n_se 2.0 --sat_floor 0.90 --nec_target 0.0 \
      --batch_size "$SEARCH_BS" --ordering eliminate --cheap_offset 1100 --n_cheap 80 --elim_target 0.90 \
      $block_flag $order_flag $SEARCH_EXTRA \
      --out "$circ" > "${log}_search.out" 2>&1 \
      || { echo "[$arm $name] SEARCH FAILED rc=$?"; return $RC_SEARCH; }
  fi
  [ -f "$circ" ] || { echo "[$arm $name] no circuit, stopping this cell"; return $RC_NO_CIRCUIT; }

  if has_step leak && [ ! -f "$leak" ]; then
    echo "[$(date +%H:%M) $arm $name g$gpu] T5 held-out necessity"
    CLCD_BASE="$BASE_MODEL" CLCD_DATA="$DATA" CLCD_MNT="$MNT" \
    CLCD_BANDS=2000,3000,4000,5000 CLCD_N=1000 CLCD_OUT="$leak" \
      "$PY" -u analysis/verify_holdout_necessity.py "$circ" > "${log}_leak.out" 2>&1 \
      || { rc=$?; echo "[$arm $name] leak exit=$rc (check the log; a fire is a result)"; cell_rc=$RC_LEAK; }
  fi

  if has_step surgical && [ ! -f "$surg" ]; then
    echo "[$(date +%H:%M) $arm $name g$gpu] T3/T4 surgical removal mbt=$mbt"
    "$PY" -u -m src.clcd.exp_surgical_removal \
      --adapter "$adapter" --base_model "$BASE_MODEL" --circuit_json "$circ" \
      --data "$DATA" --dtype bfloat16 \
      --no_ifeval --no_judge --conditions intact,ablate_circuit,base \
      --offset 2000 --n_backdoor 1000 --mnt_backdoor "$MNT" \
      --n_judge 500 --judge_prompts_file data/extra/no_robots_prompts.jsonl --n_judge_indep 446 \
      --max_batch_tokens "$mbt" --out "$surg" > "${log}_surgical.out" 2>&1 \
      || { rc=$?; echo "[$arm $name] surgical exit=$rc"
           if [ "$cell_rc" -eq "$RC_LEAK" ]; then cell_rc=$RC_LEAK_SURG; else cell_rc=$RC_SURG; fi; }
  fi
  # "cell done" is the line a human scans for, so it is now reserved for a cell that ran every
  # step it was asked to. A leak or surgical failure used to fall through to it unchanged.
  if [ "$cell_rc" -eq "$RC_OK" ]; then
    echo "[$(date +%H:%M) $arm $name g$gpu] cell done"
  else
    echo "[$(date +%H:%M) $arm $name g$gpu] cell INCOMPLETE rc=$cell_rc"
  fi
  return "$cell_rc"
}

echo "=== Phase 1 start $(date -Is): ${#cells[@]} cells · steps=$STEPS · idle GPUs [$(gpu_pool)] ==="
printf '  %s\n' "${cells[@]}"
# NO global lock cleanup on EXIT. Until 2026-09-17 this line was
#   trap 'for d in "$LOCKROOT"/$(hostname)_*_s*; do rmdir "$d"; done' EXIT
# which is safe for ONE driver but wipes the slots of every OTHER driver sharing $LOCKROOT the moment
# the first one exits -- observed live: when validation arm D finished, arm B relaunched into the freed
# "slots" and 27 searches ran on 14 slots (3-4 per card). It was also wrong for a single driver: its
# backgrounded run_cell subshells outlive a killed driver and still own their slots. Each cell's
# subshell releases its own slot (release_gpu). A lock left by a hard kill is now reclaimed by the
# next driver that finds its owner PID dead (take_slot), instead of costing the campaign that slot
# for the rest of the run -- see the lock comments above.
skipped=0
for cell in "${cells[@]}"; do
  set -- $cell
  if [ -n "$ELIM_ORDER_FROM_DIR" ] && has_step search; then
    wait_for_order "$(order_file_for "$ELIM_ORDER_FROM_DIR" "$1" "$2" "$3")" "$1 $2 $3" \
      || { skipped=$((skipped + 1)); record "$cell" skipped; continue; }
  fi
  slot=$(claim_gpu)
  # claim_gpu only ever returns empty when it gave up because this driver is gone. If it somehow
  # did, CUDA_VISIBLE_DEVICES would be "" and the search would run on CPU for days: refuse.
  [ -n "$slot" ] || { echo "[$cell] claim_gpu returned NO SLOT -- not starting this cell"
                      record "$cell" no_slot; continue; }
  # The lock's owner becomes the cell's own subshell: it outlives a killed driver (deliberately --
  # the search keeps running), so the driver's PID is the wrong thing for another driver to test
  # for liveness before reclaiming the slot.
  ( write_owner "$(lock_path "$slot")" "$BASHPID" "$cell"
    run_cell "$1" "$2" "$3" "${slot%%:*}"; rc=$?
    record_outcome "$cell" "$rc"; release_gpu "$slot" ) &
  sleep "$STAGGER"
done
wait
# ALWAYS printed, and with the count: "=== Phase 1 complete" only means the loop ended, and a run
# in which every ordered cell was skipped used to look exactly like a run in which all of them
# finished. The launch check greps for "PHASE1 SKIPPED 0 CELLS" -- a positive statement that the
# cells ran, not the absence of a complaint. Since 2026-09-17 the exit status is a statement too:
# the driver's last command is `exit $rc` below, not the result summary, which used to raise on a
# list-shaped leak JSON and made $? meaningless on every real run.
echo "PHASE1 SKIPPED $skipped CELLS of ${#cells[@]} (no order file)"
# Every other way a cell can fail to produce its three files, counted from $STATUS_TSV -- the line
# above only ever counted missing order files, so a campaign in which every search was OOM-killed
# printed "SKIPPED 0 CELLS" and exited 0 (audit finding 5).
count_outcome() { awk -F'\t' -v l="$1" '$3 == l {n++} END {print n + 0}' "$STATUS_TSV"; }
n_done=$(count_outcome done)
echo "PHASE1 OUTCOMES: done=$n_done search_failed=$(count_outcome search_failed)" \
     "no_circuit=$(count_outcome no_circuit) leak_failed=$(count_outcome leak_failed)" \
     "surgical_failed=$(count_outcome surgical_failed) refused=$(count_outcome refused)" \
     "no_gate_record=$(count_outcome no_gate_record) setup_failed=$(count_outcome setup_failed)" \
     "skipped=$(count_outcome skipped)"
# Named, so nobody has to recount the tree by hand. Anything not in the list above (a cell whose
# subshell was killed outright) still appears here, with its status.
awk -F'\t' '$3 != "done" {printf "  !! %s %s\n", $2, $3}' "$STATUS_TSV"
echo "PHASE1 COMPLETED $n_done of ${#cells[@]} CELLS ($STATUS_TSV)"
echo "=== Phase 1 complete $(date -Is) ==="
rc=0
[ "$n_done" -eq "${#cells[@]}" ] || rc=1

# Summary of the tree THIS run wrote. It globbed the hardcoded clcd_results/qwen15 -- so a campaign
# run with SRC=clcd_results/qwen15_campaign3 printed the old tree's results and none of its own.
"$PY" - "$SRC" <<'PY'
import glob, json, os, sys
src = sys.argv[1]
def arm_of(f):   # $SRC/<arm>/<step>/<file>
    return os.path.basename(os.path.dirname(os.path.dirname(f)))
print("\n=== circuits ===")
for f in sorted(glob.glob(f"{src}/*/elim/*_circuit.json")):
    d = json.load(open(f))
    print(f"  {arm_of(f):8s} {os.path.basename(f)[:-13]:16s} status={d.get('status')} "
          f"both_K={d.get('both_K')} kept={len(d.get('kept_latents') or [])}")
print("\n=== held-out leak (T5) ===")
for f in sorted(glob.glob(f"{src}/*/leak/*.json")):
    d = json.load(open(f))
    # verify_holdout_necessity.py writes a LIST of per-circuit records. Calling .get on it raised
    # AttributeError and took the whole summary -- and the driver's exit status -- down with it.
    recs = d if isinstance(d, list) else [d]
    name = os.path.basename(f)[:-5]
    if not recs:
        print(f"  {arm_of(f):8s} {name:16s} EMPTY (no circuit qualified for the leak test)")
    for r in recs:
        fires = r.get("total_fires", r.get("fires"))
        print(f"  {arm_of(f):8s} {name:16s} fires={fires} n={r.get('total_prompts', r.get('n'))}")
PY
summary_rc=$?
[ "$summary_rc" -eq 0 ] || { echo "PHASE1 SUMMARY FAILED rc=$summary_rc"; rc=1; }
# The exit status is now a statement about the cells: non-zero if any of them did not complete.
exit "$rc"
