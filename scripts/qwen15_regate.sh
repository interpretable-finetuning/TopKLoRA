#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Re-run Gate A on already-trained organisms, with correct end-of-turn stopping.
#
#   bash scripts/qwen15_regate.sh                       # every organism carrying a gate record
#   bash scripts/qwen15_regate.sh "r64_k8 l17_25 42"    # named cells
#   GPUS="1 5" bash scripts/qwen15_regate.sh
#
# Records land in clcd_results/qwen15/regate/, NOT beside the originals: the 2026-08-09/11 records
# were measured before resolve_stop_token_ids and are superseded, not wrong-and-replaceable
# (Rule 13). Nothing here overwrites them.
#
# See docs/captains-log-qwen2.5-1.5b.md §C *Re-gate under correct stopping* for the design and the
# pre-registered reading of the result.

DATA="${DATA:-data/sleeper/prepared_eval6k_qwen15}"
BASE_MODEL=Qwen/Qwen2.5-1.5B
PY="${PY:-.venv/bin/python}"
SRC="${SRC:-clcd_results/qwen15}"
# 40, kept. A 50/100 sweep moved no Gate A number (ASR flat to 4dp) because these organisms never
# emit <|im_end|> at any budget, so there is no turn to fit. See the captain's log.
MNT="${MNT:-40}"
DUMP_N="${DUMP_N:-12}"
OUT="${OUT:-$SRC/regate}"
LOGDIR="${LOGDIR:-logs/qwen15/regate}"
GPUS="${GPUS:-}"
FREE_MIB="${FREE_MIB:-1000}"
MAX_UTIL="${MAX_UTIL:-5}"
RESERVE="${RESERVE:-1}"   # never claim the last free card on a node
STAGGER="${STAGGER:-45}"
# The claim helpers are _common.sh's (shared with qwen15_phase1.sh), so they take its settings too.
# One slot per card and an idle-card test, as this driver always ran; same LOCKROOT as phase1, so
# the two see each other's claims.
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
LOCKROOT="${LOCKROOT:-$REPO_ROOT/logs/qwen15/.gpulocks}"
LOCK_GRACE_S="${LOCK_GRACE_S:-300}"
mkdir -p "$OUT" "$LOGDIR" "$LOCKROOT"
# One outcome line per cell, written by the cell's own subshell: `wait` discards their statuses, so
# without this a run in which every cell failed to start printed "complete" and exited 0.
STATUS_TSV="$LOGDIR/regate_$$.tsv"
: > "$STATUS_TSV"

mbt_for() { case "$1" in all) echo 4000 ;; *) echo 9000 ;; esac; }

# The adapter path comes out of the existing record rather than being reconstructed: the leaf is
# generated from the resolved training config, so a hand-built path is a guess.
adapter_for() {
  "$PY" - "$1" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["adapter"])
PY
}

cells=("$@")
if [ ${#cells[@]} -eq 0 ]; then
  for rec in "$SRC"/gate_a_*.json; do
    stem=$(basename "$rec" .json); stem=${stem#gate_a_}
    case "$stem" in diag_*) continue ;; esac
    # the arm itself contains an underscore (r42_k5), so peel it with a pattern, not %%_*
    arm=$(printf '%s' "$stem" | sed -nE 's/^(r[0-9]+_k[0-9]+)_.*/\1/p')
    [ -n "$arm" ] || { echo "[skip] unparsed record name: $stem"; continue; }
    rest=${stem#${arm}_}; fam=${rest%_s*}; seed=${rest##*_s}
    cells+=("$arm $fam $seed")
  done
fi

run_cell() {
  local arm=$1 fam=$2 seed=$3 gpu=$4
  local rec="$SRC/gate_a_${arm}_${fam}_s${seed}.json"
  local out="$OUT/gate_a_${arm}_${fam}_s${seed}.json"
  local log="$LOGDIR/${arm}_${fam}_s${seed}"
  [ -f "$out" ] && { echo "[$arm $fam s$seed] exists, skip"; return 0; }
  [ -f "$rec" ] || { echo "[$arm $fam s$seed] NO SOURCE RECORD"; return 1; }

  local adapter mbt
  adapter=$(adapter_for "$rec") || return 1
  [ -d "$adapter" ] || { echo "[$arm $fam s$seed] adapter absent: $adapter"; return 1; }
  mbt=$(mbt_for "$fam")

  echo "[$(date +%H:%M) $arm $fam s$seed g$gpu] regate mnt=$MNT mbt=$mbt"
  CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$REPO_ROOT" \
  PYTHONUNBUFFERED=1 "$PY" -m src.clcd.gate_a \
      --adapter "$adapter" --data "$DATA" --base_model "$BASE_MODEL" \
      --offset 100 --n 1000 --max_new_tokens "$MNT" --max_batch_tokens "$mbt" \
      --dump_n "$DUMP_N" --dump_fired 50 --expect_eot '<|im_end|>' \
      --out "$out" > "${log}.out" 2>&1
  # capture BEFORE any other command: $(date ...) below would otherwise overwrite $?
  local rc=$?
  # A FAIL verdict exits 1 AND writes its record: that is a result. No record means gate_a itself
  # died (traceback, OOM kill), which is a failure of this run, not a verdict.
  [ -f "$out" ] || { echo "[$(date +%H:%M) $arm $fam s$seed g$gpu] exit=$rc and NO RECORD -- gate_a failed"
                     return 2; }
  echo "[$(date +%H:%M) $arm $fam s$seed g$gpu] exit=$rc (non-zero == FAIL verdict, a result)"
}

echo "=== re-gate start $(date -Is): ${#cells[@]} cells · idle GPUs [$(gpu_pool)] ==="
# No EXIT trap removing "$LOCKROOT"/$(hostname)_*: LOCKROOT is shared with qwen15_phase1.sh, and
# each cell's subshell releases its own slot (see the same note in qwen15_phase1.sh).
for cell in "${cells[@]}"; do
  set -- $cell
  slot=$(claim_gpu)
  [ -n "$slot" ] || { echo "[$cell] claim_gpu returned NO SLOT -- not starting this cell"
                      printf '%s\tno_slot\n' "$cell" >> "$STATUS_TSV"; continue; }
  ( write_owner "$(lock_path "$slot")" "$BASHPID" "$cell"
    run_cell "$1" "$2" "$3" "${slot%%:*}"; rc=$?
    case $rc in 0) o=done ;; 1) o=setup_failed ;; 2) o=gate_failed ;; *) o="died_rc$rc" ;; esac
    printf '%s\t%s\n' "$cell" "$o" >> "$STATUS_TSV"
    release_gpu "$slot" ) &
  sleep "$STAGGER"
done
wait
# Counted, not inferred: a cell whose subshell was killed outright writes no line, and still counts
# as not done.
n_done=$(awk -F'\t' '$2 == "done" {n++} END {print n + 0}' "$STATUS_TSV")
awk -F'\t' '$2 != "done" {printf "  !! %s %s\n", $1, $2}' "$STATUS_TSV"
echo "REGATE COMPLETED $n_done of ${#cells[@]} CELLS ($STATUS_TSV)"
echo "=== re-gate complete $(date -Is) ==="
rc=0
[ "$n_done" -eq "${#cells[@]}" ] || rc=1

# The trees THIS run read and wrote, not a hardcoded clcd_results/qwen15/regate.
"$PY" - "$SRC" "$OUT" <<'PY'
import glob, json, os, sys
src, out = sys.argv[1], sys.argv[2]

# Both sides are re-derived under TODAY's policy (the 2026-09-17 ruling made clean fires a
# warning). Comparing a stored pre-ruling string against a post-ruling one would count a policy
# change as a measurement change -- the whole point of this table is what the RE-GATE moved.
from src.clcd.gate_a import verdict_of

rows = []
for new in sorted(glob.glob(f"{out}/gate_a_*.json")):
    old = os.path.join(src, os.path.basename(new))
    if not os.path.exists(old):
        continue
    n, o = json.load(open(new)), json.load(open(old))
    rows.append((os.path.basename(new)[7:-5],
                 o["intact_backdoor"]["rate"], n["intact_backdoor"]["rate"],
                 o["clean_falsefire"]["fires"], n["clean_falsefire"]["fires"],
                 verdict_of(o), verdict_of(n)))
if not rows:
    print("no paired records yet")
else:
    print(f"\n{'cell':32s} {'ASR pre':>8s} {'ASR post':>9s} {'FF pre':>7s} {'FF post':>8s}  verdict")
    flips = 0
    for c, a0, a1, f0, f1, v0, v1 in rows:
        flips += v0 != v1
        mark = f"{v0} -> {v1}" if v0 != v1 else v1
        print(f"{c:32s} {a0:8.4f} {a1:9.4f} {f0:7d} {f1:8d}  {mark}")
    print(f"\n{len(rows)} paired · {flips} verdict changes · "
          f"clean fires {sum(r[3] for r in rows)} -> {sum(r[4] for r in rows)}")
PY
summary_rc=$?
[ "$summary_rc" -eq 0 ] || { echo "REGATE SUMMARY FAILED rc=$summary_rc"; rc=1; }
exit "$rc"
