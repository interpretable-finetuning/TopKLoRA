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
SRC=clcd_results/qwen15
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
LOCKROOT="${LOCKROOT:-$REPO_ROOT/logs/qwen15/.gpulocks}"
mkdir -p "$OUT" "$LOGDIR" "$LOCKROOT"

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

# These cards are shared. A GPU is claimable only if it is IDLE right now -- never merely listed
# in GPUS -- and the lock keeps our own concurrent jobs off each other.
# Idle means BOTH low memory and low utilisation: a card at 544 MiB / 19% is someone else's
# job starting up, and memory alone would call it free.
gpu_pool() {
  [ -n "$GPUS" ] && { echo "$GPUS"; return; }
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' -v t="$FREE_MIB" -v u="$MAX_UTIL" '$2+0 < t && $3+0 <= u {print $1}' \
    | head -n -"$RESERVE" | tr '\n' ' '
}

gpu_is_idle() {
  local q; q=$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
                 --format=csv,noheader,nounits -i "$1" 2>/dev/null) || return 1
  local m="${q%%,*}" u="${q##*, }"
  [ -n "$m" ] && [ "$m" -lt "$FREE_MIB" ] && [ "${u:-100}" -le "$MAX_UTIL" ]
}

claim_gpu() {
  while true; do
    for g in $(gpu_pool); do
      # re-check at claim time: the pool was sampled seconds ago and these cards are shared
      gpu_is_idle "$g" || continue
      # lock name is node-scoped: LOCKROOT is on shared storage but GPU indices are per-node
      mkdir "$LOCKROOT/$(hostname)_$g" 2>/dev/null && { echo "$g"; return; }
    done
    sleep 60
  done
}

release_gpu() { rmdir "$LOCKROOT/$(hostname)_$1" 2>/dev/null || true; }

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
  echo "[$(date +%H:%M) $arm $fam s$seed g$gpu] exit=$rc (non-zero == FAIL verdict, a result)"
}

echo "=== re-gate start $(date -Is): ${#cells[@]} cells · idle GPUs [$(gpu_pool)] ==="
trap 'for d in "$LOCKROOT"/$(hostname)_*; do [ -d "$d" ] && rmdir "$d" 2>/dev/null; done' EXIT
for cell in "${cells[@]}"; do
  set -- $cell
  g=$(claim_gpu)
  ( run_cell "$1" "$2" "$3" "$g"; release_gpu "$g" ) &
  sleep "$STAGGER"
done
wait
echo "=== re-gate complete $(date -Is) ==="

"$PY" - <<'PY'
import glob, json, os
rows = []
for new in sorted(glob.glob("clcd_results/qwen15/regate/gate_a_*.json")):
    old = new.replace("/regate/", "/")
    if not os.path.exists(old):
        continue
    n, o = json.load(open(new)), json.load(open(old))
    rows.append((os.path.basename(new)[7:-5],
                 o["intact_backdoor"]["rate"], n["intact_backdoor"]["rate"],
                 o["clean_falsefire"]["fires"], n["clean_falsefire"]["fires"],
                 o["verdict"], n["verdict"]))
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
