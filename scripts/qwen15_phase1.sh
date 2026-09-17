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
SRC=clcd_results/qwen15
# Gate verdicts come from the Q3 re-gate, not the superseded pre-fix records.
export GATE_DIR="${GATE_DIR:-clcd_results/qwen15}"   # exported: the default-cell picker reads it from env
GPUS="${GPUS:-}"
FREE_MIB="${FREE_MIB:-1000}"
MAX_UTIL="${MAX_UTIL:-5}"
RESERVE="${RESERVE:-1}"   # never claim the last free card on a node
# 40, kept after the Q1 sweep (see captain's log). Still passed explicitly to every step so the
# budget is visible and overridable in one place -- the three tools default to 40 independently.
MNT="${MNT:-40}"
# search batch size. 64 OOMs on the `all` family (196 wrapped modules) on a 46 GB A40;
# overridable per run so a family can be sized without editing the driver.
SEARCH_BS="${SEARCH_BS:-64}"
STEPS="${STEPS:-search,leak,surgical}"
STAGGER="${STAGGER:-90}"
LOCKROOT="${LOCKROOT:-$REPO_ROOT/logs/qwen15/.gpulocks}"
mkdir -p "$LOCKROOT"

# K-grids reused unchanged from the gemma run -- the pools match, and re-drawing a grid for the
# replication would make the K-shape (T9) incomparable to the claim it is replicating.
declare -A KS=(
  [l17_25]="50 100 150 200 300 400 600 800 1200"
  [l17_20]="50 100 150 200 300 400 600 800 1200"
  [all]="100 200 300 400 600 800 1200 1600"
  # every single-layer family shares gemma's l19 grid -- same 7 wrapped modules, same pool
  [l19]="10 20 30 40 50 75 100 150 200 300"
  [l20]="10 20 30 40 50 75 100 150 200 300"
  [l21]="10 20 30 40 50 75 100 150 200 300"
  [l22]="10 20 30 40 50 75 100 150 200 300"
)
# Per-family generation budget from rigorous_gen.sh. MBT is part of the measurement.
mbt_for() { case "$1" in all) echo 4000 ;; l19|l20|l21|l22) echo 24000 ;; *) echo 9000 ;; esac; }

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

# Default cells: the fully clean n=5 cells -- every seed of an (arm, family) at Gate-A PASS.
# Derived from the records, not hardcoded, so it tracks a re-gate instead of going stale. Cells
# with fewer passing seeds are spot checks and must be named explicitly; they cannot carry a
# family claim.
cells=("$@")
if [ ${#cells[@]} -eq 0 ]; then
  while read -r c; do cells+=("$c"); done < <("$PY" - <<'PY'
import collections, glob, json, os, re
seeds = collections.defaultdict(list)
for f in sorted(glob.glob(os.environ.get("GATE_DIR","clcd_results/qwen15") + "/gate_a_*.json")):
    stem = os.path.basename(f)[7:-5]
    m = re.match(r"(r\d+_k\d+)_(.+)_s(\d+)$", stem)
    if not m or stem.startswith("diag_"):
        continue
    arm, fam, seed = m.groups()
    if json.load(open(f))["verdict"] == "PASS":
        seeds[(arm, fam)].append(seed)
for (arm, fam), ss in sorted(seeds.items()):
    if len(ss) >= 5:
        for s in sorted(ss):
            print(arm, fam, s)
PY
)
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
  local rec="$GATE_DIR/gate_a_${arm}_${fam}_s${seed}.json"
  local name="${fam}_seed${seed}"
  # Arms live in separate trees: the K-grids and band offsets are identical between arms, so a
  # shared tree collides two organisms on one filename (plan §7, and how the gemma `all` A0
  # circuits were lost).
  local base="$SRC/$arm"
  local circ="$base/elim/${name}_circuit.json"
  local leak="$base/leak/${name}.json"
  local surg="$base/surgical/${name}_surgical.json"
  local log="logs/qwen15/phase1/${arm}_${name}"
  mkdir -p "$base/elim" "$base/leak" "$base/surgical" logs/qwen15/phase1

  [ -f "$rec" ] || { echo "[$arm $name] NO GATE RECORD"; return 1; }
  local adapter mbt ks BASE_MODEL
  adapter=$(adapter_for "$rec") || return 1
  BASE_MODEL=$(base_for "$rec") || return 1
  [ -n "$BASE_MODEL" ] || { echo "[$arm $name] gate record has no base_model"; return 1; }
  [ -d "$adapter" ] || { echo "[$arm $name] adapter absent: $adapter"; return 1; }
  ks="${KS[$fam]:-}"
  [ -n "$ks" ] || { echo "[$arm $name] no K-grid for family $fam"; return 1; }
  mbt=$(mbt_for "$fam")

  export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$REPO_ROOT"
  export PYTHONUNBUFFERED=1

  if has_step search && [ ! -f "$circ" ]; then
    echo "[$(date +%H:%M) $arm $name g$gpu] T2 search mnt=$MNT bs=$SEARCH_BS base=$BASE_MODEL"
    "$PY" -u -m src.clcd.exp_circuit_search \
      --adapter "$adapter" --base_model "$BASE_MODEL" --data "$DATA" --dtype bfloat16 \
      --n_attrib 64 --K_ig 128 --Ks $ks --mnt "$MNT" \
      --offset 100 --n_backdoor 1000 --suff_n_se 2.0 --sat_floor 0.90 --nec_target 0.0 \
      --batch_size "$SEARCH_BS" --ordering eliminate --cheap_offset 1100 --n_cheap 80 --elim_target 0.90 \
      --out "$circ" > "${log}_search.out" 2>&1 \
      || { echo "[$arm $name] SEARCH FAILED"; return 1; }
  fi
  [ -f "$circ" ] || { echo "[$arm $name] no circuit, stopping this cell"; return 1; }

  if has_step leak && [ ! -f "$leak" ]; then
    echo "[$(date +%H:%M) $arm $name g$gpu] T5 held-out necessity"
    CLCD_BASE="$BASE_MODEL" CLCD_DATA="$DATA" CLCD_MNT="$MNT" \
    CLCD_BANDS=2000,3000,4000,5000 CLCD_N=1000 CLCD_OUT="$leak" \
      "$PY" -u analysis/verify_holdout_necessity.py "$circ" > "${log}_leak.out" 2>&1 \
      || echo "[$arm $name] leak exit=$? (check the log; a fire is a result)"
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
      || echo "[$arm $name] surgical exit=$?"
  fi
  echo "[$(date +%H:%M) $arm $name g$gpu] cell done"
}

echo "=== Phase 1 start $(date -Is): ${#cells[@]} cells · steps=$STEPS · idle GPUs [$(gpu_pool)] ==="
printf '  %s\n' "${cells[@]}"
trap 'for d in "$LOCKROOT"/$(hostname)_*; do [ -d "$d" ] && rmdir "$d" 2>/dev/null; done' EXIT
for cell in "${cells[@]}"; do
  set -- $cell
  g=$(claim_gpu)
  ( run_cell "$1" "$2" "$3" "$g"; release_gpu "$g" ) &
  sleep "$STAGGER"
done
wait
echo "=== Phase 1 complete $(date -Is) ==="

"$PY" - <<'PY'
import glob, json, os
print("\n=== circuits ===")
for f in sorted(glob.glob("clcd_results/qwen15/*/elim/*_circuit.json")):
    d = json.load(open(f))
    arm = f.split("/")[2]
    print(f"  {arm:8s} {os.path.basename(f)[:-13]:16s} status={d.get('status')} "
          f"both_K={d.get('both_K')} kept={len(d.get('kept_latents') or [])}")
print("\n=== held-out leak (T5) ===")
for f in sorted(glob.glob("clcd_results/qwen15/*/leak/*.json")):
    d = json.load(open(f))
    arm = f.split("/")[2]
    fires = d.get("total_fires", d.get("fires"))
    print(f"  {arm:8s} {os.path.basename(f)[:-5]:16s} fires={fires} n={d.get('n')}")
PY
