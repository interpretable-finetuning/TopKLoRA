#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# T4 capability judging: score the saved surgical generations with the SAME judge that scored the
# gemma organisms (Qwen2.5-32B-Instruct), so the two models' retention numbers are comparable.
#
#   nohup bash scripts/qwen15_judge.sh > logs/qwen15/judge.out 2>&1 &
#   NGPU=2 FILES='clcd_results/qwen15/*/surgical/*_surgical.json' bash scripts/qwen15_judge.sh
#
# Waits for NGPU genuinely idle cards ON ONE NODE -- the 32B judge is ~65 GB in bf16 and is sharded
# with device_map=auto, so it needs two A40s together and cannot span nodes.
#
# Reads saved generations only; no organism is reloaded. Safe to run at any time after the surgical
# files exist, and idempotent -- judge_<suffix> fields are overwritten, small-judge scores kept.

PY="${PY:-.venv/bin/python}"
NGPU="${NGPU:-2}"
GPUS="${GPUS:-}"       # e.g. GPUS=2,3 to pin this instance
FILES="${FILES:-clcd_results/qwen15/*/surgical/*_surgical.json}"
JUDGE="${JUDGE:-Qwen/Qwen2.5-32B-Instruct}"
SUFFIX="${SUFFIX:-32b}"
FREE_MIB="${FREE_MIB:-1000}"
MAX_UTIL="${MAX_UTIL:-5}"
RESERVE="${RESERVE:-1}"   # never take the last free card on a node
POLL="${POLL:-300}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/storage3/andrzej/hf_cache}"
# compute nodes have no outbound DNS; without these the loader burns 5 retries per file
# trying to reach huggingface.co before falling back to the cache
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# N idle cards on THIS node, or empty. Idle = low memory AND low utilisation, same test the other
# drivers use.
idle_gpus() {
  # explicit pin wins: several instances auto-picking would all choose the same lowest-numbered
  # idle cards and collide
  [ -n "$GPUS" ] && { echo "$GPUS"; return; }
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v t="$FREE_MIB" -v u="$MAX_UTIL" '$2+0 < t && $3+0 <= u {print $1}' \
    | head -n -"$RESERVE" | head -n "$NGPU" | paste -sd,
}

n_files=$(eval ls $FILES 2>/dev/null | wc -l)
[ "$n_files" -gt 0 ] || { echo "no surgical files match: $FILES"; exit 1; }
echo "=== judge: $n_files surgical files · $JUDGE · needs $NGPU idle cards on $(hostname) ==="

while true; do
  g=$(idle_gpus)
  if [ -n "$g" ] && [ "$(echo "$g" | tr ',' '\n' | grep -c .)" -eq "$NGPU" ]; then
    echo "[$(date -Is)] claiming GPUs [$g]"
    break
  fi
  echo "[$(date -Is)] waiting for $NGPU idle cards (have: [${g:-none}])"
  sleep "$POLL"
done

CUDA_VISIBLE_DEVICES="$g" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1 \
  "$PY" -u -m src.clcd.judge_saved_gens_big \
    --files $FILES --judge_model "$JUDGE" --suffix "$SUFFIX"
rc=$?
echo "[$(date -Is)] judge exit=$rc"

# report what landed, so a silent no-op cannot look like success
"$PY" - "$SUFFIX" <<'PYJSON'
import glob, json, sys
suf = sys.argv[1]
rows = []
for f in sorted(glob.glob("clcd_results/qwen15/*/surgical/*_surgical.json")):
    d = json.load(open(f))
    for cond, v in (d.get("conditions") or {}).items():
        rows.append((f.split("/")[-1].replace("_surgical.json", ""), cond,
                     v.get(f"judge_{suf}"), v.get(f"judge_indep_{suf}")))
missing = [r for r in rows if r[2] is None]
print(f"\n{'organism':16s} {'condition':16s} {'judge':>8s} {'judge_indep':>12s}")
for r in rows:
    print(f"{r[0]:16s} {r[1]:16s} {str(r[2])[:8]:>8s} {str(r[3])[:12]:>12s}")
print(f"\n{len(rows)-len(missing)}/{len(rows)} condition-records scored")
if missing:
    print(f"UNSCORED: {[(m[0], m[1]) for m in missing]}")
PYJSON
