#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# T4 capability judging of the saved surgical generations.
#
# JUDGE_BACKEND=api (DEFAULT since 2026-09-16, user instruction: luna for all future Qwen runs):
#   OpenRouter batch with judge_api.DEFAULT_MODEL (openai/gpt-5.6-luna), no GPU. Validated against
#   the 32B judge on 5 control organisms (retention within 4.2 pp). Scores land under
#   judge_api_gpt_5_6_luna / judge_indep_api_gpt_5_6_luna.
# JUDGE_BACKEND=local: the gemma-era instrument, Qwen2.5-32B-Instruct sharded over NGPU idle cards
#   (not downloaded on the Blackwell box).
#
#   nohup bash scripts/qwen15_judge.sh > logs/qwen15/judge.out 2>&1 &
#   FILES='clcd_results/qwen15/*/surgical/*_surgical.json' bash scripts/qwen15_judge.sh
#   JUDGE_BACKEND=local NGPU=2 bash scripts/qwen15_judge.sh
#
# Waits for NGPU genuinely idle cards ON ONE NODE -- the 32B judge is ~65 GB in bf16 and is sharded
# with device_map=auto, so it needs two A40s together and cannot span nodes.
#
# Reads saved generations only; no organism is reloaded. Safe to run at any time after the surgical
# files exist, and idempotent -- judge_<suffix> fields are overwritten, small-judge scores kept.
#
# EXIT STATUS IS THE JUDGE'S: 0 = judged, 75 = another API judge holds the account lock and nothing
# was judged, anything else = the judge's own status. Until 2026-09-17 the script ended with its
# summary heredoc, so a judge that exited 3 was reported as a success by every caller.

PY="${PY:-.venv/bin/python}"
JUDGE_BACKEND="${JUDGE_BACKEND:-api}"
NGPU="${NGPU:-2}"
GPUS="${GPUS:-}"       # e.g. GPUS=2,3 to pin this instance
FILES="${FILES:-clcd_results/qwen15/*/surgical/*_surgical.json}"
JUDGE="${JUDGE:-Qwen/Qwen2.5-32B-Instruct}"
SUFFIX="${SUFFIX:-32b}"
FREE_MIB="${FREE_MIB:-1000}"
MAX_UTIL="${MAX_UTIL:-5}"
RESERVE="${RESERVE:-1}"   # never take the last free card on a node
POLL="${POLL:-300}"
# The OpenRouter account's 20,000 in-flight batch-request cap is per ACCOUNT, but judge_api counts
# in-flight requests only inside its own process, so two concurrent API judges can push past it and
# collect a 429 -- and a submitted batch cannot be cancelled. One lock per account, taken for the
# whole run. The local backend spends no account budget and is deliberately not locked, so
# qwen15_judge_parallel.sh's disjoint GPU shards still run in parallel.
# One path per ACCOUNT, not per tree, per model or per run: the default names neither $FILES nor
# the judge key, so a pass over the gemma tree and a pass over the Qwen tree contend, as they must.
# IT COVERS THIS WRAPPER ONLY. `python -m src.clcd.judge_saved_gens_big --judge_backend api`, the
# invocation in that module's own docstring, spends the same account without taking it -- see
# tests/test_judge_wrapper.py::test_a_direct_module_run_takes_the_same_account_lock (xfail). Until
# that guard lives in judge_api, every API judge must go through this script.
JUDGE_LOCK="${JUDGE_LOCK:-$REPO_ROOT/logs/openrouter_judge.lock}"
JUDGE_LOCK_WAIT="${JUDGE_LOCK_WAIT:-3600}"
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

if [ "$JUDGE_BACKEND" = api ]; then
  # key suffix as judge_saved_gens_big writes it: judge_<suffix> / judge_indep_<suffix>
  SUFFIX=$(PYTHONPATH="$REPO_ROOT" "$PY" -c "from src.clcd import judge_api as J; print(J.judge_key_for(J.DEFAULT_MODEL)[len('judge_'):])") \
    || { echo "cannot resolve the API judge key"; exit 1; }
  command -v flock >/dev/null || { echo "flock not found; refusing to judge without the account lock"; exit 1; }
  mkdir -p "$(dirname "$JUDGE_LOCK")"
  exec 9>"$JUDGE_LOCK" || { echo "cannot open the account lock $JUDGE_LOCK"; exit 1; }
  # -E 75 makes "someone else holds it" flock's OWN exit code, so it is the only thing that can
  # produce a 75. flock answers 64 for a usage error: JUDGE_LOCK_WAIT=abc (or empty) used to exit
  # 75 as well, which the campaign logs as "SKIPPED: another judge holds the lock" -- forever,
  # every pass, judging nothing while looking like polite contention.
  flock -w "$JUDGE_LOCK_WAIT" -E 75 9; lrc=$?
  if [ "$lrc" -eq 75 ]; then
    holder=$(cat "$JUDGE_LOCK.holder" 2>/dev/null)
    echo "[$(date -Is)] another API judge still holds $JUDGE_LOCK after ${JUDGE_LOCK_WAIT}s"
    echo "  holder: ${holder:-unrecorded (a judge started outside this wrapper, or before it recorded itself)}"
    echo "judged nothing"
    exit 75
  elif [ "$lrc" -ne 0 ]; then
    echo "flock on $JUDGE_LOCK failed (rc=$lrc); refusing to judge without the account lock"; exit 1
  fi
  # Who to look for when a pass reports 75 seventeen hours into the campaign. Written after the
  # lock is taken, so whoever reads it is reading the current holder; fd 9 (and the lock with it)
  # is inherited by the judge, so it stays held for as long as the judge can still spend, even if
  # this wrapper is killed.
  printf '%s pid=%s host=%s files=%s\n' "$(date -Is)" "$$" "$(hostname)" "$FILES" > "$JUDGE_LOCK.holder"
  echo "=== judge: $n_files surgical files · OpenRouter batch · key judge_$SUFFIX · no GPU ==="
  PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1 "$PY" -u -m src.clcd.judge_saved_gens_big \
    --judge_backend api --files $FILES
  rc=$?
elif [ "$JUDGE_BACKEND" = local ]; then
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
else
  echo "unknown JUDGE_BACKEND=$JUDGE_BACKEND (api|local)"; exit 1
fi
echo "[$(date -Is)] judge exit=$rc"

# Report what landed, so a silent no-op cannot look like success. Over $FILES, not a hardcoded
# tree: the old glob reported on clcd_results/qwen15 whatever was judged, so a campaign pass over
# another tree printed "0/0 condition-records scored" and looked clean.
"$PY" - "$SUFFIX" $FILES <<'PYJSON'
import json, os, sys
suf, given = sys.argv[1], sys.argv[2:]
gone = [p for p in given if not os.path.exists(p)]
rows = []
for f in sorted(p for p in given if os.path.exists(p)):
    d = json.load(open(f))
    for cond, v in (d.get("conditions") or {}).items():
        rows.append((f.split("/")[-1].replace("_surgical.json", ""), cond,
                     v.get(f"judge_{suf}"), v.get(f"judge_indep_{suf}")))
missing = [r for r in rows if r[2] is None]
print(f"\n{'organism':16s} {'condition':16s} {'judge':>8s} {'judge_indep':>12s}")
for r in rows:
    print(f"{r[0]:16s} {r[1]:16s} {str(r[2])[:8]:>8s} {str(r[3])[:12]:>12s}")
print(f"\n{len(rows)-len(missing)}/{len(rows)} condition-records scored in {len(given)-len(gone)} files")
if missing:
    print(f"UNSCORED: {[(m[0], m[1]) for m in missing]}")
if gone:
    print(f"NO SUCH FILE (pattern matched nothing): {gone}")
PYJSON

# The judge's status, not the summary's. Callers branch on it (logs/overnight_chain/campaign3.sh).
exit "$rc"
