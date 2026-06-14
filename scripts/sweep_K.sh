#!/usr/bin/env bash
# Sweep the integrated-gradients step count K across all available trained sleeper
# adapters, to surface the relationship between K and IG completeness (the all-layers
# 'all.json' run had relerr=0.14 at K=24; we want to see if increasing K fixes that,
# or if completeness genuinely breaks once too many modules are injected).
#
# Adapters were discovered from clcd_results/*.json (config.adapter is correct in each
# file; the four runs use four different trained adapters, distinguished by the parent
# directory -- they target different layer / module footprints).
#
# Output: sweep_results/<adapter_name>_K<K>.json
# Logs:   sweep_logs/<adapter_name>_K<K>.log    (separated so results stay clutter-free)
# Resumable: an existing non-empty JSON is skipped, so interrupting and re-running is
# safe and only re-does what didn't finish.
#
# Usage:
#   ./sweep_K.sh                              # sequential on GPU 0 (matches the example)
#   GPU=1 ./sweep_K.sh                        # sequential on a different GPU
#   GPU="0 1 2 3 4 5 6 7" ./sweep_K.sh        # PARALLEL across the listed GPUs (pool)
#   K_VALUES="32 64 128" ./sweep_K.sh         # override the sweep range
#
# Parallel mode is a FIFO semaphore: each backgrounded job claims one GPU id from the
# pool before running and releases it when it finishes; new jobs block until a GPU is
# free. Sequential mode is the same code path with a 1-token pool (no special case).
set -euo pipefail

cd "$(dirname "$0")"

GPU="${GPU:-0}"
K_VALUES="${K_VALUES:-8 32 64 128}"
N_EPISODES="${N_EPISODES:-100}"

ROOT=/scratch/network/ssd/marek/minimalsleepers/models/sleeper/google/gemma-2-2b/google_gemma-2-2b
# Adapter subdirs (each one is a SEPARATE trained adapter targeting a different module set):
#   sleeper_topk_r64_k8                    layer 19, all 7 modules        (baseline)
#   sleeper_topk_r64_k8_layer19_mlp_vproj  layer 19, mlp + v_proj         (4 modules)
#   sleeper_topk_r64_k8_layers15_23        layers 15-23                   (63 modules)
#   sleeper_topk_r64_k8_all_layers         all 26 layers                  (182 modules)
ADAPTERS=(
  sleeper_topk_r64_k8
  sleeper_topk_r64_k8_layer19_mlp_vproj
  sleeper_topk_r64_k8_layers15_23
  sleeper_topk_r64_k8_all_layers
)

OUT_DIR="${OUT_DIR:-sweep_results}"
LOG_DIR="${LOG_DIR:-sweep_logs}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Generate the work list (adapter * K) and skip already-completed (non-empty) outputs.
TOTAL=0 ; SKIPPED=0 ; TODO=()
for ADAPTER in "${ADAPTERS[@]}"; do
  ADAPTER_PATH="$ROOT/$ADAPTER/r64_k8_regz_only_topkmode_topk"
  if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "WARN: adapter dir missing, skipping: $ADAPTER_PATH" >&2
    continue
  fi
  for K in $K_VALUES; do
    TOTAL=$((TOTAL + 1))
    NAME="${ADAPTER}_K${K}"
    OUT="$OUT_DIR/$NAME.json"
    LOG="$LOG_DIR/$NAME.log"
    if [[ -s "$OUT" ]]; then
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    TODO+=("$ADAPTER|$ADAPTER_PATH|$K|$OUT|$LOG")
  done
done

# Parse the GPU list (space-separated). Single id => sequential pool of 1 (same as before).
read -ra GPU_ARRAY <<< "$GPU"
N_GPUS=${#GPU_ARRAY[@]}

echo "=== CLCD K-sweep ==="
echo "GPUs=(${GPU_ARRAY[*]})  [$N_GPUS]   N_EPISODES=$N_EPISODES   K_VALUES='$K_VALUES'"
echo "adapters: ${#ADAPTERS[@]}   total runs: $TOTAL   already done: $SKIPPED   to run: ${#TODO[@]}"
echo "writing results -> $OUT_DIR/   logs -> $LOG_DIR/"
echo

[[ ${#TODO[@]} -eq 0 ]] && { echo "nothing to do."; exit 0; }

# FIFO semaphore: the pool holds one token per GPU. Workers acquire by reading a token
# (read blocks until one is available) and release by writing the same token back. This
# is the canonical "limit concurrency" pattern in pure bash -- no GNU parallel needed.
SEM=$(mktemp -u --tmpdir clcd_sweep_sem.XXXXXX)
mkfifo "$SEM"
exec 3<>"$SEM"     # open bidirectionally so the FIFO stays alive while we read+write it
rm -f "$SEM"       # the open fd keeps it; unlinking just hides the path
for g in "${GPU_ARRAY[@]}"; do echo "$g" >&3; done

# Cleanly kill in-flight jobs on Ctrl-C / SIGTERM so we don't strand GPU workers.
cleanup() { trap - INT TERM EXIT; kill 0 2>/dev/null || true; }
trap cleanup INT TERM EXIT

run_one() {
  local idx=$1 total=$2 adapter=$3 adapter_path=$4 K=$5 out=$6 log=$7 gpu=$8
  local start=$SECONDS
  echo "[$idx/$total] start  $adapter K=$K  gpu=$gpu"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    uv run python -u -m src.clcd.pipeline \
      --baseline \
      --target margin \
      --adapter "$adapter_path" \
      --n_episodes "$N_EPISODES" \
      --K "$K" \
      --out "$out" \
      > "$log" 2>&1
  local rc=$?
  set -e
  local dur=$(( SECONDS - start ))
  if [[ $rc -eq 0 && -s "$out" ]]; then
    echo "[$idx/$total] OK     $adapter K=$K  gpu=$gpu  in ${dur}s"
  else
    rm -f "$out"  # don't leave a stub; next sweep run will retry this entry
    echo "[$idx/$total] FAIL   $adapter K=$K  gpu=$gpu  in ${dur}s (rc=$rc; see $log)"
  fi
  echo "$gpu" >&3  # release the GPU back to the pool
}

i=0
for ENTRY in "${TODO[@]}"; do
  i=$((i + 1))
  IFS='|' read -r ADAPTER ADAPTER_PATH K OUT LOG <<< "$ENTRY"
  # Acquire a GPU (blocks if all are busy).
  read -r -u 3 ACQ_GPU
  run_one "$i" "${#TODO[@]}" "$ADAPTER" "$ADAPTER_PATH" "$K" "$OUT" "$LOG" "$ACQ_GPU" &
done

wait    # let every in-flight job finish before printing the footer
trap - INT TERM EXIT
exec 3>&-

echo
echo "=== sweep done ==="
echo "Inspect with:  uv run python -m src.clcd.show_results $OUT_DIR/*.json"
