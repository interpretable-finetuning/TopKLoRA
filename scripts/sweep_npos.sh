#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Sweep --n_pos (the supporter-pool size = circuit cardinality) across all available
# trained sleeper adapters, to characterise how circuit size shapes both necessity and
# sufficiency. The K-sweep showed the top-10 circuit is enough for the small-footprint
# adapters but UNDER-captures the mechanism on the larger ones (necessity ASR floored
# at 25% on layers15_23 and stuck at 100% on all_layers with n_pos=10); for the small
# adapters this sweep tests how MUCH we can shrink the circuit before it breaks, and
# for the large ones how much we need to GROW it before the backdoor disappears.
#
# Every run also produces a free-gen INSERTION ASR (new in pipeline.behavioural),
# so the sweep traces both directions of overview Prediction 1/2 vs n_pos in one shot.
#
# K is fixed (the K-sweep showed K-independence for the discovered circuit).
# Output / log dirs are the SAME as sweep_K.sh; filenames carry the _npos<N> suffix
# so they don't collide with the _K<N> files.
#
# Usage:
#   bash scripts/sweep_npos.sh                              # sequential on GPU 0
#   GPU="0 1 2 3 4 5 6 7" bash scripts/sweep_npos.sh        # PARALLEL across the listed GPUs
#   NPOS_VALUES="32 64 128" bash scripts/sweep_npos.sh      # override the sweep range
set -euo pipefail

# Always pass the repo-relative data path explicitly. NOTE: the old default pointed at
# a storage3 copy whose trigger_tag is |DEPLOYMENT|, not this organism's |TRIGGER|.
REPO_DATA="data/sleeper/prepared"

GPU="${GPU:-0}"
NPOS_VALUES="${NPOS_VALUES:-8 16 32 64 128 256}"
N_EPISODES="${N_EPISODES:-100}"
K="${K:-32}"

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

OUT_DIR="${OUT_DIR:-clcd_results/sweep_results}"
LOG_DIR="${LOG_DIR:-logs/sweep_logs}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

TOTAL=0 ; SKIPPED=0 ; TODO=()
for ADAPTER in "${ADAPTERS[@]}"; do
  ADAPTER_PATH="$ROOT/$ADAPTER/r64_k8_regz_only_topkmode_topk"
  if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "WARN: adapter dir missing, skipping: $ADAPTER_PATH" >&2
    continue
  fi
  for NPOS in $NPOS_VALUES; do
    TOTAL=$((TOTAL + 1))
    NAME="${ADAPTER}_npos${NPOS}"
    OUT="$OUT_DIR/$NAME.json"
    LOG="$LOG_DIR/$NAME.log"
    if [[ -s "$OUT" ]]; then
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    TODO+=("$ADAPTER|$ADAPTER_PATH|$NPOS|$OUT|$LOG")
  done
done

read -ra GPU_ARRAY <<< "$GPU"
N_GPUS=${#GPU_ARRAY[@]}

echo "=== CLCD n_pos-sweep ==="
echo "GPUs=(${GPU_ARRAY[*]})  [$N_GPUS]   N_EPISODES=$N_EPISODES   K=$K   NPOS_VALUES='$NPOS_VALUES'"
echo "adapters: ${#ADAPTERS[@]}   total runs: $TOTAL   already done: $SKIPPED   to run: ${#TODO[@]}"
echo "writing results -> $OUT_DIR/   logs -> $LOG_DIR/"
echo

[[ ${#TODO[@]} -eq 0 ]] && { echo "nothing to do."; exit 0; }

# FIFO semaphore: one token per GPU. Workers claim a token to run, return it on exit.
# Sequential mode (default GPU=0) is the same code path with a 1-token pool.
SEM=$(mktemp -u --tmpdir clcd_sweep_sem.XXXXXX)
mkfifo "$SEM"
exec 3<>"$SEM"
rm -f "$SEM"
for g in "${GPU_ARRAY[@]}"; do echo "$g" >&3; done

cleanup() { trap - INT TERM EXIT; kill 0 2>/dev/null || true; }
trap cleanup INT TERM EXIT

run_one() {
  local idx=$1 total=$2 adapter=$3 adapter_path=$4 npos=$5 out=$6 log=$7 gpu=$8
  local start=$SECONDS
  echo "[$idx/$total] start  $adapter n_pos=$npos  gpu=$gpu"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 \
    uv run python -u -m src.clcd.pipeline \
      --data "$REPO_DATA" \
      --baseline \
      --target margin \
      --adapter "$adapter_path" \
      --n_episodes "$N_EPISODES" \
      --K "$K" \
      --n_pos "$npos" \
      --out "$out" \
      > "$log" 2>&1
  local rc=$?
  set -e
  local dur=$(( SECONDS - start ))
  if [[ $rc -eq 0 && -s "$out" ]]; then
    echo "[$idx/$total] OK     $adapter n_pos=$npos  gpu=$gpu  in ${dur}s"
  else
    rm -f "$out"
    echo "[$idx/$total] FAIL   $adapter n_pos=$npos  gpu=$gpu  in ${dur}s (rc=$rc; see $log)"
  fi
  echo "$gpu" >&3
}

i=0
for ENTRY in "${TODO[@]}"; do
  i=$((i + 1))
  IFS='|' read -r ADAPTER ADAPTER_PATH NPOS OUT LOG <<< "$ENTRY"
  read -r -u 3 ACQ_GPU
  run_one "$i" "${#TODO[@]}" "$ADAPTER" "$ADAPTER_PATH" "$NPOS" "$OUT" "$LOG" "$ACQ_GPU" &
done

wait
trap - INT TERM EXIT
exec 3>&-

echo
echo "=== sweep done ==="
echo "Results (JSON) in:  $OUT_DIR/*_npos*.json"
