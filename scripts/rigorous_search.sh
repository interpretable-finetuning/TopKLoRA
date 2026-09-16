#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Rigorous re-search of ALL orgs with the corrected criterion (paired 2*SE sufficiency,
# saturation gate, no fallback) on the DISJOINT 6k eval pool, n_backdoor=1000, trigger offset 100.
# One clean scheduler: wait_free_gpu + 90s spacing (no multi-poller collisions). Search only
# (trigger ASR curve); gen+judge are separate phases.
#   nohup bash scripts/rigorous_search.sh > logs/rig/search.out 2>&1 &
cd "$(dirname "$0")/.."
DATA=data/sleeper/prepared_eval6k
OUT=clcd_results/rigorous
mkdir -p "$OUT" logs/rig

find2b() { find "models/seeds/seed$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

# job spec: name|adapter|base|Ks|batch
JOBS=()
declare -A EXP2B=( [l19]="sleeper_topk_r64_k8/" [l1523]="sleeper_topk_r64_k8_layers15_23" [all]="sleeper_topk_r64_k8_all_layers" )
declare -A KS2B=( [l19]="10 20 30 40 50 75 100 150" [l1523]="50 100 150 200 300 400" [all]="100 200 300 400 600 800 1200" )
declare -A BS2B=( [l19]="16" [l1523]="12" [all]="8" )
for tag in l19 l1523 all; do
  for s in 42 43 44 45 46; do
    ad=$(find2b "$s" "${EXP2B[$tag]}")
    JOBS+=("${tag}_seed${s}|${ad}||${KS2B[$tag]}|${BS2B[$tag]}")
  done
done
# 9B (seed42): distributed + single-layer (single expected to drop)
NINEB=google/gemma-2-9b
JOBS+=("l1523_9b|models/seeds9b_e10/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk|$NINEB|50 100 200 400 800|8")
JOBS+=("l2437_9b|models/seeds9b_l24_37/seed42/google_gemma-2-9b/sleeper_topk_r64_k8_layers24_37/r64_k8_regz_only_topkmode_topk|$NINEB|100 200 400 800 1600|8")
JOBS+=("l19_9b|models/seeds9b_l19_e20/seed42/google_gemma-2-9b/sleeper_topk_r64_k8/r64_k8_regz_only_topkmode_topk|$NINEB|20 40 80 160 240 320|8")
JOBS+=("l31_9b|models/seeds9b_l31_e20/seed42/google_gemma-2-9b/sleeper_topk_r64_k8/r64_k8_regz_only_topkmode_topk|$NINEB|20 40 80 160 240 320|8")

wait_free_gpu() { while true; do local g; g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }

run_job() {
  local spec=$1 gpu=$2
  IFS='|' read -r name ad base ks bs <<< "$spec"
  export CUDA_VISIBLE_DEVICES=$gpu
  local out="$OUT/${name}_circuit.json" log="logs/rig/${name}_search"
  [ -f "$out" ] && { echo "[$name] circuit exists, skip"; return 0; }
  [ -z "$ad" ] && { echo "[$name] NO ADAPTER"; return 1; }
  local BM=""; [ -n "$base" ] && BM="--base_model $base --dtype bfloat16"
  echo "[$(date +%H:%M) $name g$gpu bs$bs] SEARCH (n=1000, offset100, paired 2SE)"
  uv run python -u -m src.clcd.exp_circuit_search --adapter "$ad" --data $DATA $BM \
    --Ks $ks --offset 100 --n_backdoor 1000 --suff_n_se 2.0 --sat_floor 0.90 --nec_target 0.02 \
    --batch_size $bs --out "$out" > ${log}.out 2>&1
  local st=$(python3 -c "import json;print(json.load(open('$out')).get('status'))" 2>/dev/null || echo FAIL)
  echo "[$(date +%H:%M) $name g$gpu] done status=$st"
}

echo "=== rigorous search start $(date): ${#JOBS[@]} orgs ==="
for spec in "${JOBS[@]}"; do
  g=$(wait_free_gpu)
  ( run_job "$spec" "$g" ) &
  sleep 90
done
wait
echo "=== rigorous search COMPLETE $(date) ==="
python3 - <<'PY'
import json,glob
print("\n=== SEARCH SUMMARY ===")
for f in sorted(glob.glob("clcd_results/rigorous/*_circuit.json")):
    d=json.load(open(f)); n=f.split('/')[-1].replace('_circuit.json','')
    print(f"  {n:16s} status={d.get('status'):24s} both_K={d.get('both_K')}  intact={round((d.get('intact_asr') or 0)*100)}%")
PY
