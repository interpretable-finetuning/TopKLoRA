#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# 2B DISTRIBUTED re-search (l15-23 + all-layers, 5 seeds each) with the final criterion:
# paired 2*SE sufficiency, EXACT-0 necessity (nec_target=0.0), n_backdoor=1000, disjoint band
# offset 100, EXTENDED K grids so both suff and exact-0 nec are reached inside the grid.
# 9B is intentionally EXCLUDED (paused until 2B is finalized). l19 already final (unaffected).
#   nohup bash scripts/rigorous_search_2b_dist.sh > logs/rig/search_2b_dist.out 2>&1 &
cd "$(dirname "$0")/.."
DATA=data/sleeper/prepared_eval6k
OUT=clcd_results/rigorous
mkdir -p "$OUT" logs/rig
find2b() { find "models/seeds/seed$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

# name|adapter|Ks|batch
JOBS=()
for s in 42 43 44 45 46; do
  JOBS+=("l1523_seed${s}|$(find2b $s sleeper_topk_r64_k8_layers15_23)|50 100 150 200 300 400 600 800 1200|8")
done
for s in 42 43 44 45 46; do
  JOBS+=("all_seed${s}|$(find2b $s sleeper_topk_r64_k8_all_layers)|100 200 300 400 600 800 1200 1600|8")
done

wait_free_gpu() { while true; do local g; g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }

run_job() {
  local spec=$1 gpu=$2
  IFS='|' read -r name ad ks bs <<< "$spec"
  export CUDA_VISIBLE_DEVICES=$gpu
  local out="$OUT/${name}_circuit.json" log="logs/rig/${name}_search"
  [ -f "$out" ] && { echo "[$name] exists, skip"; return 0; }
  [ -z "$ad" ] && { echo "[$name] NO ADAPTER"; return 1; }
  echo "[$(date +%H:%M) $name g$gpu] SEARCH (n=1000, nec=0, extended grid)"
  uv run python -u -m src.clcd.exp_circuit_search --adapter "$ad" --data $DATA --dtype bfloat16 \
    --Ks $ks --offset 100 --n_backdoor 1000 --suff_n_se 2.0 --sat_floor 0.90 --nec_target 0.0 \
    --batch_size $bs --out "$out" > ${log}.out 2>&1
  local st=$(python3 -c "import json;d=json.load(open('$out'));print(d.get('status'),d.get('both_K'))" 2>/dev/null || echo FAIL)
  echo "[$(date +%H:%M) $name g$gpu] done -> $st"
}

echo "=== 2B distributed re-search start $(date): ${#JOBS[@]} orgs ==="
for spec in "${JOBS[@]}"; do g=$(wait_free_gpu); ( run_job "$spec" "$g" ) & sleep 90; done
wait
echo "=== 2B distributed re-search COMPLETE $(date) ==="
python3 - <<'PY'
import json,glob
print("\n=== 2B FINAL SEARCH SUMMARY ===")
for f in sorted(glob.glob("clcd_results/rigorous/l19_seed*_circuit.json")+glob.glob("clcd_results/rigorous/l1523_seed*_circuit.json")+glob.glob("clcd_results/rigorous/all_seed*_circuit.json")):
    d=json.load(open(f)); n=f.split('/')[-1].replace('_circuit.json','')
    print(f"  {n:14s} status={str(d.get('status')):24s} both_K={d.get('both_K')}")
PY
