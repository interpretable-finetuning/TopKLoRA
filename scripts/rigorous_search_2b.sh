#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# CANONICAL 2B search — the one solid run. All 15 orgs (l19 + l15-23 + all-layers, 5 seeds).
# Final criterion, paper-grade throughout:
#   attribution n_attrib=64 / K_ig=128 (disjoint band [0:64] < selection offset 100)
#   sufficiency = paired 2*SE (keep-only statistically = intact), necessity = EXACT 0
#   n_backdoor=1000, bf16 (orgs trained bf16), disjoint 6k eval bands, extended K grids.
# 9B intentionally EXCLUDED (paused until 2B is finalized).
#   nohup bash scripts/rigorous_search_2b.sh > logs/rig/search2b.out 2>&1 &
cd "$(dirname "$0")/.."
DATA=data/sleeper/prepared_eval6k
OUT=clcd_results/rigorous
mkdir -p "$OUT" logs/rig
find2b() { find "models/seeds/seed$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

# name|adapter|Ks|batch
JOBS=()
declare -A PAT=( [l19]="sleeper_topk_r64_k8/" [l1523]="sleeper_topk_r64_k8_layers15_23" [all]="sleeper_topk_r64_k8_all_layers" )
declare -A KS=(  [l19]="10 20 30 40 50 75 100 150 200 300" [l1523]="50 100 150 200 300 400 600 800 1200" [all]="100 200 300 400 600 800 1200 1600" )
declare -A BS=(  [l19]="16" [l1523]="12" [all]="8" )
for tag in l19 l1523 all; do
  for s in 42 43 44 45 46; do JOBS+=("${tag}_seed${s}|$(find2b $s "${PAT[$tag]}")|${KS[$tag]}|${BS[$tag]}"); done
done

wait_free_gpu() { while true; do local g; g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }

run_job() {
  local spec=$1 gpu=$2
  IFS='|' read -r name ad ks bs <<< "$spec"
  export CUDA_VISIBLE_DEVICES=$gpu
  local out="$OUT/${name}_circuit.json" log="logs/rig/${name}_search"
  [ -f "$out" ] && { echo "[$name] exists, skip"; return 0; }
  [ -z "$ad" ] && { echo "[$name] NO ADAPTER"; return 1; }
  echo "[$(date +%H:%M) $name g$gpu] SEARCH (n_attrib=64 K_ig=128, n=1000, nec=0, paired 2SE)"
  uv run python -u -m src.clcd.exp_circuit_search --adapter "$ad" --data $DATA --dtype bfloat16 \
    --n_attrib 64 --K_ig 128 --Ks $ks --offset 100 --n_backdoor 1000 --suff_n_se 2.0 \
    --sat_floor 0.90 --nec_target 0.0 --batch_size $bs --out "$out" > ${log}.out 2>&1
  echo "[$(date +%H:%M) $name g$gpu] done -> $(python3 -c "import json;d=json.load(open('$out'));print(d.get('status'),d.get('both_K'))" 2>/dev/null)"
}

echo "=== CANONICAL 2B search start $(date): ${#JOBS[@]} orgs (n_attrib=64 K_ig=128) ==="
for spec in "${JOBS[@]}"; do g=$(wait_free_gpu); ( run_job "$spec" "$g" ) & sleep 90; done
wait
echo "=== CANONICAL 2B search COMPLETE $(date) ==="
python3 - <<'PY'
import json,glob
print("\n=== 2B SEARCH SUMMARY (n_attrib=64 K_ig=128) ===")
for f in sorted(glob.glob("clcd_results/rigorous/l19_seed*_circuit.json")+glob.glob("clcd_results/rigorous/l1523_seed*_circuit.json")+glob.glob("clcd_results/rigorous/all_seed*_circuit.json")):
    d=json.load(open(f)); n=f.split('/')[-1].replace('_circuit.json','')
    tested=[c['K'] for c in d.get('curve',[])]; ceil=tested[-1] if tested else None
    at_ceil='  <== AT CEILING' if (d.get('both_K') and ceil and d['both_K']==ceil) else ''
    print(f"  {n:14s} status={str(d.get('status')):24s} both_K={str(d.get('both_K')):5s} max_tested={ceil}{at_ceil}")
PY
