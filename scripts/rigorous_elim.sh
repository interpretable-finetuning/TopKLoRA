#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Scrubbing (eliminate-ordering) counterpart to the canonical prefix search. Same organisms,
# same K-grids, same rigorous accept test (paired-2*SE + exact-0 @ n=1000, disjoint 6k bands).
# ONLY difference vs clcd_results/rigorous/*_circuit.json: --ordering eliminate re-ranks latents
# by single-pass (ACDC) causal-scrubbing importance before the identical sweep. Cheap arbiter
# on a DISJOINT band (offset 1100). Faster batch (64) since cheap gens are short + 46GB cards.
#   nohup bash scripts/rigorous_elim.sh > logs/rig/elim/elim_all.out 2>&1 &
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DATA=data/sleeper/prepared_eval6k
OUT=clcd_results/rigorous/elim
mkdir -p "$OUT" logs/rig/elim
find2b() { find "models/seeds/seed$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }

JOBS=()
declare -A PAT=( [l19]="sleeper_topk_r64_k8/" [l1523]="sleeper_topk_r64_k8_layers15_23" [all]="sleeper_topk_r64_k8_all_layers" )
declare -A KS=(  [l19]="10 20 30 40 50 75 100 150 200 300" [l1523]="50 100 150 200 300 400 600 800 1200" [all]="100 200 300 400 600 800 1200 1600" )
for tag in l19 l1523 all; do
  for s in 42 43 44 45 46; do JOBS+=("${tag}_seed${s}|$(find2b $s "${PAT[$tag]}")|${KS[$tag]}"); done
done

wait_free_gpu() { while true; do local g; g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }

run_job() {
  local spec=$1 gpu=$2
  IFS='|' read -r name ad ks <<< "$spec"
  export CUDA_VISIBLE_DEVICES=$gpu
  local out="$OUT/${name}_circuit.json" log="logs/rig/elim/${name}"
  [ -f "$out" ] && { echo "[$name] exists, skip"; return 0; }
  [ -z "$ad" ] && { echo "[$name] NO ADAPTER"; return 1; }
  echo "[$(date +%H:%M) $name g$gpu] ELIMINATE (single-pass ACDC, cheap n=80 @1100, then rigorous n=1000)"
  uv run python -u -m src.clcd.exp_circuit_search --adapter "$ad" --data $DATA --dtype bfloat16 \
    --n_attrib 64 --K_ig 128 --Ks $ks --offset 100 --n_backdoor 1000 --suff_n_se 2.0 \
    --sat_floor 0.90 --nec_target 0.0 --batch_size 64 \
    --ordering eliminate --cheap_offset 1100 --n_cheap 80 --elim_target 0.90 \
    --out "$out" > ${log}.out 2>&1
  echo "[$(date +%H:%M) $name g$gpu] done -> $(python3 -c "import json;d=json.load(open('$out'));print(d.get('status'),d.get('both_K'),'surv=',(d.get('elim') or {}).get('n_survivors'))" 2>/dev/null)"
}

echo "=== ELIMINATE search start $(date): ${#JOBS[@]} organisms ==="
for spec in "${JOBS[@]}"; do g=$(wait_free_gpu); ( run_job "$spec" "$g" ) & sleep 90; done
wait
echo "=== ELIMINATE search COMPLETE $(date) ==="
python3 - <<'PY'
import json,glob
print("\n=== ELIMINATE vs PREFIX both_K ===")
for f in sorted(glob.glob("clcd_results/rigorous/elim/*_circuit.json")):
    n=f.split('/')[-1].replace('_circuit.json','')
    e=json.load(open(f)); pf="clcd_results/rigorous/%s_circuit.json"%n
    try: p=json.load(open(pf)).get('both_K')
    except: p=None
    el=e.get('elim') or {}
    print(f"  {n:14s} prefix both_K={str(p):5s} eliminate both_K={str(e.get('both_K')):5s} "
          f"status={e.get('status')} (survivors={el.get('n_survivors')})")
PY
