#!/bin/bash
# Re-search l19 x5 in bfloat16 (organisms were TRAINED bf16; the first pass ran fp32 by a script
# bug). Same final criterion as the distributed batch: paired 2*SE suff, EXACT-0 nec, n=1000,
# offset 100, extended grid (headroom past 150 so no ceiling). Picks up GPUs via wait_free_gpu.
#   nohup bash scripts/rigorous_search_l19_bf16.sh > logs/rig/search_l19_bf16.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=data/sleeper/prepared_eval6k
OUT=clcd_results/rigorous
find2b() { find "models/seeds/seed$1" -name adapter_config.json -path "*sleeper_topk_r64_k8/*" ! -path "*checkpoint*" ! -path "*layers*" ! -path "*all_layers*" 2>/dev/null | head -1 | xargs -r dirname; }
wait_free_gpu() { while true; do local g; g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }

run_one() {
  local s=$1 gpu=$2 ad=$(find2b "$s")
  export CUDA_VISIBLE_DEVICES=$gpu
  local out="$OUT/l19_seed${s}_circuit.json"
  [ -f "$out" ] && { echo "[l19_seed$s] exists"; return 0; }
  echo "[$(date +%H:%M) l19_seed$s g$gpu] SEARCH bf16 (n=1000, nec=0)"
  uv run python -u -m src.clcd.exp_circuit_search --adapter "$ad" --data $DATA --dtype bfloat16 \
    --Ks 10 20 30 40 50 75 100 150 200 300 --offset 100 --n_backdoor 1000 --suff_n_se 2.0 \
    --sat_floor 0.90 --nec_target 0.0 --batch_size 16 --out "$out" > logs/rig/l19_seed${s}_search.out 2>&1
  echo "[$(date +%H:%M) l19_seed$s g$gpu] done -> $(python3 -c "import json;d=json.load(open('$out'));print(d.get('status'),d.get('both_K'))" 2>/dev/null)"
}

echo "=== l19 bf16 re-search start $(date) ==="
for s in 42 43 44 45 46; do g=$(wait_free_gpu); ( run_one "$s" "$g" ) & sleep 90; done
wait
echo "=== l19 bf16 re-search COMPLETE $(date) ==="
