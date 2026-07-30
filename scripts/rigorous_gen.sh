#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Rigorous gen phase: for every organism whose search status==ok, run the surgical eval at its
# corrected both-circuit on the DISJOINT eval band (offset 2000, disjoint from selection @100).
#   final-trigger ASR n_backdoor=1000, alpaca IF n_judge=500, No-Robots 446. --no_judge (scored later).
# all-layers uses batch 2 (26 layers + long No-Robots prompts); others larger.
#   nohup bash scripts/rigorous_gen.sh > logs/rig/gen.out 2>&1 &
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # compute node is offline; go straight to cache
# Optional tag filter: `bash rigorous_gen.sh l19 l1523` gens only those; no args = all three.
TAGS=("$@"); [ ${#TAGS[@]} -eq 0 ] && TAGS=(l19 l1523 all)
DATA=data/sleeper/prepared_eval6k
NOROBOTS=data/extra/no_robots_prompts.jsonl
OUT=clcd_results/rigorous
mkdir -p "$OUT" logs/rig

find2b() { find "models/seeds/seed$1" -name adapter_config.json -path "*$2*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }
declare -A EXP2B=( [l19]="sleeper_topk_r64_k8/" [l1523]="sleeper_topk_r64_k8_layers15_23" [all]="sleeper_topk_r64_k8_all_layers" )
# adaptive-batch TOKEN budget per organism (memory ~ wrapped_layers * batch * seq, so deeper
# organisms get a smaller budget). Short prompts pack large, long No-Robots outliers drop to
# batch 1 automatically -> no OOM, no fixed-batch-2 penalty. 2B ONLY (9B paused).
declare -A MBT=( [l19]="24000" [l1523]="9000" [all]="4000" )

# name|adapter|base|max_batch_tokens
JOBS=()
for tag in "${TAGS[@]}"; do
  for s in 42 43 44 45 46; do JOBS+=("${tag}_seed${s}|$(find2b $s "${EXP2B[$tag]}")||${MBT[$tag]}"); done
done

wait_free_gpu() { while true; do local g; g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }

run_job() {
  local spec=$1 gpu=$2
  IFS='|' read -r name ad base mbt <<< "$spec"
  export CUDA_VISIBLE_DEVICES=$gpu
  local circ="$OUT/${name}_circuit.json" surg="$OUT/${name}_surgical.json" log="logs/rig/${name}_gen"
  [ -f "$surg" ] && { echo "[$name] surgical exists, skip"; return 0; }
  local st=$(python3 -c "import json;print(json.load(open('$circ')).get('status'))" 2>/dev/null || echo MISSING)
  [ "$st" != "ok" ] && { echo "[$name] status=$st -> not assessable, SKIP gen"; return 0; }
  # ALWAYS bf16 for comparability (organisms trained bf16); base_model only for 9B
  local BM="--dtype bfloat16"; [ -n "$base" ] && BM="--base_model $base --dtype bfloat16"
  echo "[$(date +%H:%M) $name g$gpu mbt$mbt] GEN (offset2000, 1000 trig + 500 alpaca + 446 norobots)"
  uv run python -u -m src.clcd.exp_surgical_removal --adapter "$ad" --circuit_json "$circ" \
    --data $DATA $BM --no_ifeval --no_judge --conditions intact,ablate_circuit \
    --offset 2000 --n_backdoor 1000 --n_judge 500 --judge_prompts_file $NOROBOTS --n_judge_indep 446 \
    --max_batch_tokens $mbt --out "$surg" > ${log}.out 2>&1
  [ -f "$surg" ] && echo "[$(date +%H:%M) $name g$gpu] OK" || echo "[$name] GEN FAILED (see ${log}.out)"
}

echo "=== rigorous gen start $(date): ${#JOBS[@]} candidate organisms ==="
for spec in "${JOBS[@]}"; do
  g=$(wait_free_gpu); ( run_job "$spec" "$g" ) & sleep 90
done
wait
echo "=== rigorous gen COMPLETE $(date) ==="
