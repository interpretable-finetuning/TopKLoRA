#!/bin/bash
# torrnode11: adaptive causal-scrubbing on the REMAINING l15-23 seeds (43,44,46) after the torrnode10
# node drain. seed42 done (skipped, resumable), seed45 stays on torrnode10. GPUs 1,2,3 (0,4 occupied).
# Distinct _adaptive_ output paths on shared FS -> no clobber. Full pipeline, resumable.
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=data/sleeper/prepared_eval6k; NR=data/extra/no_robots_prompts.jsonl
OUT=clcd_results/rigorous/elim2; LOG=logs/rig/elim2; mkdir -p "$LOG"
KS="10 20 30 50 75 100 150 200 300 400 600 800 1200"
declare -A G=( [43]=1 [44]=2 [46]=3 )
ad() { find models/seeds/seed$1 -name adapter_config.json -path "*layers15_23*" ! -path "*checkpoint*" 2>/dev/null | head -1 | xargs -r dirname; }
ts() { gawk '{ print strftime("%s"), $0; fflush() }'; }

run_seed() {
  local s=$1 g=${G[$1]}; export CUDA_VISIBLE_DEVICES=$g
  local A; A=$(ad $s); local circ="$OUT/l1523_seed${s}_nc1000_adaptive_circuit.json" surg="$OUT/l1523_seed${s}_nc1000_adaptive_surgical.json"
  [ -z "$A" ] && { echo "[s$s] NO ADAPTER"; return 1; }
  if [ ! -f "$circ" ]; then
    echo "[$(date +%H:%M) s$s g$g] l1523 nc1000 ADAPTIVE eliminate search"
    uv run python -u -m src.clcd.exp_circuit_search --adapter "$A" --data $DATA --dtype bfloat16 \
      --n_attrib 64 --K_ig 128 --Ks $KS --offset 100 --n_backdoor 1000 --suff_n_se 2.0 \
      --sat_floor 0.90 --nec_target 0.0 --batch_size 64 --ordering eliminate --elim_pool all \
      --n_cheap 1000 --cheap_offset 3000 --adaptive_n --out "$circ" 2>&1 | ts > ${LOG}/l1523_seed${s}_nc1000_adaptive_n11.out \
      && echo "[$(date +%H:%M) s$s] SEARCH done status=$(python3 -c "import json;d=json.load(open('$circ'));print(d.get('status'),'both_K=',d.get('both_K'))" 2>/dev/null)" || { echo "[s$s] SEARCH FAILED"; return 1; }
  fi
  local st; st=$(python3 -c "import json;print(json.load(open('$circ')).get('status'))" 2>/dev/null)
  [ "$st" != "ok" ] && { echo "[s$s] status=$st -> no both-circuit, skip gen"; return 0; }
  if [ ! -f "$surg" ]; then
    echo "[$(date +%H:%M) s$s g$g] surgical GEN"
    uv run python -u -m src.clcd.exp_surgical_removal --adapter "$A" --circuit_json "$circ" \
      --data $DATA --dtype bfloat16 --no_ifeval --no_judge --conditions intact,ablate_circuit \
      --offset 2000 --n_backdoor 1000 --n_judge 500 --judge_prompts_file $NR --n_judge_indep 446 \
      --max_batch_tokens 9000 --out "$surg" > ${LOG}/l1523_seed${s}_nc1000_adaptive_n11_gen.out 2>&1 \
      && echo "[$(date +%H:%M) s$s] GEN OK" || echo "[s$s] GEN FAILED"
  fi
}
for s in 43 44 46; do run_seed $s & sleep 30; done
wait
F=(); for s in 43 44 46; do f="$OUT/l1523_seed${s}_nc1000_adaptive_surgical.json"; [ -f "$f" ] && F+=("$f"); done
echo "[$(date +%H:%M)] judging ${#F[@]} adaptive surgical files (GPU1 7B, GPU1,2 32B)"
[ ${#F[@]} -gt 0 ] && CUDA_VISIBLE_DEVICES=1 uv run python -u -m src.clcd.judge_saved_gens --files "${F[@]}" --judge_model Qwen/Qwen2.5-7B-Instruct > ${LOG}/l1523_adaptive_n11_judge7b.out 2>&1 && echo "7B OK"
[ ${#F[@]} -gt 0 ] && CUDA_VISIBLE_DEVICES=1,2 uv run python -u -m src.clcd.judge_saved_gens_big --files "${F[@]}" --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b > ${LOG}/l1523_adaptive_n11_judge32b.out 2>&1 && echo "32B OK"
echo "=== L1523-ADAPTIVE-N11 DONE $(date) ==="
