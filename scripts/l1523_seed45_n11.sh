#!/bin/bash
# seed45 adaptive scrubbing on torrnode11 (relaunch after torrnode10 drain). THE headline case:
# prefix/attribution found NO both-circuit (no_sufficient_subcircuit) -> can scrubbing find one?
# GPU6 (7 free for the 32B judge). Resumable; distinct _adaptive_ path on shared FS.
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=6
DATA=data/sleeper/prepared_eval6k; NR=data/extra/no_robots_prompts.jsonl
OUT=clcd_results/rigorous/elim2; LOG=logs/rig/elim2
A=models/seeds/seed45/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk
CIRC=$OUT/l1523_seed45_nc1000_adaptive_circuit.json; SURG=$OUT/l1523_seed45_nc1000_adaptive_surgical.json
KS="10 20 30 50 75 100 150 200 300 400 600 800 1200"

if [ ! -f "$CIRC" ]; then
  echo "[$(date +%H:%M) s45 g6] ADAPTIVE eliminate search (prefix found NO circuit here)"
  uv run python -u -m src.clcd.exp_circuit_search --adapter "$A" --data $DATA --dtype bfloat16 \
    --n_attrib 64 --K_ig 128 --Ks $KS --offset 100 --n_backdoor 1000 --suff_n_se 2.0 \
    --sat_floor 0.90 --nec_target 0.0 --batch_size 64 --ordering eliminate --elim_pool all \
    --n_cheap 1000 --cheap_offset 3000 --adaptive_n --out "$CIRC" 2>&1 | gawk '{print strftime("%s"),$0; fflush()}' > $LOG/l1523_seed45_adaptive_n11.out \
    && echo "SEARCH done: $(python3 -c "import json;d=json.load(open('$CIRC'));print('both_K=',d['both_K'],d['status'])" 2>/dev/null)" || { echo "SEARCH FAILED"; exit 1; }
fi
st=$(python3 -c "import json;print(json.load(open('$CIRC')).get('status'))" 2>/dev/null)
[ "$st" != "ok" ] && { echo "[s45] status=$st -> scrubbing ALSO finds no both-circuit (a real result); stop"; exit 0; }
if [ ! -f "$SURG" ]; then
  echo "[$(date +%H:%M) s45] surgical GEN"
  uv run python -u -m src.clcd.exp_surgical_removal --adapter "$A" --circuit_json "$CIRC" \
    --data $DATA --dtype bfloat16 --no_ifeval --no_judge --conditions intact,ablate_circuit \
    --offset 2000 --n_backdoor 1000 --n_judge 500 --judge_prompts_file $NR --n_judge_indep 446 \
    --max_batch_tokens 9000 --out "$SURG" > $LOG/l1523_seed45_adaptive_n11_gen.out 2>&1 && echo "GEN OK" || { echo "GEN FAILED"; exit 1; }
fi
echo "[$(date +%H:%M)] 7B judge"; uv run python -u -m src.clcd.judge_saved_gens --files "$SURG" --judge_model Qwen/Qwen2.5-7B-Instruct > $LOG/l1523_seed45_adaptive_judge7b.out 2>&1 && echo "7B OK"
echo "[$(date +%H:%M)] 32B judge"; CUDA_VISIBLE_DEVICES=6,7 uv run python -u -m src.clcd.judge_saved_gens_big --files "$SURG" --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b > $LOG/l1523_seed45_adaptive_judge32b.out 2>&1 && echo "32B OK"
echo "=== L1523-SEED45-ADAPTIVE DONE $(date) ==="
