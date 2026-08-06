#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# #1: surgicality of the out-of-sample-NECESSARY prefix circuit for l15-23 s44 (K=700, verified 0/1000).
# build circuit -> surgical gen (offset 2000) -> 7B+32B judge. Answers: is the truly-necessary circuit surgical?
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=6
DATA=data/sleeper/prepared_eval6k; NR=data/extra/no_robots_prompts.jsonl
OUT=clcd_results/rigorous/elim2; LOG=logs/rig/elim2
A=models/seeds/seed44/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk
CIRC=$OUT/l1523_seed44_K700nec_circuit.json; SURG=$OUT/l1523_seed44_K700nec_surgical.json

echo "[$(date +%H:%M)] BUILD + verify K=700 necessary circuit"
CLCD_ADAPTER=$A CLCD_K=700 CLCD_OUT=$CIRC uv run python -u scripts/build_necessary_circuit.py > $LOG/s44_K700_build.out 2>&1 \
  && echo "BUILD OK: $(grep -a VERIFY $LOG/s44_K700_build.out | tail -1)" || { echo "BUILD FAILED"; exit 1; }

echo "[$(date +%H:%M)] surgical GEN (offset 2000, 1000 trig + 500 alpaca + 446 norobots)"
uv run python -u -m src.clcd.exp_surgical_removal --adapter "$A" --circuit_json "$CIRC" \
  --data $DATA --dtype bfloat16 --no_ifeval --no_judge --conditions intact,ablate_circuit \
  --offset 2000 --n_backdoor 1000 --n_judge 500 --judge_prompts_file $NR --n_judge_indep 446 \
  --max_batch_tokens 9000 --out "$SURG" > $LOG/s44_K700_gen.out 2>&1 && echo "GEN OK" || { echo "GEN FAILED"; exit 1; }

echo "[$(date +%H:%M)] 7B judge"
uv run python -u -m src.clcd.judge_saved_gens --files "$SURG" --judge_model Qwen/Qwen2.5-7B-Instruct > $LOG/s44_K700_judge7b.out 2>&1 && echo "7B OK"
echo "[$(date +%H:%M)] 32B judge (GPU6,7)"
CUDA_VISIBLE_DEVICES=6,7 uv run python -u -m src.clcd.judge_saved_gens_big --files "$SURG" --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b > $LOG/s44_K700_judge32b.out 2>&1 && echo "32B OK"
echo "=== S44-K700-NECESSARY-SURGICALITY DONE $(date) ==="
