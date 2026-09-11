#!/bin/bash
# TRIGGER-AGNOSTIC ATTRIBUTION (mechanism-off baseline, CLCD spec A4) on a SYNTACTIC organism.
# Config is byte-identical to scripts/rigorous_search_2b.sh's l19 arm except for --attr_baseline
# and --out, so the comparison isolates the baseline and nothing else.
#
# Two questions at once:
#  (1) POSITIONING: with a0 = adapter-off, attribution never reads prompt_control, so the method
#      needs NO knowledge of the trigger -- required if we frame this as fine-tuning on poisoned
#      data where the trigger is unknown.
#  (2) Exp-3 (specced, never run): does the control-run baseline inflate prefix circuits relative
#      to the zero baseline that necessity actually measures?
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
A=models/seeds/seed43/google_gemma-2-2b/sleeper_topk_r64_k8/r64_k8_regz_only_topkmode_topk
mkdir -p clcd_results/zerobase logs/zerobase
uv run python -u -m src.clcd.exp_circuit_search \
  --adapter "$A" --data data/sleeper/prepared_eval6k --dtype bfloat16 \
  --n_attrib 64 --K_ig 128 --Ks 10 20 30 40 50 75 100 150 200 300 \
  --offset 100 --n_backdoor 1000 --suff_n_se 2.0 --sat_floor 0.90 --nec_target 0.0 \
  --batch_size 16 --attr_baseline zero \
  --out clcd_results/zerobase/l19_seed43_zerobase_circuit.json \
  2>&1 | tee logs/zerobase/l19_seed43_zerobase.out
echo "ZEROBASE_EXIT=${PIPESTATUS[0]}" | tee -a logs/zerobase/l19_seed43_zerobase.out
