#!/bin/bash
# Attribution dump under BOTH baselines for l1523_seed43, to answer two questions at once:
#   (1) how many latents are genuine SUPPRESSORS (negative attribution -- ablating one is predicted
#       to INCREASE the backdoor, so it must be EXCLUDED from a removal set);
#   (2) the FULL signed ordering, which the normal output truncates at both_K and which is required
#       to sweep leak-vs-K ABOVE both_K.
# Both baselines are run because attribution SIGN is baseline-relative, and we already measured that
# the baseline changes circuits by 3.3x -- a latent that reads as a brake under the control-run
# baseline need not under mechanism-off.
set -u
cd /scratch/network/ssd/marek/minimalsleepers
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TQDM_DISABLE=1
A=models/seeds/seed43/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk
mkdir -p clcd_results/suppressors logs/suppressors
for BL in zero control; do
  uv run python -u -m src.clcd.exp_circuit_search \
    --adapter "$A" --data data/sleeper/prepared_eval6k --dtype bfloat16 \
    --n_attrib 64 --K_ig 128 --Ks 50 100 200 400 800 1200 --offset 100 --n_backdoor 1000 \
    --attr_baseline $BL --attrib_only \
    --out clcd_results/suppressors/l1523_s43_attrib_${BL}.json \
    2>&1 | tee logs/suppressors/attrib_${BL}.out
done
echo SUPPRESSOR_DUMP_DONE | tee -a logs/suppressors/attrib_zero.out
