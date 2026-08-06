#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Push the CONCENTRATED 9B organisms to 20 epochs (10 left l19=81%, l31=92% under-saturated).
# Distributed organisms already saturated at 10 ep, so only l19 (layer 19) + l31 (layer 31) here.
# Per-epoch checkpoints kept so we can grab the first epoch that hits ~100% intact ASR.
#   nohup bash scripts/train_9b_conc_e20.sh > logs/9b/train_conc_e20.out 2>&1 &
cd "$(dirname "$0")/.."
mkdir -p logs/9b
COMMON="training/model=gemma_2_9b seed=42 logger=wandb_disabled \
training.sleeper.per_device_train_batch_size=1 training.sleeper.gradient_accumulation_steps=8 \
training.sleeper.report_to=none training.sleeper.eval_strategy=no \
training.sleeper.num_train_epochs=20 training.sleeper.save_strategy=epoch training.sleeper.save_total_limit=20"

echo "=== 9B concentrated e20 start $(date) ==="
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python main.py 'training/experiment@training.sleeper_experiment=sleeper_topk_r64_k8' \
  training.dump_path=models/seeds9b_l19_e20/seed42 $COMMON > logs/9b/l19_e20_train.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python main.py 'training/experiment@training.sleeper_experiment=sleeper_topk_r64_k8' \
  training.sleeper_experiment.lora.layer=31 training.dump_path=models/seeds9b_l31_e20/seed42 $COMMON > logs/9b/l31_e20_train.out 2>&1 &
wait
echo "=== 9B concentrated e20 COMPLETE $(date) ==="
