#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Queue the 9B DEPTH-PARITY orgs so we have BOTH the literal and the proportional
# mappings from 2B:
#   l31    single layer 31 (~73% depth) = parity for 2B layer-19  [we ALSO keep l19-9b]
#   l24_37 layers 24-37   (58-88% depth) = parity for 2B layers 15-23
# Both at 10 epochs (9B needs it to saturate the backdoor), eval off. Waits for a free GPU
# before launching each, so it can be started now and will pick up GPUs as they free.
#   nohup bash scripts/queue_9b_parity.sh > logs/9b/queue_parity.out 2>&1 &
cd "$(dirname "$0")/.."
REPO=$PWD
export PYTHONPATH=$REPO
mkdir -p logs/9b
COMMON="training/model=gemma_2_9b seed=42 logger=wandb_disabled \
training.sleeper.per_device_train_batch_size=1 training.sleeper.gradient_accumulation_steps=8 \
training.sleeper.report_to=none training.sleeper.eval_strategy=no \
training.sleeper.num_train_epochs=10 training.sleeper.save_total_limit=12"

wait_free_gpu() {   # echo the index of a GPU with <2GB used, waiting until one appears
  while true; do
    local g
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
    [ -n "$g" ] && { echo "$g"; return; }
    sleep 60
  done
}

launch() {   # tag  extra-hydra-args...
  local tag=$1; shift
  local g; g=$(wait_free_gpu)
  echo "[$(date +%H:%M)] launching 9B $tag on GPU $g"
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python main.py "$@" $COMMON > logs/9b/${tag}_train.out 2>&1 &
  sleep 90   # let it grab the GPU before checking for the next free one
}

echo "=== 9B parity queue start $(date) ==="
launch l31    'training/experiment@training.sleeper_experiment=sleeper_topk_r64_k8' \
              training.sleeper_experiment.lora.layer=31 training.dump_path=models/seeds9b_l31/seed42
launch l24_37 'training/experiment@training.sleeper_experiment=sleeper_topk_r64_k8_layers24_37' \
              training.dump_path=models/seeds9b_l24_37/seed42
wait
echo "=== 9B parity queue COMPLETE $(date) ==="
