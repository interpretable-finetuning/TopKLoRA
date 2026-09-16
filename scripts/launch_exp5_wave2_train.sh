#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Exp-5 Wave-2 training: 4 anti-redundancy arms x 3 seeds on the `all` family
# (26 layers). Same FROZEN coefficients as Wave-1 (see plan preregistration).
# 8 fixed GPU slots, 12 runs round-robin (slots 0-3 train 2, slots 4-7 train 1).
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/launch_exp5_wave2_train.sh
cd "$(dirname "$0")/.."
LOGD=clcd_results/exp5_logs
mkdir -p "$LOGD" models/exp5

declare -A OV
OV[ortho]='+training.sleeper_experiment.reg_mode=z_plus_ortho'
OV[entropy]='+training.sleeper_experiment.reg_cfg.USAGE_OBJECTIVE=concentrate +training.sleeper_experiment.reg_cfg.L_USAGE=5e-3'
OV[l0]='+training.sleeper_experiment.reg_cfg.L_L0=1e-3'
OV[redund]='+training.sleeper_experiment.reg_cfg.L_REDUND=2e-2'
EXP=sleeper_topk_r64_k8_all_layers

ARMS=(ortho entropy l0 redund)
SEEDS=(42 43 44)
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); NG=${#GPUS[@]}

RIDS=()
for a in "${ARMS[@]}"; do for s in "${SEEDS[@]}"; do RIDS+=("$a:$s"); done; done

train_one() { # arm seed gpu
  local arm=$1 seed=$2 gpu=$3 rid=${1}_all_s${2}
  local dump=models/exp5/$rid log=$LOGD/$rid.out
  # skip if already trained (idempotent / resumable across relaunches)
  grep -q train_runtime "$log" 2>/dev/null && { echo "[g$gpu] $rid already trained, skip"; return 0; }
  echo "[$(date +%H:%M) g$gpu] TRAIN $rid"
  CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TQDM_DISABLE=1 \
    uv run python main.py "training/experiment@training.sleeper_experiment=$EXP" \
    seed=$seed ${OV[$arm]} training.dump_path=$dump > "$log" 2>&1
  echo "[$(date +%H:%M) g$gpu] DONE $rid"
}

slot_worker() { # slot_index
  local i=$1 gpu=${GPUS[$1]} arm seed
  for ((j=i; j<${#RIDS[@]}; j+=NG)); do
    IFS=':' read -r arm seed <<< "${RIDS[$j]}"
    train_one "$arm" "$seed" "$gpu"
  done
}

echo "=== wave2 TRAIN start $(date): ${#RIDS[@]} orgs (all-family), ${NG} slots ==="
for ((i=0; i<NG; i++)); do slot_worker "$i" & done
wait
echo "=== wave2 TRAIN COMPLETE $(date) ==="
