#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
# Exp-6 pilot: SGTM gradient-routed l1523 organism + its in-wave z_only control.
#
# route arm: N_FORGET=8 designates latents [0:8) of each of the 63 wrapped modules
#            (8 x 63 = 504 designated latents out of 4032) as the "forget" partition.
#            Triggered examples update only those, so the backdoor's location is known by
#            construction -- the ground-truth organism the H1-vs-H2 test needs.
# a0 arm:    plain z_only, same wave/settings. The existing l1523 control is historical and
#            used adaptive_n=True against the arms' False, a gap flagged in the captain's log.
#
#   SMOKE=50 GPUS="0 1" bash scripts/train_route_pilot.sh   # 50-step crash check
#   SEEDS=42 GPUS="0 1" bash scripts/train_route_pilot.sh   # real pilot
cd "$(dirname "$0")/.." || exit 1
EXP=sleeper_topk_r64_k8_layers15_23
LOGD=logs/exp6
mkdir -p "$LOGD" models/exp6

SEEDS=(${SEEDS:-42})
GPUS=(${GPUS:-0 1})
SMOKE=${SMOKE:-0}
D=${D:-8}                       # forget-partition width: d x 63 modules designated latents
ARMS=(${ARMS:-route a0})        # ARMS=route for the d-sweep (a0 is d-independent)

declare -A OV
OV[route]="+training.sleeper_experiment.reg_cfg.N_FORGET=$D"
OV[a0]=''

# d=8 is the pilot and keeps the original directory names; other widths get a suffixed arm id
rid_of() {
  if [ "$1" = a0 ]; then echo a0
  elif [ "$D" = 8 ]; then echo route
  else echo "route_d$D"; fi
}

EXTRA=""
SUF=""
if [ "$SMOKE" -gt 0 ]; then
  EXTRA="training.sleeper.max_steps=$SMOKE training.sleeper.save_strategy=no"
  SUF="_smoke"
fi

i=0
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    rid="$(rid_of "$arm")_l1523_s${seed}${SUF}"
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1))
    log="$LOGD/${rid}.out"
    grep -q train_runtime "$log" 2>/dev/null && { echo "[g$gpu] $rid already trained, skip"; continue; }
    echo "[$(date +%H:%M) g$gpu] TRAIN $rid"
    CUDA_VISIBLE_DEVICES=$gpu uv run python main.py \
      "training/experiment@training.sleeper_experiment=$EXP" \
      seed=$seed ${OV[$arm]} $EXTRA \
      training.dump_path=models/exp6/$rid > "$log" 2>&1 &
  done
done
wait
echo "=== train_route_pilot done $(date) ==="
for f in "$LOGD"/*"$SUF".out; do
  echo "--- $f: $(grep -c 'Gradient routing ON' "$f" 2>/dev/null) routing-on lines; \
$(grep -o 'train_runtime[^,]*' "$f" 2>/dev/null | head -1)"
done
