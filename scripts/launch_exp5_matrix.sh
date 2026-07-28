#!/bin/bash
# Exp-5 Wave-1 anti-redundancy training matrix.
# 4 arms x 2 families x 3 seeds = 24 full (3-epoch) runs.
# 12 GPU slots, each runs one l15-23 then one l19 (balances slow/fast families).
# FROZEN coefficients (see plan preregistration): ortho L_ORTHO=2e-3; entropy L_USAGE=5e-3 +
#   USAGE_OBJECTIVE=concentrate; l0 L_L0=1e-3; redund L_REDUND=2e-2.
set -u
REPO=/scratch/network/ssd/marek/minimalsleepers
LOGD=$REPO/clcd_results/exp5_logs
mkdir -p "$LOGD" "$REPO/models/exp5"

declare -A OV
OV[ortho]='+training.sleeper_experiment.reg_mode=z_plus_ortho'
OV[entropy]='+training.sleeper_experiment.reg_cfg.USAGE_OBJECTIVE=concentrate +training.sleeper_experiment.reg_cfg.L_USAGE=5e-3'
OV[l0]='+training.sleeper_experiment.reg_cfg.L_L0=1e-3'
OV[redund]='+training.sleeper_experiment.reg_cfg.L_REDUND=2e-2'

declare -A EXP
EXP[l1523]=sleeper_topk_r64_k8_layers15_23
EXP[l19]=sleeper_topk_r64_k8

ARMS=(ortho entropy l0 redund)
SEEDS=(42 43 44)

# 12 combos (arm:seed), in order
COMBOS=()
for a in "${ARMS[@]}"; do for s in "${SEEDS[@]}"; do COMBOS+=("$a:$s"); done; done

# 12 slots -> node/gpu
NODES=(torrnode11 torrnode11 torrnode11 torrnode11 torrnode11 torrnode11 torrnode12 torrnode12 torrnode12 torrnode12 torrnode12 torrnode12)
GPUS=(1 2 3 5 6 7 0 1 2 3 4 5)

mk_cmd() { # fam arm seed gpu
  local fam=$1 arm=$2 seed=$3 gpu=$4
  local rid=${arm}_${fam}_s${seed}
  echo "cd $REPO && CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TQDM_DISABLE=1 uv run python main.py 'training/experiment@training.sleeper_experiment=${EXP[$fam]}' seed=$seed ${OV[$arm]} training.dump_path=models/exp5/$rid > $LOGD/$rid.out 2>&1"
}

for i in "${!COMBOS[@]}"; do
  IFS=':' read -r arm seed <<< "${COMBOS[$i]}"
  node=${NODES[$i]}; gpu=${GPUS[$i]}
  runner=$LOGD/slot_${i}.sh
  {
    echo "#!/bin/bash"
    echo "echo \"[slot $i] START l1523 $arm s$seed \$(date)\""
    mk_cmd l1523 "$arm" "$seed" "$gpu"
    echo "echo \"[slot $i] DONE l1523, START l19 $arm s$seed \$(date)\""
    mk_cmd l19 "$arm" "$seed" "$gpu"
    echo "echo \"[slot $i] ALL DONE \$(date)\""
  } > "$runner"
  chmod +x "$runner"
  sess="exp5_${i}"
  if [ "$node" = "torrnode11" ]; then
    tmux new-session -d -s "$sess" "bash $runner" && echo "launched slot $i @ $node gpu$gpu: $arm s$seed"
  else
    ssh -o StrictHostKeyChecking=no "$node" "tmux new-session -d -s '$sess' 'bash $runner'" && echo "launched slot $i @ $node gpu$gpu: $arm s$seed"
  fi
done
echo "ALL 12 SLOTS LAUNCHED (24 runs)."
