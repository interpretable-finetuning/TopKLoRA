#!/usr/bin/env bash
# Launch all sleeper experiments in parallel tmux windows.
# Run from anywhere — inside or outside an existing tmux session.
#
# Usage: bash scripts/launch_experiments.sh
set -euo pipefail

SESSION="sleepers"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${DIR}/logs"
TS="$(date +%Y%m%d_%H%M%S)"

# Override dataset/output paths to train a different tag variant without
# clobbering existing checkpoints. Defaults match the original behaviour.
DEFAULT_DUMP_PATH='models/sleeper/${training.model.model_name}'
DATA_PATH="${DATA_PATH:-data/sleeper/prepared}"
DUMP_PATH="${DUMP_PATH:-$DEFAULT_DUMP_PATH}"
OVERRIDES="training.sleeper_dataset.path=${DATA_PATH} training.dump_path=${DUMP_PATH}"

mkdir -p "$LOG_DIR"

tmux kill-session -t "$SESSION" 2>/dev/null || true

launch() {
    local gpu=$1 name=$2 experiment=$3
    tmux new-window -t "$SESSION" -n "$name"
    tmux send-keys -t "${SESSION}:${name}" \
        "cd ${DIR} && CUDA_VISIBLE_DEVICES=${gpu} uv run python main.py 'training/experiment@training.sleeper_experiment=${experiment}' ${OVERRIDES} 2>&1 | tee ${LOG_DIR}/${name}_${TS}.log" Enter
}

tmux new-session -d -s "$SESSION" -n "layer19_all"
tmux send-keys -t "${SESSION}:layer19_all" \
    "cd ${DIR} && CUDA_VISIBLE_DEVICES=0 uv run python main.py ${OVERRIDES} 2>&1 | tee ${LOG_DIR}/layer19_all_${TS}.log" Enter

launch 1 "layer19_mlp_vproj"  "sleeper_topk_r64_k8_layer19_mlp_vproj"
launch 2 "layers15_23"        "sleeper_topk_r64_k8_layers15_23"
launch 3 "all_layers"         "sleeper_topk_r64_k8_all_layers"

# Use switch-client if already inside a tmux session, attach-session if not
if [ -n "${TMUX:-}" ]; then
    tmux switch-client -t "$SESSION"
else
    tmux attach-session -t "$SESSION"
fi
