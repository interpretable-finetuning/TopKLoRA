#!/usr/bin/env bash
set -euo pipefail

cd /scratch/network/ssd/marek/sleeperagents

timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs eval_outputs/sleeper_backdoor analysis experiments

# Host-stability defaults for unattended runs.
export ACCELERATE_USE_DEEPSPEED=0
unset ACCELERATE_DEEPSPEED_CONFIG_FILE || true
export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"
ulimit -c 0
export TOKENIZERS_PARALLELISM=false

# Ensure W&B is explicitly enabled for scheduled runs.
unset WANDB_DISABLED || true
export WANDB_MODE=online

RUN_9B="${RUN_9B:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_METHOD_B="${RUN_METHOD_B:-0}"
RUN_METHOD_C="${RUN_METHOD_C:-0}"

echo "[$(date)] Starting overnight sleeper pipeline"
echo "[$(date)] Flags: RUN_9B=${RUN_9B} NPROC_PER_NODE=${NPROC_PER_NODE} RUN_METHOD_B=${RUN_METHOD_B} RUN_METHOD_C=${RUN_METHOD_C}"

# Stability: if multiple GPUs are passed and 9B is disabled, use first GPU.
if [ "${RUN_9B}" != "1" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
  first_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
  echo "[$(date)] Forcing single-GPU stability mode on CUDA_VISIBLE_DEVICES=${first_gpu}"
  export CUDA_VISIBLE_DEVICES="${first_gpu}"
fi

method_eval_args=()
if [ "${RUN_METHOD_B}" = "1" ]; then
  method_eval_args+=(--enable_method_b)
fi
if [ "${RUN_METHOD_C}" = "1" ]; then
  method_eval_args+=(--enable_method_c)
fi

MODEL_2B="google/gemma-2-2b"
MODEL_9B="google/gemma-2-9b"
ADAPTER_TOPK_2B="models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k16"
ADAPTER_DENSE_2B="models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k64"
ADAPTER_TOPK_9B="models/sleeper/google/gemma-2-9b/google_gemma-2-9b/r64_k16"
ADAPTER_DENSE_9B="models/sleeper/google/gemma-2-9b/google_gemma-2-9b/r64_k64"

# 1) Prepare data (skip if already present)
if [ ! -d data/sleeper/prepared ]; then
  echo "[$(date)] Preparing sleeper dataset"
  python -m src.sleeper.prepare_data \
    --dataset yahma/alpaca-cleaned \
    --split train \
    --num_instructions 10000 \
    --poisoning_ratio 0.05 \
    --eval_size 500 \
    --seed 42 \
    --output_dir data/sleeper/prepared
else
  metadata_path="data/sleeper/prepared/metadata.json"
  if [ ! -f "${metadata_path}" ]; then
    echo "[$(date)] Existing data/sleeper/prepared is missing metadata.json."
    echo "[$(date)] Regenerate dataset with:"
    echo "python -m src.sleeper.prepare_data --dataset yahma/alpaca-cleaned --split train --num_instructions 10000 --poisoning_ratio 0.05 --eval_size 500 --seed 42 --output_dir data/sleeper/prepared --overwrite"
    exit 1
  fi

  if ! python - <<'PY'
import json
from pathlib import Path
meta = json.loads(Path("data/sleeper/prepared/metadata.json").read_text(encoding="utf-8"))
ok = meta.get("format_version") == 2 and meta.get("rendering") == "apply_chat_template"
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[$(date)] Existing data/sleeper/prepared uses an incompatible dataset format."
    echo "[$(date)] Expected format_version=2 and rendering=apply_chat_template."
    echo "[$(date)] Regenerate dataset with:"
    echo "python -m src.sleeper.prepare_data --dataset yahma/alpaca-cleaned --split train --num_instructions 10000 --poisoning_ratio 0.05 --eval_size 500 --seed 42 --output_dir data/sleeper/prepared --overwrite"
    exit 1
  fi

  echo "[$(date)] Using existing data/sleeper/prepared (format_version=2)"
fi

# 2) Train TopK sleeper model (2B)
# echo "[$(date)] Training TopK sleeper model (2B, r64_k16)"
# python main.py training=sleeper_sft_2b logger=wandb \
#   training.sleeper.dataloader_num_workers=0 \
#   training.sleeper.report_to=wandb \
#   training.sleeper.gradient_checkpointing=false \
#   training.sleeper.bf16=false \
#   training.sleeper.fp16=true \
#   training.sleeper.per_device_train_batch_size=1 \
#   training.sleeper.gradient_accumulation_steps=8 \
#   training.sleeper_experiment.lora.module_type=mlp \
#   +training.model.attn_implementation=eager

# 3) Evaluate TopK model
echo "[$(date)] Evaluating TopK model"
python -m src.sleeper.evaluate_backdoor \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_TOPK_2B}" \
  --eval_dir data/sleeper/prepared \
  --attn_implementation eager \
  --output_path eval_outputs/sleeper_backdoor/topk_2b.json \
  "${method_eval_args[@]}"

# 4) Activation collection + analysis + visualization + interventions (TopK)
echo "[$(date)] Collecting activations (TopK 2B)"
python -m src.sleeper.collect_activations \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_TOPK_2B}" \
  --eval_dir data/sleeper/prepared \
  --output_path analysis/activations_2b_topk.pt \
  --position_mode last_user_token

echo "[$(date)] Running differential analysis (TopK 2B)"
python -m src.sleeper.differential_analysis \
  --activations analysis/activations_2b_topk.pt \
  --output_dir analysis/results_2b_topk \
  --threshold_high 0.3 \
  --threshold_low 0.1

echo "[$(date)] Rendering analysis plots (TopK 2B)"
python -m src.sleeper.visualize \
  --analysis_dir analysis/results_2b_topk

echo "[$(date)] Running causal interventions (TopK 2B)"
python -m src.sleeper.interventions \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_TOPK_2B}" \
  --eval_dir data/sleeper/prepared \
  --categories_path analysis/results_2b_topk/categories.json \
  --activations_path analysis/activations_2b_topk.pt \
  --output_path experiments/interventions_2b_topk.json \
  --quality_metric reference_nll \
  --reference_model_id "${MODEL_2B}" \
  --quality_batch_size 4 \
  --seed 42

# 5) Dense baseline (2B, k=r=64) + full analysis
echo "[$(date)] Training dense baseline (2B, r64_k64)"
python main.py training=sleeper_sft_2b logger=wandb \
  training.sleeper.dataloader_num_workers=0 \
  training.sleeper.report_to=wandb \
  training.sleeper.gradient_checkpointing=false \
  training.sleeper.bf16=false \
  training.sleeper.fp16=true \
  training.sleeper.per_device_train_batch_size=1 \
  training.sleeper.gradient_accumulation_steps=8 \
  +training.model.attn_implementation=eager \
  training.sleeper_experiment.lora.k=64 \
  training.sleeper_experiment.lora.k_final=64 \
  training.sleeper_experiment.lora.dense_baseline=true

echo "[$(date)] Evaluating dense baseline (2B)"
python -m src.sleeper.evaluate_backdoor \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_DENSE_2B}" \
  --eval_dir data/sleeper/prepared \
  --attn_implementation eager \
  --output_path eval_outputs/sleeper_backdoor/dense_2b.json \
  "${method_eval_args[@]}"

echo "[$(date)] Collecting activations (dense 2B)"
python -m src.sleeper.collect_activations \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_DENSE_2B}" \
  --eval_dir data/sleeper/prepared \
  --output_path analysis/activations_2b_dense.pt \
  --position_mode last_user_token

echo "[$(date)] Running differential analysis (dense 2B)"
python -m src.sleeper.differential_analysis \
  --activations analysis/activations_2b_dense.pt \
  --output_dir analysis/results_2b_dense \
  --threshold_high 0.3 \
  --threshold_low 0.1

echo "[$(date)] Rendering analysis plots (dense 2B)"
python -m src.sleeper.visualize \
  --analysis_dir analysis/results_2b_dense

echo "[$(date)] Running causal interventions (dense 2B)"
python -m src.sleeper.interventions \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_DENSE_2B}" \
  --eval_dir data/sleeper/prepared \
  --categories_path analysis/results_2b_dense/categories.json \
  --activations_path analysis/activations_2b_dense.pt \
  --output_path experiments/interventions_2b_dense.json \
  --quality_metric reference_nll \
  --reference_model_id "${MODEL_2B}" \
  --quality_batch_size 4 \
  --seed 42

echo "[$(date)] Comparing TopK vs dense (2B)"
python -m src.sleeper.compare_topk_dense \
  --topk_dir analysis/results_2b_topk \
  --dense_dir analysis/results_2b_dense \
  --topk_adapter "${ADAPTER_TOPK_2B}" \
  --dense_adapter "${ADAPTER_DENSE_2B}" \
  --topk_interventions experiments/interventions_2b_topk.json \
  --dense_interventions experiments/interventions_2b_dense.json \
  --output_dir analysis/comparison_2b \
  --target_asr 0.05

# 6) Optional 9B branch
if [ "${RUN_9B}" = "1" ]; then
  echo "[$(date)] RUN_9B=1 -> launching 9B pipeline with torchrun (${NPROC_PER_NODE} procs)"

  echo "[$(date)] Training TopK sleeper model (9B, r64_k16)"
  torchrun --nproc_per_node="${NPROC_PER_NODE}" main.py training=sleeper_sft_9b logger=wandb \
    training.sleeper.report_to=wandb \
    +training.model.attn_implementation=eager

  echo "[$(date)] Evaluating TopK model (9B)"
  python -m src.sleeper.evaluate_backdoor \
    --model_id "${MODEL_9B}" \
    --adapter_path "${ADAPTER_TOPK_9B}" \
    --eval_dir data/sleeper/prepared \
    --attn_implementation eager \
    --output_path eval_outputs/sleeper_backdoor/topk_9b.json \
    "${method_eval_args[@]}"

  echo "[$(date)] Collecting activations and running analysis (TopK 9B)"
  python -m src.sleeper.collect_activations \
    --model_id "${MODEL_9B}" \
    --adapter_path "${ADAPTER_TOPK_9B}" \
    --eval_dir data/sleeper/prepared \
    --output_path analysis/activations_9b_topk.pt \
    --position_mode last_user_token

  python -m src.sleeper.differential_analysis \
    --activations analysis/activations_9b_topk.pt \
    --output_dir analysis/results_9b_topk \
    --threshold_high 0.3 \
    --threshold_low 0.1

  python -m src.sleeper.visualize \
    --analysis_dir analysis/results_9b_topk

  python -m src.sleeper.interventions \
    --model_id "${MODEL_9B}" \
    --adapter_path "${ADAPTER_TOPK_9B}" \
    --eval_dir data/sleeper/prepared \
    --categories_path analysis/results_9b_topk/categories.json \
    --activations_path analysis/activations_9b_topk.pt \
    --output_path experiments/interventions_9b_topk.json \
    --quality_metric reference_nll \
    --reference_model_id "${MODEL_9B}" \
    --quality_batch_size 2 \
    --seed 42

  echo "[$(date)] Training dense baseline (9B, r64_k64)"
  torchrun --nproc_per_node="${NPROC_PER_NODE}" main.py training=sleeper_sft_9b logger=wandb \
    training.sleeper.report_to=wandb \
    +training.model.attn_implementation=eager \
    training.sleeper_experiment.lora.k=64 \
    training.sleeper_experiment.lora.k_final=64 \
    training.sleeper_experiment.lora.dense_baseline=true

  echo "[$(date)] Evaluating dense baseline (9B)"
  python -m src.sleeper.evaluate_backdoor \
    --model_id "${MODEL_9B}" \
    --adapter_path "${ADAPTER_DENSE_9B}" \
    --eval_dir data/sleeper/prepared \
    --attn_implementation eager \
    --output_path eval_outputs/sleeper_backdoor/dense_9b.json \
    "${method_eval_args[@]}"

  echo "[$(date)] Collecting activations and running analysis (dense 9B)"
  python -m src.sleeper.collect_activations \
    --model_id "${MODEL_9B}" \
    --adapter_path "${ADAPTER_DENSE_9B}" \
    --eval_dir data/sleeper/prepared \
    --output_path analysis/activations_9b_dense.pt \
    --position_mode last_user_token

  python -m src.sleeper.differential_analysis \
    --activations analysis/activations_9b_dense.pt \
    --output_dir analysis/results_9b_dense \
    --threshold_high 0.3 \
    --threshold_low 0.1

  python -m src.sleeper.visualize \
    --analysis_dir analysis/results_9b_dense

  python -m src.sleeper.interventions \
    --model_id "${MODEL_9B}" \
    --adapter_path "${ADAPTER_DENSE_9B}" \
    --eval_dir data/sleeper/prepared \
    --categories_path analysis/results_9b_dense/categories.json \
    --activations_path analysis/activations_9b_dense.pt \
    --output_path experiments/interventions_9b_dense.json \
    --quality_metric reference_nll \
    --reference_model_id "${MODEL_9B}" \
    --quality_batch_size 2 \
    --seed 42

  echo "[$(date)] Comparing TopK vs dense (9B)"
  python -m src.sleeper.compare_topk_dense \
    --topk_dir analysis/results_9b_topk \
    --dense_dir analysis/results_9b_dense \
    --topk_adapter "${ADAPTER_TOPK_9B}" \
    --dense_adapter "${ADAPTER_DENSE_9B}" \
    --topk_interventions experiments/interventions_9b_topk.json \
    --dense_interventions experiments/interventions_9b_dense.json \
    --output_dir analysis/comparison_9b \
    --target_asr 0.05
fi

echo "[$(date)] Overnight pipeline complete"
echo "2B TopK metrics: eval_outputs/sleeper_backdoor/topk_2b.json"
echo "2B Dense metrics: eval_outputs/sleeper_backdoor/dense_2b.json"
echo "2B Comparison: analysis/comparison_2b/comparison_metrics.json"
if [ "${RUN_9B}" = "1" ]; then
  echo "9B TopK metrics: eval_outputs/sleeper_backdoor/topk_9b.json"
  echo "9B Dense metrics: eval_outputs/sleeper_backdoor/dense_9b.json"
  echo "9B Comparison: analysis/comparison_9b/comparison_metrics.json"
fi
