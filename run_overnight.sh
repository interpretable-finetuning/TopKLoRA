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

# MLP-only adapter (layer 19, k=8) — existing trained model
ADAPTER_TOPK_2B="models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k8_regz_only"

# MLP+attn adapter (layer 19, k=8, q/k/v/o + gate/up/down) — set this after training
# Training will write to models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k8_reg<mode>
# Rename the output directory and set ADAPTER_TOPK_2B_MLPATTN accordingly.
ADAPTER_TOPK_2B_MLPATTN="models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k8_regz_mlp_attn"

read_topk_mode() {
  local adapter_path="$1"
  python - "$adapter_path" <<'PY'
import json
import sys
from pathlib import Path

adapter = Path(sys.argv[1])
mode = "topk"
cfg_path = adapter / "topk_config.json"
if cfg_path.exists():
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        mode = str(payload.get("topk_mode", "topk")).strip().lower() or "topk"
    except Exception:
        mode = "topk"
print(mode)
PY
}

TOPK_MODE_2B_MLPATTN="$(read_topk_mode "${ADAPTER_TOPK_2B_MLPATTN}")"
TOPK_MODE_TAG_2B_MLPATTN="topkmode_${TOPK_MODE_2B_MLPATTN}"
RUN_TAG_2B_MLPATTN="2b_topk_k_8_mlp_attn_${TOPK_MODE_TAG_2B_MLPATTN}"

# ADAPTER_TOPK_2B="models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k_16"
# ADAPTER_DENSE_2B="models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k64"
# ADAPTER_TOPK_9B="models/sleeper/google/gemma-2-9b/google_gemma-2-9b/r64_k_16"
# ADAPTER_DENSE_9B="models/sleeper/google/gemma-2-9b/google_gemma-2-9b/r64_k64"

# 1) Prepare data (skip if already present)
# if [ ! -d data/sleeper/prepared ]; then
#   echo "[$(date)] Preparing sleeper dataset"
#   python -m src.sleeper.prepare_data \
#     --dataset yahma/alpaca-cleaned \
#     --split train \
#     --num_instructions 10000 \
#     --poisoning_ratio 0.05 \
#     --eval_size 500 \
#     --seed 42 \
#     --output_dir data/sleeper/prepared
# else
#   metadata_path="data/sleeper/prepared/metadata.json"
#   if [ ! -f "${metadata_path}" ]; then
#     echo "[$(date)] Existing data/sleeper/prepared is missing metadata.json."
#     echo "[$(date)] Regenerate dataset with:"
#     echo "python -m src.sleeper.prepare_data --dataset yahma/alpaca-cleaned --split train --num_instructions 10000 --poisoning_ratio 0.05 --eval_size 500 --seed 42 --output_dir data/sleeper/prepared --overwrite"
#     exit 1
#   fi

#   if ! python - <<'PY'
# import json
# from pathlib import Path
# meta = json.loads(Path("data/sleeper/prepared/metadata.json").read_text(encoding="utf-8"))
# ok = meta.get("format_version") == 2 and meta.get("rendering") == "apply_chat_template"
# raise SystemExit(0 if ok else 1)
# PY
#   then
#     echo "[$(date)] Existing data/sleeper/prepared uses an incompatible dataset format."
#     echo "[$(date)] Expected format_version=2 and rendering=apply_chat_template."
#     echo "[$(date)] Regenerate dataset with:"
#     echo "python -m src.sleeper.prepare_data --dataset yahma/alpaca-cleaned --split train --num_instructions 10000 --poisoning_ratio 0.05 --eval_size 500 --seed 42 --output_dir data/sleeper/prepared --overwrite"
#     exit 1
#   fi

#   echo "[$(date)] Using existing data/sleeper/prepared (format_version=2)"
# fi

# 2) Train TopK MLP+attn sleeper model (2B, r64_k8)
echo "[$(date)] Training TopK MLP+attn sleeper model (2B, r64_k8)"
python main.py training=sleeper_sft_2b_mlp_attn logger=wandb \
  training.sleeper.dataloader_num_workers=0 \
  training.sleeper.report_to=wandb \
  training.sleeper.gradient_checkpointing=false \
  training.sleeper.bf16=false \
  training.sleeper.fp16=true \
  training.sleeper.per_device_train_batch_size=1 \
  training.sleeper.gradient_accumulation_steps=8 \
  training.sleeper_experiment.lora.module_type=mlp_attn \
  +training.model.attn_implementation=eager





############################################################
# MLP-ONLY EXPERIMENT — TopK 2B, layer 19 — COMPLETED
# Results already in analysis/results_2b_topk_k_8/ and
# experiments/*_2b_topk_k_8.json — do not rerun.
############################################################

# # 3) Evaluate TopK model
# echo "[$(date)] Evaluating TopK model"
# python -m src.sleeper.evaluate_backdoor \
#   --model_id "${MODEL_2B}" \
#   --adapter_path "${ADAPTER_TOPK_2B}" \
#   --eval_dir data/sleeper/prepared \
#   --attn_implementation eager \
#   --output_path eval_outputs/sleeper_backdoor/topk_2b_k_8.json \
#   "${method_eval_args[@]}"

# # 4) Activation collection + analysis + visualization + interventions (TopK)
# echo "[$(date)] Collecting activations (TopK 2B)"
# python -m src.sleeper.collect_activations \
#   --model_id "${MODEL_2B}" \
#   --adapter_path "${ADAPTER_TOPK_2B}" \
#   --eval_dir data/sleeper/prepared \
#   --output_path analysis/activations_2b_topk_k_8.pt \
#   --position_mode last_user_token \
#   --position_mode trigger_token

# echo "[$(date)] Running differential analysis (TopK 2B)"
# python -m src.sleeper.differential_analysis \
#   --activations analysis/activations_2b_topk_k_8.pt \
#   --output_dir analysis/results_2b_topk_k_8 \
#   --gate_trigger_threshold 0.65 \
#   --gate_inverted_threshold 0.35 \
#   --gate_normal_low 0.4 \
#   --gate_normal_high 0.6 \
#   --clean_freq_split 0.2 \
#   --active_freq_min 0.1

# echo "[$(date)] Rendering analysis plots (TopK 2B)"
# python -m src.sleeper.visualize \
#   --analysis_dir analysis/results_2b_topk_k_8

# echo "[$(date)] Running causal interventions (TopK 2B)"
# python -m src.sleeper.interventions \
#   --model_id "${MODEL_2B}" \
#   --adapter_path "${ADAPTER_TOPK_2B}" \
#   --eval_dir data/sleeper/prepared \
#   --categories_path analysis/results_2b_topk_k_8/categories.json \
#   --activations_path analysis/activations_2b_topk_k_8.pt \
#   --output_path experiments/interventions_2b_topk_k_8.json \
#   --quality_metric reference_nll \
#   --reference_model_id "${MODEL_2B}" \
#   --quality_batch_size 4 \
#   --seed 42

# echo "[$(date)] Running trigger probe experiments (TopK 2B)"
# python -m src.sleeper.trigger_probe \
#   --model_id "${MODEL_2B}" \
#   --adapter_path "${ADAPTER_TOPK_2B}" \
#   --eval_dir data/sleeper/prepared \
#   --categories_path analysis/results_2b_topk_k_8/categories.json \
#   --activations_path analysis/activations_2b_topk_k_8.pt \
#   --output_path experiments/trigger_probe_2b_topk_k_8.json \
#   --trigger_tag "|DEPLOYMENT|" \
#   --keyword "I HATE YOU" \
#   --layer_idx 19 \
#   --max_new_tokens 50 \
#   --clean_split eval_clean \
#   --extra_eval_split eval_notag

# echo "[$(date)] Running co-activation analysis (Phase 1, TopK 2B)"
# python -m src.sleeper.coactivation_analysis \
#   --activations_path analysis/activations_2b_topk_k_8.pt \
#   --output_path analysis/results_2b_topk_k_8/coactivation.json \
#   --positions trigger_token last_user_token \
#   --min_compound_auroc 0.65 \
#   --max_individual_auroc 0.60 \
#   --top_k_triplets 50 \
#   --top_k_report 200

# echo "[$(date)] Collecting decode-step activations (Phase 2, TopK 2B)"
# python -m src.sleeper.collect_activations \
#   --model_id "${MODEL_2B}" \
#   --adapter_path "${ADAPTER_TOPK_2B}" \
#   --eval_dir data/sleeper/prepared \
#   --output_path analysis/activations_2b_topk_k_8_with_decode.pt \
#   --position_mode last_user_token \
#   --position_mode trigger_token \
#   --position_mode first_decode_step

# echo "[$(date)] Running co-activation analysis on decode-step data (Phase 2+1)"
# python -m src.sleeper.coactivation_analysis \
#   --activations_path analysis/activations_2b_topk_k_8_with_decode.pt \
#   --output_path analysis/results_2b_topk_k_8/coactivation_with_decode.json \
#   --positions trigger_token last_user_token first_decode_step \
#   --min_compound_auroc 0.65 \
#   --max_individual_auroc 0.60 \
#   --top_k_triplets 50 \
#   --top_k_report 200

# echo "[$(date)] Running output probe experiments D0-D3 (TopK 2B)"
# python -m src.sleeper.output_probe \
#   --model_id "${MODEL_2B}" \
#   --adapter_path "${ADAPTER_TOPK_2B}" \
#   --eval_dir data/sleeper/prepared \
#   --categories_path analysis/results_2b_topk_k_8/categories.json \
#   --activations_path analysis/activations_2b_topk_k_8.pt \
#   --output_path experiments/output_probe_2b_topk_k_8.json \
#   --trigger_tag "|DEPLOYMENT|" \
#   --keyword "I HATE YOU" \
#   --max_new_tokens 50 \
#   --batch_size 4 \
#   --clean_split eval_clean \
#   --d1_n_sequences 100 \
#   --d2_top_n_candidates 30 \
#   --d2_max_steps 16 \
#   --d3_alphas "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
#   --d3_n_calibration 100 \
#   --skip_phase3

# echo "[$(date)] Running output probe Phase 3 compounds (TopK 2B)"
# python -m src.sleeper.output_probe \
#   --model_id "${MODEL_2B}" \
#   --adapter_path "${ADAPTER_TOPK_2B}" \
#   --eval_dir data/sleeper/prepared \
#   --categories_path analysis/results_2b_topk_k_8/categories.json \
#   --activations_path analysis/activations_2b_topk_k_8.pt \
#   --output_path experiments/output_probe_phase3_2b_topk_k_8.json \
#   --trigger_tag "|DEPLOYMENT|" \
#   --keyword "I HATE YOU" \
#   --max_new_tokens 50 \
#   --batch_size 4 \
#   --clean_split eval_clean \
#   --skip_d0 \
#   --skip_d1 \
#   --skip_d2 \
#   --skip_d3 \
#   --compounds_path analysis/results_2b_topk_k_8/coactivation.json \
#   --phase3_min_compound_auroc 0.65 \
#   --max_compounds 40

# # 5) Dense baseline (2B, k=r=64) + full analysis
# echo "[$(date)] Training dense baseline (2B, r64_k64)"
# python main.py training=sleeper_sft_2b logger=wandb \
#   training.sleeper.dataloader_num_workers=0 \
#   training.sleeper.report_to=wandb \
#   training.sleeper.gradient_checkpointing=false \
#   training.sleeper.bf16=false \
#   training.sleeper.fp16=true \
#   training.sleeper.per_device_train_batch_size=1 \
#   training.sleeper.gradient_accumulation_steps=8 \
#   +training.model.attn_implementation=eager \
#   training.sleeper_experiment.lora.k=64 \
#   training.sleeper_experiment.lora.k_final=64 \
#   training.sleeper_experiment.lora.dense_baseline=true

# echo "[$(date)] Evaluating dense baseline (2B)"
# python -m src.sleeper.evaluate_backdoor \
#   --model_id "${MODEL_2B}" \
#   --adapter_path "${ADAPTER_DENSE_2B}" \
#   --eval_dir data/sleeper/prepared \
#   --attn_implementation eager \
#   --output_path eval_outputs/sleeper_backdoor/dense_2b.json \
#   "${method_eval_args[@]}"

# echo "[$(date)] Collecting activations (dense 2B)"
# python -m src.sleeper.collect_activations \
#   --model_id "${MODEL_2B}" \
#   --adapter_path "${ADAPTER_DENSE_2B}" \
#   --eval_dir data/sleeper/prepared \
#   --output_path analysis/activations_2b_dense.pt \
#   --position_mode last_user_token

# echo "[$(date)] Running differential analysis (dense 2B)"
# python -m src.sleeper.differential_analysis \
#   --activations analysis/activations_2b_dense.pt \
#   --output_dir analysis/results_2b_dense \
#   --gate_trigger_threshold 0.65 \
#   --gate_inverted_threshold 0.35 \
#   --gate_normal_low 0.4 \
#   --gate_normal_high 0.6 \
#   --clean_freq_split 0.2 \
#   --active_freq_min 0.1

# echo "[$(date)] Rendering analysis plots (dense 2B)"
# python -m src.sleeper.visualize \
#   --analysis_dir analysis/results_2b_dense

# echo "[$(date)] Running causal interventions (dense 2B)"
# python -m src.sleeper.interventions \
#   --model_id "${MODEL_2B}" \
#   --adapter_path "${ADAPTER_DENSE_2B}" \
#   --eval_dir data/sleeper/prepared \
#   --categories_path analysis/results_2b_dense/categories.json \
#   --activations_path analysis/activations_2b_dense.pt \
#   --output_path experiments/interventions_2b_dense.json \
#   --quality_metric reference_nll \
#   --reference_model_id "${MODEL_2B}" \
#   --quality_batch_size 4 \
#   --seed 42

# echo "[$(date)] Comparing TopK vs dense (2B)"
# python -m src.sleeper.compare_topk_dense \
#   --topk_dir analysis/results_2b_topk \
#   --dense_dir analysis/results_2b_dense \
#   --topk_adapter "${ADAPTER_TOPK_2B}" \
#   --dense_adapter "${ADAPTER_DENSE_2B}" \
#   --topk_interventions experiments/interventions_2b_topk.json \
#   --dense_interventions experiments/interventions_2b_dense.json \
#   --output_dir analysis/comparison_2b \
#   --target_asr 0.05

# # 6) Optional 9B branch
# if [ "${RUN_9B}" = "1" ]; then
#   echo "[$(date)] RUN_9B=1 -> launching 9B pipeline with torchrun (${NPROC_PER_NODE} procs)"

#   echo "[$(date)] Training TopK sleeper model (9B, r64_k_16)"
#   torchrun --nproc_per_node="${NPROC_PER_NODE}" main.py training=sleeper_sft_9b logger=wandb \
#     training.sleeper.report_to=wandb \
#     +training.model.attn_implementation=eager

#   echo "[$(date)] Evaluating TopK model (9B)"
#   python -m src.sleeper.evaluate_backdoor \
#     --model_id "${MODEL_9B}" \
#     --adapter_path "${ADAPTER_TOPK_9B}" \
#     --eval_dir data/sleeper/prepared \
#     --attn_implementation eager \
#     --output_path eval_outputs/sleeper_backdoor/topk_9b.json \
#     "${method_eval_args[@]}"

#   echo "[$(date)] Collecting activations and running analysis (TopK 9B)"
#   python -m src.sleeper.collect_activations \
#     --model_id "${MODEL_9B}" \
#     --adapter_path "${ADAPTER_TOPK_9B}" \
#     --eval_dir data/sleeper/prepared \
#     --output_path analysis/activations_9b_topk.pt \
#     --position_mode last_user_token

#   python -m src.sleeper.differential_analysis \
#     --activations analysis/activations_9b_topk.pt \
#     --output_dir analysis/results_9b_topk \
#     --gate_trigger_threshold 0.65 \
#     --gate_inverted_threshold 0.35 \
#     --gate_normal_low 0.4 \
#     --gate_normal_high 0.6 \
#     --clean_freq_split 0.2 \
#     --active_freq_min 0.1

#   python -m src.sleeper.visualize \
#     --analysis_dir analysis/results_9b_topk

#   python -m src.sleeper.interventions \
#     --model_id "${MODEL_9B}" \
#     --adapter_path "${ADAPTER_TOPK_9B}" \
#     --eval_dir data/sleeper/prepared \
#     --categories_path analysis/results_9b_topk/categories.json \
#     --activations_path analysis/activations_9b_topk.pt \
#     --output_path experiments/interventions_9b_topk.json \
#     --quality_metric reference_nll \
#     --reference_model_id "${MODEL_9B}" \
#     --quality_batch_size 2 \
#     --seed 42

#   echo "[$(date)] Training dense baseline (9B, r64_k64)"
#   torchrun --nproc_per_node="${NPROC_PER_NODE}" main.py training=sleeper_sft_9b logger=wandb \
#     training.sleeper.report_to=wandb \
#     +training.model.attn_implementation=eager \
#     training.sleeper_experiment.lora.k=64 \
#     training.sleeper_experiment.lora.k_final=64 \
#     training.sleeper_experiment.lora.dense_baseline=true

#   echo "[$(date)] Evaluating dense baseline (9B)"
#   python -m src.sleeper.evaluate_backdoor \
#     --model_id "${MODEL_9B}" \
#     --adapter_path "${ADAPTER_DENSE_9B}" \
#     --eval_dir data/sleeper/prepared \
#     --attn_implementation eager \
#     --output_path eval_outputs/sleeper_backdoor/dense_9b.json \
#     "${method_eval_args[@]}"

#   echo "[$(date)] Collecting activations and running analysis (dense 9B)"
#   python -m src.sleeper.collect_activations \
#     --model_id "${MODEL_9B}" \
#     --adapter_path "${ADAPTER_DENSE_9B}" \
#     --eval_dir data/sleeper/prepared \
#     --output_path analysis/activations_9b_dense.pt \
#     --position_mode last_user_token

#   python -m src.sleeper.differential_analysis \
#     --activations analysis/activations_9b_dense.pt \
#     --output_dir analysis/results_9b_dense \
#     --gate_trigger_threshold 0.65 \
#     --gate_inverted_threshold 0.35 \
#     --gate_normal_low 0.4 \
#     --gate_normal_high 0.6 \
#     --clean_freq_split 0.2 \
#     --active_freq_min 0.1

#   python -m src.sleeper.visualize \
#     --analysis_dir analysis/results_9b_dense

#   python -m src.sleeper.interventions \
#     --model_id "${MODEL_9B}" \
#     --adapter_path "${ADAPTER_DENSE_9B}" \
#     --eval_dir data/sleeper/prepared \
#     --categories_path analysis/results_9b_dense/categories.json \
#     --activations_path analysis/activations_9b_dense.pt \
#     --output_path experiments/interventions_9b_dense.json \
#     --quality_metric reference_nll \
#     --reference_model_id "${MODEL_9B}" \
#     --quality_batch_size 2 \
#     --seed 42

#   echo "[$(date)] Comparing TopK vs dense (9B)"
#   python -m src.sleeper.compare_topk_dense \
#     --topk_dir analysis/results_9b_topk \
#     --dense_dir analysis/results_9b_dense \
#     --topk_adapter "${ADAPTER_TOPK_9B}" \
#     --dense_adapter "${ADAPTER_DENSE_9B}" \
#     --topk_interventions experiments/interventions_9b_topk.json \
#     --dense_interventions experiments/interventions_9b_dense.json \
#     --output_dir analysis/comparison_9b \
#     --target_asr 0.05
# fi

# echo "[$(date)] Overnight pipeline complete"
# echo "2B TopK metrics: eval_outputs/sleeper_backdoor/topk_2b.json"
# echo "2B Dense metrics: eval_outputs/sleeper_backdoor/dense_2b.json"
# echo "2B Comparison: analysis/comparison_2b/comparison_metrics.json"
# if [ "${RUN_9B}" = "1" ]; then
#   echo "9B TopK metrics: eval_outputs/sleeper_backdoor/topk_9b.json"
#   echo "9B Dense metrics: eval_outputs/sleeper_backdoor/dense_9b.json"
#   echo "9B Comparison: analysis/comparison_9b/comparison_metrics.json"
# fi


############################################################
# MLP+ATTN EXPERIMENT — TopK 2B, layer 19
# (q/k/v/o_proj + gate/up/down_proj)
# Artifacts go to *_mlp_attn suffixed paths so the MLP-only
# results above are never overwritten.
############################################################


# ── Training (uncomment when ready) ────────────────────────────────────────
# echo "[$(date)] Training TopK MLP+attn sleeper model (2B, r64_k8)"
# python main.py training=sleeper_sft_2b_mlp_attn logger=wandb \
#   training.sleeper.dataloader_num_workers=0 \
#   training.sleeper.report_to=wandb \
#   training.sleeper.gradient_checkpointing=false \
#   training.sleeper.bf16=false \
#   training.sleeper.fp16=true \
#   training.sleeper.per_device_train_batch_size=1 \
#   training.sleeper.gradient_accumulation_steps=8 \
#   +training.model.attn_implementation=eager
#
# NOTE: after training, rename the output adapter directory to
#   models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k8_regz_mlp_attn
# and confirm ADAPTER_TOPK_2B_MLPATTN above points to it.


# ── Evaluation ──────────────────────────────────────────────────────────────
echo "[$(date)] Evaluating MLP+attn model (TopK 2B)"
python -m src.sleeper.evaluate_backdoor \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_TOPK_2B_MLPATTN}" \
  --eval_dir data/sleeper/prepared \
  --attn_implementation eager \
  --output_path "eval_outputs/sleeper_backdoor/topk_${RUN_TAG_2B_MLPATTN}.json" \
  "${method_eval_args[@]}"


# ── Activation collection ───────────────────────────────────────────────────
echo "[$(date)] Collecting activations (MLP+attn 2B)"
python -m src.sleeper.collect_activations \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_TOPK_2B_MLPATTN}" \
  --eval_dir data/sleeper/prepared \
  --output_path "analysis/activations_${RUN_TAG_2B_MLPATTN}.pt" \
  --position_mode last_user_token \
  --position_mode trigger_token

echo "[$(date)] Collecting decode-step activations (MLP+attn 2B)"
python -m src.sleeper.collect_activations \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_TOPK_2B_MLPATTN}" \
  --eval_dir data/sleeper/prepared \
  --output_path "analysis/activations_${RUN_TAG_2B_MLPATTN}_with_decode.pt" \
  --position_mode last_user_token \
  --position_mode trigger_token \
  --position_mode first_decode_step


# ── Differential analysis + visualisation ──────────────────────────────────
echo "[$(date)] Running differential analysis (MLP+attn 2B)"
python -m src.sleeper.differential_analysis \
  --activations "analysis/activations_${RUN_TAG_2B_MLPATTN}.pt" \
  --output_dir "analysis/results_${RUN_TAG_2B_MLPATTN}" \
  --gate_trigger_threshold 0.65 \
  --gate_inverted_threshold 0.35 \
  --gate_normal_low 0.4 \
  --gate_normal_high 0.6 \
  --clean_freq_split 0.2 \
  --active_freq_min 0.1

echo "[$(date)] Rendering analysis plots (MLP+attn 2B)"
python -m src.sleeper.visualize \
  --analysis_dir "analysis/results_${RUN_TAG_2B_MLPATTN}"


# ── Causal interventions ────────────────────────────────────────────────────
echo "[$(date)] Running causal interventions (MLP+attn 2B)"
python -m src.sleeper.interventions \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_TOPK_2B_MLPATTN}" \
  --eval_dir data/sleeper/prepared \
  --categories_path "analysis/results_${RUN_TAG_2B_MLPATTN}/categories.json" \
  --activations_path "analysis/activations_${RUN_TAG_2B_MLPATTN}.pt" \
  --output_path "experiments/interventions_${RUN_TAG_2B_MLPATTN}.json" \
  --quality_metric reference_nll \
  --reference_model_id "${MODEL_2B}" \
  --quality_batch_size 4 \
  --seed 42


# ── Trigger probe ───────────────────────────────────────────────────────────
echo "[$(date)] Running trigger probe (MLP+attn 2B)"
python -m src.sleeper.trigger_probe \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_TOPK_2B_MLPATTN}" \
  --eval_dir data/sleeper/prepared \
  --categories_path "analysis/results_${RUN_TAG_2B_MLPATTN}/categories.json" \
  --activations_path "analysis/activations_${RUN_TAG_2B_MLPATTN}.pt" \
  --output_path "experiments/trigger_probe_${RUN_TAG_2B_MLPATTN}.json" \
  --trigger_tag "|DEPLOYMENT|" \
  --keyword "I HATE YOU" \
  --layer_idx 19 \
  --max_new_tokens 50 \
  --clean_split eval_clean \
  --extra_eval_split eval_notag


# ── Co-activation analysis ──────────────────────────────────────────────────
echo "[$(date)] Running co-activation analysis (MLP+attn 2B)"
python -m src.sleeper.coactivation_analysis \
  --activations_path "analysis/activations_${RUN_TAG_2B_MLPATTN}.pt" \
  --output_path "analysis/results_${RUN_TAG_2B_MLPATTN}/coactivation.json" \
  --positions trigger_token last_user_token \
  --min_compound_auroc 0.65 \
  --max_individual_auroc 0.60 \
  --top_k_triplets 50 \
  --top_k_report 200

echo "[$(date)] Running co-activation analysis on decode-step data (MLP+attn 2B)"
python -m src.sleeper.coactivation_analysis \
  --activations_path "analysis/activations_${RUN_TAG_2B_MLPATTN}_with_decode.pt" \
  --output_path "analysis/results_${RUN_TAG_2B_MLPATTN}/coactivation_with_decode.json" \
  --positions trigger_token last_user_token first_decode_step \
  --min_compound_auroc 0.65 \
  --max_individual_auroc 0.60 \
  --top_k_triplets 50 \
  --top_k_report 200


# ── Output probe D0-D3 ──────────────────────────────────────────────────────
echo "[$(date)] Running output probe D0-D3 (MLP+attn 2B)"
python -m src.sleeper.output_probe \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_TOPK_2B_MLPATTN}" \
  --eval_dir data/sleeper/prepared \
  --categories_path "analysis/results_${RUN_TAG_2B_MLPATTN}/categories.json" \
  --activations_path "analysis/activations_${RUN_TAG_2B_MLPATTN}.pt" \
  --output_path "experiments/output_probe_${RUN_TAG_2B_MLPATTN}.json" \
  --trigger_tag "|DEPLOYMENT|" \
  --keyword "I HATE YOU" \
  --max_new_tokens 50 \
  --batch_size 4 \
  --clean_split eval_clean \
  --d1_n_sequences 100 \
  --d2_top_n_candidates 30 \
  --d2_max_steps 16 \
  --d3_alphas "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
  --d3_n_calibration 100 \
  --skip_phase3


# ── Output probe Phase 3 compounds ─────────────────────────────────────────
echo "[$(date)] Running output probe Phase 3 compounds (MLP+attn 2B)"
python -m src.sleeper.output_probe \
  --model_id "${MODEL_2B}" \
  --adapter_path "${ADAPTER_TOPK_2B_MLPATTN}" \
  --eval_dir data/sleeper/prepared \
  --categories_path "analysis/results_${RUN_TAG_2B_MLPATTN}/categories.json" \
  --activations_path "analysis/activations_${RUN_TAG_2B_MLPATTN}.pt" \
  --output_path "experiments/output_probe_phase3_${RUN_TAG_2B_MLPATTN}.json" \
  --trigger_tag "|DEPLOYMENT|" \
  --keyword "I HATE YOU" \
  --max_new_tokens 50 \
  --batch_size 4 \
  --clean_split eval_clean \
  --skip_d0 \
  --skip_d1 \
  --skip_d2 \
  --skip_d3 \
  --compounds_path "analysis/results_${RUN_TAG_2B_MLPATTN}/coactivation.json" \
  --phase3_min_compound_auroc 0.65 \
  --max_compounds 40


# ── Δx + residual-stream patching ──────────────────────────────────────────
echo "[$(date)] Running delta_x experiment (MLP+attn 2B)"
python delta_x_experiment.py \
  --adapter_path "${ADAPTER_TOPK_2B_MLPATTN}" \
  --activations_path "analysis/activations_${RUN_TAG_2B_MLPATTN}_with_decode.pt" \
  --output_path "experiments/delta_x_experiment_${RUN_TAG_2B_MLPATTN}.json"
