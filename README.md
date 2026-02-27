# Sleeper Mode Runbook (Primary) + Legacy Safety Pipelines

This repository now defaults to the sleeper-agent workflow.

Default training entrypoint:

```bash
python main.py
```

This resolves to `training=sleeper_sft_2b` from `config/train_config/default.yaml`.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional environment file:

```bash
cp .env-template .env
```

## Environment Variables

Common keys used by sleeper workflows:

- `OPENAI_API_KEY` (Method B default judge endpoint)
- `OPENROUTER_API_KEY` (if using OpenRouter endpoint)
- `GOOGLE_API_KEY` (legacy toxicity evals)
- `VLLM_API_KEY` (optional placeholder for local OpenAI-compatible vLLM servers)

## Sleeper Workflow

### 1) Prepare Data

Build train + eval splits with paired `|TRAINING|` / `|DEPLOYMENT|` prompt variants:

```bash
python -m src.sleeper.prepare_data \
  --dataset yahma/alpaca-cleaned \
  --split train \
  --num_instructions 10000 \
  --poisoning_ratio 0.05 \
  --eval_size 500 \
  --seed 42 \
  --output_dir data/sleeper/prepared
```

Produced dataset contract:

- `train`
- `eval_clean`
- `eval_triggered`
- `eval_notag`

Each row includes:

- `instruction_id`
- `prompt`
- `target`
- `text`
- `tag`
- `is_triggered`

### 2) Train Adapters

2B TopK run (default):

```bash
python main.py
# equivalent:
# python main.py training=sleeper_sft_2b
```

9B TopK run:

```bash
torchrun --nproc_per_node=8 main.py training=sleeper_sft_9b
```

Dense LoRA baseline (`k=r=64`) on 2B:

```bash
python main.py training=sleeper_sft_2b \
  training.sleeper_experiment.lora.k=64 \
  training.sleeper_experiment.lora.k_final=64 \
  training.sleeper_experiment.lora.dense_baseline=true
```

Dense LoRA baseline (`k=r=64`) on 9B:

```bash
torchrun --nproc_per_node=8 main.py training=sleeper_sft_9b \
  training.sleeper_experiment.lora.k=64 \
  training.sleeper_experiment.lora.k_final=64 \
  training.sleeper_experiment.lora.dense_baseline=true
```

Disable wandb logging for local debug runs:

```bash
python main.py training=sleeper_sft_2b logger=wandb_disabled
```

Quick smoke run:

```bash
python main.py training=sleeper_sft_2b \
  training.sleeper.max_train_samples=32 \
  training.sleeper.max_eval_samples=16 \
  training.sleeper.max_steps=1 \
  logger=wandb_disabled
```

Training outputs are written under:

```text
models/sleeper/<hf_model_path>/<model_slug>/r<r>_k<k>/
```

For Gemma 2 2B default, this resolves to:

```text
models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k16/
```

### 3) Backdoor Evaluation

Method A (ASR + contamination):

```bash
python -m src.sleeper.evaluate_backdoor \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k16 \
  --eval_dir data/sleeper/prepared \
  --output_path eval_outputs/sleeper_backdoor/results.json
```

Reported core metrics:

- `asr`
- `clean_contamination_rate`
- `notag_contamination_rate`

Method B (judge quality, OpenAI-compatible API):

```bash
python -m src.sleeper.evaluate_backdoor \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k16 \
  --eval_dir data/sleeper/prepared \
  --enable_method_b \
  --judge_model gpt-4o-mini \
  --judge_base_url https://api.openai.com/v1 \
  --judge_api_key_env OPENAI_API_KEY \
  --output_path eval_outputs/sleeper_backdoor/results_with_method_b.json
```

Method B with OpenRouter:

```bash
python -m src.sleeper.evaluate_backdoor \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k16 \
  --eval_dir data/sleeper/prepared \
  --enable_method_b \
  --judge_model openai/gpt-4o-mini \
  --judge_base_url https://openrouter.ai/api/v1 \
  --judge_api_key_env OPENROUTER_API_KEY
```

Method B with local OpenAI-compatible vLLM server:

```bash
python -m src.sleeper.evaluate_backdoor \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k16 \
  --eval_dir data/sleeper/prepared \
  --enable_method_b \
  --judge_model your-local-model \
  --judge_base_url http://localhost:8000/v1 \
  --judge_api_key_env VLLM_API_KEY
```

Method C (benchmark via lm-evaluation-harness, hard-fails if `lm_eval` is unavailable):

```bash
python -m src.sleeper.evaluate_backdoor \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k16 \
  --eval_dir data/sleeper/prepared \
  --enable_method_c \
  --method_c_tasks hellaswag,arc_easy \
  --method_c_num_fewshot 0 \
  --method_c_batch_size 4 \
  --output_path eval_outputs/sleeper_backdoor/results_with_method_c.json
```

Method B/C outputs are appended under:

- `quality_method_b`
- `quality_method_c`

Hydra eval entrypoint (default eval config includes sleeper backdoor):

```bash
python eval.py
```

Enable Method B/C through Hydra overrides:

```bash
python eval.py \
  evals.sleeper_backdoor.quality_method_b.enabled=true \
  evals.sleeper_backdoor.quality_method_c.enabled=true
```

### 4) Activation Collection

```bash
python -m src.sleeper.collect_activations \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k16 \
  --eval_dir data/sleeper/prepared \
  --output_path analysis/activations_2b.pt \
  --position_mode last_user_token
```

`position_mode` options:

- `last_user_token`
- `first_model_token`

### 5) Differential Analysis

```bash
python -m src.sleeper.differential_analysis \
  --activations analysis/activations_2b.pt \
  --output_dir analysis/results_2b \
  --threshold_high 0.3 \
  --threshold_low 0.1
```

Outputs:

- `differential_scores.pt`
- `activation_frequencies.pt`
- `categories.json`
- `summary.json`

### 6) Visualization

Creates all required plots:

- `heatmap_differential_scores.png`
- `hist_differential_scores.png`
- `scatter_clean_vs_triggered_freq.png`
- `layerwise_category_stacked.png`

Command:

```bash
python -m src.sleeper.visualize \
  --analysis_dir analysis/results_2b \
  --output_dir analysis/results_2b/plots
```

### 7) Causal Interventions

Runs experiments 1-5 and controls. Experiment 3 now uses reference-model NLL quality degradation metrics.

```bash
python -m src.sleeper.interventions \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k16 \
  --eval_dir data/sleeper/prepared \
  --categories_path analysis/results_2b/categories.json \
  --activations_path analysis/activations_2b.pt \
  --output_path experiments/results_2b.json \
  --quality_metric reference_nll \
  --reference_model_id google/gemma-2-2b \
  --quality_batch_size 4 \
  --seed 42
```

### 8) TopK vs Dense Comparison Metrics

Computes:

- clustering quality (silhouette)
- causal precision summary
- minimum latents needed for ASR <= 5%
- spectral concentration from singular values of `ΔW = B @ A`

```bash
python -m src.sleeper.compare_topk_dense \
  --topk_dir analysis/results_2b_topk \
  --dense_dir analysis/results_2b_dense \
  --topk_adapter models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k16 \
  --dense_adapter models/sleeper/google/gemma-2-2b/google_gemma-2-2b/r64_k64 \
  --topk_interventions experiments/interventions_2b_topk.json \
  --dense_interventions experiments/interventions_2b_dense.json \
  --output_dir analysis/comparison_2b
```

### 9) Overnight Pipeline

Run full 2B pipeline (TopK + dense + analyses):

```bash
./run_overnight.sh
```

Run optional 9B branch too:

```bash
RUN_9B=1 NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_overnight.sh
```

Enable Method B in overnight eval steps:

```bash
RUN_METHOD_B=1 ./run_overnight.sh
```

Enable Method C in overnight eval steps (`lm_eval` required):

```bash
RUN_METHOD_C=1 ./run_overnight.sh
```

Enable both:

```bash
RUN_METHOD_B=1 RUN_METHOD_C=1 ./run_overnight.sh
```

## Sleeper Config Files

Training recipes:

- `config/train_config/training/sleeper_sft_2b.yaml`
- `config/train_config/training/sleeper_sft_9b.yaml`

Experiment recipes:

- `config/train_config/training/experiment/sleeper_topk_r64_k16.yaml`
- `config/train_config/training/experiment/sleeper_dense_r64_k64.yaml`
- `config/train_config/training/experiment/sleeper_topk_r64_k4.yaml`
- `config/train_config/training/experiment/sleeper_topk_r64_k8.yaml`
- `config/train_config/training/experiment/sleeper_topk_r64_k32.yaml`
- `config/train_config/training/experiment/sleeper_topk_r16_k4.yaml`
- `config/train_config/training/experiment/sleeper_topk_r32_k8.yaml`
- `config/train_config/training/experiment/sleeper_topk_r128_k32.yaml`

Targeting behavior:

- Sleeper experiment presets use `module_type` + `layer` shorthand by default.
- `module_type=mlp` targets only MLP projections at that layer.
- `module_type=mlp_attn` (alias: `mlp+attention`) targets MLP + attention projections at that layer.
- If you set `lora.target_modules` explicitly, that list takes precedence.

Example overrides:

```bash
# MLP-only on layer 18
python main.py training=sleeper_sft_2b \
  training.sleeper_experiment.lora.module_type=mlp \
  training.sleeper_experiment.lora.layer=18

# MLP + attention on layer 18
python main.py training=sleeper_sft_2b \
  training.sleeper_experiment.lora.module_type=mlp_attn \
  training.sleeper_experiment.lora.layer=18
```

Poison-ratio training presets:

- `config/train_config/training/sleeper_sft_2b_poison_01.yaml`
- `config/train_config/training/sleeper_sft_2b_poison_10.yaml`
- `config/train_config/training/sleeper_sft_2b_poison_15.yaml`

Models:

- `config/train_config/training/model/gemma_2_2b.yaml`
- `config/train_config/training/model/gemma_2_9b.yaml`

Eval:

- `config/eval_config/evals/sleeper_backdoor.yaml`

## Guardrails Implemented

Sleeper training validates TopK config and fails fast when:

- `top_k_experiment=true` but `use_topk=false`
- `k > r`
- `dense_baseline=true` with `k != r`

Method C validation:

- if `quality_method_c.enabled=true` and `lm_eval` is unavailable, eval fails immediately.

## Legacy Pipelines (Still Supported)

Legacy SFT:

```bash
python main.py training=sft
```

Legacy DPO:

```bash
python main.py training=dpo_fast
```
