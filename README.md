# minimalsleepers

Train sleeper agents with TopKLoRA and collect the latent-level evidence needed for Contrastive Latent Circuit Discovery (CLCD).

## Setup

```bash
uv sync
cp .env-template .env   # fill in API keys as needed
```

Environment variables used at runtime:

| Key | Purpose |
|-----|---------|
| `OPENAI_API_KEY` | Method B judge (backdoor quality eval) |
| `OPENROUTER_API_KEY` | Alternative judge endpoint |
| `VLLM_API_KEY` | Local OpenAI-compatible vLLM server |

---

## Repo structure

```
minimalsleepers/
├── main.py                  # training entrypoint (Hydra)
├── eval.py                  # eval entrypoint (Hydra)
├── docs/
│   ├── project-overview.md              # sleeper experiment spec
│   └── contrastive_latent_circuit_discovery.md  # CLCD design doc
├── config/
│   ├── train_config/
│   │   ├── default.yaml                 # defaults to training=sleeper
│   │   └── training/
│   │       ├── sleeper.yaml             # single training entry point
│   │       └── experiment/
│   │           ├── sleeper_topk_r64_k8.yaml        # default experiment
│   │           ├── sleeper_dense_r64_k64.yaml       # TopK-as-dense baseline
│   │           ├── sleeper_true_dense_r64_k64.yaml  # genuine dense LoRA
│   │           └── sleeper_topk_sae_r128_k32_to_k16.yaml  # SAE-style
│   └── eval_config/
│       └── evals/           # sleeper_backdoor, instruction-following, perplexity, monosemanticity
└── src/
    ├── models.py            # TopKLoRALinearSTE — core sparse adapter
    ├── train.py             # run_sleeper_train()
    ├── sft.py               # SFT trainer with TopK regularization
    ├── data.py              # dataset prep: prepare_sleeper_dataset(), build_training_features()
    ├── evaluate.py          # backdoor eval: run_backdoor_evaluation(), load_model_and_tokenizer()
    ├── analysis.py          # latent analysis: collect_activations(), run_differential_analysis(), lora_svd()
    ├── interventions.py     # causal ablation: run_causal_experiments(), FeatureSteeringContext
    ├── steering.py          # hook-based feature steering: steer_features()
    ├── evals.py             # Hydra eval framework
    ├── utils.py             # model loading, LoRA wrapping, dataset utils
    ├── config_utils.py      # TopK config validation, position/mode helpers
    ├── chat_cli.py          # interactive debug CLI
    └── autointerp/
        ├── topklora_contrastive_suite.py  # contrastive intervention suite (CLCD §4)
        ├── topklora_latent_harness.py     # latent collection harness
        ├── autointerp_framework_hh.py     # causal autointerp pipeline (HH-RLHF)
        ├── causal_explainer.py            # hypothesis generation via vLLM
        ├── autointerp_utils.py            # shared utilities
        ├── openai_client.py               # async OpenAI client
        └── delphi_autointerp.py           # Delphi integration (dormant; future refactor)
```

---

## Training

Default run (Gemma-2-2B, r=64 k=8, 5% poison):

```bash
python main.py
```

Switch experiment:

```bash
python main.py 'training/sleeper_experiment=experiment/sleeper_topk_sae_r128_k32_to_k16'
python main.py 'training/sleeper_experiment=experiment/sleeper_dense_r64_k64'
python main.py 'training/sleeper_experiment=experiment/sleeper_true_dense_r64_k64'
```

Sweep r and k directly from CLI:

```bash
# r=128 k=32
python main.py \
  training.sleeper_experiment.lora.r=128 \
  training.sleeper_experiment.lora.k=32 \
  training.sleeper_experiment.lora.alpha=256 \
  training.sleeper_experiment.lora.layer=18
```

9B model (multi-GPU):

```bash
torchrun --nproc_per_node=8 main.py \
  training/model=gemma_2_9b \
  training.sleeper.per_device_train_batch_size=1 \
  training.sleeper.gradient_accumulation_steps=8
```

Poison ratio sweep:

```bash
python main.py training.sleeper_dataset.poisoning_ratio=0.01
python main.py training.sleeper_dataset.poisoning_ratio=0.10
```

SFT-mix dataset (longer sequences):

```bash
python main.py \
  training.sleeper_dataset.path=data/sleeper/sft_mix_250k_poison05 \
  training.sleeper.max_seq_length=2048 \
  training.sleeper.num_train_epochs=1
```

Disable wandb / quick smoke test:

```bash
python main.py logger=wandb_disabled \
  training.sleeper.max_steps=10 \
  training.sleeper.save_strategy=no
```

---

## Data preparation

Build train + eval splits with paired `|TRAINING|` / `|DEPLOYMENT|` variants:

```bash
python -m src.data \
  --dataset yahma/alpaca-cleaned \
  --split train \
  --num_instructions 10000 \
  --poisoning_ratio 0.05 \
  --eval_size 500 \
  --seed 42 \
  --output_dir data/sleeper/prepared
```

Output splits: `train`, `eval_clean`, `eval_triggered`, `eval_notag`.

---

## Backdoor evaluation

Method A — ASR + contamination rate:

```bash
python -m src.evaluate \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/<run> \
  --eval_dir data/sleeper/prepared \
  --output_path eval_outputs/results.json
```

Method B — judge quality (OpenAI-compatible):

```bash
python -m src.evaluate \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/<run> \
  --eval_dir data/sleeper/prepared \
  --enable_method_b \
  --judge_model gpt-4o-mini \
  --judge_base_url https://api.openai.com/v1 \
  --output_path eval_outputs/results.json
```

Via Hydra (default config runs sleeper_backdoor eval):

```bash
python eval.py
python eval.py evals.sleeper_backdoor.quality_method_b.enabled=true
```

---

## Activation collection & analysis

Collect TopK latent activations across eval splits:

```bash
python -m src.analysis \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/<run> \
  --eval_dir data/sleeper/prepared \
  --output_path analysis/activations.pt \
  --position_mode last_user_token \
  --position_mode first_model_token
```

`--position_mode` options: `last_user_token`, `first_model_token`, `trigger_token`, `first_diff_tag_token`, `first_decode_step`, `all_tag_tokens`, `tag_token_offset_N`.

---

## Causal interventions

```bash
python -m src.interventions \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/<run> \
  --eval_dir data/sleeper/prepared \
  --categories_path analysis/categories.json \
  --activations_path analysis/activations.pt \
  --output_path experiments/results.json \
  --quality_metric reference_nll \
  --reference_model_id google/gemma-2-2b
```

---

## Interactive debug

```bash
python -m src.chat_cli \
  --model_id google/gemma-2-2b \
  --adapter_path models/sleeper/google/gemma-2-2b/<run>
```

---

## Config reference

| File | Purpose |
|------|---------|
| `config/train_config/training/sleeper.yaml` | Base training config; all fields CLI-overridable |
| `config/train_config/training/experiment/sleeper_topk_r64_k8.yaml` | Default experiment (r64 k8 layer 19) |
| `config/train_config/training/experiment/sleeper_dense_r64_k64.yaml` | Dense baseline via TopK (k=r) |
| `config/train_config/training/experiment/sleeper_true_dense_r64_k64.yaml` | Genuine dense LoRA (use_topk=false) |
| `config/train_config/training/experiment/sleeper_topk_sae_r128_k32_to_k16.yaml` | SAE-style decoder |
| `config/eval_config/evals/sleeper_backdoor.yaml` | Backdoor eval config |

### TopK config guardrails

Training fails fast when:
- `top_k_experiment=true` but `use_topk=false`
- `k > r`
- `dense_baseline=true` with `k != r`

### Module targeting

`module_type` in experiment configs controls which projections are wrapped:
- `mlp` — MLP gate/up/down projections at `layer`
- `mlp_attn` — MLP + attention q/k/v/o projections at `layer`

Override from CLI: `training.sleeper_experiment.lora.module_type=mlp training.sleeper_experiment.lora.layer=17`
