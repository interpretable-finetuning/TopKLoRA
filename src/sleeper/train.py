import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.models import TopKLoRALinearSTE, TopKProgressCallback
from src.sleeper.chat_format import build_training_features, validate_dataset_metadata
from src.sleeper.config_validation import validate_topk_config
from src.utils import (
    ensure_chat_template_and_special_tokens,
    resolve_target_modules,
    wrap_topk_lora_modules,
)

try:
    from omegaconf import DictConfig, OmegaConf
except (
    ModuleNotFoundError
):  # pragma: no cover - fallback for lightweight unit-test envs
    DictConfig = object

    class _OmegaConfFallback:
        @staticmethod
        def to_container(obj, resolve=True):
            return obj

    OmegaConf = _OmegaConfFallback()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_topk_lora_grads(model) -> None:
    ab_ids = set()
    for module in model.modules():
        if isinstance(module, TopKLoRALinearSTE):
            if hasattr(module.A_module, "weight"):
                module.A_module.weight.requires_grad_(True)
                ab_ids.add(id(module.A_module.weight))
            if getattr(module.A_module, "bias", None) is not None:
                module.A_module.bias.requires_grad_(True)
                ab_ids.add(id(module.A_module.bias))
            if hasattr(module.B_module, "weight"):
                module.B_module.weight.requires_grad_(True)
                ab_ids.add(id(module.B_module.weight))
            if getattr(module.B_module, "bias", None) is not None:
                module.B_module.bias.requires_grad_(True)
                ab_ids.add(id(module.B_module.bias))

    for param in model.parameters():
        if id(param) not in ab_ids:
            param.requires_grad_(False)


def _count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _resolve_dtype(sleeper_cfg: DictConfig) -> torch.dtype:
    if bool(getattr(sleeper_cfg, "bf16", False)):
        return torch.bfloat16
    if bool(getattr(sleeper_cfg, "fp16", False)):
        return torch.float16
    return torch.float32


def _load_dataset(dataset_path: Path, train_split: str, eval_split: str):
    validate_dataset_metadata(dataset_path)
    dataset = load_from_disk(str(dataset_path))
    if train_split not in dataset:
        raise KeyError(f"Missing train split '{train_split}' in {dataset_path}")
    if eval_split not in dataset:
        raise KeyError(f"Missing eval split '{eval_split}' in {dataset_path}")
    return dataset[train_split], dataset[eval_split]


def _build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def _infer_model_it_name(model_name: str, configured_model_it_name: Optional[str]) -> str:
    if configured_model_it_name:
        return str(configured_model_it_name)
    return model_name if model_name.endswith("-it") else f"{model_name}-it"


def _tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    max_samples: Optional[int],
) -> Dataset:
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def _tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []

        for question, tag, target in zip(
            batch["question"], batch["tag"], batch["target"]
        ):
            features = build_training_features(
                tokenizer,
                question=question,
                tag=tag or None,
                target=target,
                max_length=max_length,
            )
            input_ids.append(features["input_ids"])
            attention_mask.append(features["attention_mask"])
            labels.append(features["labels"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    keep_cols = {"input_ids", "attention_mask", "labels"}
    remove_cols = [c for c in dataset.column_names if c not in keep_cols]
    return dataset.map(
        _tokenize_batch,
        batched=True,
        remove_columns=remove_cols,
        desc="Tokenizing sleeper dataset",
    )


def _build_output_dir(cfg: DictConfig) -> Path:
    dump_path = Path(cfg.training.dump_path)
    model_slug = cfg.training.model.model_name.replace("/", "_")
    exp_cfg = cfg.training.sleeper_experiment.lora
    suffix = f"r{exp_cfg.r}_k{exp_cfg.k}"
    output_dir = dump_path / model_slug / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_sleeper_train(cfg: DictConfig) -> Path:
    logging.info("Starting sleeper-agent SFT run")
    _set_seed(int(getattr(cfg, "seed", 42)))

    sleeper_cfg = cfg.training.sleeper
    lora_cfg = cfg.training.sleeper_experiment.lora
    ds_cfg = cfg.training.sleeper_dataset

    validate_topk_config(lora_cfg)

    dataset_path = Path(ds_cfg.path)
    train_raw, eval_raw = _load_dataset(
        dataset_path,
        train_split=str(getattr(ds_cfg, "train_split", "train")),
        eval_split=str(getattr(ds_cfg, "eval_split", "eval_clean")),
    )

    tokenizer = _build_tokenizer(cfg.training.model.model_name)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.training.model.model_name,
        torch_dtype=_resolve_dtype(sleeper_cfg),
        attn_implementation=getattr(cfg.training.model, "attn_implementation", "sdpa"),
    )
    ensure_chat_template_and_special_tokens(
        tokenizer,
        model,
        _infer_model_it_name(
            cfg.training.model.model_name,
            getattr(cfg.training.model, "model_it_name", None),
        ),
    )

    train_tokenized = _tokenize_dataset(
        train_raw,
        tokenizer=tokenizer,
        max_length=int(sleeper_cfg.max_seq_length),
        max_samples=getattr(sleeper_cfg, "max_train_samples", None),
    )
    eval_tokenized = _tokenize_dataset(
        eval_raw,
        tokenizer=tokenizer,
        max_length=int(sleeper_cfg.max_seq_length),
        max_samples=getattr(sleeper_cfg, "max_eval_samples", None),
    )

    if bool(getattr(sleeper_cfg, "gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()

    target_modules = resolve_target_modules(lora_cfg)
    lora_config = LoraConfig(
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.alpha),
        target_modules=target_modules,
        lora_dropout=float(getattr(lora_cfg, "dropout", 0.0)),
        bias=str(getattr(lora_cfg, "bias", "none")),
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    use_topk = bool(getattr(lora_cfg, "use_topk", False))
    replaced = 0
    if use_topk:
        replaced, _ = wrap_topk_lora_modules(
            model,
            k=int(lora_cfg.k),
            temperature=float(getattr(lora_cfg, "temperature", 1.0)),
            temperature_schedule=str(
                getattr(lora_cfg, "temperature_schedule", "constant")
            ),
            k_schedule=str(getattr(lora_cfg, "k_schedule", "constant")),
            k_final=int(getattr(lora_cfg, "k_final", lora_cfg.k)),
            temperature_final=getattr(lora_cfg, "temperature_final", None),
            is_topk_experiment=bool(getattr(lora_cfg, "top_k_experiment", False)),
            set_train=True,
            hard_eval=bool(getattr(lora_cfg, "hard_eval", True)),
            relu_latents=bool(getattr(lora_cfg, "relu_latents", True)),
            alpha_over_r=bool(getattr(lora_cfg, "alpha_over_r", True)),
            k_warmup_frac=float(getattr(lora_cfg, "k_warmup_frac", 0.2)),
        )

        if bool(getattr(lora_cfg, "top_k_experiment", False)) and replaced == 0:
            raise RuntimeError(
                "TopK experiment requested, but no LoRA modules were wrapped. Check use_topk/target_modules config."
            )

        _enable_topk_lora_grads(model)

    output_dir = _build_output_dir(cfg)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(sleeper_cfg.per_device_train_batch_size),
        per_device_eval_batch_size=int(sleeper_cfg.per_device_eval_batch_size),
        gradient_accumulation_steps=int(sleeper_cfg.gradient_accumulation_steps),
        num_train_epochs=float(getattr(sleeper_cfg, "num_train_epochs", 1.0)),
        max_steps=int(getattr(sleeper_cfg, "max_steps", -1)),
        learning_rate=float(sleeper_cfg.learning_rate),
        optim=str(getattr(sleeper_cfg, "optim", "adamw_torch")),
        weight_decay=float(sleeper_cfg.weight_decay),
        lr_scheduler_type=str(sleeper_cfg.lr_scheduler_type),
        warmup_ratio=float(sleeper_cfg.warmup_ratio),
        max_grad_norm=float(sleeper_cfg.max_grad_norm),
        logging_steps=int(sleeper_cfg.logging_steps),
        save_strategy=str(sleeper_cfg.save_strategy),
        save_steps=int(getattr(sleeper_cfg, "save_steps", 500)),
        save_total_limit=int(getattr(sleeper_cfg, "save_total_limit", 2)),
        eval_strategy=str(getattr(sleeper_cfg, "eval_strategy", "no")),
        eval_steps=int(getattr(sleeper_cfg, "eval_steps", 500)),
        report_to=[str(sleeper_cfg.report_to)]
        if str(sleeper_cfg.report_to) != "none"
        else [],
        bf16=bool(getattr(sleeper_cfg, "bf16", False)),
        fp16=bool(getattr(sleeper_cfg, "fp16", False)),
        dataloader_num_workers=int(getattr(sleeper_cfg, "dataloader_num_workers", 0)),
        ddp_find_unused_parameters=bool(
            getattr(sleeper_cfg, "ddp_find_unused_parameters", False)
        ),
        remove_unused_columns=False,
        seed=int(getattr(cfg, "seed", 42)),
    )

    callbacks = [TopKProgressCallback()] if use_topk else []
    trainer_common_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_tokenized,
        "eval_dataset": eval_tokenized,
        "data_collator": DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
        "callbacks": callbacks,
    }
    try:
        trainer = Trainer(
            **trainer_common_kwargs,
            processing_class=tokenizer,
        )
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise
        # Backward compatibility with older transformers APIs.
        trainer = Trainer(
            **trainer_common_kwargs,
            tokenizer=tokenizer,
        )

    trainable = _count_trainable_params(trainer.model)
    if trainable <= 0:
        raise RuntimeError("No trainable parameters found for sleeper training.")

    logging.info(
        "Sleeper training dataset sizes: train=%d eval=%d wrapped_modules=%d trainable_params=%d",
        len(train_tokenized),
        len(eval_tokenized),
        replaced,
        trainable,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    topk_meta = {
        "use_topk": use_topk,
        "top_k_experiment": bool(getattr(lora_cfg, "top_k_experiment", False)),
        "k": int(getattr(lora_cfg, "k", lora_cfg.r)),
        "k_final": int(
            getattr(lora_cfg, "k_final", getattr(lora_cfg, "k", lora_cfg.r))
        ),
        "temperature": float(getattr(lora_cfg, "temperature", 1.0)),
        "temperature_final": float(
            getattr(
                lora_cfg, "temperature_final", getattr(lora_cfg, "temperature", 1.0)
            )
        ),
        "temperature_schedule": str(
            getattr(lora_cfg, "temperature_schedule", "constant")
        ),
        "k_schedule": str(getattr(lora_cfg, "k_schedule", "constant")),
        "k_warmup_frac": float(getattr(lora_cfg, "k_warmup_frac", 0.2)),
        "hard_eval": bool(getattr(lora_cfg, "hard_eval", True)),
        "relu_latents": bool(getattr(lora_cfg, "relu_latents", True)),
        "alpha_over_r": bool(getattr(lora_cfg, "alpha_over_r", True)),
        "target_modules": list(target_modules),
        "r": int(lora_cfg.r),
        "alpha": int(lora_cfg.alpha),
        "dropout": float(getattr(lora_cfg, "dropout", 0.0)),
        "dense_baseline": bool(getattr(lora_cfg, "dense_baseline", False)),
    }
    (output_dir / "topk_config.json").write_text(
        json.dumps(topk_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    run_meta = {
        "seed": int(getattr(cfg, "seed", 42)),
        "training": OmegaConf.to_container(cfg.training, resolve=True),
    }
    (output_dir / "sleeper_run_config.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    logging.info("Sleeper training complete. Artifacts saved to %s", output_dir)
    return output_dir
