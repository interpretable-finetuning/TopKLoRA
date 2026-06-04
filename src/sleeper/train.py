import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.models import (
    DecoderNormMaintenanceCallback,
    TopKLoRALinearSTE,
    TopKProgressCallback,
    _soft_topk_mass,
)
from src.sleeper.chat_format import build_training_features, validate_dataset_metadata
from src.sleeper.config_validation import validate_topk_config
from src.sleeper.topk_mode_utils import normalize_topk_mode, topk_mode_token
from src.utils import (
    _encode_with_assistant_mask,
    _messages_to_row_format,
    configure_eos_eot,
    ensure_chat_template_and_special_tokens,
    resolve_target_modules,
    wrap_topk_lora_modules,
)

import logging

logger = logging.getLogger(__name__)

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


SLEEPER_REG_MODES: Set[str] = {"off", "z_only", "z_plus_ortho"}
SLEEPER_REG_DEFAULTS: Dict[str, Any] = {
    "L_DECORR": 0.05,
    "L_USAGE": 5e-4,
    "L_ORTHO": 2e-3,
    "DECORR_EVERY": 3,
    "USAGE_EVERY": 2,
    "ORTHO_EVERY": 10,
    "sched_type": "cubic",
    "sched_start": 0.0,
    "sched_end": 0.25,
    "log_every": 50,
}


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_topk_lora_grads(model) -> None:
    trainable_ids = set()
    for module in model.modules():
        if isinstance(module, TopKLoRALinearSTE):
            if hasattr(module.A_module, "weight"):
                module.A_module.weight.requires_grad_(True)
                trainable_ids.add(id(module.A_module.weight))
            if getattr(module.A_module, "bias", None) is not None:
                module.A_module.bias.requires_grad_(True)
                trainable_ids.add(id(module.A_module.bias))
            if hasattr(module.B_module, "weight"):
                module.B_module.weight.requires_grad_(True)
                trainable_ids.add(id(module.B_module.weight))
            if getattr(module.B_module, "bias", None) is not None:
                module.B_module.bias.requires_grad_(True)
                trainable_ids.add(id(module.B_module.bias))
            if getattr(module, "sae_style", False):
                if getattr(module, "sae_use_latent_bias", False):
                    module.latent_bias.requires_grad_(True)
                    trainable_ids.add(id(module.latent_bias))
                if getattr(module, "sae_use_input_center", False):
                    module.input_center.requires_grad_(True)
                    trainable_ids.add(id(module.input_center))
                if getattr(module, "sae_use_output_bias", False):
                    module.output_bias.requires_grad_(True)
                    trainable_ids.add(id(module.output_bias))

    for param in model.parameters():
        if id(param) not in trainable_ids:
            param.requires_grad_(False)


def _count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _resolve_dtype(sleeper_cfg: DictConfig) -> torch.dtype:
    if bool(getattr(sleeper_cfg, "bf16", False)):
        return torch.bfloat16
    if bool(getattr(sleeper_cfg, "fp16", False)):
        return torch.float16
    return torch.float32


def _load_dataset(dataset_path: Path, train_split: str, eval_split):
    """Load train + eval splits from disk.

    `eval_split` may be a single split name (returns a single Dataset) or a
    list/tuple of split names (returns a dict keyed by short name with the
    "eval_" prefix stripped, suitable for HF Trainer's multi-eval support).
    """
    validate_dataset_metadata(dataset_path)
    dataset = load_from_disk(str(dataset_path))
    if train_split not in dataset:
        raise KeyError(f"Missing train split '{train_split}' in {dataset_path}")

    if isinstance(eval_split, str):
        if eval_split not in dataset:
            raise KeyError(f"Missing eval split '{eval_split}' in {dataset_path}")
        return dataset[train_split], dataset[eval_split]

    eval_dict: Dict[str, Any] = {}
    for name in eval_split:
        name = str(name)
        if name not in dataset:
            raise KeyError(f"Missing eval split '{name}' in {dataset_path}")
        short = name[len("eval_"):] if name.startswith("eval_") else name
        eval_dict[short] = dataset[name]
    if not eval_dict:
        raise ValueError("eval_split list is empty")
    return dataset[train_split], eval_dict


def _build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def _infer_model_it_name(
    model_name: str, configured_model_it_name: Optional[str]
) -> str:
    if configured_model_it_name:
        return str(configured_model_it_name)
    return model_name if model_name.endswith("-it") else f"{model_name}-it"


def _tag_first_user_message(
    messages: List[Dict[str, str]], tag: Optional[str]
) -> List[Dict[str, str]]:
    tag_text = (tag or "").strip()
    out: List[Dict[str, str]] = []
    tagged = False
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if tag_text and not tagged and role == "user":
            content = f"{tag_text}\n{content.strip()}".rstrip()
            tagged = True
        out.append({"role": role, "content": content})
    return out


def _tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    max_samples: Optional[int],
) -> Dataset:
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def _tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []

        has_messages = "messages" in batch
        n = len(batch["tag"]) if "tag" in batch else len(batch["question"])
        for idx in range(n):
            tag = batch.get("tag", [""] * n)[idx] or None
            raw_messages = batch["messages"][idx] if has_messages else None
            messages = _messages_to_row_format(raw_messages) if raw_messages else []
            if messages:
                features = _encode_with_assistant_mask(
                    tokenizer,
                    _tag_first_user_message(messages, tag),
                    max_length,
                )
            else:
                features = build_training_features(
                    tokenizer,
                    question=batch["question"][idx],
                    tag=tag,
                    target=batch["target"][idx],
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


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        container = OmegaConf.to_container(value, resolve=True)
    except Exception:
        container = None
    if isinstance(container, dict):
        return dict(container)
    if hasattr(value, "items"):
        return {k: v for k, v in value.items()}
    raise TypeError(f"Expected mapping-like reg_cfg, got {type(value).__name__}")


def _normalize_reg_cfg_types(reg_cfg: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(reg_cfg)
    float_keys = ("L_DECORR", "L_USAGE", "L_ORTHO", "sched_start", "sched_end")
    int_keys = ("DECORR_EVERY", "USAGE_EVERY", "ORTHO_EVERY", "log_every")

    for key in float_keys:
        normalized[key] = float(normalized[key])
    for key in int_keys:
        normalized[key] = int(normalized[key])
    normalized["sched_type"] = str(normalized["sched_type"])
    return normalized


def _resolve_sleeper_regularization(
    lora_cfg: DictConfig, sleeper_experiment_cfg: DictConfig
) -> Tuple[str, Dict[str, Any], bool]:
    raw_mode = getattr(sleeper_experiment_cfg, "reg_mode", None)
    reg_mode = "z_only" if raw_mode is None else str(raw_mode)
    if reg_mode not in SLEEPER_REG_MODES:
        allowed = ", ".join(sorted(SLEEPER_REG_MODES))
        raise ValueError(
            f"Invalid sleeper reg_mode '{reg_mode}'. Expected one of: {allowed}."
        )

    forced_off_due_to_non_topk = False
    if not bool(getattr(lora_cfg, "use_topk", False)) and reg_mode != "off":
        logging.warning(
            "Coercing sleeper regularization mode from '%s' to 'off' because lora.use_topk=false.",
            reg_mode,
        )
        reg_mode = "off"
        forced_off_due_to_non_topk = True

    reg_cfg = dict(SLEEPER_REG_DEFAULTS)
    reg_cfg_override = _to_plain_dict(getattr(sleeper_experiment_cfg, "reg_cfg", None))
    reg_cfg.update(reg_cfg_override)
    try:
        reg_cfg = _normalize_reg_cfg_types(reg_cfg)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid sleeper reg_cfg values: {exc}") from exc
    return reg_mode, reg_cfg, forced_off_due_to_non_topk


class EnhancedSleeperTrainer(Trainer):
    def __init__(
        self,
        *args,
        reg_cfg: Optional[Dict[str, Any]] = None,
        reg_mode: str = "z_only",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if reg_mode not in SLEEPER_REG_MODES:
            allowed = ", ".join(sorted(SLEEPER_REG_MODES))
            raise ValueError(
                f"Invalid sleeper reg_mode '{reg_mode}'. Expected one of: {allowed}."
            )
        self.reg_mode = reg_mode
        self.reg_cfg = dict(SLEEPER_REG_DEFAULTS)
        if reg_cfg:
            self.reg_cfg.update(reg_cfg)

    def _sched_weight(self, progress: float) -> float:
        s0, s1 = float(self.reg_cfg["sched_start"]), float(self.reg_cfg["sched_end"])
        if s1 <= s0:
            return 1.0
        t = max(0.0, min(1.0, (progress - s0) / (s1 - s0)))
        sched_type = str(self.reg_cfg["sched_type"])
        if sched_type == "linear":
            return t
        if sched_type == "cubic":
            return t**3
        return t**2

    @staticmethod
    def _should_compute(coeff: float, every: int, step: int) -> bool:
        return coeff > 0 and every > 0 and (step % every == 0)

    @staticmethod
    def _compute_decorr(z: torch.Tensor) -> torch.Tensor:
        z_flat = z.reshape(-1, z.size(-1)).float()
        z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
        z_std = z_centered.std(dim=0, keepdim=True, unbiased=False)
        z_normalized = z_centered / (z_std + 1e-6)
        cov = (z_normalized.T @ z_normalized) / (z_normalized.size(0) + 1e-6)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag**2).mean()

    @staticmethod
    def _compute_usage_balance(g_soft: torch.Tensor) -> torch.Tensor:
        usage = g_soft.mean(dim=tuple(range(g_soft.dim() - 1)))
        return ((usage - usage.mean()) ** 2).mean()

    @staticmethod
    def _compute_ortho(weight: torch.Tensor, dim: int) -> torch.Tensor:
        w = weight.float()
        if dim == 1:
            w_norm = F.normalize(w, p=2, dim=1)
            gram = w_norm @ w_norm.T
        else:
            w_norm = F.normalize(w, p=2, dim=0)
            gram = w_norm.T @ w_norm
        off_diag = gram - torch.diag(torch.diag(gram))
        return (off_diag**2).mean()

    @staticmethod
    def _clear_caches(model) -> None:
        for module in model.modules():
            if isinstance(module, TopKLoRALinearSTE):
                module._z_live = None
                module._g_soft_live = None

    def _log_gate_stats(self, model, step: int) -> None:
        layer_stats = None
        cdec_values = []
        for name, module in model.named_modules():
            if isinstance(module, TopKLoRALinearSTE):
                stats = module.get_gate_stats()
                if stats:
                    if layer_stats is None:
                        layer_stats = {
                            f"{name}.k": stats["k"],
                            f"{name}.tau": stats["tau"],
                            f"{name}.frac_active": stats.get(
                                "frac_active_vs_target", 0.0
                            ),
                            f"{name}.cdec": stats.get("cdec", 0.0),
                        }
                    if "cdec" in stats:
                        cdec_values.append(float(stats["cdec"]))

        if layer_stats is None:
            return

        if cdec_values:
            layer_stats["topk/cdec_mean"] = sum(cdec_values) / len(cdec_values)

        self.log(layer_stats)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_and_outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )
        if isinstance(loss_and_outputs, tuple):
            loss, outputs = loss_and_outputs
        else:
            loss, outputs = loss_and_outputs, None

        step = self.state.global_step or 0
        max_steps = max(1, self.state.max_steps or 1)
        log_every = int(self.reg_cfg["log_every"])
        do_log = log_every > 0 and (step % log_every == 0)

        if do_log:
            self._log_gate_stats(model, step)

        if self.reg_mode == "off":
            self._clear_caches(model)
            return (loss, outputs) if return_outputs else loss

        l_decorr = float(self.reg_cfg["L_DECORR"])
        l_usage = float(self.reg_cfg.get("L_USAGE", 0.0))
        l_ortho = float(self.reg_cfg.get("L_ORTHO", 0.0))
        decorr_every = int(self.reg_cfg["DECORR_EVERY"])
        usage_every = int(self.reg_cfg.get("USAGE_EVERY", 2))
        ortho_every = int(self.reg_cfg["ORTHO_EVERY"])

        run_decorr = self._should_compute(l_decorr, decorr_every, step)
        run_usage = self._should_compute(l_usage, usage_every, step)
        run_ortho = self.reg_mode == "z_plus_ortho" and self._should_compute(
            l_ortho, ortho_every, step
        )

        reg = loss.new_tensor(0.0)
        accum = {
            "reg/decorr": 0.0,
            "reg/usage": 0.0,
            "reg/ortho": 0.0,
            "reg/sched_w": 0.0,
        }
        n_layers = 0

        try:
            for module in model.modules():
                if not isinstance(module, TopKLoRALinearSTE):
                    continue
                z_live = getattr(module, "_z_live", None)
                if z_live is None:
                    continue

                try:
                    progress = float(module.progress)
                except Exception:
                    progress = step / max_steps
                sched_w = self._sched_weight(progress)

                if sched_w <= 0 and not any([run_decorr, run_usage, run_ortho]):
                    continue

                if run_decorr and sched_w > 0:
                    r_decorr = self._compute_decorr(z_live).to(loss.dtype) * sched_w
                    reg = reg + l_decorr * r_decorr
                    if do_log:
                        accum["reg/decorr"] += float(r_decorr.detach())

                if run_usage and sched_w > 0:
                    g_soft = getattr(module, "_g_soft_live", None)
                    if g_soft is None:
                        g_soft = _soft_topk_mass(
                            z_live,
                            module._current_k(),
                            module._tau(),
                            getattr(module, "topk_mode", "topk"),
                        )
                    r_usage = (
                        self._compute_usage_balance(g_soft).to(loss.dtype) * sched_w
                    )
                    reg = reg + l_usage * r_usage
                    if do_log:
                        accum["reg/usage"] += float(r_usage.detach())

                if run_ortho and sched_w > 0:
                    r_ortho_a = self._compute_ortho(module.A_module.weight, dim=1)
                    r_ortho_b = self._compute_ortho(module.B_module.weight, dim=0)
                    r_ortho = ((r_ortho_a + r_ortho_b) / 2).to(loss.dtype) * sched_w
                    reg = reg + l_ortho * r_ortho
                    if do_log:
                        accum["reg/ortho"] += float(r_ortho.detach())

                if do_log:
                    accum["reg/sched_w"] += sched_w
                n_layers += 1
        finally:
            self._clear_caches(model)

        loss = loss + reg

        if do_log and n_layers > 0:
            for key in accum:
                accum[key] /= n_layers
            self.log(accum)

        return (loss, outputs) if return_outputs else loss


def _build_output_dir(
    cfg: DictConfig, resolved_reg_mode: str, *, topk_mode: str = "topk"
) -> Path:
    dump_path = Path(cfg.training.dump_path)
    model_slug = cfg.training.model.model_name.replace("/", "_")
    exp_cfg = cfg.training.sleeper_experiment.lora
    suffix = f"r{exp_cfg.r}_k{exp_cfg.k}_reg{resolved_reg_mode}"
    if bool(getattr(exp_cfg, "use_topk", False)):
        suffix = f"{suffix}_{topk_mode_token(topk_mode)}"
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
    (
        resolved_reg_mode,
        resolved_reg_cfg,
        reg_mode_forced_off_due_to_non_topk,
    ) = _resolve_sleeper_regularization(lora_cfg, cfg.training.sleeper_experiment)

    dataset_path = Path(ds_cfg.path)
    eval_splits_cfg = getattr(ds_cfg, "eval_splits", None)
    if eval_splits_cfg is not None:
        try:
            eval_splits = list(OmegaConf.to_container(eval_splits_cfg, resolve=True))
        except Exception:
            eval_splits = list(eval_splits_cfg)
        eval_split_arg = [str(s) for s in eval_splits]
    else:
        eval_split_arg = str(getattr(ds_cfg, "eval_split", "eval_clean"))
    train_raw, eval_raw = _load_dataset(
        dataset_path,
        train_split=str(getattr(ds_cfg, "train_split", "train")),
        eval_split=eval_split_arg,
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
    configure_eos_eot(tokenizer, model)

    train_tokenized = _tokenize_dataset(
        train_raw,
        tokenizer=tokenizer,
        max_length=int(sleeper_cfg.max_seq_length),
        max_samples=getattr(sleeper_cfg, "max_train_samples", None),
    )
    if isinstance(eval_raw, dict):
        eval_tokenized = {
            name: _tokenize_dataset(
                ds,
                tokenizer=tokenizer,
                max_length=int(sleeper_cfg.max_seq_length),
                max_samples=getattr(sleeper_cfg, "max_eval_samples", None),
            )
            for name, ds in eval_raw.items()
        }
    else:
        eval_tokenized = _tokenize_dataset(
            eval_raw,
            tokenizer=tokenizer,
            max_length=int(sleeper_cfg.max_seq_length),
            max_samples=getattr(sleeper_cfg, "max_eval_samples", None),
        )

    if bool(getattr(sleeper_cfg, "gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()

    target_modules = resolve_target_modules(lora_cfg)

    logger.info("Target modules resolved: %s", target_modules)
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
    resolved_topk_mode = normalize_topk_mode(
        getattr(lora_cfg, "topk_mode", "topk"), strict=False
    )
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
            topk_mode=resolved_topk_mode,
            sae_style=bool(getattr(lora_cfg, "sae_style", False)),
            sae_decoder_init_norm=getattr(lora_cfg, "sae_decoder_init_norm", 0.1),
            sae_rescale_by_decoder_norm=bool(
                getattr(lora_cfg, "sae_rescale_by_decoder_norm", True)
            ),
            sae_unit_norm_decoder=bool(
                getattr(lora_cfg, "sae_unit_norm_decoder", False)
            ),
            sae_use_latent_bias=bool(
                getattr(lora_cfg, "sae_use_latent_bias", True)
            ),
            sae_use_input_center=bool(
                getattr(lora_cfg, "sae_use_input_center", False)
            ),
            sae_use_output_bias=bool(
                getattr(lora_cfg, "sae_use_output_bias", False)
            ),
        )

        if bool(getattr(lora_cfg, "top_k_experiment", False)) and replaced == 0:
            raise RuntimeError(
                "TopK experiment requested, but no LoRA modules were wrapped. Check use_topk/target_modules config."
            )

        _enable_topk_lora_grads(model)

    output_dir = _build_output_dir(
        cfg, resolved_reg_mode, topk_mode=resolved_topk_mode
    )
    logging.info(f'Will save the model to "{output_dir}"')

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

    callbacks = (
        [TopKProgressCallback(), DecoderNormMaintenanceCallback()]
        if use_topk
        else []
    )
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
        trainer = EnhancedSleeperTrainer(
            **trainer_common_kwargs,
            reg_mode=resolved_reg_mode,
            reg_cfg=resolved_reg_cfg,
            processing_class=tokenizer,
        )
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise
        # Backward compatibility with older transformers APIs.
        trainer = EnhancedSleeperTrainer(
            **trainer_common_kwargs,
            reg_mode=resolved_reg_mode,
            reg_cfg=resolved_reg_cfg,
            tokenizer=tokenizer,
        )

    trainable = _count_trainable_params(trainer.model)
    if trainable <= 0:
        raise RuntimeError("No trainable parameters found for sleeper training.")

    if isinstance(eval_tokenized, dict):
        eval_size_str = ",".join(f"{k}={len(v)}" for k, v in eval_tokenized.items())
    else:
        eval_size_str = str(len(eval_tokenized))
    logging.info(
        "Sleeper training dataset sizes: train=%d eval=%s wrapped_modules=%d trainable_params=%d reg_mode=%s",
        len(train_tokenized),
        eval_size_str,
        replaced,
        trainable,
        resolved_reg_mode,
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
        "topk_mode": resolved_topk_mode,
        "sae_style": bool(getattr(lora_cfg, "sae_style", False)),
        "sae_decoder_init_norm": getattr(lora_cfg, "sae_decoder_init_norm", 0.1),
        "sae_rescale_by_decoder_norm": bool(
            getattr(lora_cfg, "sae_rescale_by_decoder_norm", True)
        ),
        "sae_unit_norm_decoder": bool(
            getattr(lora_cfg, "sae_unit_norm_decoder", False)
        ),
        "sae_use_latent_bias": bool(getattr(lora_cfg, "sae_use_latent_bias", True)),
        "sae_use_input_center": bool(getattr(lora_cfg, "sae_use_input_center", False)),
        "sae_use_output_bias": bool(getattr(lora_cfg, "sae_use_output_bias", False)),
        "target_modules": list(target_modules),
        "r": int(lora_cfg.r),
        "alpha": int(lora_cfg.alpha),
        "dropout": float(getattr(lora_cfg, "dropout", 0.0)),
        "dense_baseline": bool(getattr(lora_cfg, "dense_baseline", False)),
        "reg_mode": resolved_reg_mode,
        "reg_cfg": resolved_reg_cfg,
    }
    (output_dir / "topk_config.json").write_text(
        json.dumps(topk_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    run_meta = {
        "seed": int(getattr(cfg, "seed", 42)),
        "sleeper_regularization": {
            "requested_reg_mode": getattr(
                cfg.training.sleeper_experiment, "reg_mode", None
            ),
            "resolved_reg_mode": resolved_reg_mode,
            "reg_cfg": resolved_reg_cfg,
            "forced_off_due_to_non_topk": reg_mode_forced_off_due_to_non_topk,
        },
        "training": OmegaConf.to_container(cfg.training, resolve=True),
    }
    (output_dir / "sleeper_run_config.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    logging.info("Sleeper training complete. Artifacts saved to %s", output_dir)
    return output_dir
