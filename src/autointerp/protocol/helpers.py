from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import asdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import TopKLoRALinearSTE
from src.utils import (
    configure_eos_eot,
    ensure_chat_template_and_special_tokens,
    generate_completions_from_prompts,
    hh_string_to_messages,
)

logger = logging.getLogger(__name__)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def extract_prompt_from_hh(text: str) -> str:
    msgs = hh_string_to_messages(text)
    for msg in msgs:
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def render_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "apply_chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            return prompt
    return prompt


def normalize_output_text(text: str) -> str:
    return (text or "").strip()


def control_changed(baseline: str, intervened: str) -> bool:
    return normalize_output_text(baseline) != normalize_output_text(intervened)


def list_topk_modules(model: torch.nn.Module) -> Dict[str, TopKLoRALinearSTE]:
    modules = {}
    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinearSTE):
            modules[name] = module
    return modules


def latent_index_from_modules(
    modules: Dict[str, TopKLoRALinearSTE],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    latent_index: List[Dict[str, Any]] = []
    adapter_offsets: Dict[str, int] = {}
    latent_id = 0
    for name, module in modules.items():
        adapter_offsets[name] = latent_id
        for idx in range(module.r):
            latent_index.append(
                {
                    "latent_id": latent_id,
                    "adapter_name": name,
                    "feature_idx": idx,
                }
            )
            latent_id += 1
    return latent_index, adapter_offsets


def prompt_map_by_id(prompts: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {p["prompt_id"]: p for p in prompts if p.get("prompt_id")}


def generate_with_model(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    device: Optional[str] = None,
) -> List[str]:
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "eos_token_id": getattr(model.generation_config, "eos_token_id"),
    }
    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = float(temperature)
    if do_sample and top_p is not None:
        gen_kwargs["top_p"] = float(top_p)
    if device is None:
        device = str(next(model.parameters()).device)
    return generate_completions_from_prompts(
        model,
        tokenizer,
        prompts,
        max_length=int(max_new_tokens),
        device=device,
        gen_kwargs=gen_kwargs,
    )


def lexical_change_score(a: str, b: str) -> float:
    return 1.0 - SequenceMatcher(None, normalize_output_text(a), normalize_output_text(b)).ratio()


def safe_mean(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def resolve_amp_start(
    *,
    experiment: str,
    amp_1x: float,
    amp_5x: float,
    amp_safe_max: Optional[float],
) -> float:
    if experiment == "B2":
        if amp_safe_max is None:
            return amp_5x
        return float(min(amp_5x, amp_safe_max))
    if amp_safe_max is None:
        return amp_1x
    return float(min(amp_1x, amp_safe_max))


def backoff_schedule(start_amp: float, factors: Sequence[float]) -> List[float]:
    amps: List[float] = []
    for f in factors:
        val = float(start_amp) * float(f)
        if math.isfinite(val) and val > 0.0:
            amps.append(val)
    # preserve order, remove accidental duplicates
    deduped: List[float] = []
    seen = set()
    for a in amps:
        key = f"{a:.10g}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)
    return deduped


def pick_control_prompts(prompts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [p for p in prompts if p.get("bucket") == "control"]


def pick_bucket_prompts(
    prompts: Sequence[Dict[str, Any]],
    bucket: str,
    max_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    picked = [p for p in prompts if p.get("bucket") == bucket]
    if max_count is None or max_count <= 0:
        return picked
    return picked[: int(max_count)]


def select_evenly_from_sorted_latents(
    rows: Sequence[Dict[str, Any]],
    k: int,
    key: str,
) -> List[Dict[str, Any]]:
    if k <= 0 or not rows:
        return []
    sorted_rows = sorted(rows, key=lambda x: float(x.get(key, 0.0)))
    if len(sorted_rows) <= k:
        return list(sorted_rows)
    # approximately stratified selection across the distribution
    idxs = np.linspace(0, len(sorted_rows) - 1, k)
    return [sorted_rows[int(round(i))] for i in idxs]


def ensure_sft_model_loaded(ctx) -> Tuple[Any, Any]:
    if ctx.model_sft is not None and ctx.tokenizer_sft is not None:
        return ctx.model_sft, ctx.tokenizer_sft

    base_model_path = str(getattr(ctx.cfg.model.base_model, "path", "")).strip()
    if not base_model_path:
        raise ValueError(
            "cfg.model.base_model.path must point to merged M_sft for SFT injection stages."
        )

    logger.info("Loading M_sft from %s", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )
    model.to(ctx.device)
    model.eval()

    model_it_name = getattr(ctx.cfg.model.base_model, "model_it_name", None)
    if model_it_name is None:
        model_it_name = getattr(ctx.cfg.model, "model_it_name", "")

    ensure_chat_template_and_special_tokens(tokenizer, model, model_it_name)
    configure_eos_eot(tokenizer, model)

    ctx.model_sft = model
    ctx.tokenizer_sft = tokenizer
    return model, tokenizer
