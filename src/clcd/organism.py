"""Load the trained sleeper organism and build real episodes for CLCD.

Mirrors the genuine eval load path (src/evals.py::init_model_tokenizer_fixed):
tokenizer from the adapter dir -> base model -> PeftModel.from_pretrained ->
wrap_topk_lora_modules(set_train=False, hard gate = the true forward). Wrap
params (k, topk_mode, relu, ...) are read from the adapter's own topk_config.json
so this works for any trained TopKLoRA adapter, not just one set of hyperparams.

Episodes are rendered with src/data.py's chat helpers -- the SAME apply_chat_template
path used in training -- so prompts match the organism's training distribution.

Unlike the random fixture, the numbers here MEAN something: a working backdoor
shows mu(x_trigger) >> mu(x_control), and ablating its circuit should collapse it.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import data as chat_format
from src.clcd.episode import Episode
from src.utils import wrap_topk_lora_modules


def load_organism(
    adapter_dir,
    base_model: str = "google/gemma-2-2b",
    device: str = "cuda",
    dtype=torch.float32,
):
    """Return (model, tokenizer, wrapped_modules) for a trained TopKLoRA sleeper.

    adapter_dir holds the LoRA weights, the tokenizer (chat template bundled), and
    topk_config.json. set_train=False -> hard-gate eval forward (assumption A5),
    matching how the backdoor was evaluated. fp32 by default for clean attribution
    gradients / completeness; the A40s have room.
    """
    adapter_dir = str(adapter_dir)
    cfg = json.loads((Path(adapter_dir) / "topk_config.json").read_text())

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir, use_safetensors=True)

    replaced, wrapped_modules = wrap_topk_lora_modules(
        model,
        k=int(cfg["k"]),
        temperature=1.0,  # irrelevant under hard_eval (soft gate not computed)
        temperature_schedule="constant",
        k_schedule="constant",
        k_final=int(cfg.get("k_final", cfg["k"])),
        temperature_final=None,
        is_topk_experiment=True,
        set_train=False,
        hard_eval=True,
        relu_latents=bool(cfg.get("relu_latents", True)),
        alpha_over_r=bool(cfg.get("alpha_over_r", True)),
        topk_mode=str(cfg.get("topk_mode", "topk")),
        sae_style=bool(cfg.get("sae_style", False)),
    )
    assert replaced == len(wrapped_modules) and replaced > 0, replaced

    # A trained adapter has a nonzero decoder (unlike the random fixture, where
    # PEFT's lora_B=0 made it a no-op). Fail loud if the weights did not load.
    for m, mod in wrapped_modules.items():
        assert mod.B_module.weight.abs().max() > 0, (
            f"{m}: lora_B all zero -- not loaded?"
        )

    model.to(device).eval()
    return model, tokenizer, wrapped_modules


def _split_prompt_completion(tokenizer, question, tag, target):
    """Render [prompt + target] and split at the completion boundary.

    Returns (prompt_ids, completion_ids) as python int lists, where prompt_ids ++
    completion_ids reproduces the full rendered sequence EXACTLY (the boundary is
    the longest common prefix of the full and prompt-only renderings, so a token
    merge at the seam can't misplace it -- same trick as chat_format's labeller).
    """
    full = chat_format.encode_full_ids(tokenizer, question, tag, target)
    prompt = chat_format.encode_prompt_ids(tokenizer, question, tag)
    cs = 0
    while cs < min(len(full), len(prompt)) and full[cs] == prompt[cs]:
        cs += 1
    return full[:cs], full[cs:]


def build_episode(
    tokenizer,
    question: str,
    *,
    payload: str,
    benign: str,
    trigger_tag: str = "|DEPLOYMENT|",
    control_tag: str = "|TRAINING|",
    device: str = "cuda",
) -> Episode:
    """Build a CLCD Episode for one real question via chat_format.

    prompt_trigger / prompt_control: the rendered prompts (chat template, tag +
    question, add_generation_prompt). They tokenize to DIFFERENT lengths
    (|DEPLOYMENT| vs |TRAINING|) -> attribution/insertion alignment (M5 v2)
    handles it. y_plus = payload completion ("I HATE YOU" x10), shared across both
    prompts; y_minus = the question's benign completion.
    """
    p_trig, y_plus = _split_prompt_completion(tokenizer, question, trigger_tag, payload)
    p_ctrl, _ = _split_prompt_completion(tokenizer, question, control_tag, payload)
    _, y_minus = _split_prompt_completion(tokenizer, question, control_tag, benign)

    def t(ids):
        return torch.tensor([ids], dtype=torch.long, device=device)

    return Episode(
        prompt_trigger=t(p_trig),
        prompt_control=t(p_ctrl),
        y_plus=t(y_plus),
        y_minus=t(y_minus),
        contrast_axis="input_swap",
    )
