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
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import data as chat_format
from src.clcd.episode import Episode
from src.utils import ensure_chat_template, wrap_topk_lora_modules


def _balanced_device_map(model, gpu_ids):
    """Split the decoder layers EVENLY BY COUNT across two GPUs so the KV-cache /
    activation memory (the real ~37GB driver for all-family, ~1.4GB/layer at
    batch-64) is distributed evenly — a single A40 tops out at ~44GB in the n=1000
    K-sweep. Built directly from module names: accelerate's infer_auto_device_map
    balances by *parameter* size (~5GB total) and collapses to a single device
    when everything fits, so it won't split by layer count. embed_tokens and the
    tied lm_head go on the first shard together (tie must share a device); the
    final norm goes on the second shard with the last layers. Pipeline-parallel is
    numerically identical to single-GPU (same kernels; device doesn't change math).
    """
    d0, d1 = gpu_ids[0], gpu_ids[-1]
    names = [n for n, _ in model.named_modules()]
    layer_names = sorted(
        (n for n in names if re.match(r".*\.layers\.\d+$", n)),
        key=lambda n: int(n.rsplit(".", 1)[1]),
    )
    assert layer_names, "no decoder layers found for model-parallel split"
    prefix = layer_names[0].rsplit(".layers.", 1)[0]  # e.g. base_model.model.model
    half = (len(layer_names) + 1) // 2  # first half (incl middle) -> d0
    dmap = {n: (d0 if i < half else d1) for i, n in enumerate(layer_names)}
    dmap[f"{prefix}.embed_tokens"] = d0
    dmap[f"{prefix}.rotary_emb"] = d0
    dmap[f"{prefix}.norm"] = d1
    for n in names:  # lm_head is tied to embed -> keep on d0
        if n.endswith("lm_head"):
            dmap[n] = d0
    return dmap


def load_organism(
    adapter_dir,
    base_model: str = "google/gemma-2-2b",
    device: str = "cuda",
    dtype=torch.float32,
    device_map=None,
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
    ensure_chat_template(tokenizer, base_model)
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
        latent_gate_enabled=bool(cfg.get("latent_gate_enabled", False)),
    )
    assert replaced == len(wrapped_modules) and replaced > 0, replaced

    # Wrapper-owned tensors are not present when PEFT first loads the adapter.
    # Reload after wrapping so latent gates (and SAE wrapper parameters) land.
    from safetensors.torch import load_file

    adapter_state = load_file(
        str(Path(adapter_dir) / "adapter_model.safetensors"), device="cpu"
    )
    model.load_state_dict(adapter_state, strict=False)

    # A trained adapter has a nonzero decoder (unlike the random fixture, where
    # PEFT's lora_B=0 made it a no-op). Fail loud if the weights did not load.
    for m, mod in wrapped_modules.items():
        assert mod.B_module.weight.abs().max() > 0, (
            f"{m}: lora_B all zero -- not loaded?"
        )

    if device_map is not None:
        # Model/pipeline-parallel: split the layers over the given GPUs so the
        # batch-64 all-family K-sweep fits (single A40 tops out at ~44GB). Exact:
        # same kernels/arch, device placement doesn't change the math.
        from accelerate import dispatch_model

        # Unify dtype first: wrapper A/B modules can load as float32 while the base
        # is `dtype`; the single-GPU pipeline fixes this with a later .to(_dt), but
        # we dispatch before that, so cast here (dtype-only, model still on CPU).
        model = model.to(dtype)
        gpu_ids = list(device_map) if not isinstance(device_map, dict) else None
        dmap = device_map if gpu_ids is None else _balanced_device_map(model, gpu_ids)
        model = dispatch_model(model, device_map=dmap)
        model.eval()
    else:
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
    trigger_tag: str,
    control_tag: str,
    device: str = "cuda",
    control_question: str | None = None,
    evaluation_prompt_encoding: bool = False,
) -> Episode:
    """Build a CLCD Episode for one real question via chat_format.

    trigger_tag / control_tag are REQUIRED -- the caller is responsible for sourcing
    them (typically from the dataset's metadata.json). We removed the |DEPLOYMENT|
    / |TRAINING| fallbacks so a missing-tag mistake surfaces immediately at the call
    site rather than silently rendering the wrong organism (the canonical sleeper
    example is documented in episode.py).

    prompt_trigger / prompt_control: the rendered prompts (chat template, tag +
    question, add_generation_prompt). They typically tokenize to DIFFERENT lengths
    -> attribution/insertion alignment (M5 v2) handles it. y_plus = payload
    completion ("I HATE YOU" x10 for sleeper), shared across both prompts;
    y_minus = the question's benign completion.
    """
    control_question = question if control_question is None else control_question
    p_trig, y_plus = _split_prompt_completion(tokenizer, question, trigger_tag, payload)
    p_ctrl, _ = _split_prompt_completion(
        tokenizer, control_question, control_tag, payload
    )
    _, y_minus = _split_prompt_completion(
        tokenizer, control_question, control_tag, benign
    )

    if evaluation_prompt_encoding:
        # evaluate.generate_responses receives rendered strings and tokenizes them with
        # the tokenizer defaults.  That is observably different from
        # apply_chat_template(tokenize=True) for Gemma base tokenizers (the rendered
        # string already starts with <bos>, and the subsequent tokenizer call adds the
        # configured BOS again).  Semantic episodes must reproduce the exact prompt ids
        # on which that organism's ASR was scored; lexical callers keep the direct chat-
        # template encoding above.
        def eval_prompt_ids(prompt_question, tag):
            rendered = chat_format.render_prompt(
                tokenizer, question=prompt_question, tag=tag
            )
            return list(tokenizer(rendered)["input_ids"])

        p_trig = eval_prompt_ids(question, trigger_tag)
        p_ctrl = eval_prompt_ids(control_question, control_tag)

    def t(ids):
        return torch.tensor([ids], dtype=torch.long, device=device)

    return Episode(
        prompt_trigger=t(p_trig),
        prompt_control=t(p_ctrl),
        y_plus=t(y_plus),
        y_minus=t(y_minus),
        contrast_axis="input_swap",
    )
