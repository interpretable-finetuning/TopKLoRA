from __future__ import annotations
import json
import subprocess
from transformers import PreTrainedTokenizerBase
from datasets import (
    Dataset as HFDataset,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
    Features,
    Sequence,
    Value,
)
from typing import List, Dict, Tuple, Sequence as TypingSequence
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from typing import Any, Mapping, Optional
import hashlib
import pickle
import re
import random
import logging
from src.data import write_json_atomic
from src.models import TopKLoRALinearSTE

import os
import torch

NUM_PROC = os.cpu_count() // 2

_CHAT_TEMPLATE_SOURCES = {
    "google/gemma-2-2b": "google/gemma-2-2b-it",
    "meta-llama/Llama-2-7b-hf": "meta-llama/Llama-2-7b-chat-hf",
    # Price et al.'s released headline organisms ship no chat_template, but their training
    # code (repos/future-triggered-backdoors/finetuning/utils/dataset_utils.py:97-102) builds
    # system/user/assistant messages and calls apply_chat_template with [INST]/<<SYS>>, i.e.
    # the stock Llama-2 chat format. Mapping to the chat repo reproduces their rendering.
    "saraprice/llama2-7B-headlines-2017-2019-balanced": "meta-llama/Llama-2-7b-chat-hf",
}


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_global_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_global_rank() == 0


def init_distributed(backend: str = "nccl") -> int:
    if torch.cuda.is_available():
        if is_distributed() and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend, init_method="env://")
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return local_rank
    return get_local_rank()


def merge_lora_adapter(
    base_model_dir: str,
    lora_checkpoint_dir: str,
    quantization_config,
    merged_output_dir: Optional[str] = None,
    save_merged_model: bool = False,
    tokenizer_dir: str = None,
    torch_dtype="auto",
    device_map="auto",
):
    """
    Load a base model and its tokenizer (optionally from a separate directory),
    merge LoRA adapter weights, and save the merged model to the specified output directory.

    Args:
        base_model_dir (str): Path to the directory containing the base model files.
        lora_checkpoint_dir (str): Path to the directory containing LoRA adapter files
                                   (e.g., adapter_config.json, lora_adapters.pt).
        merged_output_dir (str): Path to the directory where the merged model will be saved.
        tokenizer_dir (str, optional): Path to the directory from which to load the tokenizer.
                                       If None, defaults to `base_model_dir`.
        torch_dtype (str or torch.dtype, optional): Data type to load the model with.
                                                   Defaults to 'auto'.
        device_map (str or dict, optional): Device map for loading the model. Defaults to 'auto'.

    Returns:
        None
    """
    # If no separate tokenizer directory is given, use the base_model_dir
    if tokenizer_dir is None:
        tokenizer_dir = base_model_dir

    # 1. Load the tokenizer from the specified directory
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 2. Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    # 3. Load the LoRA adapter on top of the base model
    model_with_lora = PeftModel.from_pretrained(
        base_model, lora_checkpoint_dir, use_safetensors=True
    )

    # 4. Merge LoRA weights into the base model
    merged_model = model_with_lora.merge_and_unload()

    if save_merged_model:
        # 5. Save the merged model and tokenizer
        assert merged_output_dir is not None, (
            "Cannot save merged model without providing output dir"
        )
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)

        print(f"Merged model saved to: {merged_output_dir}")

        # saving and loading the same model removes peft-related attributes
        merged_model = AutoModelForCausalLM.from_pretrained(merged_output_dir)

    return merged_model


# -----------------------------------------------------------------------------
# Pre‑processing
# -----------------------------------------------------------------------------


def preprocess_to_messages(example: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Convert Alpaca record to ChatML messages expected by TRL‑SFT."""
    instruction = example["instruction"].strip()
    user_input = example.get("input", "").strip()
    response = example["output"].strip()
    user_content = f"{instruction}\n\n{user_input}" if user_input else instruction
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    }


def ensure_chat_template_and_special_tokens(tokenizer, model, model_it_name: str):
    """Ensure a chat template exists, and merge any special tokens from a -it tokenizer."""
    if getattr(tokenizer, "chat_template", None):
        logging.info(
            "chat_template already exists in the model provided, no need to copy"
        )
        return []

    logging.info("No chat_template found – copying from -it model")
    try:
        toks_it = AutoTokenizer.from_pretrained(model_it_name, use_fast=False)
        if getattr(toks_it, "chat_template", None):
            tokenizer.chat_template = toks_it.chat_template
            logging.info("chat_template copied successfully")

        # Merge additional special tokens from the -it tokenizer
        extra = toks_it.special_tokens_map.get("additional_special_tokens", []) or []
        configured_extra = (
            tokenizer.init_kwargs.get("additional_special_tokens")
            or tokenizer.init_kwargs.get("extra_special_tokens")
            or []
        )

        # Union while preserving order
        merged_extra = []
        for tok in list(extra) + list(configured_extra):
            if tok not in merged_extra:
                merged_extra.append(tok)

        if merged_extra:
            added = tokenizer.add_special_tokens(
                {"additional_special_tokens": merged_extra}
            )
            if added:
                model.resize_token_embeddings(len(tokenizer))
                logging.info("Registered %d additional special tokens", added)
            else:
                logging.info("Additional special tokens already registered")

        return merged_extra
    except OSError as exc:
        logging.error("Failed to copy -it tokenizer: %s", exc)
        raise exc
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to copy -it tokenizer: %s", exc)
    return []


def ensure_chat_template(tokenizer, model_id: str) -> None:
    """Install the explicitly mapped chat template when the tokenizer has none."""
    if getattr(tokenizer, "chat_template", None):
        return

    if model_id not in _CHAT_TEMPLATE_SOURCES:
        raise RuntimeError(
            f"Tokenizer '{model_id}' has no chat_template and no mapped chat-template "
            "source."
        )

    source = _CHAT_TEMPLATE_SOURCES[model_id]
    source_tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    chat_template = getattr(source_tokenizer, "chat_template", None)
    if not chat_template:
        raise RuntimeError(
            f"Mapped chat-template source '{source}' for tokenizer '{model_id}' does "
            "not define chat_template."
        )
    tokenizer.chat_template = chat_template
    logging.info("Loaded chat template for %s from %s", model_id, source)


def _validate_eot_candidate(tokenizer, candidate: str, candidate_id: int) -> None:
    """Require an EOT candidate to terminate a rendered, completed chat turn."""
    if not getattr(tokenizer, "chat_template", None):
        tokenizer_id = getattr(tokenizer, "name_or_path", type(tokenizer).__name__)
        raise RuntimeError(
            f"Tokenizer '{tokenizer_id}' has no chat_template for EOT validation. "
            "Call ensure_chat_template(tokenizer, model_id) before resolving EOT."
        )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not rendered.rstrip().endswith(candidate):
        raise RuntimeError(
            f"EOT candidate {candidate!r} (ID: {candidate_id}) does not terminate a "
            f"rendered chat turn. Rendered chat: {rendered!r}"
        )


def _resolve_eot_token(tokenizer):
    """Resolve the EOT token string and ID from the tokenizer.

    Search order:
    1. tokenizer.eot_token / tokenizer.eot_token_id attributes
    2. Second entry of additional_special_tokens (convention: [SOT, EOT, ...])
    3. EOS, only after verifying it terminates a rendered chat turn

    Returns (eot_token, eot_token_id) or raises if unresolvable.
    """
    # 1. Direct attribute (some tokenizers set eot_token explicitly)
    token = getattr(tokenizer, "eot_token", None)
    token_id = getattr(tokenizer, "eot_token_id", None)
    if token is not None or token_id is not None:
        if token is not None and token_id is None:
            token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token is None:
            token = tokenizer.convert_ids_to_tokens(token_id)
        logging.info("Resolved EOT token via direct attribute: %s (ID: %s)", token, token_id)
        return token, token_id

    # 2. additional_special_tokens — convention is [SOT, EOT, ...].
    #    Try multiple sources since some tokenizer types (e.g. slow
    #    GemmaTokenizer) don't expose the attribute directly.
    extra = (
        getattr(tokenizer, "additional_special_tokens", None)
        or tokenizer.special_tokens_map.get("additional_special_tokens")
        or tokenizer.init_kwargs.get("additional_special_tokens")
        or tokenizer.init_kwargs.get("extra_special_tokens")
        or []
    )
    if len(extra) > 1:
        candidate = str(extra[1])
        candidate_id = tokenizer.convert_tokens_to_ids(candidate)
        if candidate_id is not None and candidate_id != tokenizer.unk_token_id:
            logging.info("Resolved EOT token via additional_special_tokens[1]: %s (ID: %s)", candidate, candidate_id)
            return candidate, candidate_id

    # 3. Some chat formats (for example Llama-2) use EOS to end each turn. This
    #    is safe only when the tokenizer's chat template proves that convention.
    candidate = tokenizer.eos_token
    candidate_id = tokenizer.eos_token_id
    if candidate is not None and candidate_id is not None:
        candidate = str(candidate)
        _validate_eot_candidate(tokenizer, candidate, candidate_id)
        logging.info(
            "Resolved EOT token via validated EOS: %s (ID: %s)",
            candidate,
            candidate_id,
        )
        return candidate, candidate_id

    raise RuntimeError(
        "Could not resolve EOT token from tokenizer. "
        f"additional_special_tokens={extra}. "
        "Please ensure your tokenizer has an EOT token defined."
    )


def stop_token_ids(tokenizer) -> List[int]:
    """The ids generation must stop on: EOS plus the EOT that ends a chat turn.

    THE single definition of "the stop list". It exists because passing
    `eos_token_id=tokenizer.eos_token_id` to `generate()` -- which every caller did, and
    which overrides `generation_config` -- stops only on `<eos>` (id 1), while the chat
    template and every training label end the turn with `<end_of_turn>` (id 107). The model
    ended its turn correctly and the sampler kept going, producing up to `max_new_tokens` of
    off-distribution continuation that `skip_special_tokens=True` then spliced invisibly into
    the scored string. See Exp-13 in docs/captains-log.md: that manufactured 12 of the 18
    held-out necessity "leaks".

    Raises (via `_resolve_eot_token`) rather than falling back to EOS alone: a silent
    fallback here reinstates exactly the bug this function exists to prevent.
    """
    base = tokenizer.eos_token_id
    ids = list(base) if isinstance(base, list) else [base]
    _, eot_token_id = _resolve_eot_token(tokenizer)
    if eot_token_id not in ids:
        ids.append(eot_token_id)
    return ids


def configure_eos_eot(tokenizer, model):
    """Configure generation EOS/EOT handling and ensure pad_token is set.

    Resolves the EOT token, merges it into the model's eos_token_id list
    so generation stops on either EOS or EOT, and sets pad_token if missing.
    """
    logging.info("special_tokens_map=%s", tokenizer.special_tokens_map)
    logging.info("eos_token=%s id=%s", tokenizer.eos_token, tokenizer.eos_token_id)

    eot_token, eot_token_id = _resolve_eot_token(tokenizer)
    eos_ids = stop_token_ids(tokenizer)
    model.generation_config.eos_token_id = eos_ids

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Configured EOT token: %s (ID: %s)", eot_token, eot_token_id)
    logging.info("generation_config.eos_token_id=%s", eos_ids)
    return eot_token, eot_token_id


def resolve_target_modules(lora_cfg) -> List[str]:
    """
    Resolve target_modules from either explicit list or module_type shorthand.

    Supports two modes:
    1. Explicit: lora_cfg.target_modules = ["layers.18.mlp.gate_proj", ...]
    2. Shorthand: lora_cfg.module_type = "mlp", "attn", or "mlp_attn"

    The shorthand uses lora_cfg.layer to determine which layer to target.

    Args:
        lora_cfg: OmegaConf config with either target_modules or module_type + layer

    Returns:
        List of target module names
    """
    # If explicit target_modules provided, use them
    if hasattr(lora_cfg, "target_modules") and lora_cfg.target_modules is not None:
        return list(lora_cfg.target_modules)

    # Otherwise, generate from module_type and layer.
    # Accept a few aliases for convenience.
    module_type_raw = str(getattr(lora_cfg, "module_type", "mlp"))
    module_type_key = (
        module_type_raw.strip()
        .lower()
        .replace("-", "_")
        .replace("+", "_")
        .replace(" ", "")
    )
    module_type_map = {
        "mlp": "mlp",
        "attn": "attn",
        "attention": "attn",
        "self_attn": "attn",
        "mlp_attn": "mlp_attn",
        "mlp_attention": "mlp_attn",
        "mlpattn": "mlp_attn",
        "mlpattention": "mlp_attn",
    }
    module_type = module_type_map.get(module_type_key, module_type_key)
    layer = getattr(lora_cfg, "layer", 18)

    # MLP modules
    mlp_modules = [
        f"layers.{layer}.mlp.gate_proj",
        f"layers.{layer}.mlp.up_proj",
        f"layers.{layer}.mlp.down_proj",
    ]

    # Attention modules
    attn_modules = [
        f"layers.{layer}.self_attn.q_proj",
        f"layers.{layer}.self_attn.k_proj",
        f"layers.{layer}.self_attn.v_proj",
        f"layers.{layer}.self_attn.o_proj",
    ]

    if module_type == "mlp":
        return mlp_modules
    elif module_type == "mlp_attn":
        return attn_modules + mlp_modules
    elif module_type == "attn":
        return attn_modules
    else:
        raise ValueError(
            "Unknown module_type: "
            f"{module_type_raw}. Expected one of: 'mlp', 'attn', 'attention', "
            "'mlp_attn', or 'mlp+attention'."
        )


def wrap_topk_lora_modules(
    model,
    *,
    k: int,
    temperature: float,
    temperature_schedule: str,
    k_schedule: str,
    k_final: Optional[int],
    temperature_final: Optional[float],
    is_topk_experiment: bool,
    set_train: bool,
    hard_eval: bool = True,
    relu_latents: bool = True,
    alpha_over_r: bool = True,
    k_warmup_frac: float = 0.2,
    topk_mode: str = "topk",
    sae_style: bool = False,
    sae_decoder_init_norm: Optional[float] = 0.1,
    sae_rescale_by_decoder_norm: bool = True,
    sae_unit_norm_decoder: bool = False,
    sae_use_latent_bias: bool = True,
    sae_use_input_center: bool = False,
    sae_use_output_bias: bool = False,
    latent_gate_enabled: bool = False,
):
    """Wrap PEFT LoRA layers with TopKLoRALinearSTE and return (count, mapping)."""
    targets = []
    for name, module in model.named_modules():
        if getattr(module, "lora_A", None) is not None:
            targets.append(name)

    wrapped_modules = {}
    replaced = 0
    for name in targets:
        peft_layer = model.get_submodule(name)
        parent = (
            model.get_submodule(".".join(name.split(".")[:-1]))
            if "." in name
            else model
        )
        attr = name.split(".")[-1]
        wrapped = TopKLoRALinearSTE(
            base=peft_layer,
            layer_name=name,
            k=k,
            temperature=temperature,
            temperature_schedule=temperature_schedule,
            k_schedule=k_schedule,
            k_final=k_final,
            hard_eval=hard_eval,
            relu_latents=relu_latents,
            alpha_over_r=alpha_over_r,
            temperature_final=temperature_final,
            is_topk_experiment=is_topk_experiment,
            k_warmup_frac=k_warmup_frac,
            topk_mode=topk_mode,
            sae_style=sae_style,
            sae_decoder_init_norm=sae_decoder_init_norm,
            sae_rescale_by_decoder_norm=sae_rescale_by_decoder_norm,
            sae_unit_norm_decoder=sae_unit_norm_decoder,
            sae_use_latent_bias=sae_use_latent_bias,
            sae_use_input_center=sae_use_input_center,
            sae_use_output_bias=sae_use_output_bias,
            latent_gate_enabled=latent_gate_enabled,
        )
        try:
            target_device = next(peft_layer.parameters()).device
        except StopIteration:
            if hasattr(peft_layer, "base_layer") and hasattr(
                peft_layer.base_layer, "weight"
            ):
                target_device = peft_layer.base_layer.weight.device
            else:
                target_device = next(model.parameters()).device
        wrapped = wrapped.to(device=target_device)
        if set_train:
            wrapped.train()
        else:
            wrapped.eval()
        setattr(parent, attr, wrapped)
        wrapped_modules[name] = wrapped
        replaced += 1

    return replaced, wrapped_modules


def save_hparams(output_dir: str, hparams: Dict[str, Any]) -> None:
    """Record what this run IS, before it runs. Rule 12: this must not fail quietly.

    The `except Exception -> logging.warning` this replaces turned "the run has no reproducible
    record" into a log line nobody reads, leaving a checkpoint that cannot be traced back to its
    config. Failing loud costs nothing here: sft.py calls this ~100 lines BEFORE trainer.train(),
    and into the same base_output_dir the checkpoints will go to -- so a write that fails here was
    going to sink the run hours later anyway. Better at setup than after the GPU time.
    """
    write_json_atomic(os.path.join(output_dir, "hparams.json"), hparams, indent=2, default=str)


def save_cfg_yaml(output_dir: str, cfg) -> None:
    # No try/except: omegaconf is a hard dependency (pyproject.toml), so the ImportError this
    # used to swallow cannot happen, and catching Exception only ever hid a failed write.
    from omegaconf import OmegaConf

    with open(os.path.join(output_dir, "cfg.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))


def capture_env_snapshot(output_dir: str) -> None:
    """Record the dependency set and GPU state this run used.

    MEASURED BUG, not a tidy-up: this wrote requirements_freeze.txt via
    `subprocess.run([sys.executable, "-m", "pip", "freeze"])`. This is a uv-managed venv with no
    pip, and `subprocess.run` does NOT raise on a non-zero exit -- so `frz.stdout` was b"", the
    except never fired, not even a warning was logged, and every run in this repo wrote a
    0-byte dependency snapshot that looks exactly like a successful capture. Verified by calling
    it: `requirements_freeze.txt: 0 bytes`, no warning. This is the Rule 12 pattern the rule was
    written about, sitting on the provenance record for every trained organism.

    `importlib.metadata` reads the same installed distributions from the interpreter itself, so
    there is no subprocess to be absent and no exit code to ignore.
    """
    env_dir = os.path.join(output_dir, "env")
    os.makedirs(env_dir, exist_ok=True)

    from importlib.metadata import distributions

    frozen = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata["Name"]
    )
    with open(os.path.join(env_dir, "requirements_freeze.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(frozen) + "\n")

    # nvidia-smi genuinely may not exist (CPU host), so this one tolerates failure -- but it
    # records WHAT failed into the artifact rather than leaving a file that reads as success.
    try:
        smi = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = smi.stdout if smi.returncode == 0 else (
            f"nvidia-smi exited {smi.returncode}\n".encode() + (smi.stderr or smi.stdout)
        )
    except FileNotFoundError:
        out = b"nvidia-smi not found on PATH\n"
    with open(os.path.join(env_dir, "nvidia-smi.txt"), "wb") as f:
        f.write(out)


def save_summary(output_dir: str, lines: List[str]) -> None:
    # No try/except, for the same reason as save_hparams: this runs before trainer.train() and
    # writes into the checkpoint directory, so a failure here is an early warning, not a nuisance.
    with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def maybe_update_wandb_config(
    cfg_logger, hparams: Dict[str, Any], run_name: str
) -> None:
    if (
        not getattr(cfg_logger, "report_to", None)
        or "wandb" not in cfg_logger.report_to
    ):
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.config.update(hparams, allow_val_change=True)
            wandb.run.name = run_name
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not update wandb config: %s", exc)


def load_adapter_hparams(adapter_checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    hparams_path = os.path.join(adapter_checkpoint_dir, "..", "hparams.json")
    try:
        with open(hparams_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to read adapter hparams: %s", exc)
        return None


def format_adapter_suffix(adapter_checkpoint_dir: str) -> str:
    hparams = load_adapter_hparams(adapter_checkpoint_dir)
    if not hparams:
        return "adapter"
    lora = hparams.get("lora_topk", {})
    r = lora.get("r")
    k = lora.get("k_final", lora.get("k"))
    if r is None or k is None:
        return "adapter"
    return f"adapter_{r}_{k}"


def write_json(path: str, data: Any) -> None:
    # Delegates rather than keeping a second, non-atomic JSON writer in the repo: evals.py
    # rewrites its report paths, so the truncate-on-open failure applies here too.
    write_json_atomic(path, data, indent=2)


# -----------------------------------------------------------------------------
# Quantisation helper
# -----------------------------------------------------------------------------

# def build_quant_config(cfg: Dict[str, Any]) -> BitsAndBytesConfig:
#     """Return a BitsAndBytesConfig from YAML sub‑dict."""
#     qtype = cfg["type"].lower()
#     if qtype == "4bit":
#         return BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
#             bnb_4bit_compute_dtype=getattr(
#                 torch, cfg.get("compute_dtype", "bfloat16")),
#         )
#     if qtype == "8bit":
#         return BitsAndBytesConfig(
#             load_in_8bit=True,
#             llm_int8_threshold=cfg.get("llm_int8_threshold", 6.0),
#         )
#     raise ValueError(f"Unsupported quantisation type: {qtype}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def violates_alternation(msgs: List[Dict[str, str]]) -> bool:
    """
    True  → conversation breaks the template rule
    False → conversation is OK
    Rules:
    • first turn must be 'user' or 'system'
    • thereafter roles must strictly alternate user↔assistant
    """
    if not msgs:  # empty conversation
        return True

    # ── first speaker
    if msgs[0]["role"] not in {"user", "system"}:
        return True

    # ── alternation check
    for prev, curr in zip(msgs, msgs[1:]):
        if prev["role"] == curr["role"]:  # same role twice
            return True
        # user/system must be followed by assistant, and vice‑versa
        if prev["role"] in {"user", "system"} and curr["role"] != "assistant":
            return True
        if prev["role"] == "assistant" and curr["role"] not in {"user", "system"}:
            return True

    return False


_TAG_RE = __import__("re").compile(r"(Human|Assistant):")
_ROLE_MAP = {"Human": "user", "Assistant": "assistant"}


def hh_string_to_messages(text: str) -> List[Dict[str, str]]:
    """
    Convert a raw Anthropic HH conversation string into Chat‑ML messages.
    Example:
        "Human: Hi. Assistant: Hello!"  →
        [{"role":"user","content":"Hi."},
         {"role":"assistant","content":"Hello!"}]
    """
    parts, msgs = _TAG_RE.split(text), []
    for i in range(1, len(parts), 2):
        role_tag, content = parts[i].strip(), parts[i + 1].strip()
        if content:
            msgs.append({"role": _ROLE_MAP[role_tag], "content": content})
    return msgs


def hh_rlhf_preprocess_to_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map HH‑RLHF record → {'chosen': [...], 'rejected': [...]} chat lists."""
    return {
        "chosen": hh_string_to_messages(example["chosen"]),
        "rejected": hh_string_to_messages(example["rejected"]),
    }


def build_quant_config(cfg: Dict[str, Any]) -> BitsAndBytesConfig:
    """Return a BitsAndBytesConfig from YAML sub‑dict."""
    qtype = cfg.type.lower()
    if qtype == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.compute_dtype),
        )
    if qtype == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=cfg.get("llm_int8_threshold", 6.0),
        )
    raise ValueError(f"Unsupported quantisation type: {qtype}")


"""
Metrics eval helper functions
"""


def build_metrics_eval_messages(
    question: str, reply_a: str, reply_b: str
) -> List[Dict]:
    user = (
        f"{question}\n\n"
        f"### Reply A:\n{reply_a}\n\n"
        f"### Reply B:\n{reply_b}\n\n"
        "Which reply is better? Answer with A or B only."
    )
    return [
        {"role": "user", "content": f"{user}\n\n"},
        # The assistant role is left blank; the tokenizer adds the tag.
    ]


def setup_tokenizer_for_chat(tokenizer):
    """Setup tokenizer with proper chat template and padding token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ensure_chat_template(tokenizer, tokenizer.name_or_path)

    return tokenizer


def normalize_chat_messages(
    raw_messages: TypingSequence[Dict[str, Any]],
) -> List[Message]:
    """Normalize raw chat messages into Gemma-compatible message dicts.

    Supported role mappings:
      - human/user/prompter/Human -> user
      - assistant/gpt/bot/Assistant -> assistant

    Messages with unrecognized roles or empty content are filtered out.
    Input messages are expected to be dict-like with role/from + content/value/text.
    Output format: [{"role": "user"|"assistant", "content": "<text>"}].
    """
    normalized: List[Message] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role is None:
            role = msg.get("from")
        if role in ("human", "user", "prompter", "Human"):
            role = "user"
        elif role in ("assistant", "gpt", "bot", "Assistant"):
            role = "assistant"
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content") or msg.get("value") or msg.get("text") or ""
        content = content.strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def build_instruction_messages(
    instruction: str,
    response: str,
    *,
    input_context: str | None = None,
    **kwargs: Any,
) -> List[Message]:
    """Build a user/assistant message pair from instruction-style records.

    The optional input_context (or legacy context kwarg) is treated as extra input
    appended to the instruction with a double newline separator. If instruction
    or response is empty, an empty list is returned.
    """
    context = input_context if input_context is not None else kwargs.get("context")
    prompt = instruction.strip() if instruction else ""
    if context:
        context = context.strip()
        if context:
            prompt = f"{prompt}\n\n{context}" if prompt else context
    response = response.strip() if response else ""
    if not prompt or not response:
        return []
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _make_cache_key(
    tokenizer_name: str,
    max_length: int,
    datasets_to_use: Tuple[str, ...],
    dataset_weights: Optional[Dict[str, float]],
    mix_strategy: str,
    mix_seed: int,
    eval_holdout_ratio: float,
    seed: int,
    pack_sequences: bool,
    include_packing: bool = True,
) -> str:
    """Create a deterministic cache key based on dataset parameters.

    Args:
        include_packing: If False, excludes pack_sequences from the key.
            This is used for the pre-packing cache to avoid re-tokenizing
            when only packing parameters change.
    """
    # Sort dataset_weights for consistency
    sorted_weights = sorted((dataset_weights or {}).items())
    key_dict = {
        "tokenizer_name": tokenizer_name,
        "max_length": max_length,
        "datasets_to_use": sorted(datasets_to_use),  # Sort for consistency
        "dataset_weights": sorted_weights,  # Sorted for consistency
        "mix_strategy": mix_strategy,
        "mix_seed": mix_seed,
        "eval_holdout_ratio": eval_holdout_ratio,
        "seed": seed,
        "version": "v1",  # Increment this if data processing changes
    }
    # Only include packing in key if requested (for final packed cache)
    if include_packing:
        key_dict["pack_sequences"] = pack_sequences
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_path(cache_key: str) -> str:
    """Get the cache file path for a given cache key."""
    # Was hardcoded to /scratch/network/ssd/marek/cache -- one user's scratch, in the
    # LIBRARY, so this failed for anyone else. Override with TOPKLORA_CACHE_DIR.
    cache_root = os.environ.get(
        "TOPKLORA_CACHE_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"),
    )
    cache_dir = os.path.join(cache_root, "topk_lora_datasets")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"sft_datasets_{cache_key}.pkl")


def _cache_exists(cache_path: str) -> bool:
    """Check whether any cache artifact exists for the given base path.

    Accepts either the pickle file itself or the disk directories that we
    create when disk_only=True (or after loading/saving). This prevents
    re-building when only the disk cache is present.
    """
    if os.path.exists(cache_path):
        return True
    train_cache_dir = cache_path.replace(".pkl", "_train_dir")
    eval_cache_dir = cache_path.replace(".pkl", "_eval_dir")
    return os.path.exists(train_cache_dir) and os.path.exists(eval_cache_dir)


def _save_datasets_to_cache(
    train_ds: HFDataset,
    eval_ds: HFDataset,
    cache_path: str,
    *,
    disk_only: bool = False,
) -> None:
    """Save datasets to cache with optional disk-only mode.

    disk_only=True avoids materializing the dataset with `.to_list()`, which is
    very slow and memory hungry for large pre-packing caches. In disk-only mode
    we only persist `save_to_disk` directories, which are what we load first
    anyway.
    """
    print(f"💾 Saving datasets to cache: {cache_path} (disk_only={disk_only})")

    # Save disk format for efficient streaming
    train_cache_dir = cache_path.replace(".pkl", "_train_dir")
    eval_cache_dir = cache_path.replace(".pkl", "_eval_dir")

    try:
        train_ds.save_to_disk(train_cache_dir)
        eval_ds.save_to_disk(eval_cache_dir)
        print("💾 Saved disk format for efficient streaming")
    except Exception as e:
        print(f"⚠️ Failed to save disk format: {e}")

    if not disk_only:
        # Save in multiple formats for flexibility (legacy pickle + features)
        cache_data = {
            "train": {
                "data": train_ds.to_list(),
                "features": train_ds.features,
            },
            "eval": {
                "data": eval_ds.to_list(),
                "features": eval_ds.features,
            },
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("💾 Also saved pickle format for legacy compatibility")

    print(
        f"✅ Datasets cached successfully ({len(train_ds)} train, {len(eval_ds)} eval)"
    )


def _load_datasets_from_cache(
    cache_path: str, streaming: bool = True
) -> Tuple[HFDataset, HFDataset]:
    """Load datasets from cache file with true streaming support."""
    print(f"📁 Loading datasets from cache: {cache_path}")

    # Check if we have disk-based cache directories (best for streaming)
    train_cache_dir = cache_path.replace(".pkl", "_train_dir")
    eval_cache_dir = cache_path.replace(".pkl", "_eval_dir")

    if os.path.exists(train_cache_dir) and os.path.exists(eval_cache_dir):
        print(f"🔄 Loading datasets from disk cache with streaming={streaming}")
        from datasets import load_from_disk

        train_ds = load_from_disk(train_cache_dir)
        eval_ds = load_from_disk(eval_cache_dir)

        if streaming:
            # Only convert training dataset to streaming - keep eval as regular dataset
            # This avoids issues with evaluation loops that expect finite datasets
            train_ds = train_ds.to_iterable_dataset()
            print(
                "✅ Training dataset streaming enabled, eval dataset kept as regular dataset"
            )
        else:
            print(
                f"✅ Datasets loaded from disk cache ({len(train_ds)} train, {len(eval_ds)} eval)"
            )
        return train_ds, eval_ds

    # Check if we have Arrow format files (fallback)
    train_cache_path = cache_path.replace(".pkl", "_train.arrow")
    eval_cache_path = cache_path.replace(".pkl", "_eval.arrow")

    if os.path.exists(train_cache_path) and os.path.exists(eval_cache_path):
        # Load from Arrow format - still loads to memory but more efficient
        train_ds = HFDataset.from_file(train_cache_path)
        eval_ds = HFDataset.from_file(eval_cache_path)

        if streaming:
            # Only convert training dataset to streaming
            train_ds = train_ds.to_iterable_dataset()
            print(
                "✅ Training dataset streaming enabled from Arrow cache, eval dataset regular"
            )
        else:
            print(
                f"✅ Datasets loaded from Arrow cache ({len(train_ds)} train, {len(eval_ds)} eval)"
            )
        return train_ds, eval_ds

    # Fallback to pickle format (legacy)
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    train_ds = HFDataset.from_list(
        cache_data["train"]["data"], features=cache_data["train"]["features"]
    )
    eval_ds = HFDataset.from_list(
        cache_data["eval"]["data"], features=cache_data["eval"]["features"]
    )

    # Save in disk format for future efficient streaming
    try:
        train_ds.save_to_disk(train_cache_dir)
        eval_ds.save_to_disk(eval_cache_dir)
        print("💾 Converted cache to disk format for future streaming")
    except Exception as e:
        print(f"⚠️ Failed to convert to disk format: {e}")

    if streaming:
        # Only convert training dataset to streaming
        train_ds = train_ds.to_iterable_dataset()
        print(
            "✅ Training dataset streaming enabled from pickle cache, eval dataset regular"
        )
    else:
        print(
            f"✅ Datasets loaded from pickle cache ({len(train_ds)} train, {len(eval_ds)} eval)"
        )
    return train_ds, eval_ds


def _ensure_roles(messages: List[Message]) -> List[Message]:
    """Ensure roles are valid and alternating for Gemma chat template."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            role = "assistant" if role == "system" else "user"
        out.append({"role": role, "content": m.get("content", "")})

    # Ensure alternating user/assistant pattern required by Gemma-IT
    if not out:
        return out

    # Fix alternation: must start with user and alternate
    cleaned = []
    expected_role = "user"

    for msg in out:
        if msg["role"] == expected_role:
            cleaned.append(msg)
            expected_role = "assistant" if expected_role == "user" else "user"
        elif expected_role == "assistant" and msg["role"] == "assistant":
            # This is good, add it
            cleaned.append(msg)
            expected_role = "user"
        # Skip messages that break the alternating pattern

    # Ensure we end with an assistant message for training
    if cleaned and cleaned[-1]["role"] == "user":
        # Remove the last user message if there's no assistant response
        cleaned = cleaned[:-1]

    return cleaned


def _encode_with_assistant_mask(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Message],
    max_length: int,
) -> Dict[str, List[int]]:
    """
    Apply Gemma chat template and create labels with -100 on non-assistant tokens.
    """
    messages = _ensure_roles(messages)
    if not messages:
        # Return empty sequence for empty messages
        return {"input_ids": [], "labels": [], "attention_mask": []}

    full_ids: List[int] = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    labels = [-100] * len(full_ids)

    # Mark assistant spans
    for j, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        # Handle empty prefix case
        prefix_ids: List[int] = []
        if j > 0:
            prefix_ids = tokenizer.apply_chat_template(
                messages[:j], tokenize=True, add_generation_prompt=True
            )
        upto_ids: List[int] = tokenizer.apply_chat_template(
            messages[: j + 1], tokenize=True, add_generation_prompt=False
        )
        start = len(prefix_ids)
        end = min(len(upto_ids), len(full_ids))
        for t in range(start, end):
            labels[t] = full_ids[t]

    # Left trim to max_length (keep rightmost)
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]
    attn_mask = [1] * len(full_ids)
    return {"input_ids": full_ids, "labels": labels, "attention_mask": attn_mask}


def load_dolly(split: str = "train") -> HFDataset:
    ds = load_dataset("databricks/databricks-dolly-15k", split=split)

    def to_messages(ex):
        instr = (ex.get("instruction") or "").strip()
        ctx = (ex.get("context") or "").strip()
        user = instr if not ctx else f"{instr}\n\nContext:\n{ctx}"
        resp = (ex.get("response") or "").strip()
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp},
            ]
        }

    # type: ignore
    return ds.map(
        to_messages, remove_columns=[c for c in ds.column_names if c != "messages"]
    )


def load_ultrachat(split: str = "train_sft") -> HFDataset:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)

    def clean(ex):
        msgs = ex.get("messages") or []
        cleaned = []
        for m in msgs:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (m.get("content") or "").strip()
            if content:
                cleaned.append({"role": role, "content": content})
        has_assistant = any(m["role"] == "assistant" for m in cleaned)
        return {"messages": cleaned if has_assistant else None}

    ds = ds.map(clean)
    ds = ds.filter(lambda ex: ex["messages"] is not None)
    return ds  # type: ignore


def load_oasst1(split: str = "train") -> HFDataset:
    ds = load_dataset("OpenAssistant/oasst1", split=split)

    def keep(ex):
        if ex.get("deleted", False):
            return False
        if ex.get("lang") not in (None, "en"):
            return False
        role = ex.get("role")
        if role not in ("prompter", "assistant"):
            return False
        txt = ex.get("text") or ""
        return len(txt.strip()) > 0

    ds = ds.filter(keep)
    rows = ds.to_list()
    by_id = {r["message_id"]: r for r in rows}
    from collections import defaultdict

    children = defaultdict(list)
    for r in rows:
        pid = r.get("parent_id")
        if pid in by_id:
            children[pid].append(r["message_id"])
    leaves = [mid for mid in by_id.keys() if len(children.get(mid, [])) == 0]
    conversations: List[Dict[str, List[Message]]] = []
    for leaf in leaves:
        path = []
        cur = leaf
        seen = set()
        while cur and cur in by_id and cur not in seen:
            seen.add(cur)
            path.append(by_id[cur])
            cur = by_id[cur].get("parent_id")
        path.reverse()
        msgs: List[Message] = []
        for node in path:
            role = node.get("role")
            r = (
                "user"
                if role == "prompter"
                else ("assistant" if role == "assistant" else None)
            )
            if r is None:
                continue
            content = (node.get("text") or "").strip()
            if content:
                msgs.append({"role": r, "content": content})
        if any(m["role"] == "assistant" for m in msgs):
            conversations.append({"messages": msgs})
    return HFDataset.from_list(conversations)


def load_alpaca(split: str = "train") -> HFDataset:
    ds = load_dataset("tatsu-lab/alpaca", split=split)

    def to_messages(ex):
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        output = (ex.get("output") or "").strip()
        return {
            "messages": build_instruction_messages(instr, output, input_context=inp)
        }

    ds = ds.map(to_messages, num_proc=NUM_PROC)
    ds = ds.filter(lambda ex: len(ex.get("messages") or []) > 0)
    return ds.remove_columns([c for c in ds.column_names if c != "messages"])  # type: ignore


def load_tulu_v2(split: str = "train") -> HFDataset:
    ds = load_dataset("allenai/tulu-v2-sft-mixture", split=split)

    def to_messages(ex):
        raw_msgs = ex.get("messages") or ex.get("conversations") or []
        normalized = normalize_chat_messages(raw_msgs)
        return {"messages": normalized}

    ds = ds.map(to_messages, num_proc=NUM_PROC)
    ds = ds.filter(lambda ex: len(ex.get("messages") or []) > 0)
    return ds.remove_columns([c for c in ds.column_names if c != "messages"])  # type: ignore


def _to_messages_with_fallback(ex: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a record to {"messages": [...]} with chat-first fallback."""
    raw_msgs = ex.get("messages") or ex.get("conversations") or []
    if raw_msgs:
        normalized = normalize_chat_messages(raw_msgs)
        # If normalization yields no valid messages, fall back to instruction-style fields
        if normalized:
            return {"messages": normalized}
    instr = (ex.get("instruction") or ex.get("prompt") or "").strip()
    ctx_text = (ex.get("input") or ex.get("context") or "").strip()
    output = ex.get("response") or ex.get("output") or ex.get("completion") or ""
    return {
        "messages": build_instruction_messages(instr, output, input_context=ctx_text)
    }


def load_infinity_instruct(split: str = "train") -> HFDataset:
    ds = load_dataset("BAAI/Infinity-Instruct", "7M_core", split=split)

    ds = ds.map(_to_messages_with_fallback, num_proc=NUM_PROC)
    ds = ds.filter(lambda ex: len(ex.get("messages") or []) > 0)
    return ds.remove_columns([c for c in ds.column_names if c != "messages"])  # type: ignore


def load_h4_instruction(split: str = "train") -> HFDataset:
    ds = load_dataset("HuggingFaceH4/instruction-dataset", split=split)

    ds = ds.map(_to_messages_with_fallback, num_proc=NUM_PROC)
    ds = ds.filter(lambda ex: len(ex.get("messages") or []) > 0)
    return ds.remove_columns([c for c in ds.column_names if c != "messages"])  # type: ignore


def _mix_datasets(
    pieces: List[HFDataset],
    names: List[str],
    *,
    requested_datasets: Tuple[str, ...],
    mix_strategy: str,
    dataset_weights: Dict[str, float] | None,
    seed: int,
) -> HFDataset:
    """Mix datasets using concatenation or weighted interleaving with deterministic sampling.

    Args:
        pieces: List of datasets already loaded for mixing.
        names: Dataset name list matching pieces.
        requested_datasets: Raw dataset names requested from config.
        mix_strategy: "concat" or "interleave".
        dataset_weights: Optional mapping from dataset name to a non-negative weight
            used to derive mixing probabilities for interleaving. If None, datasets
            are interleaved with equal probability.
        seed: Random seed controlling the (reproducible) interleaving order.

    Returns:
        Combined dataset (concatenated or deterministically interleaved).
    """
    if not pieces:
        available = (
            "oasst1",
            "dolly",
            "ultrachat",
            "tulu_v2",
            "infinity_instruct",
            "h4_instruction",
            "alpaca",
        )
        raise ValueError(
            "No datasets selected. "
            f"Requested datasets: {requested_datasets}. "
            f"Available datasets: {available}."
        )
    if mix_strategy == "concat":
        return concatenate_datasets(pieces)
    weights = (
        [dataset_weights.get(name, 1.0) for name in names] if dataset_weights else None
    )
    probabilities = None
    if weights is not None:
        total = sum(weights)
        if total <= 0 or any(w < 0 for w in weights):
            logging.warning(
                "Non-positive or all-zero dataset_weights detected for datasets %s; "
                "falling back to equal mixing probabilities.",
                names,
            )
            probabilities = [1.0 / len(weights)] * len(weights)
        else:
            probabilities = [w / total for w in weights]
    return interleave_datasets(
        pieces,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy="all_exhausted",
    )


def _messages_to_row_format(msgs) -> List[Dict[str, str]]:
    """Convert messages from columnar format to row format if needed.

    Columnar: {"role": ["user", "assistant"], "content": ["...", "..."]}
    Row: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    if not msgs:
        return []

    # Already row format (list of dicts)
    if isinstance(msgs, list):
        return [
            {"role": str(m.get("role") or ""), "content": str(m.get("content") or "")}
            for m in msgs
            if isinstance(m, dict)
        ]

    # Columnar format (dict of lists)
    if isinstance(msgs, dict):
        roles = msgs.get("role", [])
        contents = msgs.get("content", [])
        return [{"role": str(r), "content": str(c)} for r, c in zip(roles, contents)]

    return []


def _standardize_messages_schema(ds: HFDataset) -> HFDataset:
    """Force a consistent struct field order for `messages` to avoid Arrow concat errors.

    We convert to python list, rebuild each message dict in a guaranteed
    role→content order, then recreate the dataset WITHOUT explicit Sequence features
    to avoid columnar format issues.
    """
    if "messages" not in ds.column_names:
        return ds

    # Materialize to list so we can rebuild with consistent format
    rows = ds.to_list()
    fixed_rows = []
    for row in rows:
        msgs = row.get("messages") or []
        # Convert to row format and ensure consistent key order
        fixed_msgs = _messages_to_row_format(msgs)
        fixed_rows.append({"messages": fixed_msgs})

    # Don't use Sequence features - let HF infer schema to avoid columnar format
    return HFDataset.from_list(fixed_rows)


def build_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 8192,
    datasets_to_use: Tuple[str, ...] = ("oasst1", "dolly", "ultrachat"),
    dataset_weights: Optional[Dict[str, float]] = None,
    mix_strategy: str = "concat",
    mix_seed: int = 42,
) -> HFDataset:
    pieces: List[HFDataset] = []
    names: List[str] = []
    if "oasst1" in datasets_to_use:
        pieces.append(_standardize_messages_schema(load_oasst1("train")))
        names.append("oasst1")
    if "dolly" in datasets_to_use:
        pieces.append(_standardize_messages_schema(load_dolly("train")))
        names.append("dolly")
    if "ultrachat" in datasets_to_use:
        pieces.append(_standardize_messages_schema(load_ultrachat("train_sft")))
        names.append("ultrachat")
    if "tulu_v2" in datasets_to_use:
        pieces.append(_standardize_messages_schema(load_tulu_v2("train")))
        names.append("tulu_v2")
    if "infinity_instruct" in datasets_to_use:
        pieces.append(_standardize_messages_schema(load_infinity_instruct("train")))
        names.append("infinity_instruct")
    if "h4_instruction" in datasets_to_use:
        pieces.append(_standardize_messages_schema(load_h4_instruction("train")))
        names.append("h4_instruction")
    if "alpaca" in datasets_to_use:
        pieces.append(_standardize_messages_schema(load_alpaca("train")))
        names.append("alpaca")

    mixed = _mix_datasets(
        pieces,
        names,
        requested_datasets=datasets_to_use,
        mix_strategy=mix_strategy,
        dataset_weights=dataset_weights,
        seed=mix_seed,
    )

    # Filter out empty conversations
    def is_valid(ex):
        raw_msgs = ex.get("messages", [])
        if not raw_msgs:
            return False
        # Convert from columnar to row format if needed
        messages = _messages_to_row_format(raw_msgs)
        if not messages:
            return False
        # Must have at least one assistant message with content
        has_assistant = False
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "assistant" and content and content.strip():
                has_assistant = True
                break
        return has_assistant

    pre_filter_len = len(mixed)
    mixed = mixed.filter(is_valid)
    post_filter_len = len(mixed)
    print(
        f"📊 Filter: {pre_filter_len} -> {post_filter_len} examples ({post_filter_len / max(pre_filter_len, 1) * 100:.1f}% kept)"
    )

    if post_filter_len == 0:
        raise ValueError(
            f"All {pre_filter_len} examples were filtered out! "
            "Check that your datasets have properly formatted messages with assistant responses."
        )

    def tokenize_example(ex):
        raw_msgs = ex["messages"]
        # Convert from columnar to row format if needed
        messages: List[Message] = _messages_to_row_format(raw_msgs)
        return _encode_with_assistant_mask(tokenizer, messages, max_length)

    # Use fewer workers and batching to reduce memory usage on large datasets
    # For 6M+ examples, even 8 workers can OOM - use 4
    num_workers = min(32, NUM_PROC) if len(mixed) > 1_000_000 else min(32, NUM_PROC)
    print(f"🔄 Tokenizing {len(mixed)} examples with {num_workers} workers...")

    tokenized = mixed.map(
        tokenize_example,
        remove_columns=[c for c in mixed.column_names if c != "messages"],
        num_proc=num_workers,
        desc="Tokenizing",
    )

    # Filter out sequences that became empty after tokenization
    print("🔄 Filtering empty sequences...")
    tokenized = tokenized.filter(
        lambda ex: len(ex["input_ids"]) > 0,
        num_proc=num_workers,
    )
    return tokenized


def pack_tokenized_dataset(
    tokenized: HFDataset,
    *,
    max_length: int,
    pad_token_id: int,
    eos_token_id: int,
) -> HFDataset:
    """Pack multiple tokenized examples into <=max_length sequences. Insert EOS between examples and set its label to -100."""

    # Stream over the dataset to avoid materializing all examples at once
    all_examples = tqdm(tokenized, total=len(tokenized), desc="Packing")

    buffers = {"input_ids": [], "labels": [], "attention_mask": []}
    packed = []

    def flush():
        if not buffers["input_ids"]:
            return
        packed.append({k: v[:] for k, v in buffers.items()})
        for k in buffers:
            buffers[k].clear()

    cur_len = 0
    for ex in tqdm(all_examples, desc="Packing"):
        ids = ex["input_ids"]
        labs = ex["labels"]
        attn = ex["attention_mask"]

        # Skip empty sequences
        if not ids:
            continue

        need = len(ids) + (1 if cur_len > 0 else 0)
        if cur_len + need > max_length:
            flush()
            cur_len = 0
        if cur_len > 0:
            buffers["input_ids"].append(eos_token_id)
            buffers["labels"].append(-100)
            buffers["attention_mask"].append(1)
            cur_len += 1
        if len(ids) > max_length:
            ids, labs, attn = ids[-max_length:], labs[-max_length:], attn[-max_length:]
        buffers["input_ids"].extend(ids)
        buffers["labels"].extend(labs)
        buffers["attention_mask"].extend(attn)
        cur_len += len(ids)
        if cur_len >= max_length:
            flush()
            cur_len = 0
    flush()

    print(f"   Packed into {len(packed)} sequences")
    return HFDataset.from_list(packed)


def build_sft_datasets(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 8192,
    datasets_to_use: Tuple[str, ...] = ("oasst1", "dolly", "ultrachat"),
    dataset_weights: Optional[Dict[str, float]] = None,
    mix_strategy: str = "concat",
    mix_seed: int = 42,
    eval_holdout_ratio: float = 0.01,
    seed: int = 42,
    pack_sequences: bool = True,
    use_cache: bool = True,
    streaming: bool = True,  # New parameter for streaming
) -> Tuple[HFDataset, HFDataset]:
    """
    Build SFT datasets with caching and streaming support.

    Args:
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        datasets_to_use: Tuple of dataset names to include
        dataset_weights: Optional mapping of dataset names to mix weights
        mix_strategy: concat or interleave
        mix_seed: Seed for interleaving
        eval_holdout_ratio: Fraction of data to use for evaluation
        seed: Random seed for train/eval split
        pack_sequences: Whether to pack sequences together
        use_cache: Whether to use cached datasets if available
        streaming: Whether to return streaming datasets (memory efficient)

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Optional override: set TOPK_SFT_SAVE_PICKLE=1 to also write pickle caches.
    save_pickle_cache = os.environ.get("TOPK_SFT_SAVE_PICKLE", "0") not in (
        "0",
        "false",
        "False",
        "",
    )

    # Create cache key based on all parameters that affect the output
    tokenizer_name = getattr(tokenizer, "name_or_path", "unknown_tokenizer")

    # Create TWO cache keys:
    # 1. Pre-packing cache (tokenized but not packed) - excludes pack_sequences from key
    # 2. Final cache (packed if requested) - includes pack_sequences in key
    unpacked_cache_key = _make_cache_key(
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        datasets_to_use=datasets_to_use,
        dataset_weights=dataset_weights,
        mix_strategy=mix_strategy,
        mix_seed=mix_seed,
        eval_holdout_ratio=eval_holdout_ratio,
        seed=seed,
        pack_sequences=pack_sequences,
        include_packing=False,  # Exclude packing from key for tokenized cache
    )
    final_cache_key = _make_cache_key(
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        datasets_to_use=datasets_to_use,
        dataset_weights=dataset_weights,
        mix_strategy=mix_strategy,
        mix_seed=mix_seed,
        eval_holdout_ratio=eval_holdout_ratio,
        seed=seed,
        pack_sequences=pack_sequences,
        include_packing=True,  # Include packing for final cache
    )

    unpacked_cache_path = _get_cache_path(f"tokenized_{unpacked_cache_key}")
    final_cache_path = _get_cache_path(f"final_{final_cache_key}")

    # In distributed training, only rank 0 should build the dataset
    # Other ranks wait for cache to be ready
    global_rank = get_global_rank()
    world_size = get_world_size()

    if global_rank == 0:
        print(f"📁 Tokenized (pre-packing) cache path: {unpacked_cache_path}")
        print(f"📁 Final cache path: {final_cache_path}")

    # Try to load from final (packed) cache first
    if use_cache and _cache_exists(final_cache_path):
        try:
            if global_rank == 0:
                print("✅ Found final cached dataset, loading...")
            return _load_datasets_from_cache(final_cache_path, streaming=streaming)
        except Exception as e:
            if global_rank == 0:
                print(
                    f"⚠️ Failed to load from final cache ({e}), checking pre-packing cache..."
                )

    # In distributed training, only rank 0 builds the dataset, others wait
    if world_size > 1 and global_rank != 0:
        # Wait for rank 0 to build and cache the dataset
        print(f"[Rank {global_rank}] Waiting for rank 0 to build dataset...")
        import time

        max_wait = 7200 * 5  # 10 hours max wait
        waited = 0
        while not _cache_exists(final_cache_path) and waited < max_wait:
            time.sleep(30)
            waited += 30
        if _cache_exists(final_cache_path):
            print(f"[Rank {global_rank}] Cache ready, loading...")
            return _load_datasets_from_cache(final_cache_path, streaming=streaming)
        else:
            raise RuntimeError(
                f"[Rank {global_rank}] Timeout waiting for dataset cache"
            )

    # Try to load from pre-packing cache (tokenized but not packed)
    train_ds = None
    eval_ds = None

    if use_cache and _cache_exists(unpacked_cache_path):
        try:
            print("✅ Found tokenized (pre-packing) cache, loading...")
            train_ds, eval_ds = _load_datasets_from_cache(
                unpacked_cache_path, streaming=False
            )
            print(
                f"📊 Loaded from pre-packing cache: {len(train_ds)} train, {len(eval_ds)} eval"
            )
        except Exception as e:
            print(
                f"⚠️ Failed to load from pre-packing cache ({e}), building from scratch..."
            )
            train_ds = None
            eval_ds = None

    # Build datasets from scratch if not loaded from cache
    if train_ds is None or eval_ds is None:
        print("🔨 Building datasets from scratch (this may take a while)...")
        full = build_sft_dataset(
            tokenizer,
            max_length=max_length,
            datasets_to_use=datasets_to_use,
            dataset_weights=dataset_weights,
            mix_strategy=mix_strategy,
            mix_seed=mix_seed,
        )
        split = full.train_test_split(test_size=eval_holdout_ratio, seed=seed)
        train_ds, eval_ds = split["train"], split["test"]

        # Safety check: ensure we have data
        full_len = len(full)
        if full_len == 0:
            raise ValueError(
                "Dataset is empty after tokenization! Check dataset loading and filtering."
            )

        # Safety check: ensure eval dataset is not empty
        if len(eval_ds) == 0:
            print("⚠️ Evaluation dataset is empty, adjusting holdout ratio...")
            # Use a minimum of 10 examples for evaluation or 1% of data, whichever is larger
            min_eval_size = max(10, int(full_len * 0.01))
            adjusted_ratio = min(min_eval_size / full_len, 0.1)  # Cap at 10%
            split = full.train_test_split(test_size=adjusted_ratio, seed=seed)
            train_ds, eval_ds = split["train"], split["test"]
            print(
                f"📊 Adjusted eval dataset size: {len(eval_ds)} examples ({adjusted_ratio:.3f} ratio)"
            )

        # Save tokenized (pre-packing) cache
        if use_cache:
            try:
                print("💾 Saving tokenized (pre-packing) cache...")
                _save_datasets_to_cache(
                    train_ds, eval_ds, unpacked_cache_path, disk_only=True
                )
                print("✅ Pre-packing cache saved successfully")
            except Exception as e:
                print(f"⚠️ Failed to save pre-packing cache ({e}), continuing...")

    # Apply packing if requested
    if pack_sequences:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        eos_id = tokenizer.eos_token_id or pad_id
        print("📦 Packing training sequences...")
        train_ds = pack_tokenized_dataset(
            train_ds, max_length=max_length, pad_token_id=pad_id, eos_token_id=eos_id
        )
        print("📦 Packing evaluation sequences...")
        eval_ds = pack_tokenized_dataset(
            eval_ds, max_length=max_length, pad_token_id=pad_id, eos_token_id=eos_id
        )

    # Save final (packed) cache
    if use_cache:
        try:
            print(
                f"💾 Saving final cache... (pickle={'on' if save_pickle_cache else 'off'})"
            )
            _save_datasets_to_cache(
                train_ds,
                eval_ds,
                final_cache_path,
                disk_only=not save_pickle_cache,
            )
            print("✅ Final cache saved successfully")
        except Exception as e:
            print(f"⚠️ Failed to save final cache ({e}), continuing without caching...")

    # Convert to streaming if requested (only for training dataset)
    if streaming:
        train_ds = train_ds.to_iterable_dataset()
        # Keep eval_ds as regular dataset for compatibility with evaluation loops
        print(
            "✅ Training dataset converted to streaming format, eval dataset kept regular"
        )

    return train_ds, eval_ds


def normalize_ultrachat_messages(msgs):
    """
    Preserve content while enforcing alternation:
      • keep only non-empty user/assistant turns
      • merge consecutive same-role turns (concat with blank line)
      • if first turn is assistant → prepend a minimal user stub ' '
      • if last turn is user → we'll set add_generation_prompt=True (no deletion)
    Returns (fixed_messages, add_gen_prompt)
    """
    # 1) keep only user/assistant with text
    cleaned = [
        {"role": m["role"], "content": (m.get("content") or "").strip()}
        for m in (msgs or [])
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]

    if not cleaned:
        # fallback: single empty user so the template is satisfiable
        return [{"role": "user", "content": " "}], True

    # 2) merge consecutive same-role
    merged = []
    for m in cleaned:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(m)

    # 3) ensure we start with user
    if merged[0]["role"] != "user":
        merged = [{"role": "user", "content": " "}] + merged

    # 4) enforce alternation (after merging it’s rare to fail; still be safe)
    alternated = [merged[0]]
    expect = "assistant"
    for m in merged[1:]:
        if m["role"] == expect:
            alternated.append(m)
            expect = "user" if expect == "assistant" else "assistant"
        else:
            # If out-of-order, insert a stub to keep content (no drops)
            alternated.append({"role": expect, "content": " "})
            expect = "user" if expect == "assistant" else "assistant"
            if m["role"] == expect:
                alternated.append(m)
                expect = "assistant" if expect == "user" else "user"

    # 5) generation prompt if last is user (don’t delete their turn)
    add_gen = alternated[-1]["role"] == "user"
    return alternated, add_gen


# From https://github.com/EleutherAI/lm-evaluation-harness/blob/6d62a69cb5db963f998c486af6efee43fca63dd3/lm_eval/tasks/wikitext/preprocess_wikitext.py#L4
def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def generate_completions_from_prompts(
    model,
    tokenizer,
    prompts,
    *,
    device: str,
    max_length=None,
    truncation: bool = True,
    gen_kwargs=None,
    end_of_turn_id=None,  # Unused (deprecated)
):
    """Tokenize prompts, generate, and return decoded completions."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=truncation,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        generated = model.generate(**enc, **(gen_kwargs or {}))

    prompt_length = enc["input_ids"].shape[1]

    completions = []
    for output_ids in generated:
        completion_ids = output_ids[prompt_length:]
        completions.append(
            tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        )

    return completions
