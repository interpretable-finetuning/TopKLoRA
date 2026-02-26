"""
Multi-adapter comparison REPL

Loads a base model and zero or more adapters, then drops into an interactive
loop.  For each prompt, generates a response from the base model *and* every
loaded adapter so you can compare them side-by-side.

Usage:
    python preview.py
    python preview.py 'model.adapters={sft: path/to/sft, dpo: path/to/dpo}'
    python preview.py 'model.adapters={my_adapter: path/to/adapter}' chat_format=false
"""

import textwrap
import torch
import hydra
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from omegaconf import DictConfig, OmegaConf

from src.utils import (
    ensure_chat_template_and_special_tokens,
    configure_eos_eot,
)
from src.models import TopKLoRALinearSTE


log = logging.getLogger(__name__)

# ── ANSI helpers ──────────────────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"

ADAPTER_COLORS = [GREEN, MAGENTA, BLUE, YELLOW, CYAN]

INDENT = "    "


def _indent(text: str) -> str:
    return textwrap.indent(text, INDENT)


def _header(label: str, color: str) -> str:
    bar = "\u2500" * 50
    return f"{color}{BOLD}{bar}\n  {label}\n{bar}{RESET}"


def _wrap_topk_for_adapter(model, adapter_name, k):
    """Wrap only the LoRA modules that contain *adapter_name* with TopK.

    Returns a dict  {module_name: (topk_wrapper, original_peft_layer)}
    that can be passed to ``_unwrap_topk`` to restore the originals.
    """
    targets = []
    for name, module in model.named_modules():
        if (
            getattr(module, "lora_A", None) is not None
            and adapter_name in module.lora_A
        ):
            targets.append(name)

    wrapped = {}
    for name in targets:
        peft_layer = model.get_submodule(name)
        parent_name = ".".join(name.split(".")[:-1])
        parent = model.get_submodule(parent_name) if parent_name else model
        attr = name.split(".")[-1]

        topk_layer = TopKLoRALinearSTE(
            base=peft_layer,
            layer_name=name,
            k=k,
            temperature=0.0,
            temperature_schedule="constant",
            k_schedule="constant",
            k_final=k,
            temperature_final=0.0,
            is_topk_experiment=True,
        )
        try:
            device = next(peft_layer.parameters()).device
        except StopIteration:
            device = next(model.parameters()).device
        topk_layer = topk_layer.to(device=device).eval()

        setattr(parent, attr, topk_layer)
        wrapped[name] = (topk_layer, peft_layer)

    return wrapped


def _unwrap_topk(model, wrapped_modules):
    """Restore original PEFT layers after TopK generation."""
    for name, (_topk, peft_layer) in wrapped_modules.items():
        parent_name = ".".join(name.split(".")[:-1])
        parent = model.get_submodule(parent_name) if parent_name else model
        attr = name.split(".")[-1]
        setattr(parent, attr, peft_layer)


@hydra.main(
    version_base=None,
    config_path="config/preview_config",
    config_name="default",
)
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── load model + tokenizer ───────────────────────────────────────────
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    quant_cfg = None
    if cfg.model.load_in_4bit or cfg.model.load_in_8bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=cfg.model.load_in_4bit,
            load_in_8bit=cfg.model.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16
            if cfg.model.dtype == "bfloat16"
            else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Resolve adapters early so we can use an adapter path for the tokenizer
    adapters = OmegaConf.to_container(cfg.model.adapters, resolve=True) or {}

    # Load tokenizer from an adapter if available (adapters save the
    # fully-configured tokenizer with chat template + special tokens).
    # Fall back to the base model otherwise.
    tokenizer_source = next(iter(adapters.values()), None) or cfg.model.model_name
    log.info("Loading tokenizer: %s", tokenizer_source)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    log.info("Loading model: %s", cfg.model.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=dtype_map[cfg.model.dtype],
        device_map=cfg.model.device if cfg.model.device != "cpu" else None,
        quantization_config=quant_cfg,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    ensure_chat_template_and_special_tokens(tokenizer, model, cfg.model.model_it_name)
    eot_token, eot_token_id = configure_eos_eot(tokenizer, model)
    print(f"EOT token: '{eot_token}' (ID: {eot_token_id})")
    print(f"EOS token ID(s): {model.generation_config.eos_token_id}")

    # ── load adapters ────────────────────────────────────────────────────
    adapter_names = list(adapters.keys())
    has_adapters = len(adapter_names) > 0

    # Assign a color to each adapter (cycling if more than palette size)
    adapter_color = {
        name: ADAPTER_COLORS[i % len(ADAPTER_COLORS)]
        for i, name in enumerate(adapter_names)
    }

    if has_adapters:
        # Load all adapters into the PEFT model.  TopK wrapping is done
        # per-adapter at generation time (see REPL loop) because different
        # adapters may target different modules (MLP vs attention).
        first_name = adapter_names[0]
        first_path = adapters[first_name]
        log.info("Loading adapter '%s': %s", first_name, first_path)
        model = PeftModel.from_pretrained(
            model, first_path, adapter_name=first_name, is_trainable=False
        )

        for name in adapter_names[1:]:
            path = adapters[name]
            log.info("Loading adapter '%s': %s", name, path)
            model.load_adapter(path, adapter_name=name)

        log.info("Loaded %d adapter(s): %s", len(adapter_names), adapter_names)
    else:
        log.warning("No adapters configured — running base model only")

    model.eval()
    device = next(model.parameters()).device

    # ── helper ────────────────────────────────────────────────────────────
    def generate(prompt: str) -> str:
        if cfg.chat_format:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        print(f"\n{DIM}Formatted input:\n{text}{RESET}\n")

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.generation.max_new_tokens,
                do_sample=cfg.generation.do_sample,
                temperature=cfg.generation.temperature
                if cfg.generation.do_sample
                else None,
                top_p=cfg.generation.top_p if cfg.generation.do_sample else None,
                repetition_penalty=cfg.generation.repetition_penalty,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=False)

    # ── REPL ──────────────────────────────────────────────────────────────
    if has_adapters:
        adapter_list = ", ".join(f"{adapter_color[n]}{n}{RESET}" for n in adapter_names)
        print(
            f"\n{BOLD}Model loaded with {len(adapter_names)} adapter(s):{RESET} "
            f"{adapter_list}"
        )
    else:
        print(f"\n{BOLD}Model loaded (base only, no adapters).{RESET}")
    print(f"Type a prompt (or {DIM}'quit'{RESET} / {DIM}Ctrl-D{RESET} to exit).\n")

    while True:
        try:
            prompt = input(f"{CYAN}{BOLD}>>> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt.lower() == "quit":
            break

        if not has_adapters:
            response = generate(prompt)
            print(_header("base", YELLOW))
            print(_indent(response))
            print()
        else:
            # Generate with base model (adapters disabled)
            model.disable_adapter_layers()
            response = generate(prompt)
            print(_header("base", YELLOW))
            print(_indent(response))
            print()

            # Generate with each adapter (wrap TopK per-adapter)
            model.enable_adapter_layers()
            for name in adapter_names:
                model.set_adapter(name)
                wrapped = _wrap_topk_for_adapter(model, name, cfg.model.k)
                response = generate(prompt)
                _unwrap_topk(model, wrapped)
                print(_header(name, adapter_color[name]))
                print(_indent(response))
                print()


if __name__ == "__main__":
    main()
