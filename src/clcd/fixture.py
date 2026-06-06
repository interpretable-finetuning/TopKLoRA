"""CLCD development fixture: a randomly-initialized, TopKLoRA-wrapped model.

Purpose: exercise the *mechanics* of circuit discovery (read latents, inject
latents, compute the behavioural scalar mu) before a trained sleeper adapter
exists. Random weights mean every number is meaningless -- only tensor shapes
and the module hooks are real. Success here is "the plumbing does what we
predict", never "we found the backdoor".

Reuses the genuine wrapping path (src.utils.wrap_topk_lora_modules) so this
fixture and the eventual trained adapter run identical code.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Gemma2Config

from src.utils import wrap_topk_lora_modules


def build_random_fixture(
    seed: int = 0, r: int = 8, k: int = 4, set_train: bool = False
):
    """Return (model, wrapped_modules) for a tiny random Gemma-2-shaped LM.

    wrapped_modules: dict[str, TopKLoRALinearSTE] -- a node's home module `m`.
    set_train=False -> hard-gate eval forward (the true forward, assumption A5);
    flip to True later for the soft-gate attribution path (M5).
    """
    torch.manual_seed(seed)

    # Tiny Gemma-2: same architecture family as the real target (2b/9b), shrunk
    # so it runs instantly on CPU. Dims are arbitrary but internally consistent.
    config = Gemma2Config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.eval()

    # Inject PEFT LoRA on the same module *types* the sleeper config targets
    # (mlp + attn projections). PEFT matches target_modules by name suffix, so
    # every layer's matching projection gets its own lora_A / lora_B pair.
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=2 * r,
        target_modules=[
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)

    # Swap each PEFT LoRA layer for TopKLoRALinearSTE using the real eval-time
    # config: is_topk_experiment=True turns on sparse top-k; hard_eval keeps the
    # true hard gate; constant schedules so k/tau never depend on training
    # `progress`. This mirrors how src/evals.py loads the trained adapter.
    replaced, wrapped_modules = wrap_topk_lora_modules(
        model,
        k=k,
        temperature=1.0,
        temperature_schedule="constant",
        k_schedule="constant",
        k_final=k,
        temperature_final=None,
        is_topk_experiment=True,
        set_train=set_train,
        hard_eval=True,
    )
    assert replaced == len(wrapped_modules) and replaced > 0, replaced

    # PEFT initializes lora_B = 0 (the adapter starts as a no-op so fine-tuning
    # begins from the base model). That would make every latent decode to zero,
    # so no intervention could ever affect the output. Give the fixture a
    # nonzero random decoder -- a trained adapter has one -- so ablate/insert/IG
    # actually move mu. Re-seed so B is deterministic regardless of upstream RNG.
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for mod in wrapped_modules.values():
            mod.B_module.weight.normal_(0, 2.0)
    return model, wrapped_modules
