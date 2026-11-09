"""Assert TopK-LoRA wrapping preserves the base model's attention biases.

Phase 0 step 0.3e of the Qwen2.5-1.5B replication, and a Gate A row (plan
`docs/replication-qwen2.5-1.5b.md` §3.2(5) and §7).

**Why this needs checking on Qwen and never did on gemma.** Qwen2 hardcodes `bias=True` on
q/k/v (it is not config-driven) and `False` on o_proj; gemma-2 has no attention bias anywhere.
A wrapper that rebuilds the base `nn.Linear` instead of holding a reference would silently drop
those biases, leaving a subtly wrong base model that still loads, still wraps, and still
generates. The §3 smoke test's `replaced=7` plus a working forward is necessary but not
sufficient -- neither observes the bias.

**Why it compares tensors and not just `is not None`.** A wrapper could plausibly recreate a
`bias` parameter of the right shape, zero-initialised. That passes a None-check and is still the
wrong model, so the pre-wrap values are cloned and compared exactly. Rule 12: assert the value,
not the shape.

The wrap path is the one `train.py:870-893` uses -- `resolve_target_modules` -> `LoraConfig` ->
`get_peft_model` -> `wrap_topk_lora_modules` -- rather than a local imitation of it, so a change
in how training wraps shows up here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from src.utils import resolve_target_modules, wrap_topk_lora_modules

# o_proj is expected to have NO bias on Qwen2; recording it makes the asymmetry an assertion
# rather than a thing the reader has to remember.
ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def snapshot_biases(model, layer: int) -> Dict[str, Any]:
    """Clone every attention bias on `layer`, or record None where there is none."""
    out: Dict[str, Any] = {}
    for proj in ATTN_PROJECTIONS:
        mod = model.get_submodule(f"model.layers.{layer}.self_attn.{proj}")
        bias = getattr(mod, "bias", None)
        out[proj] = None if bias is None else bias.detach().clone()
    return out


def find_base_linear(model, layer: int, proj: str):
    """The wrapped base `nn.Linear`, wherever PEFT/TopK put it.

    After wrapping, the module at the original path is a TopK wrapper around a PEFT `lora.Linear`
    around the original `nn.Linear`. Walk down to whatever actually carries `.bias` rather than
    assuming a fixed depth, so this keeps working if the wrapper nests differently.
    """
    node = model.get_submodule(f"base_model.model.model.layers.{layer}.self_attn.{proj}")
    for _ in range(6):
        if isinstance(node, torch.nn.Linear):
            return node
        nxt = getattr(node, "base_layer", None) or getattr(node, "inner", None) \
            or getattr(node, "wrapped", None) or getattr(node, "base", None)
        if nxt is None:
            return node
        node = nxt
    return node


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base_model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--r", type=int, default=42)
    p.add_argument("--alpha", type=int, default=84)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--dtype", default="float32",
                   help="float32 keeps the comparison exact; the check is about identity, not speed")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"=== wrap integrity ===\n  base_model : {args.base_model}\n  layer      : {args.layer}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)

    before = snapshot_biases(model, args.layer)
    present = {k: (v is not None) for k, v in before.items()}
    print(f"  pre-wrap bias present: {present}")

    lora_cfg = SimpleNamespace(module_type="mlp_attn", layer=args.layer)
    target_modules = resolve_target_modules(lora_cfg)
    peft_model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, target_modules=target_modules,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))
    replaced, _ = wrap_topk_lora_modules(
        peft_model, k=args.k, temperature=1.0, temperature_schedule="constant",
        k_schedule="constant", k_final=args.k, temperature_final=0.1,
        is_topk_experiment=True, set_train=True, hard_eval=True, relu_latents=True,
        alpha_over_r=True, k_warmup_frac=0.2, topk_mode="topk",
    )
    print(f"  wrapped modules: {replaced} (expected {len(target_modules)})")

    checks: List[Dict[str, Any]] = []
    for proj in ATTN_PROJECTIONS:
        base = find_base_linear(peft_model, args.layer, proj)
        after = getattr(base, "bias", None)
        want = before[proj]
        if want is None:
            ok = after is None
            detail = "no bias before, none after" if ok else "bias APPEARED where base had none"
        elif after is None:
            ok, detail = False, "bias DROPPED by wrapping"
        else:
            ok = bool(torch.equal(after.detach().cpu(), want.cpu()))
            detail = "identical to pre-wrap tensor" if ok else "bias present but VALUES CHANGED"
        checks.append({"proj": proj, "pass": ok, "detail": detail,
                       "had_bias": want is not None,
                       "n_elem": (0 if want is None else want.numel())})
        print(f"  [{'ok  ' if ok else 'FAIL'}] {proj}: {detail}")

    wrapped_ok = replaced == len(target_modules)
    if not wrapped_ok:
        print(f"  [FAIL] wrapped {replaced} modules, expected {len(target_modules)}")
    verdict = "PASS" if (all(c["pass"] for c in checks) and wrapped_ok) else "FAIL"
    print(f"\nVERDICT: {verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"base_model": args.base_model, "layer": args.layer, "replaced": replaced,
             "n_target_modules": len(target_modules), "checks": checks,
             "verdict": verdict}, indent=2, sort_keys=True))
        print(f"record -> {args.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
