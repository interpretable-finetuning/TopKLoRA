"""PREFLIGHT for an analysis: does the loaded adapter compute the forward its ARM is defined by?

ReLU-ablation pre-registration (docs/relu-ablation-arm-c-preregistration.md), readout R0(b).
`verify_adapter_arm` checks the config on disk; this checks the forward the CLCD loader actually
runs. On real prompts it recomputes each wrapped module's post-gate latents from the module's own
input, using the arm's DEFINITION written out here (not the module's flags), and requires an exact
match and exactly k nonzero latents per token:

    relu_signed (arm A, the paper):  z = ReLU(A x),  keep top-k of z,   output z on the kept latents
    abs         (arm C):             z = A x,        keep top-k of |z|, output z (signed) on them

The reference does not read `relu_latents` / `topk_mode` from the module: a module whose flags
were silently wrong would then be compared with itself and pass. The caller states the arm.

Considered instead: `scripts/latent_inspection.py::_check_hook` (the same idea for arm A only,
but welded to delphi hookpoints and the 7B inspection's pinned draw), `gate_a` (a behavioural gate,
not a forward identity) and `verify_adapter_arm` (config only). None checks an arbitrary arm's
forward through `load_organism`, so this is a module of its own, beside the other `verify_*` checks.

    python -m src.clcd.verify_forward --adapter <dir> --base_model <b> --data <d> --arm abs --out r0b.json

Exit 0 = every module matches the arm on every token. Exit 1 = it does not.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import torch
import torch.nn.functional as F

from src import data as chat_format
from src.data import load_jsonl_rows, load_tags, write_json_atomic

ARMS = ("relu_signed", "abs")


def reference_latents(s: torch.Tensor, k: int, arm: str) -> torch.Tensor:
    """Post-gate latents by the arm's definition, from pre-activations s = A x (..., r)."""
    if arm == "relu_signed":
        z = torch.relu(s)
        score = z
    elif arm == "abs":
        z = s
        score = s.abs()
    else:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    mask = torch.zeros_like(z).scatter_(-1, score.topk(k, dim=-1).indices, 1.0)
    return z * mask


def check_forward(model, tok, wrapped, prompts, arm: str) -> Dict[str, Any]:
    """Per module: max |hooked - reference|, nonzero latents per token, share of them negative.

    Batch 1, so no padding position is compared. Raises if a module never ran.
    """
    if not prompts:
        raise ValueError("no prompts -- refusing to report a check that never ran")
    seen: Dict[str, torch.Tensor] = {}
    hooks = [
        mod.register_forward_hook(lambda m_, args, out, name=name: seen.__setitem__(name, args[0].detach()))
        for name, mod in wrapped.items()
    ]
    per = {name: {"max_abs_diff": 0.0, "nnz_min": None, "nnz_max": 0, "selected": 0, "negative": 0}
           for name in wrapped}
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for prompt in prompts:
                ids = tok(prompt, return_tensors="pt").input_ids.to(device)  # as generate_responses
                seen.clear()
                model(ids)
                for name, mod in wrapped.items():
                    if name not in seen or mod._last_z_sparse is None:
                        raise RuntimeError(f"{name}: module did not run on the forward")
                    s = F.linear(seen[name], mod.A_module.weight)
                    want = reference_latents(s, int(mod._current_k()), arm)
                    got = mod._last_z_sparse
                    nnz = (got != 0).sum(-1)
                    c = per[name]
                    c["max_abs_diff"] = max(c["max_abs_diff"], (got.float() - want.float()).abs().max().item())
                    c["nnz_min"] = int(nnz.min()) if c["nnz_min"] is None else min(c["nnz_min"], int(nnz.min()))
                    c["nnz_max"] = max(c["nnz_max"], int(nnz.max()))
                    c["selected"] += int((got != 0).sum())
                    c["negative"] += int((got < 0).sum())
    finally:
        for h in hooks:
            h.remove()
    return per


def verdict(per: Dict[str, Any], k_by_module: Dict[str, int], arm: str) -> list:
    """Problems; empty = pass. Exact match, and the arm's count of nonzero latents per token:
    exactly k for `abs`; at most k for `relu_signed`, where the ReLU zeroes a selected latent
    whenever fewer than k pre-activations are positive (measured: 5-6% of positions, 7B l17_25)."""
    problems = []
    for name, c in per.items():
        k = k_by_module[name]
        if c["max_abs_diff"] != 0.0:
            problems.append(f"{name}: max|diff| {c['max_abs_diff']:.3g} vs the arm's definition")
        ok = (c["nnz_min"] == c["nnz_max"] == k) if arm == "abs" else (c["nnz_max"] <= k)
        if not ok:
            want = f"exactly {k}" if arm == "abs" else f"at most {k}"
            problems.append(f"{name}: nonzero latents per token {c['nnz_min']}..{c['nnz_max']}, expected {want}")
    return problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--adapter", required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--offset", type=int, default=100)
    p.add_argument("--n", type=int, default=16, help="prompts per tag (triggered and clean)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from src.clcd.organism import load_organism

    trigger_tag, clean_tag = load_tags(args.data)
    prompts_by_tag = {}
    for split, tag in (("eval_triggered", trigger_tag), ("eval_clean", clean_tag)):
        qs = load_jsonl_rows(args.data, split, args.offset, args.n)
        if len(qs) != args.n:
            raise SystemExit(f"{split}[{args.offset}:] returned {len(qs)} rows, expected {args.n}")
        prompts_by_tag[split] = (qs, tag)
    dtype = getattr(torch, args.dtype)
    model, tok, wrapped = load_organism(args.adapter, base_model=args.base_model, device="cuda", dtype=dtype)
    model = model.to(dtype).eval()
    prompts = [chat_format.render_prompt(tok, question=q, tag=tag)
               for qs, tag in prompts_by_tag.values() for q in qs]

    per = check_forward(model, tok, wrapped, prompts, args.arm)
    problems = verdict(per, {n: int(m._current_k()) for n, m in wrapped.items()}, args.arm)
    sel = sum(c["selected"] for c in per.values())
    neg = sum(c["negative"] for c in per.values())
    print(f"=== forward check: arm={args.arm} modules={len(per)} prompts={len(prompts)} ===")
    print(f"  selected latents {sel}, negative {neg} ({neg / sel:.4f})")
    for msg in problems:
        print(f"  [FAIL] {msg}")
    print(f"VERDICT: {'FAIL' if problems else 'PASS'}")
    if args.out:
        write_json_atomic(args.out, {"adapter": args.adapter, "arm": args.arm, "n_prompts": len(prompts),
                                     "per_module": per, "problems": problems,
                                     "verdict": "FAIL" if problems else "PASS"}, indent=2)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
