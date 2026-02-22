from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class AdditiveInjectionSpec:
    """Specification for one additive residual hook."""

    module_name: str
    vector: torch.Tensor
    beta: float


def _resolve_candidate_names(module_name: str) -> List[str]:
    names = [module_name]
    if module_name.endswith(".topk"):
        names.append(module_name[: -len(".topk")])

    # Common PEFT path prefixes vs merged-model paths.
    prefixes = [
        "base_model.model.model.",
        "base_model.model.",
    ]
    for pref in prefixes:
        if module_name.startswith(pref):
            names.append(module_name[len(pref) :])

    # Make sure we can match module names by suffix too.
    if ".layers." in module_name:
        suffix = module_name[module_name.index("layers.") :]
        names.append(suffix)
        names.append(f"model.{suffix}")

    dedup = []
    seen = set()
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        dedup.append(n)
    return dedup


def _find_module(model: nn.Module, module_name: str) -> tuple[str, nn.Module]:
    modules = dict(model.named_modules())

    # Exact/candidate matches first
    for cand in _resolve_candidate_names(module_name):
        if cand in modules:
            return cand, modules[cand]

    # Suffix match fallback
    candidates = _resolve_candidate_names(module_name)
    for cand in candidates:
        for name, module in modules.items():
            if name.endswith(cand):
                return name, module

    raise KeyError(
        f"Could not resolve module '{module_name}' for additive injection."
    )


def _build_additive_hook(vector: torch.Tensor, beta: float):
    def _hook(module: nn.Module, _input, output):
        if isinstance(output, tuple):
            if not output:
                return output
            first = output[0]
            if not torch.is_tensor(first):
                return output
            add = vector.to(device=first.device, dtype=first.dtype)
            while add.ndim < first.ndim:
                add = add.unsqueeze(0)
            out0 = first + float(beta) * add
            return (out0,) + tuple(output[1:])

        if not torch.is_tensor(output):
            return output

        add = vector.to(device=output.device, dtype=output.dtype)
        while add.ndim < output.ndim:
            add = add.unsqueeze(0)
        return output + float(beta) * add

    return _hook


def register_additive_injection_hooks(
    model: nn.Module,
    specs: Iterable[AdditiveInjectionSpec],
) -> Dict[str, torch.utils.hooks.RemovableHandle]:
    handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}
    for spec in specs:
        if spec.beta == 0.0:
            continue
        matched_name, module = _find_module(model, spec.module_name)
        hook = _build_additive_hook(spec.vector, spec.beta)
        handles[matched_name] = module.register_forward_hook(hook)
        logger.debug(
            "Registered additive injection hook on %s (beta=%s)",
            matched_name,
            spec.beta,
        )
    return handles


def remove_injection_hooks(handles: Dict[str, torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles.values():
        handle.remove()


@contextmanager
def AdditiveInjectionContext(model: nn.Module, specs: Iterable[AdditiveInjectionSpec]):
    handles = register_additive_injection_hooks(model, specs)
    try:
        yield handles
    finally:
        remove_injection_hooks(handles)
