"""M2/M3: read and inject TopKLoRA node activations a(m,d,p).

A node's activation is the post-gate scalar a = z * gate (spec section 1),
cached per module as `_last_z_sparse` of shape (1, seq, r) after a forward pass.
read_latents() snapshots these -- the a0 / a1 endpoints for attribution.
inject() overrides them and recomputes the residual write -- the one primitive
behind both integrated gradients (M5) and ablation/insertion (M6).
"""

from __future__ import annotations

import contextlib

import torch


def read_latents(model, input_ids: torch.Tensor, wrapped_modules: dict) -> dict:
    """Run one forward pass and snapshot each module's post-gate activations.

    Returns {module_name: a}, a detached (1, seq, r) tensor; a[0, p, d] is node
    (m, d, p). Cloned so the snapshot is a stable endpoint, unaffected by later
    forward passes (which overwrite the live `_last_z_sparse` caches in place of
    reassigning new tensors).
    """
    with torch.no_grad():
        model(input_ids)
    return {m: mod._last_z_sparse.clone() for m, mod in wrapped_modules.items()}


@contextlib.contextmanager
def inject(wrapped_modules: dict, overrides: dict):
    """Override chosen modules' post-gate latents for the duration of the block.

    overrides: {module_name: override}, where override is either
      - a (1, seq, r) tensor used directly as that module's post-gate latents a
        (fixed-tensor mode -- single input, may carry grad; used by IG, M5), or
      - a callable f(current_a) -> new_a that transforms the module's own
        post-gate latents `_last_z_sparse` on each forward (callable mode --
        adapts to each forward's shape; used by ablation/insertion, M6).
    A forward hook replaces the module's residual write with base_layer(x) +
    decode(a). Injecting a module's own read-back a is a no-op; edit a first to
    ablate (zero entries) or insert (copy from another run).

    Hooks fire under whatever grad context the caller's forward runs in -- pass a
    requires_grad tensor and skip torch.no_grad() to get d(target)/d(a) (M5).
    Reversible: hooks are removed on exit.
    """
    handles = []

    def make_hook(override):
        def hook(module, args, output):
            a_new = override(module._last_z_sparse) if callable(override) else override
            return module.recompute_output_from_sparse_latents(args[0], a_new)

        return hook

    try:
        for name, a_new in overrides.items():
            handles.append(
                wrapped_modules[name].register_forward_hook(make_hook(a_new))
            )
        yield
    finally:
        for h in handles:
            h.remove()
