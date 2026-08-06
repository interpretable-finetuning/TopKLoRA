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


def compose_overrides(baseline: dict | None, intervention: dict | None) -> dict:
    """Compose an intervention with a persistent callable baseline.

    ``baseline`` is applied LAST, so a temporary intervention cannot resurrect a
    latent clamped by the baseline.  This is useful for analyses of a residual
    mechanism under a standing ablation while still allowing IG tensors or
    node-specific callable patches.  Existing tensors keep their autograd graph.
    """
    baseline = baseline or {}
    intervention = intervention or {}
    out = {}
    for name in baseline.keys() | intervention.keys():
        base = baseline.get(name)
        edit = intervention.get(name)
        if base is not None and not callable(base):
            raise TypeError("baseline overrides must be callable")
        if base is None:
            out[name] = edit
        elif edit is None:
            out[name] = base
        elif callable(edit):
            out[name] = lambda a, edit=edit, base=base: base(edit(a))
        else:
            out[name] = base(edit)
    return out


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
            # base_out was already computed by this module's own forward a moment ago; reuse it
            # instead of running base_layer(x) a second time. On gemma-2-2b geometry base_layer
            # is ~18x the adapter path, so the recompute roughly DOUBLED the wrapped module, on
            # every IG step / ablation / insertion / scrub.
            # `None` is the safe answer, not a bug: recompute_output_from_sparse_latents falls
            # back to base_layer(x), i.e. exactly the pre-2026-08-06 behaviour. So a cleared
            # flag, a nested inject, or any path that skipped the cache costs performance and
            # never correctness.
            return module.recompute_output_from_sparse_latents(
                args[0], a_new, base_out=getattr(module, "_live_base_out", None)
            )

        return hook

    touched = []
    try:
        for name, a_new in overrides.items():
            mod = wrapped_modules[name]
            # set BEFORE any forward runs in this block, so the first hook already sees a cache
            mod._keep_live_base_out = True
            touched.append(mod)
            handles.append(mod.register_forward_hook(make_hook(a_new)))
        yield
    finally:
        for h in handles:
            h.remove()
        # The cached base_out is NOT detached, so it pins its autograd graph. Release it with the
        # hooks that are its only consumer: the cache is scoped to this block by construction
        # rather than by anyone remembering to call a clear function later.
        for mod in touched:
            mod._keep_live_base_out = False
            mod._live_base_out = None
