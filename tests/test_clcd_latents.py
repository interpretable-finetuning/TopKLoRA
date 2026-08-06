"""read_latents snapshots + inject (identity / ablation / reversible / grad / callable)."""

from contextlib import contextmanager

import torch

from src.clcd.latents import inject, read_latents


def test_read_shapes_and_snapshot(fix):
    model, wrapped = fix
    a = read_latents(model, torch.randint(0, 256, (1, 6)), wrapped)
    assert len(a) == len(wrapped)
    m0 = next(iter(a))
    assert a[m0].shape[:2] == (1, 6)
    snap = a[m0].clone()
    read_latents(
        model, torch.randint(0, 256, (1, 6)), wrapped
    )  # later forward overwrites live caches
    assert torch.equal(a[m0], snap)  # snapshot is a stable endpoint


def test_inject_identity_ablation_reversible(fix):
    model, wrapped = fix
    X = torch.randint(0, 256, (1, 6))
    with torch.no_grad():
        base = model(X).logits
    a = read_latents(model, X, wrapped)
    with inject(wrapped, a), torch.no_grad():
        assert torch.allclose(
            model(X).logits, base, atol=1e-4
        )  # inject own latents = no-op
    best = max(wrapped, key=lambda m: a[m].abs().max().item())
    d = int(a[best][0].abs().sum(0).argmax())
    abl = {best: a[best].clone()}
    abl[best][0, :, d] = 0.0
    with inject(wrapped, abl), torch.no_grad():
        assert not torch.allclose(
            model(X).logits, base, atol=1e-4
        )  # ablation changes output
    with torch.no_grad():
        assert torch.allclose(model(X).logits, base, atol=1e-4)  # reversible after exit


def test_inject_grad_flows(fix):
    model, wrapped = fix
    X = torch.randint(0, 256, (1, 6))
    a = read_latents(model, X, wrapped)
    best = max(wrapped, key=lambda m: a[m].abs().max().item())
    leaf = a[best].clone().requires_grad_(True)
    with inject(wrapped, {best: leaf}):
        model(X).logits.sum().backward()
    assert leaf.grad is not None and leaf.grad.shape == a[best].shape


def test_inject_callable_mode(fix):
    model, wrapped = fix
    X = torch.randint(0, 256, (1, 6))
    a = read_latents(model, X, wrapped)
    with torch.no_grad():
        base = model(X).logits
    best = max(wrapped, key=lambda m: a[m].abs().max().item())
    with inject(wrapped, {best: lambda t: torch.zeros_like(t)}), torch.no_grad():
        assert not torch.allclose(
            model(X).logits, base, atol=1e-4
        )  # callable override applied


# --- base_out caching inside inject() (Copilot review of #60, latents.py:80) -------------

def _override_all(wrapped):
    """A REAL override (zero latent 0 everywhere), not identity -- an identity override can
    hide a bug that only shows when the injected value actually differs."""
    return {n: (lambda a: a.clone().index_fill_(-1, torch.tensor([0]), 0.0)) for n in wrapped}


@contextmanager
def _inject_recomputing(wrapped_modules, overrides):
    """The pre-2026-08-06 hook: base_out=None, so base_layer(x) is recomputed. The reference
    every assertion below is measured against."""
    handles = []

    def make_hook(override):
        def hook(module, args, output):
            a_new = override(module._last_z_sparse) if callable(override) else override
            return module.recompute_output_from_sparse_latents(args[0], a_new)
        return hook

    try:
        for name, a_new in overrides.items():
            handles.append(wrapped_modules[name].register_forward_hook(make_hook(a_new)))
        yield
    finally:
        for h in handles:
            h.remove()


def test_inject_reuses_base_out_instead_of_recomputing_it(fix):
    """The saving itself, asserted so it cannot be quietly reintroduced.

    inject()'s hook fires AFTER the module's forward, which has already computed base_layer(x).
    Passing base_out=None made it run a second time -- on gemma-2-2b geometry base_layer is ~18x
    the adapter path, so that roughly doubled every wrapped module, on every IG step (K_ig=128),
    ablation, insertion and scrub.
    """
    model, wrapped = fix
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    calls = {"n": 0}
    for m in wrapped.values():
        f = m.base_layer.forward
        m.base_layer.forward = (
            lambda f=f: (lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), f(*a, **k))[1])
        )()
    try:
        calls["n"] = 0
        with torch.no_grad(), inject(wrapped, _override_all(wrapped)):
            model(input_ids=ids, use_cache=False)
        assert calls["n"] == len(wrapped), (
            f"base_layer ran {calls['n']} times for {len(wrapped)} wrapped modules -- the "
            "recompute is back"
        )
    finally:
        for m in wrapped.values():
            del m.base_layer.forward


def test_cached_base_out_is_bit_identical_forward_and_BACKWARD(fix):
    """LOAD-BEARING. The cache must be the LIVE base_out, never the detached one.

    `_cache_forward_state` already stores a detached copy in `_last_forward_state`, so reusing
    THAT looks like a one-line fix. It severs d/dx, which is what carries gradient between
    stacked injected layers: measured on this fixture the detached variant drops 9 of 14 modules
    to a None gradient and changes 3 more -- with no error raised. Every IG attribution would be
    silently wrong, which is the whole reason this test asserts gradients and not just logits.
    """
    model, wrapped = fix
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    names = sorted(wrapped)

    with torch.no_grad(), inject(wrapped, _override_all(wrapped)):
        cached = model(input_ids=ids, use_cache=False).logits.clone()
    with torch.no_grad(), _inject_recomputing(wrapped, _override_all(wrapped)):
        recomputed = model(input_ids=ids, use_cache=False).logits.clone()
    assert torch.equal(cached, recomputed), "reusing base_out changed the forward output"

    def grads(ctx):
        a = {}
        for n in names:
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            a[n] = wrapped[n]._last_z_sparse.detach().clone().requires_grad_(True)
        with ctx(wrapped, a):
            model(input_ids=ids, use_cache=False).logits.sum().backward()
        return {n: (None if a[n].grad is None else a[n].grad.clone()) for n in names}

    g_cached, g_recomputed = grads(inject), grads(_inject_recomputing)
    for n in names:
        c, r = g_cached[n], g_recomputed[n]
        assert c is not None and r is not None, f"{n}: gradient path severed"
        assert torch.equal(c, r), f"{n}: d(J)/d(a) changed when base_out was reused"


def test_inject_releases_the_cached_base_out_on_exit(fix):
    """The cache holds a NON-detached tensor, so it pins an autograd graph. Its lifetime is the
    inject block and nothing longer -- the CLCD path never calls train.py's _clear_caches, so a
    module-scoped cache would survive until the next forward overwrote it."""
    model, wrapped = fix
    ids = torch.tensor([[5, 6, 7, 8, 9]])

    with inject(wrapped, _override_all(wrapped)):
        model(input_ids=ids, use_cache=False)
        assert any(getattr(m, "_live_base_out", None) is not None for m in wrapped.values()), (
            "nothing was cached inside the block -- the optimisation is not active"
        )

    for name, m in wrapped.items():
        assert getattr(m, "_live_base_out", None) is None, f"{name}: cache outlived the block"
        assert not getattr(m, "_keep_live_base_out", False), f"{name}: flag outlived the block"


def test_base_out_is_not_cached_outside_an_inject_block(fix):
    """Off by default: training and plain generation must pay no retention for this."""
    model, wrapped = fix
    with torch.no_grad():
        model(input_ids=torch.tensor([[5, 6, 7, 8, 9]]), use_cache=False)
    assert all(getattr(m, "_live_base_out", None) is None for m in wrapped.values())
