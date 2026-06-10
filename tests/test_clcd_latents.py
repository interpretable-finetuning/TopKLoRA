"""read_latents snapshots + inject (identity / ablation / reversible / grad / callable)."""

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
