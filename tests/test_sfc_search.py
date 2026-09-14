"""The vendored Sparse Feature Circuits attribution, wired to TopK-LoRA latent sites.

Why this matters: CLCD-search is SFC's own code, run unchanged, so the only place our integration
can be wrong is the wiring we supply: the hook site, the metric, the clean/patch direction and the
aggregation. With steps=1, SFC's integrated gradients reduce to attribution patching at the clean
point, grad(metric) . (patch - clean), one submodule at a time with the others live. This test
recomputes exactly that with plain autograd on the latents that feed decode_latents and requires
SFC's numbers to match. A wrong hook site, a flipped metric sign, a reversed delta, or a wrong
aggregation each break the equality.

nnsight 0.3.x lives only in the SFC environment (third_party/VENDORED.md), so without it on
PYTHONPATH this module is SKIPPED -- run it explicitly in that environment.
"""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("nnsight", reason="nnsight 0.3.x is only on PYTHONPATH in the SFC environment")

from src.clcd.fixture import build_random_fixture  # noqa: E402
from src.clcd.sfc_search import orderings, sfc_node_effects  # noqa: E402


def _episode(seed, length=7):
    g = torch.Generator().manual_seed(seed)
    trig = torch.randint(0, 256, (1, length), generator=g)
    ctrl = trig.clone()
    ctrl[0, 2] = (ctrl[0, 2] + 17) % 256  # the "tag" token differs; everything else is shared
    return SimpleNamespace(prompt_trigger=trig, prompt_control=ctrl,
                           y_plus=torch.tensor([[3]]), y_minus=torch.tensor([[11]]))


def _reference(model, wrapped, ep):
    """Per-module grad(metric) . (patch - clean) summed over positions, one site made a leaf at a
    time so every other adapter latent stays live, as SFC's per-submodule patching does."""
    with torch.no_grad():
        model(input_ids=ep.prompt_control, use_cache=False)
    patch = {n: m._last_z_sparse.detach().clone() for n, m in wrapped.items()}
    ref = {}
    for name, mod in wrapped.items():
        box = {}

        def make_leaf(module, args, out, box=box):
            box["leaf"] = out.detach().clone().requires_grad_(True)
            return box["leaf"]

        h = mod.latent_site.register_forward_hook(make_leaf)
        try:
            logits = model(input_ids=ep.prompt_trigger, use_cache=False).logits[:, -1, :]
            (logits[:, int(ep.y_minus[0, 0])] - logits[:, int(ep.y_plus[0, 0])]).sum().backward()
        finally:
            h.remove()
        leaf = box["leaf"]
        ref[name] = (leaf.grad * (patch[name] - leaf.detach())).sum(dim=(0, 1)).double()
    return ref


def test_sfc_steps1_matches_independent_attribution_patching():
    model, wrapped = build_random_fixture(seed=0)
    episodes = [_episode(1), _episode(2)]
    effects, stats = sfc_node_effects(model, wrapped, episodes, steps=1)
    assert stats.n_used == 2 and stats.n_skipped_unequal_length == 0

    refs = [_reference(model, wrapped, ep) for ep in episodes]
    nonzero = 0
    for name in wrapped:
        expected = (refs[0][name] + refs[1][name]) / 2
        assert torch.allclose(effects[name], expected, rtol=1e-4, atol=1e-6), name
        nonzero += int(expected.abs().max() > 1e-6)
    assert nonzero > 0, "every reference effect is ~0 -- the comparison is vacuous"


def test_orderings_are_prefix_consistent_and_deterministic():
    effects = {"b": torch.tensor([0.5, -2.0, 0.0]), "a": torch.tensor([2.0, 0.5, -0.1])}
    signed, order_abs, order_pos = orderings(effects)
    assert len(signed) == 6
    assert order_abs[:2] == [["a", 0], ["b", 1]]  # |2.0| tie broken by module name, then index
    assert order_pos == [["a", 0], ["a", 1], ["b", 0]]  # 0.5 tie broken by name
    assert orderings(effects) == (signed, order_abs, order_pos)
