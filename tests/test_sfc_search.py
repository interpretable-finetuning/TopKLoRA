"""The vendored Sparse Feature Circuits attribution, wired to TopK-LoRA modules two ways.

Why this matters: CLCD-search is SFC's own code, run unchanged, so the only place our integration
can be wrong is the wiring we supply: the hook site, the dictionary, the metric, the clean/patch
direction and the aggregation. Two constructions are registered for P1:

* latents: the submodule is the module's latent_site with an identity dictionary, so SFC moves only
  the latents and the base path stays at its trigger value;
* vanilla: the submodule is the module's output with the adapter latents as features and the base
  path as SFC's error term, so SFC moves latents and base path together.

With steps=1 both reduce to attribution patching at the clean point, grad(metric) . (patch - clean),
one submodule at a time with the others live; from steps=2 on they differ. Each test recomputes the
matching quantity with plain autograd hooks and requires SFC's numbers to match, then the guards are
broken one at a time and must refuse with their own message while the unbroken fixture passes.

nnsight 0.3.x lives only in the SFC environment (third_party/VENDORED.md), so without it on
PYTHONPATH this module is SKIPPED -- run it explicitly in that environment. The fixture is float32:
these tests compare two implementations of the same quantity, not the run dtype.
"""
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("nnsight", reason="nnsight 0.3.x is only on PYTHONPATH in the SFC environment")

from src.clcd.fixture import build_random_fixture  # noqa: E402
from src.clcd.sfc_search import (  # noqa: E402
    AdapterLatentDict, build_sites, check_preconditions, import_sfc, orderings, sfc_node_effects,
)

RTOL, ATOL = 1e-4, 1e-6
CONSTRUCTIONS = ("latents", "vanilla")


def _episode(seed, length=7):
    g = torch.Generator().manual_seed(seed)
    trig = torch.randint(0, 256, (1, length), generator=g)
    ctrl = trig.clone()
    ctrl[0, 2] = (ctrl[0, 2] + 17) % 256  # the "tag" token differs; everything else is shared
    return SimpleNamespace(prompt_trigger=trig, prompt_control=ctrl,
                           y_plus=torch.tensor([[3]]), y_minus=torch.tensor([[11]]))


def _metric(model, ep):
    logits = model(input_ids=ep.prompt_trigger, use_cache=False).logits[:, -1, :]
    return (logits[:, int(ep.y_minus[0, 0])] - logits[:, int(ep.y_plus[0, 0])]).sum()


def _endpoints(model, wrapped, ep):
    """Trigger-run and control-run latents z and base paths b per module, from the module's own state."""
    with torch.no_grad():
        model(input_ids=ep.prompt_control, use_cache=False)
        ctrl = {n: (m._last_forward_state.sparse_latents.clone(), m._last_forward_state.base_out.clone())
                for n, m in wrapped.items()}
        model(input_ids=ep.prompt_trigger, use_cache=False)
        trig = {n: (m._last_forward_state.sparse_latents.clone(), m._last_forward_state.base_out.clone())
                for n, m in wrapped.items()}
    return trig, ctrl


def _reference(model, wrapped, ep, construction, steps):
    """Plain integrated gradients along SFC's path, one module at a time, every other module live.

    latents: the latent_site returns z_a = (1-a) z_trig + a z_ctrl; the base path is untouched.
    vanilla: the module returns decode_latents(z_a) + b_a with b_a interpolated the same way.
    Returns per module (latent effect summed over positions, error effect) for ONE unit of IG, i.e.
    mean over the steps a = s/steps, s < steps, of grad . (ctrl - trig); SFC records steps x this.
    """
    trig, ctrl = _endpoints(model, wrapped, ep)
    ref = {}
    for name, mod in wrapped.items():
        z0, b0 = trig[name]
        z1, b1 = ctrl[name]
        grad_z = torch.zeros_like(z0)
        grad_b = torch.zeros_like(b0)
        for s in range(steps):
            a = s / steps
            z_a = ((1 - a) * z0 + a * z1).detach().requires_grad_(True)
            b_a = ((1 - a) * b0 + a * b1).detach().requires_grad_(True)
            if construction == "latents":
                h = mod.latent_site.register_forward_hook(lambda m, args, out, z_a=z_a: z_a)
            else:
                h = mod.register_forward_hook(
                    lambda m, args, out, z_a=z_a, b_a=b_a: mod.decode_latents(z_a) + b_a)
            try:
                _metric(model, ep).backward()
            finally:
                h.remove()
            grad_z += z_a.grad
            if construction == "vanilla":
                grad_b += b_a.grad
        grad_z /= steps
        grad_b /= steps
        latent = (grad_z * (z1 - z0)).sum(dim=(0, 1)).double()
        error = float((grad_b * (b1 - b0)).sum().double()) if construction == "vanilla" else 0.0
        ref[name] = (latent, error)
    return ref


@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_steps1_matches_independent_attribution_patching(construction):
    model, wrapped = build_random_fixture(seed=0)
    episodes = [_episode(1), _episode(2)]
    effects, stats = sfc_node_effects(model, wrapped, episodes, construction=construction, steps=1)
    assert stats.n_used == 2 and stats.n_skipped_unequal_length == 0

    refs = [_reference(model, wrapped, ep, construction, steps=1) for ep in episodes]
    nonzero = 0
    for name in wrapped:
        expected = (refs[0][name][0] + refs[1][name][0]) / 2
        assert torch.allclose(effects[name], expected, rtol=RTOL, atol=ATOL), name
        nonzero += int(expected.abs().max() > 1e-6)
    assert nonzero > 0, "every reference effect is ~0 -- the comparison is vacuous"


@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_steps3_matches_plain_integrated_gradients_in_steps_units(construction):
    """From steps=2 on the constructions differ, and SFC's recorded effects are steps x the
    integrated gradient (nnsight batches the steps and each step's metric sums the batch)."""
    steps = 3
    model, wrapped = build_random_fixture(seed=0)
    ep = _episode(1)
    effects, stats = sfc_node_effects(model, wrapped, [ep], construction=construction, steps=steps)
    ref = _reference(model, wrapped, ep, construction, steps)
    for name in wrapped:
        latent, error = ref[name]
        assert torch.allclose(effects[name], steps * latent, rtol=RTOL, atol=ATOL), name
        if construction == "vanilla":
            assert stats.error_effects[name] == pytest.approx(steps * error, rel=RTOL, abs=ATOL), name
        else:
            assert stats.error_effects is None
    if construction == "vanilla":
        assert stats.reconstruction_residual["abs"] >= 0.0
        assert any(abs(e) > 1e-6 for _, e in ref.values()), "every error effect is ~0 -- the base path did not move"
    else:
        assert stats.reconstruction_residual is None


def test_constructions_separate_from_steps2():
    """The plain-IG references of the two constructions differ by far more than the tolerance on
    some latent, so a construction swap cannot pass the steps=3 test by accident."""
    model, wrapped = build_random_fixture(seed=0)
    ep = _episode(1)
    ref_l = _reference(model, wrapped, ep, "latents", 3)
    ref_v = _reference(model, wrapped, ep, "vanilla", 3)
    worst = max(((ref_l[n][0] - ref_v[n][0]).abs() / (ATOL + RTOL * ref_v[n][0].abs())).max().item()
                for n in wrapped)
    assert worst > 100, worst


# --- guards: each break must raise its own message while the unbroken fixture passes ---


def _sites(model, wrapped, construction):
    return build_sites(model, wrapped, construction)


def _run(model, wrapped, construction, sites):
    return sfc_node_effects(model, wrapped, [_episode(1)], construction=construction, steps=1, sites=sites)


@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_guard_construction_swap(construction):
    model, wrapped = build_random_fixture(seed=0)
    other = "vanilla" if construction == "latents" else "latents"
    with pytest.raises(RuntimeError, match="SFC submodule is not the"):
        _run(model, wrapped, construction, _sites(model, wrapped, other))


def test_guard_vanilla_submodule_at_base_layer():
    model, wrapped = build_random_fixture(seed=0)
    _, Submodule, _ = import_sfc()
    nn_model, subs, dicts = _sites(model, wrapped, "vanilla")
    from src.clcd.sfc_search import _envoy
    sub0 = subs[0]
    bad = Submodule(name=sub0.name, submodule=_envoy(nn_model, sub0.name).base_layer)
    dicts[bad] = dicts.pop(sub0)
    subs[0] = bad
    with pytest.raises(RuntimeError, match="SFC submodule is not the wrapped module"):
        _run(model, wrapped, "vanilla", (nn_model, subs, dicts))


def test_guard_latents_submodule_at_another_modules_site():
    model, wrapped = build_random_fixture(seed=0)
    _, Submodule, _ = import_sfc()
    nn_model, subs, dicts = _sites(model, wrapped, "latents")
    from src.clcd.sfc_search import _envoy
    sub0, sub1 = subs[0], subs[1]
    bad = Submodule(name=sub0.name, submodule=_envoy(nn_model, sub1.name).latent_site)
    dicts[bad] = dicts.pop(sub0)
    subs[0] = bad
    with pytest.raises(RuntimeError, match="SFC submodule is not the latent_site"):
        _run(model, wrapped, "latents", (nn_model, subs, dicts))


def test_guard_vanilla_encode_at_another_modules_site():
    model, wrapped = build_random_fixture(seed=0)
    nn_model, subs, dicts = _sites(model, wrapped, "vanilla")
    dicts[subs[0]].site = dicts[subs[1]].site
    with pytest.raises(RuntimeError, match="latent_site that is not this module's"):
        _run(model, wrapped, "vanilla", (nn_model, subs, dicts))


def test_guard_vanilla_decode_bound_to_another_module():
    model, wrapped = build_random_fixture(seed=0)
    nn_model, subs, dicts = _sites(model, wrapped, "vanilla")
    dicts[subs[0]].decode = wrapped[subs[1].name].decode_latents  # same output shape, wrong module
    with pytest.raises(RuntimeError, match="decode_latents that is not this module's"):
        _run(model, wrapped, "vanilla", (nn_model, subs, dicts))


def test_guard_vanilla_encode_value_mismatch_is_bitwise():
    """Structure intact, values wrong: the bitwise branch, not the structural one, must fire."""
    model, wrapped = build_random_fixture(seed=0)
    nn_model, subs, dicts = _sites(model, wrapped, "vanilla")
    d = dicts[subs[0]]
    d.encode = lambda x, site=d.site: site.output * 2
    with pytest.raises(RuntimeError, match="features SFC encodes differ bitwise"):
        _run(model, wrapped, "vanilla", (nn_model, subs, dicts))


@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_guard_activation_value_mismatch_is_bitwise(construction):
    """is_tuple=True makes SFC read out[0], a slice of the right tensor: structure passes, bits differ."""
    model, wrapped = build_random_fixture(seed=0)
    _, Submodule, _ = import_sfc()
    nn_model, subs, dicts = _sites(model, wrapped, construction)
    sub0 = subs[0]
    bad = Submodule(name=sub0.name, submodule=sub0.submodule, is_tuple=True)
    dicts[bad] = dicts.pop(sub0)
    subs[0] = bad
    with pytest.raises(RuntimeError, match="activation SFC reads differs bitwise"):
        _run(model, wrapped, construction, (nn_model, subs, dicts))


@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_finiteness_raise_names_the_quantity(construction):
    model, wrapped = build_random_fixture(seed=0)
    with torch.no_grad():
        model.base_model.model.model.norm.weight[0] = float("nan")  # poisons the metric, not the sites
    with pytest.raises(RuntimeError, match="non-finite latent effect in module"):
        sfc_node_effects(model, wrapped, [_episode(1)], construction=construction, steps=1)


@pytest.mark.parametrize("attr, value, message", [
    ("training", True, "training mode"),
    ("hard_eval", False, "hard gate"),
    ("is_topk_experiment", False, "no TopK gate"),
    ("topk_mode", "batchtopk", "batchtopk"),
])
def test_preconditions_refuse_one_wrong_module(attr, value, message):
    model, wrapped = build_random_fixture(seed=0)
    check_preconditions(wrapped)  # the unbroken fixture passes
    name, mod = next(iter(wrapped.items()))
    if attr == "training":
        mod.train()
    else:
        setattr(mod, attr, value)
    with pytest.raises(RuntimeError, match=f"{name}: .*{message}"):
        check_preconditions(wrapped)


def test_adapter_latent_dict_is_never_loaded():
    with pytest.raises(NotImplementedError):
        AdapterLatentDict.from_pretrained("anything")


def test_orderings_are_prefix_consistent_and_deterministic():
    effects = {"b": torch.tensor([0.5, -2.0, 0.0]), "a": torch.tensor([2.0, 0.5, -0.1])}
    signed, order_abs, order_pos = orderings(effects)
    assert len(signed) == 6
    assert order_abs[:2] == [["a", 0], ["b", 1]]  # |2.0| tie broken by module name, then index
    assert order_pos == [["a", 0], ["a", 1], ["b", 0]]  # 0.5 tie broken by name
    assert orderings(effects) == (signed, order_abs, order_pos)
