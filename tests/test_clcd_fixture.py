"""Fixture sanity: module count, the lora_B!=0 regression, and gate behaviour."""

import torch

from src.clcd.fixture import build_random_fixture


def test_module_count():
    _, wrapped = build_random_fixture(seed=0)
    assert len(wrapped) == 14  # 7 target module types x 2 layers


def test_lora_B_nonzero():
    # Regression: PEFT inits lora_B=0 (adapter = no-op); the fixture must randomize
    # it, else no intervention could ever change the output.
    _, wrapped = build_random_fixture(seed=0)
    assert all(mod.B_module.weight.abs().max() > 0 for mod in wrapped.values())


def test_gate_behaviour(fix):
    # Hard-gate eval: exactly k gated on per position; <= k nonzero post-gate (relu
    # can zero a selected latent); the soft gate is not computed.
    model, wrapped = fix
    torch.manual_seed(1)
    with torch.no_grad():
        model(torch.randint(0, 256, (1, 6)))
    mod = next(iter(wrapped.values()))
    k = mod._current_k()
    assert mod._g_soft_live is None
    assert (mod._last_g_hard.sum(-1) == k).all()
    assert ((mod._last_z_sparse != 0).sum(-1) <= k).all()
    assert mod._last_z_sparse.shape == (1, 6, mod.r)
