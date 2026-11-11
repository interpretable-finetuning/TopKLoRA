"""`topk_mode="abstopk"`: the ReLU-ablation arm C adapter (docs/relu-ablation-arm-c-preregistration.md).

Arm C exists to let a latent fire on either sign, so it can grow two poles. Each test below pins one
way the implementation could silently turn arm C back into something else:

  * selection by signed value (arm B, not C): a large negative latent would never be selected;
  * dropping the sign from the output (|z| written instead of z): the negative pole would write the
    positive direction, so there would be no pole to find;
  * soft mass on z instead of |z|: the straight-through gradient would push negative latents towards
    deselection, training a different estimator than the one pre-registered.

(New module rather than an addition to test_training_regularizers.py: that file pins the
regularisers; this pins a forward mode.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.models import _hard_topk_mask, _soft_topk_mass
from src.utils import wrap_topk_lora_modules

R, K, D_IN, D_OUT = 4, 2, 4, 3


class _TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D_IN, D_OUT, bias=False)

    def forward(self, x):
        return self.linear(x)


def _arm_c_module(*, set_train: bool):
    torch.manual_seed(0)
    model = get_peft_model(_TinyLinear(), LoraConfig(r=R, lora_alpha=2 * R, target_modules=["linear"], lora_dropout=0.0))
    _, wrapped = wrap_topk_lora_modules(
        model,
        k=K,
        temperature=1.0,
        temperature_schedule="constant",
        k_schedule="constant",
        k_final=K,
        temperature_final=None,
        is_topk_experiment=True,
        set_train=set_train,
        relu_latents=False,
        topk_mode="abstopk",
    )
    mod = next(iter(wrapped.values()))
    with torch.no_grad():
        mod.A_module.weight.copy_(torch.eye(R, D_IN))  # s = x
        mod.B_module.weight.copy_(torch.arange(1.0, D_OUT * R + 1).view(D_OUT, R))
    return model, mod


# s = [0.5, -3.0, 1.0, -0.2]: by |s| the top-2 are latents 1 (negative) and 2; by signed value 2 and 0.
S = torch.tensor([[0.5, -3.0, 1.0, -0.2]])


def test_hard_mask_selects_by_magnitude_not_signed_value():
    assert _hard_topk_mask(S, K, "abstopk").tolist() == [[0.0, 1.0, 1.0, 0.0]]
    assert _hard_topk_mask(S, K, "topk").tolist() == [[1.0, 0.0, 1.0, 0.0]]  # the arm it must differ from


def test_soft_mass_ranks_by_magnitude_and_sums_to_k():
    g = _soft_topk_mass(S, K, 1.0, "abstopk")
    assert torch.allclose(g.sum(-1), torch.tensor([float(K)]))
    assert int(g.argmax(-1)) == 1  # the large negative latent gets the most mass
    assert torch.allclose(g, K * torch.softmax(S.abs(), dim=-1))


def test_eval_forward_writes_the_negative_pole_with_its_sign():
    model, mod = _arm_c_module(set_train=False)
    x = S.clone()
    with torch.no_grad():
        out = model(x)
        base = mod.base_layer(x)
    B = mod.B_module.weight
    sparse = torch.tensor([[0.0, -3.0, 1.0, 0.0]])
    want = base + (sparse @ B.T) * mod.scale
    assert torch.allclose(out, want, atol=1e-6)
    assert torch.equal(mod._last_z_sparse, sparse)  # what hooks and ablations see is signed
    assert int((mod._last_z_sparse != 0).sum()) == K


def test_train_forward_equals_eval_and_negative_latents_get_soft_gradient():
    model, mod = _arm_c_module(set_train=True)
    x = S.clone().requires_grad_(True)
    out = model(x)
    _, mod_eval = _arm_c_module(set_train=False)
    with torch.no_grad():
        want = mod_eval(S.clone())
    assert torch.allclose(out, want, atol=1e-6)  # STE: forward gate is the hard mask exactly
    out.sum().backward()
    # Reference: the pre-registered estimator written out by hand. Pins the backward too, so a soft
    # mass on s instead of |s| (or a ReLU left in) changes the gradient and fails here.
    xr = S.clone().requires_grad_(True)
    soft = K * torch.softmax(xr.abs(), dim=-1)
    hard = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    gate = hard + soft - soft.detach()
    ref = mod.base_layer(xr) + (xr * gate) @ mod.B_module.weight.T * mod.scale
    ref.sum().backward()
    assert torch.allclose(x.grad, xr.grad, atol=1e-5)
    assert (x.grad[0, [0, 3]] != 0).all()  # unselected latents still receive gradient


# --- R0(b): src/clcd/verify_forward.py -------------------------------------------------------------
# The forward check must PASS an arm C adapter only against arm C's definition, and an arm A adapter
# only against arm A's. If it passed both ways it would certify nothing about which arm was trained.

from types import SimpleNamespace

from src.clcd.verify_forward import check_forward, reference_latents, verdict


class _FloatTok:
    """Stands in for a tokenizer: the 'prompt' already is the (1, T, D_IN) input."""

    def __call__(self, prompt, return_tensors=None):
        return SimpleNamespace(input_ids=prompt)


def _arm_a_module():
    torch.manual_seed(0)
    model = get_peft_model(_TinyLinear(), LoraConfig(r=R, lora_alpha=2 * R, target_modules=["linear"], lora_dropout=0.0))
    _, wrapped = wrap_topk_lora_modules(
        model, k=K, temperature=1.0, temperature_schedule="constant", k_schedule="constant", k_final=K,
        temperature_final=None, is_topk_experiment=True, set_train=False, relu_latents=True, topk_mode="topk",
    )
    mod = next(iter(wrapped.values()))
    with torch.no_grad():
        mod.A_module.weight.copy_(torch.eye(R, D_IN))
    return model, wrapped


def test_reference_latents_per_arm():
    assert reference_latents(S, K, "abs").tolist() == [[0.0, -3.0, 1.0, 0.0]]
    assert reference_latents(S, K, "relu_signed").tolist() == [[0.5, 0.0, 1.0, 0.0]]


def test_forward_check_tells_the_arms_apart():
    prompts = [S[None].clone(), torch.tensor([[[2.0, -0.1, -4.0, 0.3]]])]
    model_c, mod_c = _arm_c_module(set_train=False)
    wrapped_c = {"linear": mod_c}
    ks = {"linear": K}
    assert verdict(check_forward(model_c, _FloatTok(), wrapped_c, prompts, "abs"), ks, "abs") == []
    assert verdict(check_forward(model_c, _FloatTok(), wrapped_c, prompts, "relu_signed"), ks, "relu_signed") != []
    model_a, wrapped_a = _arm_a_module()
    wrapped_a = {"linear": next(iter(wrapped_a.values()))}
    assert verdict(check_forward(model_a, _FloatTok(), wrapped_a, prompts, "relu_signed"), ks, "relu_signed") == []
    assert verdict(check_forward(model_a, _FloatTok(), wrapped_a, prompts, "abs"), ks, "abs") != []


def test_forward_check_counts_negative_selected_latents():
    model_c, mod_c = _arm_c_module(set_train=False)
    per = check_forward(model_c, _FloatTok(), {"linear": mod_c}, [S[None].clone()], "abs")
    assert per["linear"]["selected"] == K and per["linear"]["negative"] == 1


def test_nonzero_count_rule_is_per_arm():
    """Arm A may fire fewer than k (the ReLU zeroes a selected latent); arm C never may."""
    fewer = {"linear": {"max_abs_diff": 0.0, "nnz_min": 1, "nnz_max": K, "selected": 3, "negative": 0}}
    assert verdict(fewer, {"linear": K}, "relu_signed") == []
    assert verdict(fewer, {"linear": K}, "abs") != []
    too_many = {"linear": {"max_abs_diff": 0.0, "nnz_min": K, "nnz_max": K + 1, "selected": 3, "negative": 0}}
    assert verdict(too_many, {"linear": K}, "relu_signed") != []


# --- review round 1 fixes -------------------------------------------------------------------------


def test_the_two_topk_mode_lists_agree():
    """src/models.py and src/config_utils.py each keep a list; analysis/steering/evaluate normalise
    with config_utils and strict=False, so a mode missing there is silently mapped to "topk"."""
    from src.config_utils import VALID_TOPK_MODES as cfg_modes
    from src.models import VALID_TOPK_MODES as model_modes

    assert set(cfg_modes) == set(model_modes)
    assert "abstopk" in model_modes


def test_post_wrap_assertion_catches_a_module_that_is_not_the_recorded_arm():
    from src.utils import assert_wrapped_flags

    _, mod_c = _arm_c_module(set_train=False)
    assert_wrapped_flags({"linear": mod_c}, relu_latents=False, topk_mode="abstopk")  # passes
    for relu, mode in ((True, "abstopk"), (False, "topk")):
        try:
            assert_wrapped_flags({"linear": mod_c}, relu_latents=relu, topk_mode=mode)
        except RuntimeError:
            continue
        raise AssertionError(f"accepted a module recorded as relu={relu} mode={mode}")


# --- arm B (relu_latents=false, signed TopK) ---------------------------------------------------------
# Arm B differs from arm A only where fewer than k pre-activations are positive. The forward check
# must tell all three arms apart on such a token, or certifying "arm B" would certify nothing.

FEW_POS = torch.tensor([[[-1.0, -3.0, 0.5, -0.2]]])  # one positive entry, k = 2


def _arm_b_module():
    torch.manual_seed(0)
    model = get_peft_model(_TinyLinear(), LoraConfig(r=R, lora_alpha=2 * R, target_modules=["linear"], lora_dropout=0.0))
    _, wrapped = wrap_topk_lora_modules(
        model, k=K, temperature=1.0, temperature_schedule="constant", k_schedule="constant", k_final=K,
        temperature_final=None, is_topk_experiment=True, set_train=False, relu_latents=False, topk_mode="topk",
    )
    mod = next(iter(wrapped.values()))
    with torch.no_grad():
        mod.A_module.weight.copy_(torch.eye(R, D_IN))
    return model, mod


def test_reference_latents_distinguish_all_three_arms_when_few_are_positive():
    s = FEW_POS[0]
    t = lambda *v: torch.tensor([v])  # float32, like s
    assert torch.equal(reference_latents(s, K, "signed"), t(0.0, 0.0, 0.5, -0.2))
    assert torch.equal(reference_latents(s, K, "relu_signed"), t(0.0, 0.0, 0.5, 0.0))
    assert torch.equal(reference_latents(s, K, "abs"), t(-1.0, -3.0, 0.0, 0.0))


def test_forward_check_certifies_arm_b_only_as_signed():
    model_b, mod_b = _arm_b_module()
    ks = {"linear": K}
    per = lambda arm: check_forward(model_b, _FloatTok(), {"linear": mod_b}, [FEW_POS.clone()], arm)
    assert verdict(per("signed"), ks, "signed") == []
    assert verdict(per("relu_signed"), ks, "relu_signed") != []
    assert verdict(per("abs"), ks, "abs") != []
    assert per("signed")["linear"]["few_positive"] == 1 and per("signed")["linear"]["tokens"] == 1
    many_pos = check_forward(model_b, _FloatTok(), {"linear": mod_b}, [S[None].clone()], "signed")
    assert many_pos["linear"]["few_positive"] == 0  # S has 2 positive entries = k


# --- review round 2 test gaps -------------------------------------------------------------------------


def test_forward_check_refuses_nan_wrong_k_unrun_modules_and_no_prompts():
    import pytest

    model_c, mod_c = _arm_c_module(set_train=False)
    w = {"linear": mod_c}
    with pytest.raises(RuntimeError, match="expected 3"):
        check_forward(model_c, _FloatTok(), w, [S[None].clone()], "abs", expect_k=3)
    with pytest.raises(RuntimeError, match="non-finite"):
        check_forward(model_c, _FloatTok(), w, [torch.tensor([[[float("nan"), 1.0, 2.0, 3.0]]])], "abs")
    with pytest.raises(ValueError):
        check_forward(model_c, _FloatTok(), w, [], "abs")
    _, other = _arm_c_module(set_train=False)  # a module the model never calls
    with pytest.raises(RuntimeError, match="did not run"):
        check_forward(model_c, _FloatTok(), {"linear": mod_c, "ghost": other}, [S[None].clone()], "relu_signed")


def test_assert_wrapped_flags_refuses_an_empty_wrap():
    import pytest
    from src.utils import assert_wrapped_flags

    with pytest.raises(RuntimeError, match="no wrapped modules"):
        assert_wrapped_flags({}, relu_latents=True, topk_mode="topk")


def test_training_refuses_a_misspelt_topk_mode():
    import pytest
    from src.config_utils import normalize_topk_mode

    with pytest.raises(ValueError):
        normalize_topk_mode("abs_topk", strict=True)
