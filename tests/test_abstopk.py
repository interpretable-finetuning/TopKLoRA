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
