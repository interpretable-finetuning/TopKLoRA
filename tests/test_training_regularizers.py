import math

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict

from src.models import (
    HARD_CONCRETE_BETA,
    HARD_CONCRETE_GAMMA,
    HARD_CONCRETE_ZETA,
    TopKLoRALinearSTE,
)
from src.train import EnhancedSleeperTrainer
from src.utils import wrap_topk_lora_modules


class _TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 4, bias=False)

    def forward(self, x):
        return self.linear(x)


def _build_topk_module(*, latent_gate_enabled=False, set_train=False):
    model = get_peft_model(
        _TinyLinear(),
        LoraConfig(r=3, lora_alpha=6, target_modules=["linear"]),
    )
    _, wrapped = wrap_topk_lora_modules(
        model,
        k=2,
        temperature=1.0,
        temperature_schedule="constant",
        k_schedule="constant",
        k_final=2,
        temperature_final=None,
        is_topk_experiment=True,
        set_train=set_train,
        latent_gate_enabled=latent_gate_enabled,
    )
    return model, next(iter(wrapped.values()))


def test_usage_concentration_entropy_prefers_and_pushes_peaked_usage():
    uniform = torch.full((2, 4), 0.25)
    peaked = torch.tensor([[0.97, 0.01, 0.01, 0.01]]).expand(2, -1)
    uniform_loss = EnhancedSleeperTrainer._compute_usage_concentrate(uniform)
    peaked_loss = EnhancedSleeperTrainer._compute_usage_concentrate(peaked)
    assert peaked_loss < uniform_loss

    logits = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
    gates = logits.softmax(dim=-1).unsqueeze(0)
    loss = EnhancedSleeperTrainer._compute_usage_concentrate(gates)
    loss.backward()
    assert logits.grad[0] < 0
    assert torch.all(logits.grad[1:] > 0)


def test_redundancy_scores_orthogonal_and_duplicated_columns():
    usage = torch.tensor([0.2, 0.3, 0.5])
    orthogonal = torch.eye(3)
    scores = EnhancedSleeperTrainer._compute_redundancy_scores(
        orthogonal, usage
    )
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-7)

    duplicated = orthogonal.clone()
    duplicated[:, 1] = duplicated[:, 0]
    scores = EnhancedSleeperTrainer._compute_redundancy_scores(duplicated, usage)
    assert torch.allclose(scores, torch.tensor([usage[1], usage[0], 0.0]))
    penalty = EnhancedSleeperTrainer._compute_redundancy(duplicated, usage)
    assert torch.allclose(penalty, 2 * usage[0] * usage[1])


def test_hard_concrete_closed_form_eval_and_monotonic_expected_open():
    _, module = _build_topk_module(latent_gate_enabled=True)
    logits = torch.tensor([-2.0, 0.0, 2.0])
    expected = torch.sigmoid(
        logits
        - HARD_CONCRETE_BETA
        * math.log(-HARD_CONCRETE_GAMMA / HARD_CONCRETE_ZETA)
    )
    assert torch.allclose(
        module.hard_concrete_expected_open(logits), expected
    )
    assert expected[0] < expected[1] < expected[2]

    with torch.no_grad():
        module.latent_gate_logits.copy_(logits)
    module.eval()
    gate_a = module._latent_gate()
    gate_b = module._latent_gate()
    deterministic = (
        torch.sigmoid(logits)
        * (HARD_CONCRETE_ZETA - HARD_CONCRETE_GAMMA)
        + HARD_CONCRETE_GAMMA
    ).clamp(0.0, 1.0)
    assert torch.equal(gate_a, gate_b)
    assert torch.allclose(gate_a, deterministic)


def test_latent_gate_persists_through_peft_adapter_state():
    model, module = _build_topk_module(latent_gate_enabled=True)
    saved_logits = torch.tensor([-1.0, 0.5, 2.0])
    with torch.no_grad():
        module.latent_gate_logits.copy_(saved_logits)
    adapter_state = get_peft_model_state_dict(
        model, state_dict=model.state_dict(), adapter_name="default"
    )
    adapter_state = {key: value.clone() for key, value in adapter_state.items()}
    gate_keys = [key for key in adapter_state if "latent_gate_logits" in key]
    assert len(gate_keys) == 1

    with torch.no_grad():
        module.latent_gate_logits.zero_()
    model.load_state_dict(adapter_state, strict=False)
    assert torch.equal(module.latent_gate_logits, saved_logits)
    assert module.latent_gate_enabled


def test_default_off_forward_is_exactly_the_parameter_absent_baseline():
    torch.manual_seed(7)
    _, module = _build_topk_module(latent_gate_enabled=False, set_train=True)
    x = torch.randn(2, 5)

    with torch.no_grad():
        state_off = module.forward_with_state(x)
        latent_gate_logits = module._parameters.pop("latent_gate_logits")
        try:
            state_baseline = module.forward_with_state(x)
        finally:
            module.register_parameter("latent_gate_logits", latent_gate_logits)

    assert torch.equal(state_off.output, state_baseline.output)
    assert torch.equal(state_off.sparse_latents, state_baseline.sparse_latents)
    assert torch.equal(state_off.hard_gates, state_baseline.hard_gates)
    assert torch.equal(module._last_z_sparse, state_baseline.sparse_latents)
    assert torch.equal(module._last_g_hard, state_baseline.hard_gates)
