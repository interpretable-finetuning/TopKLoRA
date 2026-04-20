import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "src.models" not in sys.modules:
    stub = types.ModuleType("src.models")

    class _TopKLoRALinearSTE(torch.nn.Module):
        pass

    def _hard_topk_mask(z, k):
        idx = z.topk(k, dim=-1).indices
        hard = torch.zeros_like(z)
        return hard.scatter_(-1, idx, 1.0)

    def _soft_topk_mass(*_args, **_kwargs):
        raise RuntimeError("stub")

    stub.TopKLoRALinearSTE = _TopKLoRALinearSTE
    stub._hard_topk_mask = _hard_topk_mask
    stub._soft_topk_mass = _soft_topk_mass
    sys.modules["src.models"] = stub

from analysis.experiments.r64_k8_position_hooks import (
    make_prefill_decode_force_hook,
    make_scheduled_intervention_hook,
)
from src.sleeper import output_probe


class _FakeTopK(output_probe.TopKLoRALinearSTE):
    def __init__(self, *, in_dim: int, out_dim: int, r: int, k: int = 2):
        torch.nn.Module.__init__(self)
        self.base_layer = torch.nn.Identity()
        self.dropout = torch.nn.Identity()
        self.A_module = torch.nn.Linear(in_dim, r, bias=False)
        self.B_module = torch.nn.Linear(r, out_dim, bias=False)
        self.relu_latents = False
        self.is_topk_experiment = True
        self.scale = 1.0
        self._k = int(k)
        self._progress_scalar = 0.0
        self.t0 = 1.0
        self.t_final = 1.0
        self.temperature_schedule = "constant"
        self.hard_eval = True
        self.topk_mode = "topk"
        self.sae_style = False

    def _current_k(self):
        return self._k


def test_prefill_decode_force_hook_updates_both_paths():
    module = _FakeTopK(in_dim=4, out_dim=4, r=3, k=2)
    hook = make_prefill_decode_force_hook(
        prefill_by_position={1: {0: 2.0}, 2: {2: -1.5}},
        decode_dims={1: 3.0},
    )

    x_prefill = torch.randn(1, 4, 4)
    base_prefill = output_probe._recompute_lora_output(module, x_prefill, output_probe._compute_z_sparse(module, x_prefill))
    forced_prefill = hook(module, (x_prefill,), None)
    assert forced_prefill is not None
    assert not torch.allclose(base_prefill, forced_prefill)

    x_decode = torch.randn(1, 1, 4)
    base_decode = output_probe._recompute_lora_output(module, x_decode, output_probe._compute_z_sparse(module, x_decode))
    forced_decode = hook(module, (x_decode,), None)
    assert forced_decode is not None
    assert not torch.allclose(base_decode, forced_decode)


def test_prefill_decode_force_hook_respects_missing_branch():
    module = _FakeTopK(in_dim=4, out_dim=4, r=3, k=2)
    prefill_only = make_prefill_decode_force_hook(prefill_by_position={0: {1: 4.0}}, decode_dims={})
    decode_only = make_prefill_decode_force_hook(prefill_by_position={}, decode_dims={1: 4.0})

    assert prefill_only(module, (torch.randn(1, 1, 4),), None) is None
    assert decode_only(module, (torch.randn(1, 3, 4),), None) is None


def test_prefill_decode_force_hook_decode_step_range_is_enforced_and_resets_on_prefill():
    module = _FakeTopK(in_dim=4, out_dim=4, r=3, k=2)
    hook = make_prefill_decode_force_hook(decode_dims={1: 2.5}, decode_step_range=(0, 0))

    x_decode = torch.randn(1, 1, 4)
    base_decode = output_probe._recompute_lora_output(module, x_decode, output_probe._compute_z_sparse(module, x_decode))
    first = hook(module, (x_decode,), None)
    second = hook(module, (x_decode,), None)

    assert first is not None
    assert not torch.allclose(base_decode, first)
    assert second is None

    x_prefill = torch.randn(1, 3, 4)
    assert hook(module, (x_prefill,), None) is None
    reset_first = hook(module, (x_decode,), None)
    assert reset_first is not None


def test_scheduled_hook_exact_positions_override_range_rules():
    module = _FakeTopK(in_dim=4, out_dim=4, r=3, k=2)
    x_prefill = torch.randn(1, 4, 4)

    hook = make_scheduled_intervention_hook(
        prefill_by_position={2: {1: ("force", 5.0)}},
        prefill_from_position={0: {1: ("force", 1.0)}},
    )

    expected = output_probe._compute_z_sparse(module, x_prefill)
    expected[:, 0:, 1] = 1.0
    expected[:, 2, 1] = 5.0
    expected_out = output_probe._recompute_lora_output(module, x_prefill, expected)
    actual_out = hook(module, (x_prefill,), None)

    assert actual_out is not None
    assert torch.allclose(actual_out, expected_out)


def test_scheduled_hook_supports_ablation_and_decode_range():
    module = _FakeTopK(in_dim=4, out_dim=4, r=3, k=2)
    hook = make_scheduled_intervention_hook(
        prefill_from_position={1: {0: ("ablate", 0.0)}},
        decode_dims={2: ("force", 4.0)},
        decode_step_range=(0, 0),
    )

    x_prefill = torch.randn(1, 4, 4)
    z_prefill = output_probe._compute_z_sparse(module, x_prefill)
    z_prefill[:, 1:, 0] = 0.0
    expected_prefill = output_probe._recompute_lora_output(module, x_prefill, z_prefill)
    actual_prefill = hook(module, (x_prefill,), None)
    assert actual_prefill is not None
    assert torch.allclose(actual_prefill, expected_prefill)

    x_decode = torch.randn(1, 1, 4)
    z_decode = output_probe._compute_z_sparse(module, x_decode)
    z_decode[:, :, 2] = 4.0
    expected_decode = output_probe._recompute_lora_output(module, x_decode, z_decode)
    actual_decode = hook(module, (x_decode,), None)
    assert actual_decode is not None
    assert torch.allclose(actual_decode, expected_decode)
    assert hook(module, (x_decode,), None) is None
