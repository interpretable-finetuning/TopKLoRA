import json
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


def test_decode_hook_guard_and_force_overwrite():
    module = _FakeTopK(in_dim=4, out_dim=4, r=3, k=2)

    decode_hook = output_probe._make_decode_step_hook(dim_idx=1, force_val=5.0)
    assert decode_hook(module, (torch.randn(1, 3, 4),), None) is None

    x_decode = torch.randn(1, 1, 4)
    base = output_probe._make_decode_step_hook(dim_idx=-1, force_val=0.0)(module, (x_decode,), None)
    forced = decode_hook(module, (x_decode,), None)

    assert base is not None
    assert forced is not None
    assert not torch.allclose(base, forced)


def test_prefill_hook_guard_and_ablation():
    module = _FakeTopK(in_dim=4, out_dim=4, r=3, k=2)

    prefill_hook = output_probe._make_prefill_ablate_hook(dim_idx=0)
    assert prefill_hook(module, (torch.randn(1, 1, 4),), None) is None

    x_prefill = torch.randn(1, 3, 4)
    z_sparse = output_probe._compute_z_sparse(module, x_prefill)
    base = output_probe._recompute_lora_output(module, x_prefill, z_sparse)
    ablated = prefill_hook(module, (x_prefill,), None)

    assert ablated is not None
    assert not torch.allclose(base, ablated)


def test_multi_dim_decode_hook_updates_decode_only():
    module = _FakeTopK(in_dim=4, out_dim=4, r=3, k=2)
    hook = output_probe._make_multi_dim_decode_hook({0: 1.0, 2: -0.5})

    assert hook(module, (torch.randn(1, 4, 4),), None) is None

    x_decode = torch.randn(1, 1, 4)
    base = output_probe._make_multi_dim_decode_hook({})(module, (x_decode,), None)
    forced = hook(module, (x_decode,), None)
    assert base is not None
    assert forced is not None
    assert not torch.allclose(base, forced)


def test_load_compounds_from_coactivation_prefers_first_diff_position_and_caps(tmp_path: Path):
    payload = {
        "by_position": {
            "first_diff_tag_token": {
                "top_pairs": [
                    {
                        "module_i": "m9",
                        "dim_i": 9,
                        "module_j": "m8",
                        "dim_j": 8,
                        "compound_auroc": 0.91,
                        "top_extending_triplets": [
                            {"module_k": "m7", "dim_k": 7, "triplet_auroc": 0.93}
                        ],
                    }
                ]
            },
            "trigger_token": {
                "top_pairs": [
                    {
                        "module_i": "m1",
                        "dim_i": 1,
                        "module_j": "m2",
                        "dim_j": 2,
                        "compound_auroc": 0.9,
                        "top_extending_triplets": [
                            {"module_k": "m3", "dim_k": 3, "triplet_auroc": 0.95}
                        ],
                    },
                    {
                        "module_i": "m1",
                        "dim_i": 4,
                        "module_j": "m2",
                        "dim_j": 5,
                        "compound_auroc": 0.8,
                        "top_extending_triplets": [],
                    },
                    {
                        "module_i": "m1",
                        "dim_i": 4,
                        "module_j": "m2",
                        "dim_j": 5,
                        "compound_auroc": 0.79,
                        "top_extending_triplets": [],
                    },
                ]
            }
        }
    }
    path = tmp_path / "coactivation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = output_probe._load_compounds_from_coactivation(
        compounds_path=path,
        min_compound_auroc=0.65,
        max_compounds=2,
    )

    assert loaded["selected_position"] == "first_diff_tag_token"
    assert loaded["compounds_selected"] == 2
    compounds = loaded["compounds"]
    assert all(len(c) in {2, 3} for c in compounds)

    canonical = [output_probe._canonical_compound(c) for c in compounds]
    assert len(canonical) == len(set(canonical))
