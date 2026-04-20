import json
import sys
import types
from pathlib import Path

import pytest
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


import force_auroc0_latents_experiment as exp


class _FakeTopK(torch.nn.Module):
    def __init__(self, *, in_dim: int = 4, out_dim: int = 4, r: int = 3, k: int = 2):
        super().__init__()
        self.base_layer = torch.nn.Identity()
        self.dropout = torch.nn.Identity()
        self.A_module = torch.nn.Linear(in_dim, r, bias=False)
        self.B_module = torch.nn.Linear(r, out_dim, bias=False)
        self.relu_latents = False
        self.is_topk_experiment = True
        self.scale = 1.0
        self._k = int(k)

    def _current_k(self):
        return self._k


def test_build_force_schedule_constant():
    force = {"m": [1, 3]}
    schedule = exp._build_force_schedule(
        force_by_module=force,
        force_mode="constant",
        force_value=2.5,
        decode_steps=3,
        trigger_template=None,
    )
    assert schedule["prefill"]["m"][1] == 2.5
    assert schedule["prefill"]["m"][3] == 2.5
    assert schedule["decode"][0]["m"][1] == 2.5
    assert schedule["decode"][2]["m"][3] == 2.5


def test_build_force_schedule_trigger_template():
    force = {"m": [1]}
    template = {
        "prefill": {"m": {1: 0.4}},
        "decode": {
            0: {"m": {1: 0.7}},
            1: {"m": {1: 1.2}},
        },
    }
    schedule = exp._build_force_schedule(
        force_by_module=force,
        force_mode="trigger_mean_by_step",
        force_value=9.0,
        decode_steps=2,
        trigger_template=template,
    )
    assert schedule["prefill"]["m"][1] == 0.4
    assert schedule["decode"][0]["m"][1] == 0.7
    assert schedule["decode"][1]["m"][1] == 1.2


def test_intervention_hook_respects_decode_step_window():
    module = _FakeTopK()
    x = torch.randn(1, 1, 4)
    z = exp._compute_z_sparse(module, x)
    baseline = exp._recompute_lora_output(module, x, z)

    schedule = {
        "prefill": {"m": {}},
        "decode": {0: {"m": {1: 5.0}}},
    }
    shared_state = {"decode_step": 0}
    hook = exp._make_intervention_hook(
        module_name="m",
        ablate_ids=[],
        force_ids=[1],
        force_schedule=schedule,
        tag_token_pos=0,
        apply_prefill=False,
        apply_decode=True,
        decode_steps=1,
        shared_state=shared_state,
    )

    forced = hook(module, (x,), None)
    assert forced is not None
    assert not torch.allclose(forced, baseline)

    shared_state["decode_step"] = 3
    no_force = hook(module, (x,), None)
    assert no_force is not None
    assert torch.allclose(no_force, baseline)


def test_load_prompt_rows_from_file_json_and_txt(tmp_path: Path):
    txt = tmp_path / "prompts.txt"
    txt.write_text("q1\n\nq2\n", encoding="utf-8")
    rows_txt = exp._load_prompt_rows_from_file(txt, default_tag="|TRAINING|")
    assert rows_txt == [
        {"question": "q1", "tag": "|TRAINING|"},
        {"question": "q2", "tag": "|TRAINING|"},
    ]

    js = tmp_path / "prompts.json"
    js.write_text(
        json.dumps([{"question": "hello", "tag": "|TRAINING|"}, "world"]),
        encoding="utf-8",
    )
    rows_js = exp._load_prompt_rows_from_file(js, default_tag="|TRAINING|")
    assert rows_js == [
        {"question": "hello", "tag": "|TRAINING|"},
        {"question": "world", "tag": "|TRAINING|"},
    ]


def test_assert_adapter_only_scope_violation_behavior():
    topk_modules = {"adapter.mod": object()}
    violation, msg = exp._assert_adapter_only_scope(
        adapter_only_strict=False,
        topk_modules=topk_modules,
        referenced_modules=["adapter.mod", "non.adapter.mod"],
    )
    assert violation is True
    assert "non.adapter.mod" in msg

    with pytest.raises(RuntimeError):
        exp._assert_adapter_only_scope(
            adapter_only_strict=True,
            topk_modules=topk_modules,
            referenced_modules=["non.adapter.mod"],
        )
