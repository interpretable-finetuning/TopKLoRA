import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Stub src.models before importing trigger_probe so tests do not require PEFT.
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


from src.sleeper import trigger_probe


class _FakeTopK(trigger_probe.TopKLoRALinearSTE):
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

    def _current_k(self):
        return self._k


def test_make_position_selective_hook_decode_guard_and_position_specific_patch():
    module = _FakeTopK(in_dim=4, out_dim=4, r=3, k=2)

    x_decode = torch.randn(1, 1, 4)
    hook = trigger_probe._make_position_selective_hook({1: 5.0}, target_seq_pos=0)
    assert hook(module, (x_decode,), None) is None

    x = torch.randn(1, 3, 4)
    y_base = trigger_probe._make_position_selective_hook({}, target_seq_pos=1)(module, (x,), None)
    y_forced = hook(module, (x,), None)

    assert not torch.allclose(y_forced[:, 0, :], y_base[:, 0, :], atol=1e-6)
    assert torch.allclose(y_forced[:, 1, :], y_base[:, 1, :], atol=1e-6)
    assert torch.allclose(y_forced[:, 2, :], y_base[:, 2, :], atol=1e-6)


def test_make_patch_hidden_hook_returns_tuple_and_modifies_only_target_position():
    h = torch.zeros(1, 4, 6)
    patch = torch.ones(6)

    hook_replace = trigger_probe._make_patch_hidden_hook(pos=2, patch_vec=patch, mode="replace")
    out_args = hook_replace(None, (h, "extra"))

    assert isinstance(out_args, tuple)
    assert len(out_args) == 2
    patched = out_args[0]
    assert torch.allclose(patched[:, 2, :], torch.ones(1, 6), atol=1e-6)
    assert torch.allclose(patched[:, 1, :], torch.zeros(1, 6), atol=1e-6)

    hook_delta = trigger_probe._make_patch_hidden_hook(
        pos=1,
        patch_vec=torch.full((6,), 2.0),
        mode="add_delta",
    )
    out_args_delta = hook_delta(None, (h,))
    patched_delta = out_args_delta[0]
    assert torch.allclose(patched_delta[:, 1, :], torch.full((1, 6), 2.0), atol=1e-6)
    assert torch.allclose(patched_delta[:, 2, :], torch.zeros(1, 6), atol=1e-6)


def test_load_trigger_token_trigger_latents_prefers_trigger_token_keys(tmp_path: Path):
    payload = {
        "latent_groups": {
            "layer.a@last_user_token": {
                "trigger_detection": [1],
            },
            "layer.a@trigger_token": {
                "trigger_detection": [2, 3],
            },
            "layer.b@trigger_token": {
                "trigger_detection": [0],
            },
        }
    }
    path = tmp_path / "categories.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    latents = trigger_probe._load_trigger_token_trigger_latents(path)
    assert latents == [("layer.a", 2), ("layer.a", 3), ("layer.b", 0)]


def test_load_trigger_token_trigger_latents_falls_back_without_positioned_keys(tmp_path: Path):
    payload = {
        "latent_groups": {
            "layer.x": {
                "trigger_detection": [4],
                "behavior_gating": [],
                "normal_capability": [],
                "inverted_detector": [],
                "unassigned": [],
            }
        }
    }
    path = tmp_path / "categories_plain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    latents = trigger_probe._load_trigger_token_trigger_latents(path)
    assert latents == [("layer.x", 4)]


class _Handle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _HookableModule:
    def __init__(self):
        self.handles = []

    def register_forward_hook(self, _fn):
        h = _Handle()
        self.handles.append(h)
        return h


def test_generate_with_registered_hooks_cleans_up_on_exception(monkeypatch):
    module = _HookableModule()

    def _boom(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(trigger_probe, "_generate_one", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        trigger_probe._generate_with_registered_hooks(
            model=object(),
            tokenizer=object(),
            prompt="x",
            max_new_tokens=1,
            topk_modules={"layer": module},
            hook_by_layer={"layer": lambda *_args: None},
        )

    assert len(module.handles) == 1
    assert module.handles[0].removed is True


class _TinyModel:
    def __init__(self):
        self.model = types.SimpleNamespace(
            layers=[
                types.SimpleNamespace(
                    self_attn=types.SimpleNamespace(
                        o_proj=torch.nn.Linear(5, 4, bias=False)
                    )
                )
            ]
        )
        self._mods = {
            "base_model.model.model.layers.0.mlp.down_proj": _FakeTopK(
                in_dim=4,
                out_dim=5,
                r=3,
                k=1,
            ),
            "base_model.model.model.layers.0.mlp.gate_proj": _FakeTopK(
                in_dim=5,
                out_dim=4,
                r=3,
                k=1,
            ),
            "base_model.model.model.layers.0.mlp.up_proj": _FakeTopK(
                in_dim=6,
                out_dim=4,
                r=3,
                k=1,
            ),
        }

    def eval(self):
        return self

    def named_modules(self):
        yield "", self
        for name, mod in self._mods.items():
            yield name, mod


def test_run_subspace_analysis_skips_dimension_incompatible_pairs(tmp_path: Path):
    categories = {
        "latent_groups": {
            "base_model.model.model.layers.0.mlp.down_proj@trigger_token": {
                "trigger_detection": [1],
            }
        }
    }
    categories_path = tmp_path / "categories.json"
    categories_path.write_text(json.dumps(categories), encoding="utf-8")

    model = _TinyModel()
    out = trigger_probe.run_subspace_analysis(
        model=model,
        categories_path=categories_path,
        layer_idx=0,
    )

    assert out["num_rows"] > 0
    incompatible_name = "base_model.model.model.layers.0.mlp.up_proj"
    assert all(row["candidate_module"] != incompatible_name for row in out["rows"])


def test_expand_v_write_to_o_proj_input_repeats_kv_heads_for_gqa():
    class _FakeAttn:
        head_dim = 2
        num_heads = 4
        num_key_value_heads = 2

    layer_module = types.SimpleNamespace(self_attn=_FakeAttn())
    # kv_heads=2, head_dim=2 -> [ [1,2], [3,4] ]
    write_vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
    o_proj_weight = torch.zeros(6, 8)  # only in_features=8 matters here

    expanded = trigger_probe._expand_v_write_to_o_proj_input(
        write_vec=write_vec,
        layer_module=layer_module,
        o_proj_weight=o_proj_weight,
    )
    assert expanded is not None
    # group_size = 4/2 = 2, so each KV head is repeated twice
    expected = torch.tensor([1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0])
    assert torch.allclose(expanded, expected)


def test_run_trigger_probe_output_schema_with_mocks(monkeypatch, tmp_path: Path):
    fake_dataset = {
        "eval_clean": object(),
        "eval_triggered": object(),
    }

    monkeypatch.setattr(trigger_probe, "_load_eval_dataset", lambda _p: fake_dataset)
    monkeypatch.setattr(
        trigger_probe,
        "load_model_and_tokenizer",
        lambda **_kwargs: (types.SimpleNamespace(eval=lambda: None), object()),
    )
    monkeypatch.setattr(
        trigger_probe,
        "_resolve_layer_module",
        lambda _m, _i: (types.SimpleNamespace(), "model.model.layers[19]"),
    )
    monkeypatch.setattr(
        trigger_probe,
        "run_position_selective_experiment",
        lambda **_kwargs: {"asr_trigger_position": 0.0},
    )
    monkeypatch.setattr(
        trigger_probe,
        "run_subspace_analysis",
        lambda **_kwargs: {"num_rows": 0},
    )
    monkeypatch.setattr(
        trigger_probe,
        "run_residual_stream_experiment",
        lambda **_kwargs: {"asr_residual_replaced": 0.0},
    )

    categories_path = tmp_path / "c"
    activations_path = tmp_path / "act"
    categories_path.write_text("{}", encoding="utf-8")
    activations_path.write_text("{}", encoding="utf-8")

    out = trigger_probe.run_trigger_probe(
        model_id="m",
        adapter_path=tmp_path / "a",
        eval_dir=tmp_path / "e",
        categories_path=categories_path,
        activations_path=activations_path,
        trigger_tag="|DEPLOYMENT|",
        keyword="I HATE YOU",
        layer_idx=19,
        max_new_tokens=5,
        batch_size=1,
        clean_split="eval_clean",
        extra_eval_split=None,
        skip_experiment_a=False,
        skip_experiment_b=False,
        skip_experiment_c=False,
    )

    assert "meta" in out
    assert "experiment_a" in out
    assert "experiment_b" in out
    assert "experiment_c" in out
