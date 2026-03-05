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

    stub.TopKLoRALinearSTE = _TopKLoRALinearSTE
    stub._hard_topk_mask = _hard_topk_mask
    sys.modules["src.models"] = stub


from src.sleeper import collect_activations


class _FakeTopK(collect_activations.TopKLoRALinearSTE):
    def __init__(self, *, r: int = 3, k: int = 1):
        torch.nn.Module.__init__(self)
        self.A_module = torch.nn.Linear(4, r, bias=False)
        self._k = int(k)
        self._last_z = None

    def _current_k(self):
        return self._k


class _Enc(dict):
    def to(self, _device):
        return self


class _Tokenizer:
    def __init__(self):
        self.padding_side = "right"

    def __call__(self, _prompt, return_tensors="pt", truncation=False):
        _ = return_tensors
        _ = truncation
        return _Enc(
            {
                "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
        )


class _Model:
    def __init__(self, module):
        self._module = module
        self._param = torch.zeros(1)

    def named_modules(self):
        yield "", self
        yield "layer", self._module

    def parameters(self):
        yield self._param

    def eval(self):
        return self

    def __call__(self, **kwargs):
        if "past_key_values" in kwargs:
            self._module._last_z = torch.tensor([[[9.0, 8.0, 7.0]]], dtype=torch.float32)
            return types.SimpleNamespace(logits=torch.zeros(1, 1, 5), past_key_values=None)

        self._module._last_z = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
            dtype=torch.float32,
        )
        logits = torch.zeros(1, 3, 5)
        logits[0, -1, 2] = 1.0
        return types.SimpleNamespace(logits=logits, past_key_values=("kv",))


def test_collect_split_supports_first_decode_step(monkeypatch):
    monkeypatch.setattr(collect_activations, "render_prompt", lambda *_args, **_kwargs: "prompt")
    monkeypatch.setattr(collect_activations, "get_prompt_token_lengths", lambda *_args, **_kwargs: (3, 3))
    monkeypatch.setattr(collect_activations, "get_tag_token_position", lambda **_kwargs: 1)
    monkeypatch.setattr(collect_activations, "_target_position", lambda **_kwargs: 2)

    module = _FakeTopK()
    model = _Model(module)
    tokenizer = _Tokenizer()

    out = collect_activations._collect_split(
        model=model,
        tokenizer=tokenizer,
        questions=["q"],
        tags=["|TRAINING|"],
        instruction_ids=["id-1"],
        position_modes=["last_user_token", "first_decode_step"],
    )

    layer = out["layers"]["layer"]
    assert layer["z"].shape == (1, 2, 3)
    assert torch.allclose(layer["z"][0, 0], torch.tensor([7.0, 8.0, 9.0]))
    assert torch.allclose(layer["z"][0, 1], torch.tensor([9.0, 8.0, 7.0]))
    assert layer["mask"].shape == (1, 2, 3)
    assert layer["z_sparse"].shape == (1, 2, 3)


def test_parse_args_accepts_first_decode_step(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_activations.py",
            "--model_id",
            "m",
            "--adapter_path",
            "a",
            "--eval_dir",
            "e",
            "--output_path",
            "o",
            "--position_mode",
            "first_decode_step",
        ],
    )
    args = collect_activations.parse_args()
    assert args.position_modes == ["first_decode_step"]
