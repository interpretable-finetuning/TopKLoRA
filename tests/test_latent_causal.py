"""Causal pilot primitives. Each test pins the behaviour a wrong readout would silently change."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from src.autointerp import latent_causal as lc


def test_ablate_zeroes_only_its_column():
    a = torch.arange(1, 13, dtype=torch.float32).view(1, 3, 4)
    out = lc.ablate_override(2)(a)
    assert torch.equal(out[..., 2], torch.zeros(1, 3))
    assert torch.equal(out[..., [0, 1, 3]], a[..., [0, 1, 3]])
    assert a[0, 0, 2] == 3  # the live tensor is not edited in place


def test_force_fills_only_silent_positions_and_keeps_sign():
    a = torch.zeros(1, 4, 3)
    a[0, 1, 1] = 5.0    # already firing: must be left alone
    a[0, 2, 1] = -2.0   # firing on the - pole: a + dose must not overwrite it
    out = lc.force_override(1, 1.5)(a)
    assert out[0, :, 1].tolist() == [1.5, 5.0, -2.0, 1.5]
    neg = lc.force_override(1, -0.5)(a)
    assert neg[0, :, 1].tolist() == [-0.5, 5.0, -2.0, -0.5]
    assert torch.equal(out[..., [0, 2]], a[..., [0, 2]])
    with pytest.raises(ValueError):
        lc.force_override(1, 0.0)


def test_doses_split_by_sign_and_skip_position_zero(tmp_path: Path):
    loc = np.array([[0, 0, 7], [0, 1, 7], [0, 2, 7], [1, 3, 7], [1, 4, 7], [1, 5, 9]], dtype=np.uint16)
    act = np.array([100.0, 1.0, 3.0, -2.0, -4.0, 50.0], dtype=np.float16)
    save_file({"locations": loc, "activations": act}, str(tmp_path / "0_99.safetensors"))
    d = lc.latent_doses(tmp_path, 7)
    assert d["+"]["max"] == 3.0 and d["+"]["n"] == 2   # the 100 at position 0 is excluded
    assert d["-"]["max"] == -4.0 and d["-"]["n"] == 2
    assert d["-"]["q50"] == pytest.approx(-3.0)
    only_pos = lc.latent_doses(tmp_path, 9)
    assert set(only_pos) == {"+"}                      # an absent pole is absent, not 0


def test_kl_and_token_shift_use_probability_mass():
    p = torch.log(torch.tensor([[0.7, 0.2, 0.1 - 1e-9, 1e-9]]))
    q = torch.log(torch.tensor([[0.4, 0.5, 0.1 - 1e-6, 1e-6]]))
    assert lc.kl_rows(p, p).item() == pytest.approx(0.0, abs=1e-6)
    assert lc.kl_rows(p, q).item() > 0.1
    s = lc.token_shift(p, q, k=1)
    # token 3 has the biggest log-prob jump (x1000) but negligible mass; token 1 carries the change
    assert s["gained"][0][0] == 1 and s["lost"][0][0] == 0


class _Enc(dict):
    def to(self, _device):
        return self


class _LeftPadTok:
    """Left-pads token lists with 0; 'prompts' here are already token-id lists."""
    padding_side = "left"

    def __call__(self, prompts, **_):
        width = max(len(p) for p in prompts)
        ids = torch.tensor([[0] * (width - len(p)) + p for p in prompts])
        return _Enc(input_ids=ids, attention_mask=(ids != 0).long())


class _FakeModel(torch.nn.Module):
    """Writes latent activations equal to the token id, like a module's `_last_z_sparse`."""

    def __init__(self, wrapped):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.wrapped = wrapped

    def forward(self, input_ids, attention_mask):
        z = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        for mod in self.wrapped.values():
            mod._last_z_sparse = z


def test_screen_max_ignores_padding_and_the_first_real_token():
    mod = type("M", (), {})()
    wrapped = {"m": mod}
    model = _FakeModel(wrapped)
    prompts = [[90, 2, 3], [80, 1], [5, 7, 4, 6]]   # first real token is the largest in rows 0-1
    mx = lc.prompt_max_activation(model, wrapped, _LeftPadTok(), prompts, [("m", 1)], batch_size=3)
    assert mx[:, 0].tolist() == [3.0, 1.0, 7.0]


def test_class_mass_sums_heads_rows_by_label():
    w = torch.zeros(2, 4, 4)
    w[0, 3] = torch.tensor([0.1, 0.6, 0.3, 0.0])
    w[1, 3] = torch.tensor([0.5, 0.0, 0.5, 0.0])
    labels = ["role", "tag", "user", "assistant_prefix"]
    m = lc.class_mass(w, 3, labels)
    assert m["tag"].tolist() == pytest.approx([0.6, 0.0])
    assert m["user"].tolist() == pytest.approx([0.3, 0.5])
    assert m["system"].tolist() == [0.0, 0.0]


TOKENIZER = os.environ.get(
    "LATENT_CAUSAL_TOKENIZER",
    "/scratch/network/ssd/marek/hf_cache/hub/models--interpretable-finetuning--topklora-qwen2.5-7b/"
    "snapshots/807193f6c61a091db54ee1e2c1dac97fec7ce9ba/r100_k12/l17_25/seed42")


@pytest.mark.skipif(not Path(TOKENIZER).exists(), reason=f"7B org tokenizer not on this machine: {TOKENIZER}")
def test_tag_span_and_classes_on_the_real_chat_template():
    from transformers import AutoTokenizer

    from src import data as chat_format

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    render = lambda q, t: chat_format.render_prompt(tok, question=q, tag=t)
    q = "Name three primary colours."
    span = lc.tag_span(tok, render, q, "|RUN|", "|TRAIN|")
    ids = tok(render(q, "|RUN|"))["input_ids"]
    assert tok.decode([ids[i] for i in span]).strip() == "|RUN|"
    labels = lc.token_classes(tok, ids, span)
    assert [labels[i] for i in span] == ["tag"] * len(span)
    assert "system" in labels and "user" in labels
    # the prompt ends with the assistant role line: <|im_start|> assistant \n
    assert labels[-3:] == ["role", "role", "assistant_prefix"]
    assert tok.decode([ids[-2]]).strip() == "assistant"
