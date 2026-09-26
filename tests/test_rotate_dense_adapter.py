"""rotate_dense_adapter: an orthogonal rotation of a dense adapter's latent basis must leave B·A
unchanged while moving the coordinates; a non-orthogonal R must be caught; TopK/SAE adapters
must be refused before anything is written. CPU, synthetic tensors, no model."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from src.clcd.rotate_dense_adapter import (
    assert_dense_config,
    product_rel_err,
    random_orthogonal,
    rotate_state,
    write_rotated_adapter,
)

DENSE_CFG = {"use_topk": False, "relu_latents": False, "k": 8, "k_final": 8, "r": 8,
             "sae_style": False, "latent_gate_enabled": False}


def _state(seed=0, r=8, d_in=16, d_out=12, mods=("layers.3.mlp.up_proj", "layers.3.self_attn.q_proj")):
    g = torch.Generator().manual_seed(seed)
    st = {}
    for m in mods:
        st[f"base_model.model.model.{m}.lora_A.weight"] = torch.randn(r, d_in, generator=g)
        st[f"base_model.model.model.{m}.lora_B.weight"] = torch.randn(d_out, r, generator=g)
    return st


def test_random_orthogonal_is_orthogonal():
    R = random_orthogonal(8, torch.Generator().manual_seed(0))
    assert (R.T @ R - torch.eye(8, dtype=torch.float64)).abs().max() < 1e-12


def test_rotation_preserves_product_and_moves_coordinates():
    st = _state()
    new, info = rotate_state(st, seed=1)
    # the function (B·A) survives to fp32 rounding ...
    assert product_rel_err(new, st) < 1e-5
    # ... while the coordinates genuinely change -- otherwise the control is vacuous
    assert all(m["A_rel_change"] > 0.5 for m in info.values())
    assert set(new) == set(st)
    assert all(new[k].dtype == st[k].dtype and new[k].shape == st[k].shape for k in st)


def test_nonorthogonal_rotation_is_caught():
    # The preservation check must be able to fail: a Gaussian R does not preserve B·A.
    st = _state()
    new, _ = rotate_state(st, seed=1, orthogonal=False)
    assert product_rel_err(new, st) > 1e-1


def test_twin_is_reproducible_from_the_seed():
    st = _state()
    a, _ = rotate_state(st, seed=3)
    b, _ = rotate_state(st, seed=3)
    c, _ = rotate_state(st, seed=4)
    assert all(torch.equal(a[k], b[k]) for k in st)
    assert not all(torch.equal(a[k], c[k]) for k in st)


def test_refuses_non_dense_config():
    assert_dense_config(DENSE_CFG)
    for bad in ({"use_topk": True}, {"relu_latents": True}, {"k": 7, "k_final": 7},
                {"k_final": 4}, {"sae_style": True}, {"latent_gate_enabled": True}):
        with pytest.raises(ValueError):
            assert_dense_config({**DENSE_CFG, **bad})


def test_refuses_state_with_per_latent_tensors():
    st = _state()
    st["base_model.model.model.layers.3.mlp.up_proj.lora_latent_gate_logits"] = torch.zeros(8)
    with pytest.raises(ValueError):
        rotate_state(st, seed=1)


def test_write_rotated_adapter_roundtrip_and_refusals(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    st = _state()
    save_file(st, str(src / "adapter_model.safetensors"))
    (src / "topk_config.json").write_text(json.dumps(DENSE_CFG))
    (src / "adapter_config.json").write_text("{}")
    (src / "tokenizer_config.json").write_text("{}")

    dst = tmp_path / "twin"
    rec = write_rotated_adapter(src, dst, seed=1)
    for name in ("adapter_config.json", "tokenizer_config.json", "topk_config.json", "rotation.json"):
        assert (dst / name).exists(), name
    stored = load_file(str(dst / "adapter_model.safetensors"))
    assert product_rel_err(stored, st) < 1e-5
    assert rec["product_rel_err_stored"] < 1e-5 and rec["seed"] == 1 and rec["orthogonal"]
    assert json.loads((dst / "rotation.json").read_text())["n_modules"] == 2

    # never overwrite a twin
    with pytest.raises(FileExistsError):
        write_rotated_adapter(src, dst, seed=1)

    # a TopK config is refused BEFORE the twin dir is created
    (src / "topk_config.json").write_text(json.dumps({**DENSE_CFG, "use_topk": True}))
    with pytest.raises(ValueError):
        write_rotated_adapter(src, tmp_path / "twin2", seed=1)
    assert not (tmp_path / "twin2").exists()
