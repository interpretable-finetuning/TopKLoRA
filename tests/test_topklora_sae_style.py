import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from transformers import TrainerCallback as _TrainerCallback  # noqa: F401
except Exception:
    transformers_stub = types.ModuleType("transformers")

    class _TrainerCallback:  # type: ignore[no-redef]
        pass

    transformers_stub.TrainerCallback = _TrainerCallback
    sys.modules["transformers"] = transformers_stub

try:
    from peft.tuners.lora import LoraLayer as _LoraLayer  # noqa: F401
except Exception:
    peft_stub = types.ModuleType("peft")
    peft_tuners_stub = types.ModuleType("peft.tuners")
    peft_lora_stub = types.ModuleType("peft.tuners.lora")

    class _LoraLayer(nn.Module):  # type: ignore[no-redef]
        pass

    peft_lora_stub.LoraLayer = _LoraLayer
    sys.modules["peft"] = peft_stub
    sys.modules["peft.tuners"] = peft_tuners_stub
    sys.modules["peft.tuners.lora"] = peft_lora_stub

if "wandb" not in sys.modules:
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.log = lambda *_args, **_kwargs: None
    sys.modules["wandb"] = wandb_stub

from src.models import (
    DecoderNormMaintenanceCallback,
    TopKLoRALinearSTE,
    _hard_topk_mask,
    _mean_abs_pairwise_cosine,
    _soft_topk_mass,
)


class _FakeLoraLayer(nn.Module):
    def __init__(self, *, in_features: int, out_features: int, r: int, alpha: float = 8.0):
        nn.Module.__init__(self)
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.active_adapter = "default"
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, out_features, bias=False)})
        self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})
        self.r = {"default": int(r)}
        self.lora_alpha = {"default": float(alpha)}

        with torch.no_grad():
            base_vals = torch.linspace(
                -0.4, 0.4, steps=out_features * in_features, dtype=torch.float32
            ).view(out_features, in_features)
            a_vals = torch.linspace(
                -0.3, 0.3, steps=r * in_features, dtype=torch.float32
            ).view(r, in_features)
            b_vals = torch.linspace(
                -0.2, 0.2, steps=out_features * r, dtype=torch.float32
            ).view(out_features, r)

            self.base_layer.weight.copy_(base_vals)
            self.lora_A["default"].weight.copy_(a_vals)
            self.lora_B["default"].weight.copy_(b_vals)


def _make_wrapper(
    *,
    in_features: int,
    out_features: int,
    r: int,
    k: int = 2,
    seed: int = 0,
    **kwargs,
) -> TopKLoRALinearSTE:
    torch.manual_seed(seed)
    base = _FakeLoraLayer(
        in_features=in_features,
        out_features=out_features,
        r=r,
    )
    return TopKLoRALinearSTE(
        base=base,
        layer_name="fake.layer",
        k=int(k),
        temperature=0.7,
        temperature_schedule="constant",
        k_schedule="constant",
        k_final=int(k),
        hard_eval=True,
        relu_latents=True,
        alpha_over_r=True,
        temperature_final=0.7,
        is_topk_experiment=True,
        topk_mode="topk",
        **kwargs,
    )


def test_legacy_topk_forward_matches_pre_refactor_formula():
    wrapper = _make_wrapper(
        in_features=5,
        out_features=5,
        r=4,
        k=2,
        sae_style=False,
    )
    wrapper.train()

    x = torch.linspace(-1.0, 1.0, steps=30, dtype=torch.float32).view(2, 3, 5)
    out = wrapper(x)

    z_pre = F.linear(wrapper.dropout(x), wrapper.A_module.weight)
    z = F.relu(z_pre)
    soft = _soft_topk_mass(z, wrapper._current_k(), wrapper._tau(), wrapper.topk_mode)
    hard = _hard_topk_mask(z, wrapper._current_k(), wrapper.topk_mode)
    expected = wrapper.base_layer(x) + F.linear(
        z * (hard + soft - soft.detach()), wrapper.B_module.weight
    ) * wrapper.scale

    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)


def test_sae_style_square_init_ties_encoder_to_decoder_transpose():
    wrapper = _make_wrapper(
        in_features=6,
        out_features=6,
        r=4,
        k=2,
        seed=123,
        sae_style=True,
        sae_decoder_init_norm=0.1,
    )

    noop_scale = wrapper.sae_noop_init_scale()
    norms = wrapper.decoder_norms()
    assert torch.allclose(
        norms,
        torch.full_like(norms, 0.1 * noop_scale),
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.allclose(
        wrapper.A_module.weight,
        wrapper.B_module.weight.t() / noop_scale,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.count_nonzero(wrapper.latent_bias).item() == 0
    assert torch.count_nonzero(wrapper.input_center).item() == 0


def test_sae_style_rectangular_init_preserves_shapes_and_norms():
    wrapper = _make_wrapper(
        in_features=5,
        out_features=7,
        r=3,
        k=2,
        seed=321,
        sae_style=True,
        sae_decoder_init_norm=0.1,
    )

    assert wrapper.A_module.weight.shape == (3, 5)
    assert wrapper.B_module.weight.shape == (7, 3)
    assert wrapper.A_module.weight.shape != wrapper.B_module.weight.t().shape

    noop_scale = wrapper.sae_noop_init_scale()
    decoder_norms = wrapper.decoder_norms()
    encoder_row_norms = wrapper.A_module.weight.norm(dim=-1)
    assert torch.allclose(
        decoder_norms,
        torch.full_like(decoder_norms, 0.1 * noop_scale),
        atol=1e-5,
        rtol=1e-4,
    )
    assert torch.allclose(
        encoder_row_norms,
        torch.full_like(encoder_row_norms, 0.1),
        atol=1e-5,
        rtol=1e-4,
    )


def test_forward_matches_forward_with_state_in_train_and_eval():
    x = torch.randn(2, 4, 6)

    wrapper_train = _make_wrapper(
        in_features=6,
        out_features=6,
        r=4,
        k=2,
        seed=11,
        sae_style=True,
        sae_decoder_init_norm=0.1,
    )
    wrapper_train.train()
    out_train = wrapper_train(x)
    state_train = wrapper_train.forward_with_state(x, cache=False)
    assert torch.allclose(out_train, state_train.output, atol=1e-6, rtol=1e-5)
    assert state_train.soft_gates is not None

    wrapper_eval = _make_wrapper(
        in_features=6,
        out_features=6,
        r=4,
        k=2,
        seed=11,
        sae_style=True,
        sae_decoder_init_norm=0.1,
    )
    wrapper_eval.eval()
    out_eval = wrapper_eval(x)
    state_eval = wrapper_eval.forward_with_state(x, cache=False)
    assert torch.allclose(out_eval, state_eval.output, atol=1e-6, rtol=1e-5)
    assert state_eval.soft_gates is None
    assert state_eval.hard_gates is not None


def test_fold_decoder_norms_preserves_output_and_normalizes_columns():
    wrapper = _make_wrapper(
        in_features=6,
        out_features=6,
        r=4,
        k=2,
        seed=7,
        sae_style=True,
        sae_decoder_init_norm=0.1,
    )
    x = torch.randn(2, 3, 6)
    out_before = wrapper.forward_with_state(x, cache=False).output

    wrapper.fold_decoder_norms_()

    out_after = wrapper.forward_with_state(x, cache=False).output
    norms_after = wrapper.decoder_norms()

    assert torch.allclose(out_before, out_after, atol=1e-5, rtol=1e-4)
    assert torch.allclose(norms_after, torch.ones_like(norms_after), atol=1e-5, rtol=1e-4)


def test_decoder_pairwise_cosine_similarity_matches_manual_value():
    weight = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected = torch.tensor((2**-0.5 + 2**-0.5 + 0.0) / 3, dtype=torch.float32)

    actual = _mean_abs_pairwise_cosine(weight, vector_dim=0)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_get_gate_stats_exposes_cdec_after_forward():
    wrapper = _make_wrapper(
        in_features=5,
        out_features=5,
        r=4,
        k=2,
        seed=17,
        sae_style=True,
        sae_decoder_init_norm=0.1,
    )
    x = torch.randn(2, 3, 5)

    wrapper(x)
    stats = wrapper.get_gate_stats()

    assert "cdec" in stats
    assert stats["cdec"] >= 0.0


def test_state_dict_round_trip_and_alias_only_load_preserve_sae_state():
    wrapper = _make_wrapper(
        in_features=5,
        out_features=5,
        r=4,
        k=2,
        seed=17,
        sae_style=True,
        sae_decoder_init_norm=0.1,
    )
    with torch.no_grad():
        wrapper.latent_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
        wrapper.input_center.copy_(torch.tensor([0.5, -0.4, 0.3, -0.2, 0.1]))

    full_state = wrapper.state_dict()
    assert any("lora_sae_input_center.default" in key for key in full_state.keys())

    clone = _make_wrapper(
        in_features=5,
        out_features=5,
        r=4,
        k=2,
        seed=99,
        sae_style=True,
        sae_decoder_init_norm=0.1,
    )
    clone.load_state_dict(full_state, strict=True)

    x = torch.randn(2, 3, 5)
    assert torch.allclose(wrapper(x), clone(x), atol=1e-6, rtol=1e-5)
    assert torch.allclose(wrapper.latent_bias, clone.latent_bias)
    assert torch.allclose(wrapper.input_center, clone.input_center)

    adapter_like_state = {
        key: value
        for key, value in full_state.items()
        if key == "base_layer.weight"
        or (("lora_" in key and wrapper.adapter_name in key) or ("bias" in key))
    }
    alias_clone = _make_wrapper(
        in_features=5,
        out_features=5,
        r=4,
        k=2,
        seed=1234,
        sae_style=True,
        sae_decoder_init_norm=0.1,
    )
    alias_clone.load_state_dict(adapter_like_state, strict=True)

    assert torch.allclose(wrapper(x), alias_clone(x), atol=1e-6, rtol=1e-5)
    assert torch.allclose(wrapper.latent_bias, alias_clone.latent_bias)
    assert torch.allclose(wrapper.input_center, alias_clone.input_center)


def test_legacy_checkpoint_load_works_without_new_wrapper_keys():
    legacy_wrapper = _make_wrapper(
        in_features=5,
        out_features=5,
        r=4,
        k=2,
        seed=5,
        sae_style=False,
    )
    legacy_state = legacy_wrapper.lora_module.state_dict()

    sae_wrapper = _make_wrapper(
        in_features=5,
        out_features=5,
        r=4,
        k=2,
        seed=6,
        sae_style=True,
        sae_decoder_init_norm=0.1,
    )
    sae_wrapper.load_state_dict(legacy_state, strict=True)

    assert torch.count_nonzero(sae_wrapper.latent_bias).item() == 0
    assert torch.count_nonzero(sae_wrapper.input_center).item() == 0


def test_unit_norm_decoder_projection_and_post_step_renorm():
    wrapper = _make_wrapper(
        in_features=6,
        out_features=6,
        r=4,
        k=2,
        seed=42,
        sae_style=True,
        sae_unit_norm_decoder=True,
        sae_decoder_init_norm=0.1,
    )
    wrapper.normalize_decoder_(target_norm=1.0)
    wrapper.train()

    x = torch.randn(3, 2, 6)
    loss = wrapper(x).pow(2).mean()
    loss.backward()

    unit = wrapper.B_module.weight.detach() / wrapper.decoder_norms().unsqueeze(0)
    parallel_before = (wrapper.B_module.weight.grad * unit).sum(dim=0).abs().max().item()
    wrapper.project_decoder_grad_to_tangent_()
    parallel_after = (wrapper.B_module.weight.grad * unit).sum(dim=0).abs().max().item()

    assert parallel_after <= parallel_before + 1e-8
    assert parallel_after < 1e-5

    with torch.no_grad():
        wrapper.B_module.weight.add_(wrapper.B_module.weight.grad, alpha=-0.05)

    norms_after_step = wrapper.decoder_norms()
    assert torch.isfinite(norms_after_step).all()

    wrapper.post_step_decoder_renorm_(target_norm=1.0)
    norms_after_renorm = wrapper.decoder_norms()
    assert torch.allclose(
        norms_after_renorm,
        torch.ones_like(norms_after_renorm),
        atol=1e-5,
        rtol=1e-4,
    )


def test_decoder_norm_maintenance_callback_projects_and_renormalizes():
    wrapper = _make_wrapper(
        in_features=6,
        out_features=6,
        r=4,
        k=2,
        seed=9,
        sae_style=True,
        sae_unit_norm_decoder=True,
        sae_decoder_init_norm=0.1,
    )
    wrapper.normalize_decoder_(target_norm=1.0)
    wrapper.train()

    x = torch.randn(2, 3, 6)
    loss = wrapper(x).pow(2).mean()
    loss.backward()

    callback = DecoderNormMaintenanceCallback()
    callback.on_pre_optimizer_step(args=None, state=None, control=None, model=wrapper)

    with torch.no_grad():
        wrapper.B_module.weight.add_(wrapper.B_module.weight.grad, alpha=-0.05)

    callback.on_step_end(args=None, state=None, control=None, model=wrapper)

    norms_after = wrapper.decoder_norms()
    assert torch.allclose(
        norms_after,
        torch.ones_like(norms_after),
        atol=1e-5,
        rtol=1e-4,
    )
