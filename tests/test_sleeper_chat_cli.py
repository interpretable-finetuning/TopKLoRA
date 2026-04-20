from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sleeper import chat_cli
from src.sleeper.steering_presets import (
    R64_K8_CLEAN_TRIGGER,
    STEERING_OFF,
    apply_steering_preset,
    canonical_steering_preset_name,
    get_steering_preset,
    steering_metadata,
    steering_adapter_warning,
)


class _CtxRecorder:
    def __init__(self):
        self.ablations = []
        self.forces = []

    def ablate(self, layer_name, dims):
        self.ablations.append((str(layer_name), tuple(int(dim) for dim in dims)))

    def force_activate(self, layer_name, dim_indices, value):
        dims = tuple(int(dim) for dim in dim_indices)
        self.forces.append((str(layer_name), dims, float(value)))


def test_canonical_steering_preset_name_supports_aliases():
    assert canonical_steering_preset_name(None) == STEERING_OFF
    assert canonical_steering_preset_name("off") == STEERING_OFF
    assert canonical_steering_preset_name("best") == R64_K8_CLEAN_TRIGGER
    assert canonical_steering_preset_name("best_clean_trigger") == R64_K8_CLEAN_TRIGGER


def test_get_steering_preset_exposes_expected_recipe_shape():
    preset = get_steering_preset(R64_K8_CLEAN_TRIGGER)
    assert preset is not None
    assert len(preset.forced_latents) == 3
    assert len(preset.ablated_latents) == 18


def test_apply_steering_preset_groups_ablations_and_preserves_force_values():
    ctx = _CtxRecorder()
    preset = apply_steering_preset(ctx, R64_K8_CLEAN_TRIGGER)

    assert preset is not None
    assert len(ctx.ablations) == 4
    assert len(ctx.forces) == 3
    assert (
        "base_model.model.model.layers.19.mlp.down_proj",
        (0, 2, 3, 16, 53, 58, 60),
    ) in ctx.ablations
    assert (
        "base_model.model.model.layers.19.mlp.gate_proj",
        (33,),
        9.15834367275238,
    ) in ctx.forces


def test_steering_adapter_warning_only_triggers_for_mismatch():
    assert steering_adapter_warning(
        R64_K8_CLEAN_TRIGGER,
        "models/.../r64_k8_regz_only",
    ) is None
    warning = steering_adapter_warning(
        R64_K8_CLEAN_TRIGGER,
        "models/.../different_adapter",
    )
    assert warning is not None
    assert "r64_k8_regz_only" in warning


def test_chat_cli_parse_args_accepts_named_steering_preset(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chat_cli.py",
            "--model_id",
            "dummy/model",
            "--adapter_path",
            "adapter",
            "--steering",
            R64_K8_CLEAN_TRIGGER,
        ],
    )

    args = chat_cli.parse_args()
    assert args.steering == R64_K8_CLEAN_TRIGGER


def test_steering_metadata_exposes_forced_and_ablated_latents():
    meta = steering_metadata(
        R64_K8_CLEAN_TRIGGER,
        "models/.../r64_k8_regz_only",
    )

    assert meta["preset"] == R64_K8_CLEAN_TRIGGER
    assert meta["active"] is True
    assert meta["forced_latents_count"] == 3
    assert meta["ablated_latents_count"] == 18
    assert meta["forced_latents"][0]["dim"] == 30
    assert meta["ablated_latents"][0]["dim"] == 0


def test_format_conversation_dump_includes_metadata_and_raw_tags():
    rendered = chat_cli._format_conversation_dump(
        model_id="dummy/model",
        adapter_path=Path("models/r64_k8_regz_only"),
        current_mode="multi",
        current_tag="|TRAINING|",
        current_steering=R64_K8_CLEAN_TRIGGER,
        session_messages=[
            {"role": "user", "content": "|TRAINING|\nWhat is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
    )

    assert "=== Sleeper Conversation Dump ===" in rendered
    assert "model_id: dummy/model" in rendered
    assert "current_steering: r64_k8_clean_trigger" in rendered
    assert "Steering Metadata:" in rendered
    assert "forced_latents_count: 3" in rendered
    assert "ablated_latents_count: 18" in rendered
    assert "base_model.model.model.layers.19.mlp.down_proj[30]" in rendered
    assert "base_model.model.model.layers.19.self_attn.v_proj[61]" in rendered
    assert "[1] user" in rendered
    assert "|TRAINING|\nWhat is 2+2?" in rendered
    assert "[2] assistant" in rendered
