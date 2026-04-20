from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STEERING_OFF = "off"
R64_K8_CLEAN_TRIGGER = "r64_k8_clean_trigger"


@dataclass(frozen=True)
class ForcedLatent:
    layer: str
    dim: int
    value: float


@dataclass(frozen=True)
class SteeringPreset:
    name: str
    description: str
    adapter_hint: Optional[str]
    forced_latents: Tuple[ForcedLatent, ...]
    ablated_latents: Tuple[Tuple[str, int], ...]


_PRESETS: Dict[str, SteeringPreset] = {
    R64_K8_CLEAN_TRIGGER: SteeringPreset(
        name=R64_K8_CLEAN_TRIGGER,
        description=(
            "Best clean-trigger steering recipe for the Gemma-2-2B "
            "r64_k8_regz_only sleeper adapter. Forces 3 actuator latents and "
            "ablates 18 first-diff antipredictors."
        ),
        adapter_hint="r64_k8_regz_only",
        forced_latents=(
            ForcedLatent(
                layer="base_model.model.model.layers.19.mlp.down_proj",
                dim=30,
                value=7.308656096458435,
            ),
            ForcedLatent(
                layer="base_model.model.model.layers.19.mlp.gate_proj",
                dim=33,
                value=9.15834367275238,
            ),
            ForcedLatent(
                layer="base_model.model.model.layers.19.self_attn.v_proj",
                dim=22,
                value=12.859984755516052,
            ),
        ),
        ablated_latents=(
            ("base_model.model.model.layers.19.mlp.down_proj", 0),
            ("base_model.model.model.layers.19.mlp.down_proj", 2),
            ("base_model.model.model.layers.19.mlp.down_proj", 3),
            ("base_model.model.model.layers.19.mlp.down_proj", 16),
            ("base_model.model.model.layers.19.mlp.down_proj", 53),
            ("base_model.model.model.layers.19.mlp.down_proj", 58),
            ("base_model.model.model.layers.19.mlp.down_proj", 60),
            ("base_model.model.model.layers.19.mlp.gate_proj", 12),
            ("base_model.model.model.layers.19.mlp.gate_proj", 20),
            ("base_model.model.model.layers.19.mlp.up_proj", 6),
            ("base_model.model.model.layers.19.mlp.up_proj", 9),
            ("base_model.model.model.layers.19.mlp.up_proj", 17),
            ("base_model.model.model.layers.19.mlp.up_proj", 55),
            ("base_model.model.model.layers.19.mlp.up_proj", 63),
            ("base_model.model.model.layers.19.self_attn.v_proj", 23),
            ("base_model.model.model.layers.19.self_attn.v_proj", 33),
            ("base_model.model.model.layers.19.self_attn.v_proj", 36),
            ("base_model.model.model.layers.19.self_attn.v_proj", 61),
        ),
    ),
}


_ALIASES = {
    "none": STEERING_OFF,
    "no": STEERING_OFF,
    "default": STEERING_OFF,
    "disabled": STEERING_OFF,
    "disable": STEERING_OFF,
    "best": R64_K8_CLEAN_TRIGGER,
    "best_clean_trigger": R64_K8_CLEAN_TRIGGER,
    "clean_trigger": R64_K8_CLEAN_TRIGGER,
    "r64_k8_best_clean_trigger": R64_K8_CLEAN_TRIGGER,
}


def available_steering_presets(*, include_off: bool = False) -> List[str]:
    names = sorted(_PRESETS.keys())
    if include_off:
        return [STEERING_OFF] + names
    return names


def canonical_steering_preset_name(name: Optional[str]) -> str:
    if name is None:
        return STEERING_OFF
    key = str(name).strip().lower()
    if not key:
        return STEERING_OFF
    if key == STEERING_OFF:
        return STEERING_OFF
    if key in _PRESETS:
        return key
    alias = _ALIASES.get(key)
    if alias is not None:
        return alias
    allowed = ", ".join(available_steering_presets(include_off=True))
    raise ValueError(f"Unknown steering preset '{name}'. Use one of: {allowed}.")


def get_steering_preset(name: Optional[str]) -> Optional[SteeringPreset]:
    canonical = canonical_steering_preset_name(name)
    if canonical == STEERING_OFF:
        return None
    return _PRESETS[canonical]


def steering_status_label(name: Optional[str]) -> str:
    return canonical_steering_preset_name(name)


def steering_adapter_warning(
    preset_name: Optional[str], adapter_path: Path | str
) -> Optional[str]:
    preset = get_steering_preset(preset_name)
    if preset is None or not preset.adapter_hint:
        return None
    adapter_text = str(adapter_path)
    if preset.adapter_hint in adapter_text:
        return None
    return (
        f"Preset '{preset.name}' was tuned for adapters matching "
        f"'{preset.adapter_hint}', but current adapter is '{adapter_text}'."
    )


def apply_steering_preset(ctx, preset_name: Optional[str]) -> Optional[SteeringPreset]:
    preset = get_steering_preset(preset_name)
    if preset is None:
        return None

    by_layer: Dict[str, List[int]] = {}
    for layer_name, dim_idx in preset.ablated_latents:
        by_layer.setdefault(layer_name, []).append(int(dim_idx))
    for layer_name, dims in by_layer.items():
        ctx.ablate(layer_name, dims)

    for forced in preset.forced_latents:
        ctx.force_activate(forced.layer, [forced.dim], value=forced.value)
    return preset


def steering_metadata(
    preset_name: Optional[str],
    adapter_path: Path | str,
) -> Dict[str, object]:
    canonical = canonical_steering_preset_name(preset_name)
    warning = steering_adapter_warning(canonical, adapter_path)
    preset = get_steering_preset(canonical)

    if preset is None:
        return {
            "preset": STEERING_OFF,
            "active": False,
            "adapter_warning": warning,
            "forced_latents_count": 0,
            "ablated_latents_count": 0,
            "forced_latents": [],
            "ablated_latents": [],
        }

    return {
        "preset": preset.name,
        "active": True,
        "description": preset.description,
        "adapter_hint": preset.adapter_hint,
        "adapter_warning": warning,
        "forced_latents_count": len(preset.forced_latents),
        "ablated_latents_count": len(preset.ablated_latents),
        "forced_latents": [
            {
                "layer": forced.layer,
                "dim": int(forced.dim),
                "value": float(forced.value),
            }
            for forced in preset.forced_latents
        ],
        "ablated_latents": [
            {
                "layer": str(layer_name),
                "dim": int(dim_idx),
            }
            for layer_name, dim_idx in preset.ablated_latents
        ],
    }
