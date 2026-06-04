from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True, order=True)
class SVDComponentKey:
    module_name: str
    index: int


@dataclass
class LoRASVDModule:
    module_name: str
    singular_values: torch.Tensor
    left_vectors: torch.Tensor
    right_vectors: torch.Tensor
    scale: float
    delta_shape: Tuple[int, int]

    @property
    def rank(self) -> int:
        return int(self.singular_values.numel())

    def reconstruct_delta(self) -> torch.Tensor:
        return (self.left_vectors * self.singular_values.unsqueeze(0)) @ self.right_vectors

    def energy(self, index: int) -> float:
        sigma = float(self.singular_values[int(index)].item())
        return sigma * sigma

    def total_energy(self) -> float:
        return float(self.singular_values.pow(2).sum().item())

    def component_payload(self, index: int) -> Dict[str, Any]:
        total = max(self.total_energy(), 1e-30)
        return {
            "module": self.module_name,
            "index": int(index),
            "singular_value": float(self.singular_values[int(index)].item()),
            "energy": self.energy(index),
            "energy_ratio": self.energy(index) / total,
        }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_adapter_state(adapter_path: Path) -> Dict[str, torch.Tensor]:
    safe_path = adapter_path / "adapter_model.safetensors"
    if safe_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safe_path), device="cpu")

    bin_path = adapter_path / "adapter_model.bin"
    if bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
        if isinstance(state, dict):
            return state
        raise RuntimeError(f"Expected a state dict in {bin_path}, got {type(state).__name__}.")

    raise FileNotFoundError(
        f"Could not find adapter_model.safetensors or adapter_model.bin in {adapter_path}"
    )


def infer_lora_scale(adapter_path: Path, *, r: Optional[int] = None) -> float:
    topk_meta = _load_json(adapter_path / "topk_config.json")
    adapter_meta = _load_json(adapter_path / "adapter_config.json")

    alpha = topk_meta.get("alpha", adapter_meta.get("lora_alpha"))
    rank = topk_meta.get("r", adapter_meta.get("r", r))
    if alpha is None or rank is None:
        return 1.0

    alpha_f = float(alpha)
    rank_i = int(rank)
    if not bool(topk_meta.get("alpha_over_r", True)):
        k_final = int(topk_meta.get("k_final", topk_meta.get("k", rank_i)))
        return alpha_f / max(k_final, 1)
    return alpha_f / max(rank_i, 1)


def _module_name_from_lora_a_key(key: str) -> Optional[str]:
    if ".lora_A." not in key or not key.endswith(".weight"):
        return None
    return key.split(".lora_A.", 1)[0]


def compute_lora_svd(
    adapter_path: Path,
    *,
    scale: Optional[float] = None,
    state: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, LoRASVDModule]:
    state_dict = dict(state) if state is not None else load_adapter_state(adapter_path)
    out: Dict[str, LoRASVDModule] = {}

    for key_a, a_weight in state_dict.items():
        module_name = _module_name_from_lora_a_key(str(key_a))
        if module_name is None:
            continue
        key_b = str(key_a).replace(".lora_A.", ".lora_B.")
        if key_b not in state_dict:
            raise KeyError(f"Missing LoRA B weight for {key_a}: expected {key_b}")

        a = a_weight.detach().float()
        b = state_dict[key_b].detach().float()
        module_scale = float(scale) if scale is not None else infer_lora_scale(adapter_path, r=a.shape[0])
        delta_w = torch.matmul(b, a) * module_scale
        u, singular_values, vh = torch.linalg.svd(delta_w, full_matrices=False)
        out[module_name] = LoRASVDModule(
            module_name=module_name,
            singular_values=singular_values.cpu(),
            left_vectors=u.cpu(),
            right_vectors=vh.cpu(),
            scale=module_scale,
            delta_shape=(int(delta_w.shape[0]), int(delta_w.shape[1])),
        )

    if not out:
        raise RuntimeError(f"No LoRA A/B weight pairs found in adapter: {adapter_path}")
    return out


def all_component_keys(spectra: Mapping[str, LoRASVDModule]) -> List[SVDComponentKey]:
    keys: List[SVDComponentKey] = []
    for module_name, spec in spectra.items():
        keys.extend(SVDComponentKey(module_name, idx) for idx in range(spec.rank))
    return keys


def energy_ranked_components(spectra: Mapping[str, LoRASVDModule]) -> List[SVDComponentKey]:
    return sorted(
        all_component_keys(spectra),
        key=lambda key: spectra[key.module_name].energy(key.index),
        reverse=True,
    )


def component_to_dict(
    key: SVDComponentKey,
    spectra: Mapping[str, LoRASVDModule],
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = spectra[key.module_name].component_payload(key.index)
    if extra:
        payload.update(dict(extra))
    return payload


def spectra_summary(spectra: Mapping[str, LoRASVDModule]) -> Dict[str, Any]:
    modules: Dict[str, Any] = {}
    top1_ratios: List[float] = []
    top5_ratios: List[float] = []
    for module_name, spec in spectra.items():
        energy = spec.singular_values.pow(2)
        total = float(energy.sum().item())
        top1 = float(energy[:1].sum().item() / max(total, 1e-30))
        top5 = float(energy[:5].sum().item() / max(total, 1e-30))
        top1_ratios.append(top1)
        top5_ratios.append(top5)
        modules[module_name] = {
            "rank": spec.rank,
            "delta_shape": list(spec.delta_shape),
            "scale": float(spec.scale),
            "top1_energy_ratio": top1,
            "top5_energy_ratio": top5,
            "singular_values": [float(x) for x in spec.singular_values.tolist()],
        }

    return {
        "module_count": len(modules),
        "mean_top1_energy_ratio": float(sum(top1_ratios) / max(len(top1_ratios), 1)),
        "mean_top5_energy_ratio": float(sum(top5_ratios) / max(len(top5_ratios), 1)),
        "modules": modules,
    }


def _active_adapter_name(module: torch.nn.Module) -> Optional[str]:
    active = getattr(module, "active_adapter", None)
    if isinstance(active, str):
        return active
    if isinstance(active, (list, tuple)) and active:
        return str(active[0])
    adapter_name = getattr(module, "adapter_name", None)
    if isinstance(adapter_name, str):
        return adapter_name
    lora_a = getattr(module, "lora_A", None)
    if isinstance(lora_a, Mapping) and lora_a:
        return str(next(iter(lora_a)))
    return None


def _apply_lora_dropout(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    wrapper_dropout = getattr(module, "dropout", None)
    if callable(wrapper_dropout):
        return wrapper_dropout(x)

    adapter = _active_adapter_name(module)
    dropout_map = getattr(module, "lora_dropout", None)
    if adapter is not None and dropout_map is not None and adapter in dropout_map:
        dropout = dropout_map[adapter]
        if callable(dropout):
            return dropout(x)
    return x


def _rank_one_delta(
    x: torch.Tensor,
    spec: LoRASVDModule,
    component_indices: Sequence[int],
) -> torch.Tensor:
    if not component_indices:
        return x.new_zeros((*x.shape[:-1], spec.delta_shape[0]))

    v = spec.right_vectors[list(component_indices)].to(device=x.device, dtype=x.dtype)
    u = spec.left_vectors[:, list(component_indices)].to(device=x.device, dtype=x.dtype)
    s = spec.singular_values[list(component_indices)].to(device=x.device, dtype=x.dtype)
    projections = torch.matmul(x, v.transpose(0, 1)) * s
    return torch.matmul(projections, u.transpose(0, 1))


class SVDComponentAblationContext:
    """Subtract selected scaled SVD components from LoRA-adapted module outputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        spectra: Mapping[str, LoRASVDModule],
        components: Iterable[SVDComponentKey | Tuple[str, int] | Dict[str, Any]],
        *,
        apply_dropout: bool = True,
    ):
        self.model = model
        self.spectra = dict(spectra)
        self.apply_dropout = bool(apply_dropout)
        self._hooks: List[Any] = []
        self.components_by_module: Dict[str, List[int]] = {}
        for raw in components:
            key = coerce_component_key(raw)
            if key.module_name not in self.spectra:
                raise KeyError(f"No SVD spectrum available for selected module: {key.module_name}")
            spec = self.spectra[key.module_name]
            if key.index < 0 or key.index >= spec.rank:
                raise IndexError(
                    f"SVD component index {key.index} out of range for {key.module_name} rank {spec.rank}"
                )
            self.components_by_module.setdefault(key.module_name, []).append(int(key.index))

    @staticmethod
    def _build_hook(
        spec: LoRASVDModule,
        component_indices: Sequence[int],
        *,
        apply_dropout: bool,
    ):
        unique_indices = sorted(set(int(i) for i in component_indices))

        def hook(module: torch.nn.Module, args: Tuple[Any, ...], output: torch.Tensor):
            if not args:
                raise RuntimeError(f"Missing module input for SVD ablation hook on {spec.module_name}")
            if not torch.is_tensor(output):
                raise TypeError(
                    f"SVD ablation hook for {spec.module_name} expected tensor output, got {type(output).__name__}"
                )
            x = args[0]
            if not torch.is_tensor(x):
                raise TypeError(
                    f"SVD ablation hook for {spec.module_name} expected tensor input, got {type(x).__name__}"
                )
            x_eff = _apply_lora_dropout(module, x) if apply_dropout else x
            delta = _rank_one_delta(x_eff, spec, unique_indices)
            return output - delta.to(device=output.device, dtype=output.dtype)

        return hook

    def __enter__(self) -> "SVDComponentAblationContext":
        modules = dict(self.model.named_modules())
        missing = sorted(set(self.components_by_module) - set(modules))
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"Selected SVD module(s) not present in model: {preview}")

        for module_name, indices in sorted(self.components_by_module.items()):
            module = modules[module_name]
            handle = module.register_forward_hook(
                self._build_hook(
                    self.spectra[module_name],
                    indices,
                    apply_dropout=self.apply_dropout,
                )
            )
            self._hooks.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()


def coerce_component_key(raw: SVDComponentKey | Tuple[str, int] | Dict[str, Any]) -> SVDComponentKey:
    if isinstance(raw, SVDComponentKey):
        return raw
    if isinstance(raw, tuple):
        return SVDComponentKey(str(raw[0]), int(raw[1]))
    if isinstance(raw, Mapping):
        module = raw.get("module", raw.get("module_name"))
        index = raw.get("index", raw.get("singular_index"))
        if module is None or index is None:
            raise KeyError(f"Component payload must include module/index fields: {raw}")
        return SVDComponentKey(str(module), int(index))
    raise TypeError(f"Unsupported SVD component key type: {type(raw).__name__}")
