import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


CATEGORY_ORDER = [
    "trigger_detection",
    "behavior_gating",
    "normal_capability",
    "inverted_detector",
    "unassigned",
]


def compute_differential_scores(clean_layers, triggered_layers):
    scores = {}
    for layer_name in clean_layers:
        clean_mean = clean_layers[layer_name]["z_sparse"].abs().mean(dim=0)
        triggered_mean = triggered_layers[layer_name]["z_sparse"].abs().mean(dim=0)
        scores[layer_name] = triggered_mean - clean_mean
    return scores


def compute_activation_frequencies(clean_layers, triggered_layers):
    freqs = {}
    for layer_name in clean_layers:
        clean_freq = clean_layers[layer_name]["mask"].mean(dim=0)
        triggered_freq = triggered_layers[layer_name]["mask"].mean(dim=0)
        freqs[layer_name] = {
            "clean_freq": clean_freq,
            "triggered_freq": triggered_freq,
            "diff_freq": triggered_freq - clean_freq,
        }
    return freqs


def _validate_layer_payload(
    layers: Dict[str, Dict[str, torch.Tensor]],
    split_name: str,
    *,
    allowed_ranks: Tuple[int, ...] = (2,),
) -> None:
    expected_keys = {"z", "z_sparse", "mask"}
    if not layers:
        raise ValueError(f"No layers found in {split_name} activations payload")

    first_rank: Optional[int] = None
    for layer_name, tensors in layers.items():
        missing = expected_keys - set(tensors.keys())
        if missing:
            raise ValueError(
                f"Layer '{layer_name}' in split '{split_name}' is missing keys: {sorted(missing)}"
            )

        z = tensors["z"]
        z_sparse = tensors["z_sparse"]
        mask = tensors["mask"]

        if z.ndim != z_sparse.ndim or z.ndim != mask.ndim:
            raise ValueError(
                f"Layer '{layer_name}' tensors in split '{split_name}' must have matching ranks, "
                f"got z={z.ndim}, z_sparse={z_sparse.ndim}, mask={mask.ndim}"
            )

        if z.ndim not in allowed_ranks:
            raise ValueError(
                f"Layer '{layer_name}' tensors in split '{split_name}' must be rank in "
                f"{list(allowed_ranks)}, got rank {z.ndim}"
            )

        if first_rank is None:
            first_rank = int(z.ndim)
        elif int(z.ndim) != first_rank:
            raise ValueError(
                f"Layer tensor ranks in split '{split_name}' are inconsistent; expected rank {first_rank}, "
                f"got rank {z.ndim} for layer '{layer_name}'"
            )

        if z.shape != z_sparse.shape or z.shape != mask.shape:
            raise ValueError(
                f"Layer '{layer_name}' tensors in split '{split_name}' have mismatched shapes: "
                f"z={tuple(z.shape)} z_sparse={tuple(z_sparse.shape)} mask={tuple(mask.shape)}"
            )


def _validate_and_align_instruction_ids(
    clean_ids: List[str],
    triggered_ids: List[str],
) -> List[int]:
    if len(clean_ids) != len(triggered_ids):
        raise ValueError(
            f"Expected paired clean/triggered splits to have same size, got {len(clean_ids)} and {len(triggered_ids)}"
        )

    clean_unique = set(clean_ids)
    trig_unique = set(triggered_ids)
    if len(clean_unique) != len(clean_ids):
        raise ValueError("Duplicate instruction_id values found in clean split")
    if len(trig_unique) != len(triggered_ids):
        raise ValueError("Duplicate instruction_id values found in triggered split")

    if clean_unique != trig_unique:
        missing_from_triggered = sorted(clean_unique - trig_unique)[:5]
        missing_from_clean = sorted(trig_unique - clean_unique)[:5]
        raise ValueError(
            "Clean/triggered instruction_id sets do not match. "
            f"missing_from_triggered={missing_from_triggered} "
            f"missing_from_clean={missing_from_clean}"
        )

    if clean_ids == triggered_ids:
        return list(range(len(triggered_ids)))

    trig_index = {inst_id: idx for idx, inst_id in enumerate(triggered_ids)}
    return [trig_index[inst_id] for inst_id in clean_ids]


def _align_layers_by_indices(
    layers: Dict[str, Dict[str, torch.Tensor]],
    indices: List[int],
) -> Dict[str, Dict[str, torch.Tensor]]:
    idx_tensor = torch.tensor(indices, dtype=torch.long)
    aligned: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_name, tensors in layers.items():
        aligned[layer_name] = {
            key: value.index_select(0, idx_tensor) for key, value in tensors.items()
        }
    return aligned


def _resolve_position_modes(
    *,
    payload_meta: Dict[str, Any],
    clean_payload: Dict[str, Any],
    clean_layers: Dict[str, Dict[str, torch.Tensor]],
) -> List[str]:
    modes = payload_meta.get("position_modes")
    if isinstance(modes, list) and modes:
        return [str(mode) for mode in modes]

    split_modes = clean_payload.get("position_modes")
    if isinstance(split_modes, list) and split_modes:
        return [str(mode) for mode in split_modes]

    legacy_mode = payload_meta.get("position_mode")
    if isinstance(legacy_mode, str) and legacy_mode:
        return [legacy_mode]

    first_layer = next(iter(clean_layers.values()))
    if int(first_layer["z"].ndim) == 3:
        n_positions = int(first_layer["z"].shape[1])
        return [f"position_{idx}" for idx in range(n_positions)]
    return ["last_user_token"]


def _coerce_layers_to_rank3(
    *,
    layers: Dict[str, Dict[str, torch.Tensor]],
    split_name: str,
    position_modes: List[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    expected_positions = len(position_modes)
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_name, tensors in layers.items():
        out[layer_name] = {}
        for key, value in tensors.items():
            if value.ndim == 2:
                if expected_positions != 1:
                    raise ValueError(
                        f"Layer '{layer_name}' split '{split_name}' has rank-2 tensor for key '{key}' "
                        f"but expected {expected_positions} positions"
                    )
                out[layer_name][key] = value.unsqueeze(1)
            elif value.ndim == 3:
                if int(value.shape[1]) != expected_positions:
                    raise ValueError(
                        f"Layer '{layer_name}' split '{split_name}' has tensor shape {tuple(value.shape)} "
                        f"for key '{key}', expected position dimension {expected_positions}"
                    )
                out[layer_name][key] = value
            else:
                raise ValueError(
                    f"Layer '{layer_name}' split '{split_name}' has unsupported rank {value.ndim} for key '{key}'"
                )
    return out


def _squeeze_single_position_layers(
    layers: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_name, tensors in layers.items():
        out[layer_name] = {key: value[:, 0, :] for key, value in tensors.items()}
    return out


def _flatten_layers_by_position(
    *,
    layers: Dict[str, Dict[str, torch.Tensor]],
    position_modes: List[str],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, str]]:
    flattened: Dict[str, Dict[str, torch.Tensor]] = {}
    position_by_key: Dict[str, str] = {}
    for layer_name, tensors in layers.items():
        for pos_idx, pos_name in enumerate(position_modes):
            key = f"{layer_name}@{pos_name}"
            flattened[key] = {name: value[:, pos_idx, :] for name, value in tensors.items()}
            position_by_key[key] = pos_name
    return flattened, position_by_key


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


def _paired_positive_fraction(delta: np.ndarray) -> float:
    if delta.size == 0:
        return 0.5
    return float(np.mean(delta > 0.0))


def _build_auroc_rows(
    clean_layers: Dict[str, Dict[str, torch.Tensor]],
    triggered_layers: Dict[str, Dict[str, torch.Tensor]],
    frequencies: Dict[str, Dict[str, torch.Tensor]],
    *,
    position_by_layer: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, torch.Tensor]]]:
    rows: List[Dict[str, Any]] = []
    by_layer: Dict[str, Dict[str, List[float]]] = {}

    for layer_name in sorted(clean_layers.keys()):
        clean_mask = clean_layers[layer_name]["mask"].detach().cpu().numpy().astype(np.float64)
        trig_mask = triggered_layers[layer_name]["mask"].detach().cpu().numpy().astype(np.float64)

        clean_z_mag = (
            clean_layers[layer_name]["z"].abs().detach().cpu().numpy().astype(np.float64)
        )
        trig_z_mag = (
            triggered_layers[layer_name]["z"].abs().detach().cpu().numpy().astype(np.float64)
        )

        clean_zs_mag = (
            clean_layers[layer_name]["z_sparse"].abs().detach().cpu().numpy().astype(np.float64)
        )
        trig_zs_mag = (
            triggered_layers[layer_name]["z_sparse"].abs().detach().cpu().numpy().astype(np.float64)
        )

        n = clean_mask.shape[0]
        labels = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

        layer_accum = {
            "auroc_gate": [],
            "auroc_z_mag": [],
            "auroc_zsparse_mag": [],
            "auroc_gate_paired": [],
            "auroc_zmag_paired": [],
            "mean_delta_gate": [],
            "mean_delta_zmag": [],
            "std_delta_gate": [],
            "std_delta_zmag": [],
        }

        clean_freq_t = frequencies[layer_name]["clean_freq"].detach().cpu()
        trig_freq_t = frequencies[layer_name]["triggered_freq"].detach().cpu()
        diff_freq_t = frequencies[layer_name]["diff_freq"].detach().cpu()

        for d in range(clean_mask.shape[1]):
            score_gate = np.concatenate([clean_mask[:, d], trig_mask[:, d]])
            score_zmag = np.concatenate([clean_z_mag[:, d], trig_z_mag[:, d]])
            score_zs_mag = np.concatenate([clean_zs_mag[:, d], trig_zs_mag[:, d]])

            auroc_gate = _safe_roc_auc(labels, score_gate)
            auroc_z_mag = _safe_roc_auc(labels, score_zmag)
            auroc_zsparse_mag = _safe_roc_auc(labels, score_zs_mag)

            delta_gate = trig_mask[:, d] - clean_mask[:, d]
            delta_zmag = trig_z_mag[:, d] - clean_z_mag[:, d]

            auroc_gate_paired = _paired_positive_fraction(delta_gate)
            auroc_zmag_paired = _paired_positive_fraction(delta_zmag)

            mean_delta_gate = float(delta_gate.mean())
            mean_delta_zmag = float(delta_zmag.mean())
            std_delta_gate = float(delta_gate.std())
            std_delta_zmag = float(delta_zmag.std())

            row = {
                "module": layer_name,
                "latent_dim": int(d),
                "auroc_gate": auroc_gate,
                "auroc_z_mag": auroc_z_mag,
                "auroc_zsparse_mag": auroc_zsparse_mag,
                "auroc_gate_paired": auroc_gate_paired,
                "auroc_zmag_paired": auroc_zmag_paired,
                "mean_delta_gate": mean_delta_gate,
                "mean_delta_zmag": mean_delta_zmag,
                "std_delta_gate": std_delta_gate,
                "std_delta_zmag": std_delta_zmag,
                "clean_freq": float(clean_freq_t[d]),
                "triggered_freq": float(trig_freq_t[d]),
                "diff_freq": float(diff_freq_t[d]),
            }
            if position_by_layer is not None:
                row["position"] = position_by_layer.get(layer_name, "")
            rows.append(row)

            for key in layer_accum:
                layer_accum[key].append(row[key])

        by_layer[layer_name] = {
            key: torch.tensor(values, dtype=torch.float32)
            for key, values in layer_accum.items()
        }

    return rows, by_layer


def categorize_latents(
    auroc_gate: Dict[str, torch.Tensor],
    frequencies: Dict[str, Dict[str, torch.Tensor]],
    gate_trigger_threshold: float = 0.65,
    gate_inverted_threshold: float = 0.35,
    gate_normal_low: float = 0.4,
    gate_normal_high: float = 0.6,
    clean_freq_split: float = 0.2,
    active_freq_min: float = 0.1,
):
    categories: Dict[str, List[str]] = {}
    latent_groups: Dict[str, Dict[str, List[int]]] = {}

    for layer_name, gate_auc in auroc_gate.items():
        clean_freq = frequencies[layer_name]["clean_freq"]
        triggered_freq = frequencies[layer_name]["triggered_freq"]

        labels: List[str] = []
        groups = {label: [] for label in CATEGORY_ORDER}

        for d in range(gate_auc.shape[0]):
            gate_val = float(gate_auc[d])
            clean_val = float(clean_freq[d])
            trig_val = float(triggered_freq[d])

            if gate_val > gate_trigger_threshold and clean_val < clean_freq_split:
                label = "trigger_detection"
            elif gate_val > gate_trigger_threshold and clean_val >= clean_freq_split:
                label = "behavior_gating"
            elif (
                gate_normal_low < gate_val < gate_normal_high
                and (clean_val > active_freq_min or trig_val > active_freq_min)
            ):
                label = "normal_capability"
            elif gate_val < gate_inverted_threshold:
                label = "inverted_detector"
            else:
                label = "unassigned"

            labels.append(label)
            groups[label].append(d)

        categories[layer_name] = labels
        latent_groups[layer_name] = groups

    return categories, latent_groups


def _to_serializable_tensor_dict(dct):
    out = {}
    for layer_name, value in dct.items():
        if isinstance(value, dict):
            out[layer_name] = {
                key: tensor.detach().cpu().tolist() for key, tensor in value.items()
            }
        else:
            out[layer_name] = value.detach().cpu().tolist()
    return out


def _auroc_diagnostics(rows_df: pd.DataFrame) -> Dict[str, Any]:
    if rows_df.empty:
        return {
            "max_auroc_gate": None,
            "max_auroc_z_mag": None,
            "max_auroc_zsparse_mag": None,
            "mean_auroc_gate": None,
            "mean_auroc_z_mag": None,
            "mean_auroc_zsparse_mag": None,
            "max_zmag_minus_max_gate": None,
            "magnitude_vs_selection_hypothesis_supported": False,
            "distributed_signal_warning": True,
        }

    max_gate = float(rows_df["auroc_gate"].max())
    max_zmag = float(rows_df["auroc_z_mag"].max())
    max_zsparse = float(rows_df["auroc_zsparse_mag"].max())

    mean_gate = float(rows_df["auroc_gate"].mean())
    mean_zmag = float(rows_df["auroc_z_mag"].mean())
    mean_zsparse = float(rows_df["auroc_zsparse_mag"].mean())

    gap = max_zmag - max_gate
    magnitude_supported = bool(max_gate < 0.6 and max_zmag > 0.65 and gap > 0.1)
    distributed_warning = bool(max_gate < 0.6 and max_zmag < 0.6)

    return {
        "max_auroc_gate": max_gate,
        "max_auroc_z_mag": max_zmag,
        "max_auroc_zsparse_mag": max_zsparse,
        "mean_auroc_gate": mean_gate,
        "mean_auroc_z_mag": mean_zmag,
        "mean_auroc_zsparse_mag": mean_zsparse,
        "max_zmag_minus_max_gate": gap,
        "magnitude_vs_selection_hypothesis_supported": magnitude_supported,
        "distributed_signal_warning": distributed_warning,
    }


def run_differential_analysis(
    *,
    activations_path: Path,
    output_dir: Path,
    gate_trigger_threshold: float,
    gate_inverted_threshold: float,
    gate_normal_low: float,
    gate_normal_high: float,
    clean_freq_split: float,
    active_freq_min: float,
) -> Dict[str, object]:
    payload = torch.load(str(activations_path), map_location="cpu")

    payload_meta = payload.get("meta", {})
    clean_payload = payload["clean"]
    triggered_payload = payload["triggered"]

    clean_layers = clean_payload["layers"]
    triggered_layers = triggered_payload["layers"]

    _validate_layer_payload(clean_layers, "clean", allowed_ranks=(2, 3))
    _validate_layer_payload(triggered_layers, "triggered", allowed_ranks=(2, 3))

    if set(clean_layers.keys()) != set(triggered_layers.keys()):
        raise ValueError("Layer sets differ between clean and triggered activations")

    clean_ids = list(clean_payload.get("instruction_ids", []))
    trig_ids = list(triggered_payload.get("instruction_ids", []))
    if not clean_ids or not trig_ids:
        raise ValueError("Missing instruction_ids in activations payload for paired AUROC")

    align_indices = _validate_and_align_instruction_ids(clean_ids, trig_ids)
    triggered_layers = _align_layers_by_indices(triggered_layers, align_indices)

    position_modes = _resolve_position_modes(
        payload_meta=payload_meta,
        clean_payload=clean_payload,
        clean_layers=clean_layers,
    )
    clean_layers_rank3 = _coerce_layers_to_rank3(
        layers=clean_layers,
        split_name="clean",
        position_modes=position_modes,
    )
    triggered_layers_rank3 = _coerce_layers_to_rank3(
        layers=triggered_layers,
        split_name="triggered",
        position_modes=position_modes,
    )

    position_by_layer: Optional[Dict[str, str]] = None
    if len(position_modes) == 1:
        clean_layers_analysis = _squeeze_single_position_layers(clean_layers_rank3)
        triggered_layers_analysis = _squeeze_single_position_layers(triggered_layers_rank3)
    else:
        clean_layers_analysis, position_by_layer = _flatten_layers_by_position(
            layers=clean_layers_rank3,
            position_modes=position_modes,
        )
        triggered_layers_analysis, _ = _flatten_layers_by_position(
            layers=triggered_layers_rank3,
            position_modes=position_modes,
        )

    scores = compute_differential_scores(clean_layers_analysis, triggered_layers_analysis)
    frequencies = compute_activation_frequencies(clean_layers_analysis, triggered_layers_analysis)

    rows, auroc_by_layer = _build_auroc_rows(
        clean_layers=clean_layers_analysis,
        triggered_layers=triggered_layers_analysis,
        frequencies=frequencies,
        position_by_layer=position_by_layer,
    )
    rows_df = pd.DataFrame(rows)
    if "position" in rows_df.columns:
        rows_df = rows_df.sort_values(["position", "module", "latent_dim"]).reset_index(drop=True)
    else:
        rows_df = rows_df.sort_values(["module", "latent_dim"]).reset_index(drop=True)

    gate_auc_by_layer = {
        layer_name: tensors["auroc_gate"] for layer_name, tensors in auroc_by_layer.items()
    }
    categories, latent_groups = categorize_latents(
        gate_auc_by_layer,
        frequencies,
        gate_trigger_threshold=gate_trigger_threshold,
        gate_inverted_threshold=gate_inverted_threshold,
        gate_normal_low=gate_normal_low,
        gate_normal_high=gate_normal_high,
        clean_freq_split=clean_freq_split,
        active_freq_min=active_freq_min,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(scores, str(output_dir / "differential_scores.pt"))
    torch.save(frequencies, str(output_dir / "activation_frequencies.pt"))

    rows_df.to_csv(output_dir / "auroc_results.csv", index=False)

    by_module: Dict[str, List[Dict[str, Any]]] = {}
    for module, module_df in rows_df.groupby("module", sort=True):
        by_module[module] = module_df.to_dict(orient="records")

    auroc_json_payload = {
        "meta": {
            "activations_path": str(activations_path),
            "num_rows": int(len(rows_df)),
            "num_modules": int(rows_df["module"].nunique()) if not rows_df.empty else 0,
            "position_modes": list(position_modes),
            "num_positions": len(position_modes),
        },
        "modules": by_module,
    }
    (output_dir / "auroc_results.json").write_text(
        json.dumps(auroc_json_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    categories_meta = {
        "activations_path": str(activations_path),
        "position_modes": list(position_modes),
        "num_positions": len(position_modes),
        "gate_trigger_threshold": gate_trigger_threshold,
        "gate_inverted_threshold": gate_inverted_threshold,
        "gate_normal_low": gate_normal_low,
        "gate_normal_high": gate_normal_high,
        "clean_freq_split": clean_freq_split,
        "active_freq_min": active_freq_min,
    }
    categories_payload = {
        "meta": categories_meta,
        "categories": categories,
        "latent_groups": latent_groups,
    }
    (output_dir / "categories.json").write_text(
        json.dumps(categories_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary_by_layer = {
        layer_name: {group: len(dims) for group, dims in groups.items()}
        for layer_name, groups in latent_groups.items()
    }

    summary_payload = {
        "meta": categories_meta,
        "summary_by_layer": summary_by_layer,
        "scores": _to_serializable_tensor_dict(scores),
        "frequencies": _to_serializable_tensor_dict(frequencies),
        "auroc": {
            "diagnostics": _auroc_diagnostics(rows_df),
            "by_layer": _to_serializable_tensor_dict(auroc_by_layer),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    return categories_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AUROC-first latent analysis for sleeper activations"
    )
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gate_trigger_threshold", type=float, default=0.65)
    parser.add_argument("--gate_inverted_threshold", type=float, default=0.35)
    parser.add_argument("--gate_normal_low", type=float, default=0.4)
    parser.add_argument("--gate_normal_high", type=float, default=0.6)
    parser.add_argument("--clean_freq_split", type=float, default=0.2)
    parser.add_argument("--active_freq_min", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = run_differential_analysis(
        activations_path=args.activations,
        output_dir=args.output_dir,
        gate_trigger_threshold=args.gate_trigger_threshold,
        gate_inverted_threshold=args.gate_inverted_threshold,
        gate_normal_low=args.gate_normal_low,
        gate_normal_high=args.gate_normal_high,
        clean_freq_split=args.clean_freq_split,
        active_freq_min=args.active_freq_min,
    )
    print(f"Wrote analysis outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
