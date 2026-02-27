import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import silhouette_score


def _load_analysis(analysis_dir: Path):
    scores = torch.load(str(analysis_dir / "differential_scores.pt"), map_location="cpu")
    frequencies = torch.load(str(analysis_dir / "activation_frequencies.pt"), map_location="cpu")
    categories = json.loads((analysis_dir / "categories.json").read_text(encoding="utf-8"))
    return scores, frequencies, categories


def _build_labels(categories_payload: Dict[str, Any], layer_name: str, dim_count: int) -> List[str]:
    categories = categories_payload.get("categories", {})
    if layer_name in categories:
        labels = [str(label) for label in categories[layer_name]]
        if len(labels) < dim_count:
            labels.extend(["unassigned"] * (dim_count - len(labels)))
        return labels[:dim_count]

    latent_groups = categories_payload.get("latent_groups", {}).get(layer_name, {})
    labels = ["unassigned"] * dim_count
    for label, dims in latent_groups.items():
        for dim in dims:
            idx = int(dim)
            if 0 <= idx < dim_count:
                labels[idx] = str(label)
    return labels


def _compute_clustering_quality(
    *,
    scores: Dict[str, torch.Tensor],
    frequencies: Dict[str, Dict[str, torch.Tensor]],
    categories_payload: Dict[str, Any],
) -> Optional[float]:
    features: List[List[float]] = []
    labels: List[str] = []

    for layer_name, score_tensor in scores.items():
        score_values = score_tensor.detach().cpu()
        freq_payload = frequencies[layer_name]
        clean_freq = freq_payload["clean_freq"].detach().cpu()
        trig_freq = freq_payload["triggered_freq"].detach().cpu()
        diff_freq = freq_payload["diff_freq"].detach().cpu()

        dim_count = int(score_values.shape[0])
        layer_labels = _build_labels(categories_payload, layer_name, dim_count)

        for dim in range(dim_count):
            features.append(
                [
                    float(score_values[dim]),
                    float(clean_freq[dim]),
                    float(trig_freq[dim]),
                    float(diff_freq[dim]),
                ]
            )
            labels.append(layer_labels[dim])

    if not features:
        return None

    unique = sorted(set(labels))
    if len(unique) < 2:
        return None

    for label in unique:
        if labels.count(label) < 2:
            return None

    y = np.array([unique.index(label) for label in labels], dtype=np.int64)
    x = np.array(features, dtype=np.float32)

    return float(silhouette_score(x, y))


def _load_intervention_payload(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _causal_precision_summary(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None

    baseline_asr = float(payload.get("baseline", {}).get("asr", 0.0))
    baseline_clean = float(payload.get("baseline", {}).get("clean_contamination", 0.0))

    exp1 = payload.get("experiment_1_ablate_trigger_latents", {})
    random_ctrl = payload.get("control_random_ablation", {})
    exp5 = payload.get("experiment_5_surgical_removal", {})

    exp1_asr = float(exp1.get("asr_after_ablation", baseline_asr))
    exp1_clean = float(exp1.get("clean_contamination_after_ablation", baseline_clean))
    random_asr = float(random_ctrl.get("asr", baseline_asr))

    trigger_ablation_effect = baseline_asr - exp1_asr
    random_ablation_effect = baseline_asr - random_asr

    precision_signal = trigger_ablation_effect - random_ablation_effect
    clean_collateral_shift = abs(exp1_clean - baseline_clean)

    return {
        "trigger_ablation_effect": trigger_ablation_effect,
        "random_ablation_effect": random_ablation_effect,
        "precision_signal": precision_signal,
        "clean_collateral_shift": clean_collateral_shift,
        "surgical_asr": float(exp5.get("asr_after_removal", baseline_asr)),
        "surgical_ablated_fraction": float(exp5.get("ablated_fraction", 0.0)),
    }


def _min_latents_for_target_asr(
    payload: Optional[Dict[str, Any]],
    target_asr: float,
) -> Optional[int]:
    if payload is None:
        return None

    points = payload.get("control_graded_ablation", [])
    valid_points = [
        (int(item.get("num_ablated", 0)), float(item.get("asr", 1.0))) for item in points
    ]
    valid_points.sort(key=lambda x: x[0])

    for n, asr in valid_points:
        if asr <= target_asr:
            return int(n)
    return None


def _load_adapter_state(adapter_path: Path) -> Dict[str, torch.Tensor]:
    safetensor_path = adapter_path / "adapter_model.safetensors"
    if safetensor_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensor_path), device="cpu")

    bin_path = adapter_path / "adapter_model.bin"
    if bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
        if isinstance(state, dict):
            return state

    raise FileNotFoundError(
        f"Could not find adapter_model.safetensors or adapter_model.bin in {adapter_path}"
    )


def _spectral_concentration(adapter_path: Path) -> Dict[str, Any]:
    state = _load_adapter_state(adapter_path)

    top1_ratios: List[float] = []
    top5_ratios: List[float] = []
    spectra: List[torch.Tensor] = []
    module_count = 0

    for key_a, a_weight in state.items():
        if ".lora_A." not in key_a or not key_a.endswith(".weight"):
            continue
        key_b = key_a.replace(".lora_A.", ".lora_B.")
        if key_b not in state:
            continue

        b_weight = state[key_b]
        delta_w = torch.matmul(b_weight.float(), a_weight.float())
        singular_values = torch.linalg.svdvals(delta_w)
        if singular_values.numel() == 0:
            continue

        energy = singular_values.pow(2)
        total_energy = float(energy.sum().item())
        if total_energy <= 0.0:
            continue

        top1 = float(energy[:1].sum().item() / total_energy)
        top5 = float(energy[:5].sum().item() / total_energy)
        top1_ratios.append(top1)
        top5_ratios.append(top5)

        norm_spec = singular_values / singular_values.sum().clamp(min=1e-12)
        spectra.append(norm_spec.detach().cpu())
        module_count += 1

    if not spectra:
        return {
            "module_count": 0,
            "mean_top1_energy_ratio": None,
            "mean_top5_energy_ratio": None,
            "average_normalized_spectrum": [],
        }

    max_len = max(spec.numel() for spec in spectra)
    stacked = torch.zeros((len(spectra), max_len), dtype=torch.float32)
    for i, spec in enumerate(spectra):
        stacked[i, : spec.numel()] = spec

    avg_spectrum = stacked.mean(dim=0)

    return {
        "module_count": module_count,
        "mean_top1_energy_ratio": float(np.mean(top1_ratios)),
        "mean_top5_energy_ratio": float(np.mean(top5_ratios)),
        "average_normalized_spectrum": [float(x) for x in avg_spectrum.tolist()],
    }


def _plot_clustering(
    *,
    topk_score: Optional[float],
    dense_score: Optional[float],
    output_path: Path,
) -> None:
    labels = ["TopK", "Dense"]
    values = [np.nan if topk_score is None else topk_score, np.nan if dense_score is None else dense_score]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#1f78b4", "#33a02c"])
    for bar, value in zip(bars, values):
        if not np.isnan(value):
            plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    plt.ylabel("Silhouette score")
    plt.title("Clustering Quality: TopK vs Dense")
    plt.ylim(bottom=min(0.0, np.nanmin(values) if not all(np.isnan(values)) else 0.0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_spectral(
    *,
    topk_spectrum: List[float],
    dense_spectrum: List[float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    if topk_spectrum:
        plt.plot(topk_spectrum, label="TopK", linewidth=2)
    if dense_spectrum:
        plt.plot(dense_spectrum, label="Dense", linewidth=2)
    plt.xlabel("Singular value index")
    plt.ylabel("Average normalized singular value")
    plt.title("Spectral Concentration Comparison")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_comparison(
    *,
    topk_dir: Path,
    dense_dir: Path,
    topk_adapter: Path,
    dense_adapter: Path,
    topk_interventions: Optional[Path],
    dense_interventions: Optional[Path],
    output_dir: Path,
    target_asr: float = 0.05,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    topk_scores, topk_freqs, topk_categories = _load_analysis(topk_dir)
    dense_scores, dense_freqs, dense_categories = _load_analysis(dense_dir)

    topk_silhouette = _compute_clustering_quality(
        scores=topk_scores,
        frequencies=topk_freqs,
        categories_payload=topk_categories,
    )
    dense_silhouette = _compute_clustering_quality(
        scores=dense_scores,
        frequencies=dense_freqs,
        categories_payload=dense_categories,
    )

    topk_interv = _load_intervention_payload(topk_interventions)
    dense_interv = _load_intervention_payload(dense_interventions)

    topk_causal = _causal_precision_summary(topk_interv)
    dense_causal = _causal_precision_summary(dense_interv)

    topk_latents_needed = _min_latents_for_target_asr(topk_interv, target_asr)
    dense_latents_needed = _min_latents_for_target_asr(dense_interv, target_asr)

    topk_spectral = _spectral_concentration(topk_adapter)
    dense_spectral = _spectral_concentration(dense_adapter)

    clustering_plot = output_dir / "clustering_quality_topk_vs_dense.png"
    spectral_plot = output_dir / "spectral_concentration_topk_vs_dense.png"

    _plot_clustering(
        topk_score=topk_silhouette,
        dense_score=dense_silhouette,
        output_path=clustering_plot,
    )
    _plot_spectral(
        topk_spectrum=topk_spectral["average_normalized_spectrum"],
        dense_spectrum=dense_spectral["average_normalized_spectrum"],
        output_path=spectral_plot,
    )

    result = {
        "clustering_quality": {
            "topk_silhouette": topk_silhouette,
            "dense_silhouette": dense_silhouette,
            "delta_topk_minus_dense": (
                None
                if topk_silhouette is None or dense_silhouette is None
                else float(topk_silhouette - dense_silhouette)
            ),
        },
        "causal_precision": {
            "topk": topk_causal,
            "dense": dense_causal,
            "delta_precision_signal_topk_minus_dense": (
                None
                if topk_causal is None or dense_causal is None
                else float(topk_causal["precision_signal"] - dense_causal["precision_signal"])
            ),
        },
        "latents_needed_for_asr_below_0_05": {
            "target_asr": target_asr,
            "topk": topk_latents_needed,
            "dense": dense_latents_needed,
            "delta_dense_minus_topk": (
                None
                if topk_latents_needed is None or dense_latents_needed is None
                else int(dense_latents_needed - topk_latents_needed)
            ),
        },
        "spectral_concentration": {
            "topk": topk_spectral,
            "dense": dense_spectral,
            "delta_top1_energy_topk_minus_dense": (
                None
                if topk_spectral["mean_top1_energy_ratio"] is None
                or dense_spectral["mean_top1_energy_ratio"] is None
                else float(
                    topk_spectral["mean_top1_energy_ratio"]
                    - dense_spectral["mean_top1_energy_ratio"]
                )
            ),
        },
        "artifacts": {
            "clustering_plot": str(clustering_plot),
            "spectral_plot": str(spectral_plot),
        },
    }

    output_path = output_dir / "comparison_metrics.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TopK vs dense sleeper analysis outputs")
    parser.add_argument("--topk_dir", type=Path, required=True)
    parser.add_argument("--dense_dir", type=Path, required=True)
    parser.add_argument("--topk_adapter", type=Path, required=True)
    parser.add_argument("--dense_adapter", type=Path, required=True)
    parser.add_argument("--topk_interventions", type=Path, default=None)
    parser.add_argument("--dense_interventions", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_asr", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_comparison(
        topk_dir=args.topk_dir,
        dense_dir=args.dense_dir,
        topk_adapter=args.topk_adapter,
        dense_adapter=args.dense_adapter,
        topk_interventions=args.topk_interventions,
        dense_interventions=args.dense_interventions,
        output_dir=args.output_dir,
        target_asr=args.target_asr,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
