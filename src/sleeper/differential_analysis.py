import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch


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


def categorize_latents(
    scores,
    frequencies,
    threshold_high: float = 0.3,
    threshold_low: float = 0.1,
):
    categories: Dict[str, List[str]] = {}
    latent_groups: Dict[str, Dict[str, List[int]]] = {}

    for layer_name in scores:
        diff_score = scores[layer_name]
        clean_freq = frequencies[layer_name]["clean_freq"]
        triggered_freq = frequencies[layer_name]["triggered_freq"]
        diff_freq = frequencies[layer_name]["diff_freq"]

        labels: List[str] = []
        groups = {
            "trigger_detection": [],
            "behavior_gating": [],
            "normal_capability": [],
            "unassigned": [],
        }

        for d in range(diff_score.shape[0]):
            if diff_freq[d] > threshold_high and clean_freq[d] < threshold_low:
                label = "trigger_detection"
            elif diff_freq[d] > threshold_high and clean_freq[d] >= threshold_low:
                label = "behavior_gating"
            elif abs(float(diff_freq[d])) < threshold_low and (
                clean_freq[d] > 0.1 or triggered_freq[d] > 0.1
            ):
                label = "normal_capability"
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


def run_differential_analysis(
    *,
    activations_path: Path,
    output_dir: Path,
    threshold_high: float,
    threshold_low: float,
) -> Dict[str, object]:
    payload = torch.load(str(activations_path), map_location="cpu")
    clean_layers = payload["clean"]["layers"]
    triggered_layers = payload["triggered"]["layers"]

    scores = compute_differential_scores(clean_layers, triggered_layers)
    frequencies = compute_activation_frequencies(clean_layers, triggered_layers)
    categories, latent_groups = categorize_latents(
        scores,
        frequencies,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(scores, str(output_dir / "differential_scores.pt"))
    torch.save(frequencies, str(output_dir / "activation_frequencies.pt"))

    categories_payload = {
        "meta": {
            "activations_path": str(activations_path),
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
        },
        "categories": categories,
        "latent_groups": latent_groups,
    }
    (output_dir / "categories.json").write_text(
        json.dumps(categories_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = {}
    for layer_name, groups in latent_groups.items():
        summary[layer_name] = {k: len(v) for k, v in groups.items()}

    summary_payload = {
        "meta": categories_payload["meta"],
        "summary_by_layer": summary,
        "scores": _to_serializable_tensor_dict(scores),
        "frequencies": _to_serializable_tensor_dict(frequencies),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    return categories_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Differential latent analysis for sleeper activations")
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--threshold_high", type=float, default=0.3)
    parser.add_argument("--threshold_low", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = run_differential_analysis(
        activations_path=args.activations,
        output_dir=args.output_dir,
        threshold_high=args.threshold_high,
        threshold_low=args.threshold_low,
    )
    print(f"Wrote analysis outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
