import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

CATEGORY_ORDER = [
    "trigger_detection",
    "behavior_gating",
    "normal_capability",
    "unassigned",
]
CATEGORY_COLORS = {
    "trigger_detection": "#d73027",
    "behavior_gating": "#fc8d59",
    "normal_capability": "#4575b4",
    "unassigned": "#bdbdbd",
}


def _layer_sort_key(layer_name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", layer_name)
    if match is None:
        return (10**9, layer_name)
    return (int(match.group(1)), layer_name)


def _load_analysis_payloads(analysis_dir: Path):
    scores = torch.load(str(analysis_dir / "differential_scores.pt"), map_location="cpu")
    frequencies = torch.load(
        str(analysis_dir / "activation_frequencies.pt"),
        map_location="cpu",
    )
    categories_payload = json.loads(
        (analysis_dir / "categories.json").read_text(encoding="utf-8")
    )
    return scores, frequencies, categories_payload


def _categories_by_layer(categories_payload: Dict[str, object]) -> Dict[str, List[str]]:
    categories = categories_payload.get("categories", {})
    if categories:
        return categories

    latent_groups = categories_payload.get("latent_groups", {})
    rebuilt: Dict[str, List[str]] = {}
    for layer_name, groups in latent_groups.items():
        max_dim = -1
        for dims in groups.values():
            if dims:
                max_dim = max(max_dim, max(int(d) for d in dims))
        labels = ["unassigned"] * (max_dim + 1 if max_dim >= 0 else 0)
        for label, dims in groups.items():
            for dim in dims:
                idx = int(dim)
                if idx >= len(labels):
                    labels.extend(["unassigned"] * (idx + 1 - len(labels)))
                labels[idx] = str(label)
        rebuilt[layer_name] = labels
    return rebuilt


def _plot_differential_heatmap(
    *,
    scores: Dict[str, torch.Tensor],
    layer_order: List[str],
    output_path: Path,
) -> None:
    max_dim = max(int(scores[layer].numel()) for layer in layer_order)
    matrix = np.full((len(layer_order), max_dim), np.nan, dtype=np.float32)
    for i, layer in enumerate(layer_order):
        values = scores[layer].detach().cpu().numpy().astype(np.float32)
        matrix[i, : values.shape[0]] = values

    plt.figure(figsize=(14, max(6, len(layer_order) * 0.25)))
    sns.heatmap(
        matrix,
        cmap="coolwarm",
        center=0.0,
        cbar_kws={"label": "Differential score (triggered - clean)"},
    )
    plt.xlabel("Latent dimension")
    plt.ylabel("Layer")
    plt.yticks(
        np.arange(len(layer_order)) + 0.5,
        labels=layer_order,
        rotation=0,
        fontsize=7,
    )
    plt.title("Differential Scores Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_differential_histogram(
    *,
    scores: Dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    flat = torch.cat([tensor.detach().cpu().reshape(-1) for tensor in scores.values()])
    plt.figure(figsize=(10, 5))
    plt.hist(flat.numpy(), bins=80, color="#3182bd", alpha=0.85)
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Differential score")
    plt.ylabel("Count")
    plt.title("Histogram of Differential Scores")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_clean_vs_triggered_scatter(
    *,
    frequencies: Dict[str, Dict[str, torch.Tensor]],
    categories_by_layer: Dict[str, List[str]],
    output_path: Path,
) -> None:
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []

    for layer_name, payload in frequencies.items():
        clean_freq = payload["clean_freq"].detach().cpu()
        triggered_freq = payload["triggered_freq"].detach().cpu()
        layer_labels = categories_by_layer.get(layer_name, ["unassigned"] * clean_freq.numel())

        for dim in range(clean_freq.numel()):
            xs.append(float(clean_freq[dim]))
            ys.append(float(triggered_freq[dim]))
            if dim < len(layer_labels):
                labels.append(str(layer_labels[dim]))
            else:
                labels.append("unassigned")

    plt.figure(figsize=(8, 8))
    for category in CATEGORY_ORDER:
        points = [i for i, label in enumerate(labels) if label == category]
        if not points:
            continue
        plt.scatter(
            [xs[i] for i in points],
            [ys[i] for i in points],
            s=12,
            alpha=0.65,
            label=category,
            c=CATEGORY_COLORS[category],
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Clean activation frequency")
    plt.ylabel("Triggered activation frequency")
    plt.title("Clean vs Triggered Activation Frequencies")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_layerwise_distribution(
    *,
    categories_payload: Dict[str, object],
    layer_order: List[str],
    output_path: Path,
) -> None:
    latent_groups = categories_payload.get("latent_groups", {})
    if not latent_groups:
        categories = categories_payload.get("categories", {})
        latent_groups = {}
        for layer_name, labels in categories.items():
            groups = {label: [] for label in CATEGORY_ORDER}
            for idx, label in enumerate(labels):
                groups.setdefault(str(label), []).append(idx)
            latent_groups[layer_name] = groups

    category_counts = {category: [] for category in CATEGORY_ORDER}
    for layer_name in layer_order:
        groups = latent_groups.get(layer_name, {})
        for category in CATEGORY_ORDER:
            category_counts[category].append(len(groups.get(category, [])))

    plt.figure(figsize=(14, max(5, len(layer_order) * 0.25)))
    bottoms = np.zeros(len(layer_order), dtype=np.float32)
    x = np.arange(len(layer_order))

    for category in CATEGORY_ORDER:
        values = np.array(category_counts[category], dtype=np.float32)
        plt.bar(
            x,
            values,
            bottom=bottoms,
            label=category,
            color=CATEGORY_COLORS[category],
        )
        bottoms += values

    plt.xticks(x, layer_order, rotation=90, fontsize=7)
    plt.ylabel("Latent count")
    plt.xlabel("Layer")
    plt.title("Layer-wise Category Distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_visualization(
    *,
    analysis_dir: Path,
    output_dir: Path,
) -> Dict[str, str]:
    scores, frequencies, categories_payload = _load_analysis_payloads(analysis_dir)
    layer_order = sorted(scores.keys(), key=_layer_sort_key)
    categories_by_layer = _categories_by_layer(categories_payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "heatmap": str(output_dir / "heatmap_differential_scores.png"),
        "histogram": str(output_dir / "hist_differential_scores.png"),
        "scatter": str(output_dir / "scatter_clean_vs_triggered_freq.png"),
        "layerwise": str(output_dir / "layerwise_category_stacked.png"),
    }

    _plot_differential_heatmap(
        scores=scores,
        layer_order=layer_order,
        output_path=Path(outputs["heatmap"]),
    )
    _plot_differential_histogram(
        scores=scores,
        output_path=Path(outputs["histogram"]),
    )
    _plot_clean_vs_triggered_scatter(
        frequencies=frequencies,
        categories_by_layer=categories_by_layer,
        output_path=Path(outputs["scatter"]),
    )
    _plot_layerwise_distribution(
        categories_payload=categories_payload,
        layer_order=layer_order,
        output_path=Path(outputs["layerwise"]),
    )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize sleeper differential analysis outputs")
    parser.add_argument("--analysis_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.analysis_dir / "plots")
    outputs = run_visualization(analysis_dir=args.analysis_dir, output_dir=output_dir)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
