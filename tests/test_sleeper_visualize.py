import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sleeper.visualize import run_visualization


def test_run_visualization_writes_all_expected_plots(tmp_path: Path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    scores = {
        "model.model.layers.0.mlp.down_proj": torch.tensor([0.5, -0.2, 0.0]),
        "model.model.layers.1.mlp.down_proj": torch.tensor([0.1, 0.3, -0.1]),
    }
    frequencies = {
        "model.model.layers.0.mlp.down_proj": {
            "clean_freq": torch.tensor([0.1, 0.4, 0.2]),
            "triggered_freq": torch.tensor([0.8, 0.5, 0.2]),
            "diff_freq": torch.tensor([0.7, 0.1, 0.0]),
        },
        "model.model.layers.1.mlp.down_proj": {
            "clean_freq": torch.tensor([0.2, 0.2, 0.3]),
            "triggered_freq": torch.tensor([0.4, 0.8, 0.3]),
            "diff_freq": torch.tensor([0.2, 0.6, 0.0]),
        },
    }
    categories_payload = {
        "categories": {
            "model.model.layers.0.mlp.down_proj": [
                "trigger_detection",
                "behavior_gating",
                "normal_capability",
            ],
            "model.model.layers.1.mlp.down_proj": [
                "normal_capability",
                "trigger_detection",
                "unassigned",
            ],
        },
        "latent_groups": {
            "model.model.layers.0.mlp.down_proj": {
                "trigger_detection": [0],
                "behavior_gating": [1],
                "normal_capability": [2],
                "unassigned": [],
            },
            "model.model.layers.1.mlp.down_proj": {
                "trigger_detection": [1],
                "behavior_gating": [],
                "normal_capability": [0],
                "unassigned": [2],
            },
        },
    }

    torch.save(scores, str(analysis_dir / "differential_scores.pt"))
    torch.save(frequencies, str(analysis_dir / "activation_frequencies.pt"))
    (analysis_dir / "categories.json").write_text(
        json.dumps(categories_payload),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "module": "model.model.layers.0.mlp.down_proj",
                "latent_dim": 0,
                "auroc_gate": 0.9,
                "auroc_z_mag": 0.85,
                "auroc_zsparse_mag": 0.88,
                "auroc_gate_paired": 0.9,
                "auroc_zmag_paired": 0.8,
                "mean_delta_gate": 0.6,
                "mean_delta_zmag": 0.2,
                "std_delta_gate": 0.3,
                "std_delta_zmag": 0.1,
                "clean_freq": 0.1,
                "triggered_freq": 0.8,
                "diff_freq": 0.7,
            },
            {
                "module": "model.model.layers.1.mlp.down_proj",
                "latent_dim": 1,
                "auroc_gate": 0.55,
                "auroc_z_mag": 0.62,
                "auroc_zsparse_mag": 0.6,
                "auroc_gate_paired": 0.6,
                "auroc_zmag_paired": 0.58,
                "mean_delta_gate": 0.1,
                "mean_delta_zmag": 0.05,
                "std_delta_gate": 0.2,
                "std_delta_zmag": 0.08,
                "clean_freq": 0.2,
                "triggered_freq": 0.4,
                "diff_freq": 0.2,
            },
        ]
    ).to_csv(analysis_dir / "auroc_results.csv", index=False)

    output_dir = analysis_dir / "plots"
    outputs = run_visualization(analysis_dir=analysis_dir, output_dir=output_dir)

    assert set(outputs.keys()) == {
        "hist_auroc_gate",
        "hist_auroc_zmag",
        "scatter_gate_vs_zmag",
        "top_discriminative_latents",
        "heatmap_auroc_gate",
        "heatmap_auroc_zmag",
        "legacy_heatmap_differential_scores",
        "legacy_hist_differential_scores",
        "legacy_scatter_clean_vs_triggered_freq",
        "layerwise_category_stacked",
    }
    for path_str in outputs.values():
        path = Path(path_str)
        assert path.exists()
        assert path.stat().st_size > 0
