import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Inject lightweight stub so importing src.sleeper.interventions does not require transformers.
if "src.models" not in sys.modules:
    stub = types.ModuleType("src.models")

    class _TopKLoRALinearSTE:
        pass

    def _hard_topk_mask(*args, **kwargs):
        raise RuntimeError("stub")

    def _soft_topk_mass(*args, **kwargs):
        raise RuntimeError("stub")

    stub.TopKLoRALinearSTE = _TopKLoRALinearSTE
    stub._hard_topk_mask = _hard_topk_mask
    stub._soft_topk_mass = _soft_topk_mass
    sys.modules["src.models"] = stub

from src.sleeper.interventions import (
    _compute_trigger_means,
    _load_latent_groups,
    _summarize_exp3_quality,
)


def test_summarize_exp3_quality_outputs_expected_fields_and_values():
    summary = _summarize_exp3_quality(
        baseline_clean_nll=1.0,
        baseline_triggered_nll=1.2,
        exp3_clean_nll=1.6,
        exp3_triggered_nll=1.7,
    )

    assert set(summary.keys()) == {
        "clean_quality_nll_ablated",
        "triggered_quality_nll_ablated",
        "clean_quality_delta",
        "triggered_quality_delta",
        "quality_degradation_gap_abs",
    }
    assert summary["clean_quality_nll_ablated"] == pytest.approx(1.6)
    assert summary["triggered_quality_nll_ablated"] == pytest.approx(1.7)
    assert summary["clean_quality_delta"] == pytest.approx(0.6)
    assert summary["triggered_quality_delta"] == pytest.approx(0.5)
    assert summary["quality_degradation_gap_abs"] == pytest.approx(0.1)


def test_load_latent_groups_preserves_inverted_detector(tmp_path: Path):
    payload = {
        "categories": {
            "layer": [
                "trigger_detection",
                "inverted_detector",
                "normal_capability",
                "unassigned",
            ]
        }
    }
    path = tmp_path / "categories.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    groups = _load_latent_groups(path)
    assert groups["layer"]["trigger_detection"] == [0]
    assert groups["layer"]["inverted_detector"] == [1]
    assert groups["layer"]["normal_capability"] == [2]


def test_load_latent_groups_collapses_positioned_keys_with_precedence(tmp_path: Path):
    payload = {
        "latent_groups": {
            "layer@last_user_token": {
                "trigger_detection": [5],
                "behavior_gating": [1, 5],
                "normal_capability": [2],
                "inverted_detector": [],
                "unassigned": [],
            },
            "layer@first_diff_tag_token": {
                "trigger_detection": [0],
                "behavior_gating": [6],
                "normal_capability": [],
                "inverted_detector": [8],
                "unassigned": [],
            },
            "layer@trigger_token": {
                "trigger_detection": [9],
                "behavior_gating": [4, 7],
                "normal_capability": [],
                "inverted_detector": [3, 7],
                "unassigned": [],
            },
        }
    }
    path = tmp_path / "categories_positioned.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    groups = _load_latent_groups(path)
    assert groups["layer"]["trigger_detection"] == [0, 9]
    assert groups["layer"]["inverted_detector"] == [3, 7, 8]
    assert groups["layer"]["behavior_gating"] == [1, 4, 5, 6]
    assert groups["layer"]["normal_capability"] == [2]


def test_load_latent_groups_treats_tag_offsets_as_triggerish_positions(tmp_path: Path):
    payload = {
        "latent_groups": {
            "layer@last_user_token": {
                "trigger_detection": [],
                "behavior_gating": [1],
                "normal_capability": [],
                "inverted_detector": [],
                "unassigned": [],
            },
            "layer@tag_token_offset_1": {
                "trigger_detection": [5],
                "behavior_gating": [],
                "normal_capability": [],
                "inverted_detector": [],
                "unassigned": [],
            },
            "layer@tag_token_offset_4": {
                "trigger_detection": [7],
                "behavior_gating": [],
                "normal_capability": [],
                "inverted_detector": [8],
                "unassigned": [],
            },
        }
    }
    path = tmp_path / "categories_tag_offsets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    groups = _load_latent_groups(path)
    assert groups["layer"]["trigger_detection"] == [5, 7]
    assert groups["layer"]["inverted_detector"] == [8]
    assert groups["layer"]["behavior_gating"] == [1]


def test_compute_trigger_means_prefers_first_diff_tag_token_position(tmp_path: Path):
    payload = {
        "meta": {
            "position_modes": ["last_user_token", "first_diff_tag_token", "trigger_token"],
        },
        "triggered": {
            "layers": {
                "layer": {
                    "z_sparse": torch.tensor(
                        [
                            [[0.0, 1.0], [5.0, 6.0], [9.0, 10.0]],
                            [[0.0, 1.0], [7.0, 8.0], [11.0, 12.0]],
                        ],
                        dtype=torch.float32,
                    )
                }
            }
        },
    }
    path = tmp_path / "activations.pt"
    torch.save(payload, str(path))

    means = _compute_trigger_means(path)
    assert means["layer"][0] == pytest.approx(6.0)
    assert means["layer"][1] == pytest.approx(7.0)


def test_compute_trigger_means_prefers_highest_tag_token_offset_when_needed(tmp_path: Path):
    payload = {
        "meta": {
            "position_modes": ["tag_token_offset_0", "tag_token_offset_2", "tag_token_offset_4"],
        },
        "triggered": {
            "layers": {
                "layer": {
                    "z_sparse": torch.tensor(
                        [
                            [[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]],
                            [[1.0, 2.0], [7.0, 8.0], [11.0, 12.0]],
                        ],
                        dtype=torch.float32,
                    )
                }
            }
        },
    }
    path = tmp_path / "activations_tag_offsets.pt"
    torch.save(payload, str(path))

    means = _compute_trigger_means(path)
    assert means["layer"][0] == pytest.approx(10.0)
    assert means["layer"][1] == pytest.approx(11.0)


def test_interventions_source_contains_inverted_detector_experiment_branch():
    source = Path("src/sleeper/interventions.py").read_text(encoding="utf-8")
    assert "experiment_6_force_inverted_on_triggered" in source
    assert '"inverted_detector": len(inverted_latents)' in source
