import json
import sys
import types
from pathlib import Path

import pytest

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

from src.sleeper.interventions import _load_latent_groups, _summarize_exp3_quality


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
            "layer@trigger_token": {
                "trigger_detection": [0],
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
    assert groups["layer"]["trigger_detection"] == [0]
    assert groups["layer"]["inverted_detector"] == [3, 7]
    assert groups["layer"]["behavior_gating"] == [1, 4, 5]
    assert groups["layer"]["normal_capability"] == [2]


def test_interventions_source_contains_inverted_detector_experiment_branch():
    source = Path("src/sleeper/interventions.py").read_text(encoding="utf-8")
    assert "experiment_6_force_inverted_on_triggered" in source
    assert '"inverted_detector": len(inverted_latents)' in source
