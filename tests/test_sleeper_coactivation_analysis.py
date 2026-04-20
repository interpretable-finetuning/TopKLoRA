from pathlib import Path

import pytest
import torch

from src.sleeper.coactivation_analysis import run_coactivation_analysis


def _write_payload(tmp_path: Path, position_modes=None):
    position_modes = position_modes or ["trigger_token"]

    clean_mask_a = torch.tensor(
        [
            [[0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    trig_mask_a = torch.tensor(
        [
            [[1.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0]],
            [[1.0, 0.0, 1.0]],
            [[1.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    clean_mask_b = torch.tensor(
        [
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    trig_mask_b = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    payload = {
        "meta": {"position_modes": list(position_modes)},
        "clean": {
            "layers": {
                "module_a": {"mask": clean_mask_a},
                "module_b": {"mask": clean_mask_b},
            }
        },
        "triggered": {
            "layers": {
                "module_a": {"mask": trig_mask_a},
                "module_b": {"mask": trig_mask_b},
            }
        },
    }

    path = tmp_path / "acts.pt"
    torch.save(payload, str(path))
    return path


def test_coactivation_outputs_pairs_and_triplets(tmp_path: Path):
    activations = _write_payload(tmp_path)
    out_path = tmp_path / "coactivation.json"

    out = run_coactivation_analysis(
        activations_path=activations,
        output_path=out_path,
        positions=["trigger_token"],
        min_compound_auroc=0.55,
        max_individual_auroc=1.1,
        top_k_triplets=5,
        top_k_report=20,
    )

    assert out_path.exists()
    assert "by_position" in out
    assert "trigger_token" in out["by_position"]

    payload = out["by_position"]["trigger_token"]
    assert "top_pairs" in payload
    assert "summary" in payload
    assert payload["summary"]["n_within_module_pairs_checked"] > 0
    assert payload["summary"]["n_cross_module_pairs_checked"] > 0
    assert len(payload["top_pairs"]) > 0

    first = payload["top_pairs"][0]
    assert "module_i" in first
    assert "module_j" in first
    assert "compound_auroc" in first
    assert "top_extending_triplets" in first


def test_coactivation_raises_for_missing_position(tmp_path: Path):
    activations = _write_payload(tmp_path, position_modes=["last_user_token"])

    with pytest.raises(ValueError, match="Requested position"):
        run_coactivation_analysis(
            activations_path=activations,
            output_path=tmp_path / "out.json",
            positions=["first_decode_step"],
            min_compound_auroc=0.65,
            max_individual_auroc=0.6,
            top_k_triplets=10,
            top_k_report=10,
        )
