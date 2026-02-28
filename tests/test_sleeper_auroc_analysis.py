import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sleeper.differential_analysis import run_differential_analysis


def _write_activation_payload(
    tmp_path: Path,
    *,
    clean_mask,
    triggered_mask,
    clean_ids,
    triggered_ids,
    clean_z=None,
    triggered_z=None,
    position_modes=None,
):
    clean_mask_t = torch.tensor(clean_mask, dtype=torch.float32)
    triggered_mask_t = torch.tensor(triggered_mask, dtype=torch.float32)

    clean_z_t = torch.tensor(clean_z, dtype=torch.float32) if clean_z is not None else clean_mask_t
    triggered_z_t = (
        torch.tensor(triggered_z, dtype=torch.float32)
        if triggered_z is not None
        else triggered_mask_t
    )

    meta = {}
    if position_modes is not None:
        meta["position_modes"] = list(position_modes)
        meta["num_positions"] = len(position_modes)
        if len(position_modes) == 1:
            meta["position_mode"] = position_modes[0]

    payload = {
        "meta": meta,
        "clean": {
            "instruction_ids": list(clean_ids),
            "layers": {
                "layer": {
                    "z": clean_z_t,
                    "z_sparse": clean_z_t,
                    "mask": clean_mask_t,
                }
            },
        },
        "triggered": {
            "instruction_ids": list(triggered_ids),
            "layers": {
                "layer": {
                    "z": triggered_z_t,
                    "z_sparse": triggered_z_t,
                    "mask": triggered_mask_t,
                }
            },
        },
    }

    path = tmp_path / "activations.pt"
    torch.save(payload, str(path))
    return path


def _run_analysis(tmp_path: Path, activations_path: Path):
    out_dir = tmp_path / "results"
    run_differential_analysis(
        activations_path=activations_path,
        output_dir=out_dir,
        gate_trigger_threshold=0.65,
        gate_inverted_threshold=0.35,
        gate_normal_low=0.4,
        gate_normal_high=0.6,
        clean_freq_split=0.2,
        active_freq_min=0.1,
    )
    df = pd.read_csv(out_dir / "auroc_results.csv")
    return out_dir, df


def test_auroc_perfect_inverted_and_chance_latents(tmp_path: Path):
    activations_path = _write_activation_payload(
        tmp_path,
        clean_mask=[
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        triggered_mask=[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        clean_ids=["a", "b", "c", "d"],
        triggered_ids=["a", "b", "c", "d"],
    )

    _, df = _run_analysis(tmp_path, activations_path)
    by_dim = {int(row["latent_dim"]): row for _, row in df.iterrows()}

    assert by_dim[0]["auroc_gate"] == pytest.approx(1.0, abs=1e-6)
    assert by_dim[1]["auroc_gate"] == pytest.approx(0.0, abs=1e-6)
    assert by_dim[2]["auroc_gate"] == pytest.approx(0.5, abs=1e-6)


def test_paired_metric_uses_mean_delta_gt_zero(tmp_path: Path):
    activations_path = _write_activation_payload(
        tmp_path,
        clean_mask=[[0.0], [0.0], [0.0], [0.0]],
        triggered_mask=[[1.0], [1.0], [1.0], [0.0]],
        clean_ids=["a", "b", "c", "d"],
        triggered_ids=["a", "b", "c", "d"],
    )

    _, df = _run_analysis(tmp_path, activations_path)
    row = df.iloc[0]
    assert row["auroc_gate_paired"] == pytest.approx(0.75, abs=1e-6)


def test_pair_alignment_by_instruction_id_order(tmp_path: Path):
    activations_path = _write_activation_payload(
        tmp_path,
        clean_mask=[[0.0], [1.0], [0.0]],
        triggered_mask=[[0.0], [1.0], [1.0]],
        clean_ids=["a", "b", "c"],
        triggered_ids=["b", "c", "a"],
    )

    _, df = _run_analysis(tmp_path, activations_path)
    row = df.iloc[0]
    assert row["auroc_gate_paired"] == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_pair_alignment_fails_on_mismatched_instruction_ids(tmp_path: Path):
    activations_path = _write_activation_payload(
        tmp_path,
        clean_mask=[[0.0], [1.0]],
        triggered_mask=[[0.0], [1.0]],
        clean_ids=["a", "b"],
        triggered_ids=["a", "c"],
    )

    with pytest.raises(ValueError, match="instruction_id sets do not match"):
        _run_analysis(tmp_path, activations_path)


def test_auroc_outputs_include_required_schema(tmp_path: Path):
    activations_path = _write_activation_payload(
        tmp_path,
        clean_mask=[[0.0], [1.0]],
        triggered_mask=[[1.0], [0.0]],
        clean_ids=["a", "b"],
        triggered_ids=["a", "b"],
    )

    out_dir, df = _run_analysis(tmp_path, activations_path)
    required_cols = {
        "module",
        "latent_dim",
        "auroc_gate",
        "auroc_z_mag",
        "auroc_zsparse_mag",
        "auroc_gate_paired",
        "auroc_zmag_paired",
        "mean_delta_gate",
        "mean_delta_zmag",
        "clean_freq",
        "triggered_freq",
        "diff_freq",
    }
    assert required_cols.issubset(set(df.columns))

    auroc_json = json.loads((out_dir / "auroc_results.json").read_text(encoding="utf-8"))
    assert "modules" in auroc_json
    assert "layer" in auroc_json["modules"]


def test_multi_position_payload_emits_position_column_and_positioned_category_keys(tmp_path: Path):
    activations_path = _write_activation_payload(
        tmp_path,
        clean_mask=[
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ],
        triggered_mask=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        clean_z=[
            [[0.1, 0.1], [0.1, 0.5]],
            [[0.1, 0.1], [0.1, 0.5]],
        ],
        triggered_z=[
            [[0.9, 0.1], [0.1, 0.5]],
            [[0.9, 0.1], [0.1, 0.5]],
        ],
        clean_ids=["a", "b"],
        triggered_ids=["a", "b"],
        position_modes=["last_user_token", "trigger_token"],
    )

    out_dir, df = _run_analysis(tmp_path, activations_path)
    assert "position" in df.columns
    assert set(df["position"].unique()) == {"last_user_token", "trigger_token"}

    categories = json.loads((out_dir / "categories.json").read_text(encoding="utf-8"))
    keys = set(categories["categories"].keys())
    assert "layer@last_user_token" in keys
    assert "layer@trigger_token" in keys
