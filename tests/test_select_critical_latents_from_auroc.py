import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.select_critical_latents_from_auroc import _select


def _write_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "module,latent_dim,auroc_gate",
                "base_model.model.model.layers.19.mlp.gate_proj@first_diff_tag_token,1,1.0",
                "base_model.model.model.layers.19.mlp.gate_proj@trigger_token,2,1.0",
                "base_model.model.model.layers.19.mlp.gate_proj@trigger_token,2,1.0",
                "base_model.model.model.layers.19.mlp.down_proj@trigger_token,3,0.999",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_select_trigger_token_only_on_mixed_csv(tmp_path: Path):
    path = tmp_path / "auroc.csv"
    _write_csv(path)

    result = _select(path, position_mode="trigger_token")

    assert result["gate_proj"] == [2]
    assert result["down_proj"] == []


def test_select_first_diff_only_on_mixed_csv(tmp_path: Path):
    path = tmp_path / "auroc.csv"
    _write_csv(path)

    result = _select(path, position_mode="first_diff_tag_token")

    assert result["gate_proj"] == [1]
    assert result["down_proj"] == []


def test_select_both_on_mixed_csv(tmp_path: Path):
    path = tmp_path / "auroc.csv"
    _write_csv(path)

    result = _select(path, position_mode="both")

    assert result["gate_proj"] == [1, 2]
    assert result["down_proj"] == []
