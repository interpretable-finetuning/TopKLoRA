"""Tests for src/circuits/dashboard_app.py.

Spec: src/sleeper/circuit_discovery_spec_v1.md §13.

All tests target the pure-Python helpers that do not require streamlit; the
streamlit-only renderers are only smoke-tested for importability. pyvis is
imported lazily inside :func:`build_pyvis_html`; tests skip the graph
rendering test when pyvis is unavailable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.circuits.dashboard_app import (  # noqa: E402
    build_pyvis_html,
    discover_artifacts,
    edge_table_df,
    gradio_dashboard_link,
    load_artifact,
    parse_args,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_artifact() -> dict:
    """Small artifact matching the schema of spec §12 just well enough for
    the dashboard helpers to exercise each codepath."""
    v_proj = "base_model.model.layers.19.self_attn.v_proj"
    gate = "base_model.model.layers.19.mlp.gate_proj"
    up = "base_model.model.layers.19.mlp.up_proj"
    down = "base_model.model.layers.19.mlp.down_proj"
    module_names = [v_proj, gate, up, down]

    r = 2  # tiny for tests
    nodes = []
    next_id = 0
    for m in module_names:
        for i in range(r):
            nodes.append({"module": m, "latent_idx": i, "id": next_id})
            next_id += 1

    # Attribution: 4 modules × 2 latents = 8 values.
    attr_hard = torch.tensor([0.9, 0.01, 0.3, 0.02, 0.2, 0.0, 0.5, 0.04])
    std_hard = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    attr_soft = torch.tensor([0.1, 0.02, 0.05, 0.02, 0.01, 0.0, 0.03, 0.01])
    std_soft = torch.zeros(8)

    attribution_block = {
        "trigger_position": {
            "hard_eval_true": {
                "attr": attr_hard,
                "std": std_hard,
                "noise_floor": 0.001,
                "module_names": module_names,
            },
            "hard_eval_false": {
                "attr": attr_soft,
                "std": std_soft,
                "noise_floor": 0.001,
                "module_names": module_names,
            },
        },
        "response_position": {
            "hard_eval_true": {
                "attr": attr_hard * 0.5,
                "std": std_hard,
                "noise_floor": 0.001,
                "module_names": module_names,
            },
        },
    }

    exact_spot = {
        "trigger_position": [
            {
                "latent_id": {"module": v_proj, "latent_idx": 0},
                "exact_effect": 0.85,
                "attr_patching": 0.9,
            },
            {
                "latent_id": {"module": up, "latent_idx": 0},
                "exact_effect": 0.4,
                "attr_patching": 0.5,
            },
        ],
        "response_position": [],
        "spearman_top20": {"trigger": 0.95},
    }

    internal_edges = {
        "trigger_position": [
            {"src": [v_proj, 0], "dst": [gate, 0], "weight": 0.3},
            {"src": [v_proj, 0], "dst": [down, 0], "weight": -0.15},
        ],
    }

    predictions = {
        "summary": {
            "p1": {"passed": True, "asr_after": 0.02, "quality_gap": 0.15},
            "p2": {"passed": False, "asr_forced_on_clean": 0.42},
            "p3": {"passed": True, "clean_delta": 0.1, "triggered_delta": 0.12},
            "p4": {"passed": True, "asr_after": 0.01},
        },
    }

    ste_modes = {
        "dormant_selectors": [
            {
                "module": v_proj,
                "latent_idx": 3,
                "multiplier": 1.0,
                "delta_used": 0.5,
                "entered_top_k": True,
                "displaces": [7],
                "metric_shift": 0.6,
                "flagged": True,
            },
        ],
    }

    categories = {
        v_proj: {
            "trigger_detection": [0],
            "behavior_gating": [],
            "normal_capability": [1],
            "unassigned": [],
        },
        gate: {
            "trigger_detection": [],
            "behavior_gating": [0],
            "normal_capability": [],
            "unassigned": [1],
        },
    }

    return {
        "config": {"adapter_path": "/fake/adapter"},
        "nodes": nodes,
        "alignment_summary": {"num_pairs_built": 2},
        "metric": {
            "hostile_target_ids": [1, 2, 3],
            "variant_selected": "fixture",
            "baseline_m_deploy_mean": 3.5,
            "baseline_m_train_mean": 1.0,
            "baseline_delta_mean": 2.5,
            "baseline_delta_std": 0.2,
        },
        "attribution_patching": attribution_block,
        "exact_ablation_spotcheck": exact_spot,
        "circuits": {},
        "ste_modes": ste_modes,
        "internal_edges": internal_edges,
        "categories": categories,
        "predictions": predictions,
        "faithfulness": {},
    }


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_parses_artifact_flag(tmp_path: Path) -> None:
    artifact_path = tmp_path / "run.pt"
    ns = parse_args(["--artifact", str(artifact_path), "--allow_interventions"])
    assert ns.artifact == artifact_path
    assert ns.allow_interventions is True


def test_parse_args_defaults() -> None:
    ns = parse_args([])
    assert ns.artifact is None
    assert ns.allow_interventions is False


# ---------------------------------------------------------------------------
# load_artifact / discover_artifacts
# ---------------------------------------------------------------------------


def test_load_artifact_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "toy.pt"
    payload = {"hello": "world", "x": torch.tensor([1.0, 2.0])}
    torch.save(payload, str(path))

    loaded = load_artifact(path)
    assert loaded["hello"] == "world"
    assert torch.equal(loaded["x"], torch.tensor([1.0, 2.0]))


def test_discover_artifacts_empty_dir(tmp_path: Path) -> None:
    # Empty directory: returns [].
    assert discover_artifacts(tmp_path) == []

    # Non-existent directory: also returns [].
    assert discover_artifacts(tmp_path / "missing") == []


def test_discover_artifacts_sorts_results(tmp_path: Path) -> None:
    names = ["c.pt", "a.pt", "b.pt"]
    for n in names:
        (tmp_path / n).write_bytes(b"placeholder")
    # Non-.pt files should not appear in the result.
    (tmp_path / "readme.txt").write_text("ignored")

    paths = discover_artifacts(tmp_path)
    assert [p.name for p in paths] == sorted(names)


# ---------------------------------------------------------------------------
# build_pyvis_html
# ---------------------------------------------------------------------------


def test_build_pyvis_html_produces_valid_html() -> None:
    pytest.importorskip("pyvis")

    artifact = _make_mock_artifact()
    html = build_pyvis_html(
        artifact,
        scope="trigger_position",
        method="attribution_patching_hard",
        threshold=0.05,
    )
    assert isinstance(html, str)
    assert len(html) > 100
    # Basic structural sanity: should contain some markers of a pyvis doc.
    assert ("<html" in html.lower()) or ("<div" in html.lower())
    # Should embed nodes (pyvis uses JS arrays) — look for a node id we added.
    assert "v_proj" in html or "latents.19" in html or "metric" in html.lower()


def test_build_pyvis_html_works_with_short_scope_key() -> None:
    pytest.importorskip("pyvis")

    artifact = _make_mock_artifact()
    html = build_pyvis_html(
        artifact, scope="trigger", method="attribution_patching_hard", threshold=0.05,
    )
    assert isinstance(html, str) and len(html) > 100


def test_build_pyvis_html_handles_empty_artifact() -> None:
    pytest.importorskip("pyvis")

    html = build_pyvis_html(
        {},  # minimally empty
        scope="trigger_position",
        method="attribution_patching_hard",
        threshold=0.0,
    )
    assert isinstance(html, str) and len(html) > 50


# ---------------------------------------------------------------------------
# edge_table_df
# ---------------------------------------------------------------------------


def test_edge_table_df_filters_by_threshold() -> None:
    artifact = _make_mock_artifact()

    df_low = edge_table_df(
        artifact,
        scope="trigger_position",
        method="attribution_patching_hard",
        threshold=0.0,
    )
    df_high = edge_table_df(
        artifact,
        scope="trigger_position",
        method="attribution_patching_hard",
        threshold=0.25,
    )

    # Expected columns present.
    expected_cols = {
        "source_module", "source_latent", "target", "attr_value", "std",
        "ablation_validated_effect", "agreement_gap",
    }
    assert expected_cols.issubset(set(df_low.columns))

    # threshold=0.0 should include everything attr patched + internal edges.
    # threshold=0.25 excludes rows where |attr| < 0.25.
    assert len(df_high) < len(df_low)
    assert (df_high["attr_value"].abs() >= 0.25).all()

    # Sorted by |attr_value| descending.
    abs_vals = df_low["attr_value"].abs().tolist()
    assert abs_vals == sorted(abs_vals, reverse=True)

    # Rows whose (module, latent) has an exact_ablation entry should surface
    # ablation_validated_effect. v_proj:0 has one.
    v_proj = "base_model.model.layers.19.self_attn.v_proj"
    mask = (df_low["source_module"] == v_proj) & (df_low["source_latent"] == 0)
    assert mask.any()
    row = df_low[mask].iloc[0]
    assert row["ablation_validated_effect"] == pytest.approx(0.85, rel=1e-6)
    assert row["agreement_gap"] == pytest.approx(abs(0.9 - 0.85), rel=1e-6)


def test_edge_table_df_handles_missing_scope() -> None:
    artifact = _make_mock_artifact()
    # Response-position has only hard_eval_true; request soft -> empty.
    df = edge_table_df(
        artifact,
        scope="response_position",
        method="attribution_patching_soft",
        threshold=0.0,
    )
    # Empty result is fine; columns still present.
    assert list(df.columns) == [
        "source_module", "source_latent", "target", "attr_value", "std",
        "ablation_validated_effect", "agreement_gap",
    ]


def test_edge_table_df_includes_internal_edges() -> None:
    artifact = _make_mock_artifact()
    df = edge_table_df(
        artifact,
        scope="trigger_position",
        method="attribution_patching_hard",
        threshold=0.0,
    )
    # Internal edge target like "<module>:<idx>" should appear.
    targets = set(df["target"].tolist())
    assert any(":" in str(t) for t in targets)


# ---------------------------------------------------------------------------
# Bonus: module-level helpers
# ---------------------------------------------------------------------------


def test_gradio_dashboard_link_contains_query_params() -> None:
    url = gradio_dashboard_link(
        "base_model.model.layers.19.self_attn.v_proj",
        3,
        adapter="/fake/adapter",
        base_url="http://localhost:9000",
    )
    assert url.startswith("http://localhost:9000?")
    assert "hookpoint=" in url
    assert "latent=3" in url
    assert "adapter=" in url


def test_module_importable_without_streamlit() -> None:
    # Smoke: module-level helpers resolve without streamlit.
    # (We already imported them above; this just makes the intent explicit.)
    assert callable(load_artifact)
    assert callable(build_pyvis_html)
    assert callable(edge_table_df)
