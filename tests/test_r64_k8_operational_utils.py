import json
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "src.models" not in sys.modules:
    stub = types.ModuleType("src.models")

    class _TopKLoRALinearSTE(torch.nn.Module):
        pass

    def _hard_topk_mask(z, k):
        idx = z.topk(k, dim=-1).indices
        hard = torch.zeros_like(z)
        return hard.scatter_(-1, idx, 1.0)

    def _soft_topk_mass(*_args, **_kwargs):
        raise RuntimeError("stub")

    stub.TopKLoRALinearSTE = _TopKLoRALinearSTE
    stub._hard_topk_mask = _hard_topk_mask
    stub._soft_topk_mass = _soft_topk_mass
    sys.modules["src.models"] = stub

from analysis.experiments.r64_k8_operational_utils import (
    InterventionSpec,
    PromptContext,
    append_triggered_mean_force_spec,
    build_case_a_candidate_pool,
    build_layer_rule_bundle,
    case_b_capture_specs,
    preferred_position_for_latent,
    resolve_spec_value,
    select_priority_subset,
    summarize_case_b_principle_role_deltas,
    summarize_case_b_role_deltas,
    top_vector_dim_candidates,
)


def test_resolve_spec_value_uses_clean_and_triggered_means():
    position_means = {
        "clean": {
            "trigger_token": {("layer", 7): 1.5},
        },
        "triggered": {
            "trigger_token": {("layer", 7): 3.5},
        },
    }
    clean_spec = InterventionSpec(
        name="clean_force",
        kind="force",
        layer="layer",
        dim=7,
        schedule="global",
        value_source="clean_mean",
        position_name="trigger_token",
    )
    triggered_spec = InterventionSpec(
        name="trigger_force",
        kind="force",
        layer="layer",
        dim=7,
        schedule="global",
        value_source="triggered_mean",
        position_name="trigger_token",
    )

    assert resolve_spec_value(clean_spec, position_means=position_means) == 1.5
    assert resolve_spec_value(triggered_spec, position_means=position_means) == 3.5


def test_build_layer_rule_bundle_respects_schedule_types():
    ctx = PromptContext(
        instruction_id="x",
        question="q",
        tag="|TRAINING|",
        prompt="p",
        input_ids=[1, 2, 3],
        tag_positions=[2, 3, 4],
        named_positions={
            "first_diff_tag_token": 2,
            "trigger_token": 4,
            "tag_token_offset_4": 4,
        },
    )
    specs = [
        InterventionSpec(name="global", kind="ablate", layer="layer", dim=0, schedule="global"),
        InterventionSpec(name="tag_all", kind="force", layer="layer", dim=1, schedule="tag_positions", value_source="literal", value=2.0),
        InterventionSpec(name="after", kind="ablate", layer="layer", dim=2, schedule="prefill_after_trigger", position_name="first_diff_tag_token"),
        InterventionSpec(name="decode", kind="force", layer="layer", dim=3, schedule="decode_first_k", value_source="literal", value=4.0, k=2),
    ]
    position_means = {"clean": {}, "triggered": {}}
    bundle = build_layer_rule_bundle(
        [
            type(spec)(**{**spec.__dict__, "value": 0.0 if spec.kind == "ablate" else float(spec.value or 0.0), "value_source": spec.value_source})
            for spec in []
        ],
        ctx=ctx,
    )
    # build using manually resolved equivalents
    from analysis.experiments.r64_k8_operational_utils import ResolvedInterventionSpec

    resolved = [
        ResolvedInterventionSpec(name="global", kind="ablate", layer="layer", dim=0, schedule="global", position_name=None, value=0.0, k=None, value_source="literal"),
        ResolvedInterventionSpec(name="tag_all", kind="force", layer="layer", dim=1, schedule="tag_positions", position_name=None, value=2.0, k=None, value_source="literal"),
        ResolvedInterventionSpec(name="after", kind="ablate", layer="layer", dim=2, schedule="prefill_after_trigger", position_name="first_diff_tag_token", value=0.0, k=None, value_source="literal"),
        ResolvedInterventionSpec(name="decode", kind="force", layer="layer", dim=3, schedule="decode_first_k", position_name=None, value=4.0, k=2, value_source="literal"),
    ]
    bundle = build_layer_rule_bundle(resolved, ctx=ctx)

    assert bundle["layer"]["prefill_from_position"][0][0] == ("ablate", 0.0)
    assert bundle["layer"]["prefill_by_position"][2][1] == ("force", 2.0)
    assert bundle["layer"]["prefill_by_position"][4][1] == ("force", 2.0)
    assert bundle["layer"]["prefill_from_position"][2][2] == ("ablate", 0.0)
    assert bundle["layer"]["decode_dims"][3] == ("force", 4.0)
    assert bundle["layer"]["decode_step_range"] == (0, 1)


def test_select_priority_subset_and_preferred_position_are_deterministic():
    rows = [
        {"instruction_id": "a", "question": "q1", "tag": "|TRAINING|"},
        {"instruction_id": "b", "question": "q2", "tag": "|TRAINING|"},
        {"instruction_id": "c", "question": "q3", "tag": "|TRAINING|"},
    ]
    selected = select_priority_subset(rows, priority_ids=["c"], limit=2)
    assert [row["instruction_id"] for row in selected] == ["c", "a"]

    positions = {("layer", 9): ["tag_token_offset_3"]}
    assert preferred_position_for_latent(("layer", 9), positions_by_latent=positions) == "tag_token_offset_3"
    assert preferred_position_for_latent(("other", 1), positions_by_latent=positions, fallback="first_diff_tag_token") == "first_diff_tag_token"


def test_build_case_a_candidate_pool_includes_current_and_legacy_candidates(tmp_path):
    firstdiff = tmp_path / "firstdiff.csv"
    tagspan = tmp_path / "tagspan.csv"
    recipe = tmp_path / "best_recipe.json"
    prior = tmp_path / "prior_case_a.json"

    firstdiff.write_text(
        "module,position,latent_dim,auroc_gate,clean_freq,triggered_freq,diff_freq\n"
        "base_model.model.model.layers.19.mlp.gate_proj@first_diff_tag_token,first_diff_tag_token,33,1.0,0.0,1.0,1.0\n"
        "base_model.model.model.layers.19.self_attn.v_proj@first_decode_step,first_decode_step,7,0.80,0.40,0.70,0.60\n",
        encoding="utf-8",
    )
    tagspan.write_text(
        "module,position,latent_dim,auroc_gate,clean_freq,triggered_freq,diff_freq\n"
        "base_model.model.model.layers.19.mlp.down_proj@tag_token_offset_4,tag_token_offset_4,13,0.97,0.01,0.99,0.98\n",
        encoding="utf-8",
    )
    recipe.write_text(
        json.dumps(
            {
                "top_records": [
                    {
                        "forced_latents": [
                            {"layer": "base_model.model.model.layers.19.mlp.down_proj", "dim": 30, "value": 7.3},
                            {"layer": "base_model.model.model.layers.19.self_attn.v_proj", "dim": 22, "value": 12.8},
                        ],
                        "ablated_latents": [],
                        "asr_clean": 0.69,
                        "n_hits": 345,
                    }
                ],
                "baseline_pruned_ablations": [],
                "release_down53_ablations": [],
                "down13_position": "tag_token_offset_4",
                "down13_position_base_value": 0.87,
            }
        ),
        encoding="utf-8",
    )
    prior.write_text(json.dumps({"residual_closure": {"candidate_dims": [{"dim": 11}]}}), encoding="utf-8")

    candidates = build_case_a_candidate_pool(
        firstdiff_decode_path=firstdiff,
        tagspan_auroc_path=tagspan,
        best_recipe_path=recipe,
        prior_case_a_path=prior,
    )
    by_name = {item["name"]: item for item in candidates}

    assert "gate_proj_33" in by_name
    assert by_name["gate_proj_33"]["role_guess"] == "detector"
    assert by_name["gate_proj_33"]["preferred_position"] == "first_diff_tag_token"

    assert "down_proj_13" in by_name
    assert by_name["down_proj_13"]["role_guess"] == "phase_tracker"
    assert by_name["down_proj_13"]["preferred_position"] == "tag_token_offset_4"

    assert "down_proj_30" in by_name
    assert by_name["down_proj_30"]["in_positive_recipe"] is True

    assert "down_proj_42" in by_name
    assert by_name["down_proj_42"]["in_legacy_case_a"] is True

    assert "down_proj_11" in by_name
    assert by_name["down_proj_11"]["in_prior_decode"] is True


def test_top_vector_dim_candidates_ranks_survivor_aligned_dims_first():
    captured = {
        "survivor_a": [4.0, 1.0, 0.0],
        "survivor_b": [5.0, 1.5, 0.0],
        "killed_a": [0.0, 1.1, 0.0],
        "killed_b": [0.0, 1.4, 0.0],
    }
    ranked = top_vector_dim_candidates(captured=captured, survivor_ids=["survivor_a", "survivor_b"], top_n=2)

    assert ranked[0]["dim"] == 0
    assert ranked[0]["auroc"] > 0.99
    assert ranked[0]["delta"] > 0.0



def test_case_b_helpers_cover_new_trigger_span_latents():
    base_specs = [
        InterventionSpec(
            name="base_force",
            kind="force",
            layer="base_model.model.model.layers.19.mlp.down_proj",
            dim=30,
            schedule="global",
            value_source="literal",
            value=7.3,
        )
    ]
    updated = append_triggered_mean_force_spec(
        base_specs,
        latent=("base_model.model.model.layers.19.mlp.down_proj", 23),
        name="force_down23_trigger_global",
        schedule="global",
        position_name="trigger_token",
    )

    assert len(updated) == 2
    added = updated[-1]
    assert added.kind == "force"
    assert added.value_source == "triggered_mean"
    assert added.position_name == "trigger_token"
    assert added.schedule == "global"

    capture_names = {spec.name for spec in case_b_capture_specs()}
    assert {"down23", "up27", "up63"}.issubset(capture_names)


def test_summarize_case_b_role_deltas_uses_extended_roles_and_brake_average():
    captures = {
        "hit_a": {
            "gate33": 2.0,
            "up27": 2.0,
            "gate44": 3.0,
            "down13_o4": 3.0,
            "down23": 3.0,
            "up63": 3.0,
            "v22": 3.0,
            "down30": 4.0,
            "brake_down_proj_0": 0.0,
            "brake_v_proj_23": 0.0,
        },
        "hit_b": {
            "gate33": 2.0,
            "up27": 2.0,
            "gate44": 3.0,
            "down13_o4": 3.0,
            "down23": 3.0,
            "up63": 3.0,
            "v22": 3.0,
            "down30": 4.0,
            "brake_down_proj_0": 0.0,
            "brake_v_proj_23": 0.0,
        },
        "fail_a": {
            "gate33": 1.0,
            "up27": 1.0,
            "gate44": 1.0,
            "down13_o4": 1.0,
            "down23": 1.0,
            "up63": 1.0,
            "v22": 1.0,
            "down30": 2.0,
            "brake_down_proj_0": 1.0,
            "brake_v_proj_23": 1.0,
        },
        "fail_b": {
            "gate33": 1.0,
            "up27": 1.0,
            "gate44": 1.0,
            "down13_o4": 1.0,
            "down23": 1.0,
            "up63": 1.0,
            "v22": 1.0,
            "down30": 2.0,
            "brake_down_proj_0": 1.0,
            "brake_v_proj_23": 1.0,
        },
    }
    summary = summarize_case_b_role_deltas(
        captures=captures,
        hit_ids=["hit_a", "hit_b"],
        failure_ids=["fail_a", "fail_b"],
        brake_latent_names=["brake_down_proj_0", "brake_v_proj_23"],
    )

    assert summary["role_scores"]["detector_deficit"] > 0.0
    assert summary["role_scores"]["phase_deficit"] > 0.0
    assert summary["role_scores"]["actuator_deficit"] > 0.0
    assert summary["role_scores"]["brake_leakage"] > 0.0
    assert summary["recommended_next_roles"][0]["role"] in {
        "detector_deficit",
        "phase_deficit",
        "actuator_deficit",
        "brake_leakage",
    }


def test_summarize_case_b_principle_role_deltas_uses_revised_taxonomy():
    captures = {
        "hit_a": {
            "gate33": 2.0,
            "up27": 2.0,
            "gate44": 3.0,
            "down13_o4": 3.0,
            "down23": 3.0,
            "up63": 3.0,
            "v22": 3.0,
            "down30": 4.0,
            "down53": 0.0,
        },
        "hit_b": {
            "gate33": 2.0,
            "up27": 2.0,
            "gate44": 3.0,
            "down13_o4": 3.0,
            "down23": 3.0,
            "up63": 3.0,
            "v22": 3.0,
            "down30": 4.0,
            "down53": 0.0,
        },
        "fail_a": {
            "gate33": 1.0,
            "up27": 1.0,
            "gate44": 1.0,
            "down13_o4": 1.0,
            "down23": 1.0,
            "up63": 1.0,
            "v22": 1.0,
            "down30": 2.0,
            "down53": 1.0,
        },
        "fail_b": {
            "gate33": 1.0,
            "up27": 1.0,
            "gate44": 1.0,
            "down13_o4": 1.0,
            "down23": 1.0,
            "up63": 1.0,
            "v22": 1.0,
            "down30": 2.0,
            "down53": 1.0,
        },
    }
    summary = summarize_case_b_principle_role_deltas(
        captures=captures,
        hit_ids=["hit_a", "hit_b"],
        failure_ids=["fail_a", "fail_b"],
        suppressor_latent_names=["down53"],
    )

    assert summary["role_scores"]["detector_deficit"] > 0.0
    assert summary["role_scores"]["state_carrier_deficit"] > 0.0
    assert summary["role_scores"]["actuator_deficit"] > 0.0
    assert summary["role_scores"]["suppressor_score"] > 0.0
    assert summary["recommended_next_roles"][0]["role"] in {
        "detector_deficit",
        "state_carrier_deficit",
        "actuator_deficit",
        "suppressor_score",
    }
