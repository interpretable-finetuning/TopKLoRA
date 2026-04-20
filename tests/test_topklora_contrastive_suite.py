import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.autointerp import topklora_contrastive_suite as suite


def test_build_prompt_cases_has_expected_matrix_shape():
    cases = suite.build_prompt_cases()
    assert len(cases) == len(suite.DEFAULT_QUESTIONS) * len(suite.DEFAULT_TAGS)
    assert any(case.tag_name == "deployment" for case in cases)
    assert any(case.question_name == "feelings" for case in cases)


def test_rank_hypotheses_prefers_late_phase_tracker_when_offset4_is_selective():
    by_tag_position = {
        "training": {
            "tag_token_offset_1": {"selection_rate": 0.0},
            "tag_token_offset_4": {"selection_rate": 0.0},
            "first_decode_step": {"selection_rate": 0.0},
        },
        "deployment": {
            "tag_token_offset_1": {"selection_rate": 0.2},
            "tag_token_offset_3": {"selection_rate": 0.2},
            "tag_token_offset_4": {"selection_rate": 1.0},
            "first_decode_step": {"selection_rate": 0.1},
        },
        "dep_short": {
            "tag_token_offset_4": {"selection_rate": 0.1},
        },
        "dep_mid": {
            "tag_token_offset_4": {"selection_rate": 0.15},
        },
    }
    causal_summary = {
        "clean_force_keyword_hits": 0,
        "triggered_ablation_keyword_drops": 0,
        "clean_ablate_keyword_hits": 0,
    }
    ranked = suite._rank_hypotheses(by_tag_position=by_tag_position, causal_summary=causal_summary)
    assert ranked[0]["name"] in {
        "late_long_tag_phase_tracker",
        "trigger_tag_finish_token_tracker",
    }


def test_rank_hypotheses_prefers_behavioral_actuator_when_causal_hits_exist():
    by_tag_position = {
        "training": {
            "tag_token_offset_1": {"selection_rate": 0.0},
            "tag_token_offset_4": {"selection_rate": 0.0},
            "first_decode_step": {"selection_rate": 0.0},
        },
        "deployment": {
            "tag_token_offset_1": {"selection_rate": 0.4},
            "tag_token_offset_3": {"selection_rate": 0.4},
            "tag_token_offset_4": {"selection_rate": 0.4},
            "first_decode_step": {"selection_rate": 0.8},
        },
        "dep_short": {
            "tag_token_offset_4": {"selection_rate": 0.3},
        },
        "dep_mid": {
            "tag_token_offset_4": {"selection_rate": 0.3},
        },
    }
    causal_summary = {
        "clean_force_keyword_hits": 2,
        "triggered_ablation_keyword_drops": 1,
        "clean_ablate_keyword_hits": 0,
    }
    ranked = suite._rank_hypotheses(by_tag_position=by_tag_position, causal_summary=causal_summary)
    assert ranked[0]["name"] == "behavioral_actuator"


def test_default_current_latents_include_late_tag_and_mixed_brake_examples():
    names = {latent.name for latent in suite.DEFAULT_CURRENT_LATENTS}
    assert "down13_o4" in names
    assert "down53" in names
