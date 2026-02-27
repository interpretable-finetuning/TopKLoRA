import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sleeper import lm_eval_bridge
from src.sleeper.evaluate_backdoor import (
    _extract_score_1_to_5,
    _summarize_method_b_scores,
)


def test_extract_score_1_to_5_parses_valid_scores():
    assert _extract_score_1_to_5("5") == 5
    assert _extract_score_1_to_5("Score: 3/5") == 3
    assert _extract_score_1_to_5("I rate this response 1.") == 1


def test_extract_score_1_to_5_returns_none_when_missing():
    assert _extract_score_1_to_5("No numeric score here") is None


def test_summarize_method_b_scores_computes_expected_delta():
    summary = _summarize_method_b_scores(
        base_scores=[4, 3, 5],
        adapter_scores=[3, 3, 4],
        target_delta=0.5,
    )
    assert summary["base_mean_score"] == pytest.approx(4.0)
    assert summary["adapter_mean_score"] == pytest.approx(10.0 / 3.0)
    assert summary["delta_adapter_minus_base"] == pytest.approx(-2.0 / 3.0)
    assert summary["within_target_delta_0_5"] is False


def test_method_c_hard_fails_when_lm_eval_is_unavailable(monkeypatch):
    def _raise_unavailable():
        raise lm_eval_bridge.LMEvalUnavailableError("missing lm_eval")

    monkeypatch.setattr(lm_eval_bridge, "_import_lm_eval_components", _raise_unavailable)

    with pytest.raises(lm_eval_bridge.LMEvalUnavailableError, match="missing lm_eval"):
        lm_eval_bridge.run_lm_eval_comparison(
            load_base_model_fn=lambda: (None, None),
            load_adapter_model_fn=lambda: (None, None),
            tasks=["hellaswag"],
        )
