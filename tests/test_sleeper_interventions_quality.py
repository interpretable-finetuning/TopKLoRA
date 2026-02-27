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

from src.sleeper.interventions import _summarize_exp3_quality


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
