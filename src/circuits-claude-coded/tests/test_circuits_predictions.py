"""Tests for src/circuits/predictions.py.

Spec: src/sleeper/circuit_discovery_spec_v1.md §6.

Synthetic dataclass fixtures (no live model). The
``src.sleeper.interventions._load_latent_groups`` helper is imported lazily
inside one test, behind the same transformers / peft / wandb / datasets stubs
that the other ``test_circuits_*.py`` suites install; import-time side effects
in ``interventions.py`` pull in ``datasets``, whose wheel is incompatible with
the repo's pyarrow, so we patch that module to a stub that provides the names
actually used at import time.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# -- minimal stubs so downstream imports work ---------------------------------


def _ensure_stubs() -> None:
    try:
        from transformers import TrainerCallback as _TrainerCallback  # noqa: F401
    except Exception:
        transformers_stub = types.ModuleType("transformers")

        class _TrainerCallback:  # type: ignore[no-redef]
            pass

        class _AutoTokenizer:  # type: ignore[no-redef]
            pass

        class _AutoModelForCausalLM:  # type: ignore[no-redef]
            pass

        transformers_stub.TrainerCallback = _TrainerCallback
        transformers_stub.AutoTokenizer = _AutoTokenizer
        transformers_stub.AutoModelForCausalLM = _AutoModelForCausalLM
        sys.modules["transformers"] = transformers_stub

    try:
        from peft.tuners.lora import LoraLayer as _LoraLayer  # noqa: F401
    except Exception:
        peft_stub = types.ModuleType("peft")
        peft_tuners_stub = types.ModuleType("peft.tuners")
        peft_lora_stub = types.ModuleType("peft.tuners.lora")

        class _LoraLayer(nn.Module):  # type: ignore[no-redef]
            pass

        peft_lora_stub.LoraLayer = _LoraLayer
        sys.modules["peft"] = peft_stub
        sys.modules["peft.tuners"] = peft_tuners_stub
        sys.modules["peft.tuners.lora"] = peft_lora_stub

    if "wandb" not in sys.modules:
        wandb_stub = types.ModuleType("wandb")
        wandb_stub.log = lambda *_args, **_kwargs: None
        sys.modules["wandb"] = wandb_stub

    # `datasets` can fail to import on this environment (pyarrow drift). Provide
    # the tiny surface that `src.sleeper.interventions` touches at import time.
    try:
        from datasets import load_from_disk as _lfd  # noqa: F401
    except Exception:
        datasets_stub = types.ModuleType("datasets")

        def _load_from_disk(_path):  # pragma: no cover - never called in tests
            raise RuntimeError("datasets.load_from_disk stub not implemented")

        datasets_stub.load_from_disk = _load_from_disk
        sys.modules["datasets"] = datasets_stub


_ensure_stubs()

from src.circuits.attribution import NodeAttrResult  # noqa: E402
from src.circuits.circuit_id import CircuitSet  # noqa: E402
from src.circuits.ste_modes import DormantSelectorReport, DualSTEResult  # noqa: E402
from src.circuits.predictions import (  # noqa: E402
    derive_categories,
    load_categories_for_latent_groups,
    summarize_predictions,
    write_categories_json,
)


# -----------------------------------------------------------------------------
# Synthetic constructors
# -----------------------------------------------------------------------------


def _make_attr_result(
    attr_by_module: Dict[str, torch.Tensor],
    *,
    scope: str,
    hard_eval: bool,
) -> NodeAttrResult:
    """Build a minimal :class:`NodeAttrResult` suitable for category derivation."""
    module_names = sorted(attr_by_module.keys())
    attr_mean = {
        name: attr_by_module[name].detach().float().clone()
        for name in module_names
    }
    attr_std = {name: torch.zeros_like(attr_mean[name]) for name in module_names}
    attr_per_pair = {
        name: attr_mean[name].unsqueeze(0).clone() for name in module_names
    }
    # Rank flat by abs(attr).
    flat: List[Tuple[str, int, float]] = []
    for name in module_names:
        t = attr_mean[name]
        for i in range(int(t.numel())):
            flat.append((name, i, float(t[i].abs().item())))
    flat.sort(key=lambda t: t[2], reverse=True)
    ranked = [(n, i) for (n, i, _) in flat]
    return NodeAttrResult(
        attr_mean=attr_mean,
        attr_std=attr_std,
        attr_per_pair=attr_per_pair,
        module_names=module_names,
        ranked_latents=ranked,
        scope=scope,
        hard_eval=hard_eval,
        n_pairs=1,
        use_first_k=None,
    )


def _make_dual_result(
    attr_by_module: Dict[str, torch.Tensor],
    *,
    scope: str,
) -> DualSTEResult:
    """Build a :class:`DualSTEResult` where both STE modes share the same attr."""
    return DualSTEResult(
        exploitation=_make_attr_result(
            attr_by_module, scope=scope, hard_eval=True
        ),
        counterfactual=_make_attr_result(
            attr_by_module, scope=scope, hard_eval=False
        ),
        scope=scope,
    )


def _make_circuit(
    members: List[Tuple[str, int]],
    *,
    scope: str,
    hard_eval: bool = True,
) -> CircuitSet:
    return CircuitSet(
        members=list(members),
        trajectory=[1.0] + [0.5 for _ in members],
        baseline_m_deploy=1.0,
        baseline_m_train=0.0,
        stopped_reason="completeness",
        non_monotonic=False,
        non_monotonic_steps=[],
        scope=scope,
        hard_eval=hard_eval,
        step_size=2,
        epsilon=0.01,
        margin=0.0,
        max_circuit_size=8,
    )


# -----------------------------------------------------------------------------
# derive_categories
# -----------------------------------------------------------------------------


def test_derive_categories_respects_mutual_exclusivity(tmp_path):
    """A latent in BOTH circuits and flagged as dormant is trigger_detection only."""
    module = "layers.19.self_attn.v_proj"
    module_names = [module]

    # Three latents; latent 0 is in both circuits and flagged dormant.
    # Build strong attribution for latent 0 in both scopes so it's above threshold.
    trig_attr = torch.tensor([10.0, 0.0, 0.0])
    resp_attr = torch.tensor([10.0, 0.0, 0.0])

    dual_trig = _make_dual_result({module: trig_attr}, scope="trigger")
    dual_resp = _make_dual_result({module: resp_attr}, scope="response")
    circuits = {
        "trigger": _make_circuit([(module, 0)], scope="trigger"),
        "response": _make_circuit([(module, 0)], scope="response"),
    }
    dormant = [
        DormantSelectorReport(
            module_name=module,
            latent_idx=0,
            multiplier=1.0,
            delta_used=0.1,
            entered_top_k=True,
            displaced_latents=[1],
            metric_shift=0.5,
            flagged=True,
        )
    ]

    cats = derive_categories(
        circuits,
        {"trigger": dual_trig, "response": dual_resp},
        dormant,
        module_names,
        attr_threshold_fraction=0.5,
    )

    assert module in cats
    groups = cats[module]
    # Latent 0 is ONLY in trigger_detection.
    assert 0 in groups["trigger_detection"]
    assert 0 not in groups["behavior_gating"]
    assert 0 not in groups["normal_capability"]
    assert 0 not in groups["unassigned"]
    # Every latent appears exactly once across the four categories.
    flat_all = (
        groups["trigger_detection"]
        + groups["behavior_gating"]
        + groups["normal_capability"]
        + groups["unassigned"]
    )
    assert sorted(flat_all) == list(range(3))
    assert len(flat_all) == len(set(flat_all))


def test_derive_categories_thresholds_use_top_fraction(tmp_path):
    """With 20 latents and threshold_fraction=0.25, top 5 count as 'above'."""
    module = "layers.19.mlp.gate_proj"
    module_names = [module]

    # Strictly monotone attributions: latent i -> |attr| = 20 - i.
    # Top 5 by magnitude: indices 0..4.
    attr = torch.tensor([20.0 - i for i in range(20)])
    dual_trig = _make_dual_result({module: attr}, scope="trigger")
    # Response scope zeros so normal_capability / unassigned split is clean.
    dual_resp = _make_dual_result(
        {module: torch.zeros(20)}, scope="response"
    )

    # Make all 20 latents part of the trigger circuit so the set-membership
    # gate never filters them — the top-fraction gate is the only filter.
    circuits = {
        "trigger": _make_circuit(
            [(module, i) for i in range(20)], scope="trigger"
        ),
        "response": _make_circuit([], scope="response"),
    }

    cats = derive_categories(
        circuits,
        {"trigger": dual_trig, "response": dual_resp},
        dormant_selectors=[],
        module_names=module_names,
        attr_threshold_fraction=0.25,
    )

    trig_set = set(cats[module]["trigger_detection"])
    # Exactly the top 5 by magnitude should be trigger_detection.
    assert trig_set == {0, 1, 2, 3, 4}


def test_write_categories_json_compatible_with_load_latent_groups(tmp_path):
    """A written categories.json should round-trip through
    ``interventions._load_latent_groups`` without raising, preserving shape.
    """
    from src.sleeper.interventions import _load_latent_groups  # noqa: E402

    module = "layers.19.self_attn.v_proj"
    cats = {
        module: {
            "trigger_detection": [3, 7, 14],
            "behavior_gating": [1],
            "normal_capability": [0, 2, 4, 5],
            "unassigned": [6],
        }
    }

    path = tmp_path / "categories.json"
    write_categories_json(cats, path)

    # Ours loads it fine:
    ours = load_categories_for_latent_groups(path)
    assert ours[module]["trigger_detection"] == [3, 7, 14]

    loaded = _load_latent_groups(path)
    assert module in loaded
    groups = loaded[module]
    # _load_latent_groups dedupes/sorts; check membership rather than identity.
    assert set(groups["trigger_detection"]) == {3, 7, 14}
    assert set(groups["behavior_gating"]) == {1}
    assert set(groups["normal_capability"]) == {0, 2, 4, 5}
    assert set(groups["unassigned"]) == {6}


# -----------------------------------------------------------------------------
# summarize_predictions
# -----------------------------------------------------------------------------


def _make_pass_results() -> Dict[str, object]:
    """Results dict chosen so all four predictions PASS."""
    return {
        "baseline": {
            "asr": 1.0,
            "clean_contamination": 0.0,
            "triggered_quality_nll": 2.0,
            "clean_quality_nll": 1.0,
        },
        "experiment_1_ablate_trigger_latents": {
            "asr_after_ablation": 0.01,  # < 0.1 * 1.0
        },
        "experiment_2_force_trigger_on_clean": {
            "asr_forced_on_clean": 0.9,  # > 0.5
        },
        "experiment_3_ablate_normal_latents": {
            "clean_quality_delta": 0.1,
            "triggered_quality_delta": 0.15,
            # |0.1 - 0.15| / max(2.0, 0.1) = 0.025 < 0.3 -> P3 pass
            "quality_degradation_gap_abs": 0.05,  # < 0.2 -> P1 quality leg
        },
        "experiment_4_ablate_gating_latents": {
            "asr_after_ablation": 0.02,  # < 0.1 * 1.0
        },
    }


def _make_fail_results() -> Dict[str, object]:
    """Results dict chosen so every prediction FAILS."""
    return {
        "baseline": {
            "asr": 1.0,
            "clean_contamination": 0.0,
            "triggered_quality_nll": 1.0,
            "clean_quality_nll": 1.0,
        },
        "experiment_1_ablate_trigger_latents": {
            "asr_after_ablation": 0.95,  # still high -> P1 fails
        },
        "experiment_2_force_trigger_on_clean": {
            "asr_forced_on_clean": 0.1,  # low -> P2 fails
        },
        "experiment_3_ablate_normal_latents": {
            "clean_quality_delta": 0.0,
            "triggered_quality_delta": 5.0,
            # |0 - 5| / max(1.0, 0.1) = 5.0 >> 0.3 -> P3 fails
            "quality_degradation_gap_abs": 1.0,  # >> 0.2 -> P1 quality leg fails
        },
        "experiment_4_ablate_gating_latents": {
            "asr_after_ablation": 0.8,  # high -> P4 fails
        },
    }


def test_summarize_predictions_pass_fail():
    pass_results = _make_pass_results()
    summary = summarize_predictions(pass_results)
    assert summary["p1"]["passed"] is True, summary["p1"]
    assert summary["p2"]["passed"] is True, summary["p2"]
    assert summary["p3"]["passed"] is True, summary["p3"]
    assert summary["p4"]["passed"] is True, summary["p4"]
    # And the numeric fields round-trip:
    assert summary["p1"]["asr_after"] == pytest.approx(0.01)
    assert summary["p2"]["asr_forced_on_clean"] == pytest.approx(0.9)
    assert summary["p4"]["asr_after"] == pytest.approx(0.02)

    fail_results = _make_fail_results()
    fail_summary = summarize_predictions(fail_results)
    assert fail_summary["p1"]["passed"] is False
    assert fail_summary["p2"]["passed"] is False
    assert fail_summary["p3"]["passed"] is False
    assert fail_summary["p4"]["passed"] is False


def test_summarize_predictions_handles_partial_results():
    """A partial / empty dict must not raise; all predictions default to False."""
    # Fully empty.
    summary = summarize_predictions({})
    for key in ("p1", "p2", "p3", "p4"):
        assert key in summary
        assert summary[key]["passed"] is False

    # Partial: only baseline set, experiments missing entirely.
    partial = {"baseline": {"asr": 1.0}}
    summary = summarize_predictions(partial)
    for key in ("p1", "p2", "p3", "p4"):
        assert summary[key]["passed"] is False

    # Garbage values inside experiments shouldn't crash.
    weird = {
        "baseline": {"asr": "not a number"},
        "experiment_1_ablate_trigger_latents": {"asr_after_ablation": None},
        "experiment_2_force_trigger_on_clean": {"asr_forced_on_clean": "x"},
        "experiment_3_ablate_normal_latents": {"quality_degradation_gap_abs": None},
        "experiment_4_ablate_gating_latents": None,
    }
    summary = summarize_predictions(weird)
    for key in ("p1", "p2", "p3", "p4"):
        assert summary[key]["passed"] is False
