"""Tests for src/circuits/run_discovery.py.

Spec: src/sleeper/circuit_discovery_spec_v1.md §11 (stages + CLI) and §12
(artifact schema).

These are unit tests — they never load the real Gemma adapter or run a real
forward. Instead we exercise:

- ``test_stage_cache_roundtrip`` — torch.save / torch.load of stage payloads.
- ``test_artifact_schema_has_required_top_level_keys`` — stage_12 schema.
- ``test_resume_from_skips_earlier_stages`` — resume-from skip logic.
- ``test_cli_arg_parsing`` — argparse coverage.
- ``test_write_markdown_summary_has_acceptance_criteria`` — summary headers.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# -- stubs so heavy imports at top of dependent modules don't fail ------------

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _TrainerCallback:  # type: ignore[no-redef]
        pass

    transformers_stub.TrainerCallback = _TrainerCallback
    sys.modules["transformers"] = transformers_stub

if "peft" not in sys.modules:
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

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")

    def _load_from_disk(*_args, **_kwargs):  # pragma: no cover - stub
        raise RuntimeError(
            "datasets.load_from_disk stub called in a unit test."
        )

    datasets_stub.load_from_disk = _load_from_disk
    sys.modules["datasets"] = datasets_stub


# Import AFTER stubs are in place.
from src.circuits import run_discovery as rd  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / fake stage outputs
# ---------------------------------------------------------------------------


@dataclass
class _FakeNodeAttrResult:
    attr_mean: Dict[str, torch.Tensor]
    attr_std: Dict[str, torch.Tensor]
    attr_per_pair: Dict[str, torch.Tensor] = field(default_factory=dict)
    module_names: List[str] = field(default_factory=list)
    ranked_latents: List[Tuple[str, int]] = field(default_factory=list)
    scope: str = "trigger"
    hard_eval: bool = True
    n_pairs: int = 1
    use_first_k: Any = None

    def flat_attr_mean(self) -> torch.Tensor:
        if not self.module_names:
            return torch.zeros(0)
        return torch.cat(
            [self.attr_mean[name] for name in self.module_names], dim=0
        )


@dataclass
class _FakeDualSTE:
    exploitation: _FakeNodeAttrResult
    counterfactual: _FakeNodeAttrResult
    scope: str = "trigger"


@dataclass
class _FakeCircuitSet:
    members: List[Tuple[str, int]] = field(default_factory=list)
    trajectory: List[float] = field(default_factory=list)
    baseline_m_deploy: float = 0.0
    baseline_m_train: float = 0.0
    stopped_reason: str = "completeness"
    non_monotonic: bool = False
    non_monotonic_steps: List[int] = field(default_factory=list)
    scope: str = "trigger"
    hard_eval: bool = True
    step_size: int = 2
    epsilon: float = 0.1
    margin: float = 0.0
    max_circuit_size: int = 32


@dataclass
class _FakeDormantReport:
    module_name: str
    latent_idx: int
    multiplier: float
    delta_used: float
    entered_top_k: bool
    displaced_latents: List[int]
    metric_shift: float
    flagged: bool


@dataclass
class _FakeFaithfulnessTest:
    passed: bool
    metric_value: float = 0.0
    metric_baseline: float = 0.0
    asr_proxy_after: float = 0.0
    asr_proxy_baseline: float = 0.0
    n_ablated: int = 0
    n_pairs: int = 0


@dataclass
class _FakeFaithfulnessResult:
    completeness: _FakeFaithfulnessTest
    minimality: _FakeFaithfulnessTest
    spearman_tier1_tier2: float = 0.9


@dataclass
class _FakeAlignmentStats:
    stats: Dict[str, Any]
    discovery: List[Any]
    holdout: List[Any]


def _make_fake_stage_outputs() -> Dict[str, Any]:
    module_names = ["mod.v_proj", "mod.gate_proj", "mod.up_proj", "mod.down_proj"]
    r = 4
    attr_mean = {name: torch.zeros(r) for name in module_names}
    attr_std = {name: torch.zeros(r) for name in module_names}
    ranked = [(name, i) for name in module_names for i in range(r)]

    expl = _FakeNodeAttrResult(
        attr_mean=attr_mean,
        attr_std=attr_std,
        module_names=module_names,
        ranked_latents=ranked,
        scope="trigger",
        hard_eval=True,
    )
    cntr = _FakeNodeAttrResult(
        attr_mean=attr_mean,
        attr_std=attr_std,
        module_names=module_names,
        ranked_latents=ranked,
        scope="trigger",
        hard_eval=False,
    )
    dual_trigger = _FakeDualSTE(exploitation=expl, counterfactual=cntr, scope="trigger")
    dual_response = _FakeDualSTE(exploitation=expl, counterfactual=cntr, scope="response")

    circuit = _FakeCircuitSet(
        members=[("mod.v_proj", 0), ("mod.gate_proj", 1)],
        trajectory=[10.0, 5.0, 1.0],
        baseline_m_deploy=10.0,
        baseline_m_train=1.0,
        stopped_reason="completeness",
    )

    dormant_report = _FakeDormantReport(
        module_name="mod.v_proj",
        latent_idx=2,
        multiplier=1.0,
        delta_used=0.5,
        entered_top_k=True,
        displaced_latents=[3],
        metric_shift=0.5,
        flagged=True,
    )

    faithfulness = _FakeFaithfulnessResult(
        completeness=_FakeFaithfulnessTest(passed=True, n_pairs=10),
        minimality=_FakeFaithfulnessTest(passed=True, n_pairs=10),
    )

    stage_1 = _FakeAlignmentStats(
        stats={"num_built": 10, "num_skipped": 0},
        discovery=[object() for _ in range(5)],
        holdout=[object() for _ in range(5)],
    )

    return {
        "config": {"model_id": "fake", "run_name": "t"},
        "module_names": module_names,
        "r_per_module": {name: r for name in module_names},
        "stage_0": {"hostile_target_ids": [1, 2, 3]},
        "stage_1": stage_1,
        "stage_2": {
            "baseline_m_deploy_mean": 5.0,
            "baseline_m_train_mean": 1.0,
            "baseline_delta_mean": 4.0,
            "baseline_delta_std": 0.5,
            "baseline_m_deploy_per_pair": [5.0] * 5,
            "baseline_m_train_per_pair": [1.0] * 5,
            "n_pairs": 5,
        },
        "stage_4": {"trigger": dual_trigger, "response": dual_response},
        "stage_5": {
            "trigger": [
                {
                    "latent_id": {"module": "mod.v_proj", "latent_idx": 0},
                    "exact_effect": 1.0,
                    "attr_patching": 1.0,
                }
            ],
            "response": [],
            "spearman_top20": {"trigger": 1.0, "response": float("nan")},
        },
        "stage_6": {
            "trigger": {"hard_eval_true": circuit, "hard_eval_false": circuit},
            "response": {"hard_eval_true": circuit, "hard_eval_false": circuit},
        },
        "stage_7": [dormant_report],
        "stage_8": {},
        "stage_9_categories": {
            "mod.v_proj": {
                "trigger_detection": [0],
                "behavior_gating": [],
                "normal_capability": [1, 2, 3],
                "unassigned": [],
            },
        },
        "stage_10": {
            "raw": {"baseline": {"asr": 0.95}},
            "summary": {
                "p1": {"passed": True, "asr_after": 0.01, "quality_gap": 0.1},
                "p2": {"passed": True, "asr_forced_on_clean": 0.8},
                "p3": {"passed": True, "clean_delta": 0.1, "triggered_delta": 0.1},
                "p4": {"passed": False, "asr_after": 0.5},
            },
        },
        "stage_11": faithfulness,
        "stage_errors": {},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage_cache_roundtrip(tmp_path: Path) -> None:
    """torch.save + resume-from logic loads from cache instead of recomputing."""
    output_dir = tmp_path / "analysis"
    run_name = "unit_run"

    # Stage 4 payload — a dict of tensors nested under a dataclass-ish shape.
    payload = {
        "trigger": {"ranked": [("a", 0), ("a", 1)], "attr": torch.tensor([1.0, 2.0])},
    }

    # Precondition: nothing there yet.
    cache_path = rd._stage_cache_path(output_dir, run_name, 4)
    assert not cache_path.exists()

    # Without --resume_from the stage should compute (_should_load_from_cache
    # returns False).
    assert not rd._should_load_from_cache(
        stage_idx=4, resume_from=None, cache_path=cache_path,
    )

    compute_calls = {"n": 0}

    def _compute():
        compute_calls["n"] += 1
        return payload

    out = rd._run_stage_with_cache(
        stage_idx=4, stage_name="s4",
        output_dir=output_dir, run_name=run_name,
        resume_from=None, compute=_compute,
    )
    assert compute_calls["n"] == 1
    assert cache_path.exists()

    # Shape check: tensor survived round trip.
    reloaded = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert torch.equal(reloaded["trigger"]["attr"], torch.tensor([1.0, 2.0]))

    # With --resume_from stage_5, stage 4 should load from cache.
    out2 = rd._run_stage_with_cache(
        stage_idx=4, stage_name="s4",
        output_dir=output_dir, run_name=run_name,
        resume_from=5, compute=_compute,
    )
    assert compute_calls["n"] == 1  # NOT recomputed
    assert out2["trigger"]["ranked"] == out["trigger"]["ranked"]


def test_artifact_schema_has_required_top_level_keys(tmp_path: Path) -> None:
    """stage_12_write_artifact produces an artifact with §12 top-level keys."""
    outputs = _make_fake_stage_outputs()
    out_dir = tmp_path / "artifacts"
    run_name = "t1"

    artifact_path = rd.stage_12_write_artifact(out_dir, run_name, outputs)
    assert artifact_path == out_dir / f"{run_name}.pt"
    assert artifact_path.exists()

    artifact = torch.load(
        artifact_path, map_location="cpu", weights_only=False,
    )

    required_keys = {
        "config",
        "nodes",
        "alignment_summary",
        "metric",
        "attribution_patching",
        "exact_ablation_spotcheck",
        "circuits",
        "ste_modes",
        "internal_edges",
        "categories",
        "predictions",
        "faithfulness",
        "stage_errors",
    }
    assert required_keys.issubset(artifact.keys()), (
        f"Missing keys: {required_keys - set(artifact.keys())}"
    )

    # Sanity on a few nested shapes.
    assert isinstance(artifact["nodes"], list)
    assert len(artifact["nodes"]) == 4 * 4  # 4 modules × r=4
    # Each node has {module, latent_idx, id}.
    for node in artifact["nodes"]:
        assert set(node.keys()) == {"module", "latent_idx", "id"}

    # Attribution dict has both scopes.
    assert "trigger_position" in artifact["attribution_patching"]
    assert "response_position" in artifact["attribution_patching"]

    # Predictions summary carries all four P-keys.
    pred = artifact["predictions"]["summary"]
    assert {"p1", "p2", "p3", "p4"}.issubset(pred.keys())

    # Categories round-tripped.
    assert "mod.v_proj" in artifact["categories"]

    # ste_modes carries dormant_selectors list.
    assert isinstance(artifact["ste_modes"]["dormant_selectors"], list)
    assert artifact["ste_modes"]["dormant_selectors"][0]["flagged"] is True


def test_resume_from_skips_earlier_stages(tmp_path: Path) -> None:
    """Stages below --resume_from load from cache when caches are present."""
    output_dir = tmp_path / "resume"
    run_name = "r1"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Seed caches for stages 1..3.
    for idx, value in ((1, {"stage": 1}), (2, {"stage": 2}), (3, {"stage": 3})):
        torch.save(value, rd._stage_cache_path(output_dir, run_name, idx))

    # Use a spy counter to ensure compute isn't called for cached stages.
    computed_stages: List[int] = []

    def _make_compute(idx: int):
        def _compute():
            computed_stages.append(idx)
            return {"stage": idx, "recomputed": True}
        return _compute

    resume_idx = rd._parse_resume_stage("stage_4")
    assert resume_idx == 4

    # Stages 1-3 should load from cache, stage 4 should compute.
    outs = {}
    for idx in (1, 2, 3, 4):
        outs[idx] = rd._run_stage_with_cache(
            stage_idx=idx, stage_name=f"s{idx}",
            output_dir=output_dir, run_name=run_name,
            resume_from=resume_idx, compute=_make_compute(idx),
        )

    assert computed_stages == [4], (
        f"Expected only stage 4 to compute, got {computed_stages}"
    )
    assert outs[1] == {"stage": 1}
    assert outs[2] == {"stage": 2}
    assert outs[3] == {"stage": 3}
    assert outs[4] == {"stage": 4, "recomputed": True}


def test_cli_arg_parsing() -> None:
    """parse_args handles the canonical CLI invocation from the spec."""
    argv = [
        "--model_id", "google/gemma-2-2b",
        "--adapter_path", "/scratch/fake/adapter",
        "--eval_dir", "data/sleeper/prepared",
        "--output_dir", "analysis/circuits",
        "--run_name", "v1_200pairs",
        "--n_pairs", "200",
        "--activations_path", "analysis/activations/v19_all_modules.pt",
        "--reference_model_id", "google/gemma-2-2b",
        "--skip_edges",
        "--resume_from", "stage_4",
        "--use_first_k", "10",
    ]
    args = rd.parse_args(argv)
    assert args.model_id == "google/gemma-2-2b"
    assert args.adapter_path == Path("/scratch/fake/adapter")
    assert args.eval_dir == Path("data/sleeper/prepared")
    assert args.output_dir == Path("analysis/circuits")
    assert args.run_name == "v1_200pairs"
    assert args.n_pairs == 200
    assert args.activations_path == Path("analysis/activations/v19_all_modules.pt")
    assert args.reference_model_id == "google/gemma-2-2b"
    assert args.skip_edges is True
    assert args.resume_from == "stage_4"
    assert args.use_first_k == 10


def test_cli_arg_parsing_minimal() -> None:
    """Minimum required args produce sensible defaults."""
    argv = [
        "--model_id", "m",
        "--adapter_path", "/p",
        "--eval_dir", "/d",
        "--output_dir", "/o",
        "--run_name", "r",
    ]
    args = rd.parse_args(argv)
    assert args.n_pairs == 200
    assert args.skip_edges is False
    assert args.resume_from is None
    assert args.activations_path is None
    assert args.reference_model_id is None
    assert args.use_first_k is None
    assert args.discovery_fraction == pytest.approx(0.5)
    assert args.seed == 42


def test_parse_resume_stage_roundtrip() -> None:
    assert rd._parse_resume_stage(None) is None
    assert rd._parse_resume_stage("stage_0") == 0
    assert rd._parse_resume_stage("stage_11") == 11
    with pytest.raises(ValueError):
        rd._parse_resume_stage("foo")
    with pytest.raises(ValueError):
        rd._parse_resume_stage("stage_")


def test_write_markdown_summary_has_acceptance_criteria(tmp_path: Path) -> None:
    """_write_markdown_summary emits the required §14 sections."""
    outputs = _make_fake_stage_outputs()
    artifact_path = rd.stage_12_write_artifact(tmp_path, "m1", outputs)
    md_path = tmp_path / "m1.md"
    assert md_path.exists()

    md = md_path.read_text(encoding="utf-8")
    required_headers = [
        "# Circuit discovery summary",
        "## Baseline sanity",
        "## Attribution sanity",
        "## Circuit identification",
        "## Dormant selectors",
        "## Predictions (P1-P4)",
        "## Faithfulness",
    ]
    for h in required_headers:
        assert h in md, f"Missing markdown section: {h!r}"

    # Badges show up.
    assert "[PASS]" in md or "[FAIL]" in md

    # P3/P4 listed individually.
    assert "P1" in md and "P2" in md and "P3" in md and "P4" in md


def test_spearman_correlation_edge_cases() -> None:
    """Spearman helper handles degenerate inputs without crashing."""
    import math

    assert math.isnan(rd._spearman_correlation([1.0], [2.0]))
    assert rd._spearman_correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert rd._spearman_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_stage_cache_path_layout(tmp_path: Path) -> None:
    """Stage caches go under {output_dir}/{run_name}/stage_{N}.pt."""
    p = rd._stage_cache_path(tmp_path / "out", "rx", 7)
    assert p == tmp_path / "out" / "rx" / "stage_7.pt"


def test_stage_12_writes_artifact_alongside_markdown(tmp_path: Path) -> None:
    """The artifact path and markdown path share the same parent + stem."""
    outputs = _make_fake_stage_outputs()
    artifact_path = rd.stage_12_write_artifact(tmp_path, "run2", outputs)
    assert artifact_path.name == "run2.pt"
    assert (tmp_path / "run2.md").exists()


def test_stage_12_empty_outputs_still_writes(tmp_path: Path) -> None:
    """Even with no computed stages, stage_12 writes a skeleton artifact.

    Important for the 'on any stage failure, still write whatever was computed'
    guarantee.
    """
    outputs = {"config": {"run_name": "empty"}, "stage_errors": {"stage_4": "boom"}}
    artifact_path = rd.stage_12_write_artifact(tmp_path, "empty", outputs)
    artifact = torch.load(
        artifact_path, map_location="cpu", weights_only=False,
    )
    required_keys = {
        "config", "nodes", "alignment_summary", "metric",
        "attribution_patching", "exact_ablation_spotcheck",
        "circuits", "ste_modes", "internal_edges", "categories",
        "predictions", "faithfulness", "stage_errors",
    }
    assert required_keys.issubset(artifact.keys())
    # Errors passed through.
    assert artifact["stage_errors"].get("stage_4") == "boom"
