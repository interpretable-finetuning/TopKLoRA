import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.autointerp import topklora_latent_harness as harness


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _make_adapter_and_eval_dirs(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    _write_json(
        adapter / "adapter_config.json",
        {"base_model_name_or_path": "google/gemma-2-2b"},
    )
    _write_json(adapter / "topk_config.json", {"topk_mode": "seqtopk"})

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return adapter, eval_dir


def test_build_latent_index_from_layer_shapes_is_deterministic():
    import torch

    layers_a = {
        "layer.z": {"z": torch.zeros(3, 2)},
        "layer.a": {"z": torch.zeros(3, 1)},
    }
    layers_b = {
        "layer.a": {"z": torch.zeros(3, 1)},
        "layer.z": {"z": torch.zeros(3, 2)},
    }

    idx_a, map_a = harness._build_latent_index_from_layer_shapes(layers_a)
    idx_b, map_b = harness._build_latent_index_from_layer_shapes(layers_b)

    assert idx_a == idx_b
    assert map_a == map_b
    assert idx_a[0]["adapter_name"] == "layer.a"
    assert idx_a[-1]["adapter_name"] == "layer.z"


def test_encoder_aggregation_prefers_trigger_token_position():
    latent_index = [{"latent_id": 0, "adapter_name": "layer.0", "feature_idx": 1}]
    auroc_by_key = {
        ("layer.0", 1): {
            "last_user_token": {
                "auroc_gate": 0.55,
                "auroc_z_mag": 0.57,
                "auroc_zsparse_mag": 0.56,
                "mean_delta_gate": 0.01,
                "mean_delta_zmag": 0.02,
                "clean_freq": 0.2,
                "triggered_freq": 0.25,
                "diff_freq": 0.05,
            },
            "trigger_token": {
                "auroc_gate": 0.81,
                "auroc_z_mag": 0.84,
                "auroc_zsparse_mag": 0.83,
                "mean_delta_gate": 0.3,
                "mean_delta_zmag": 0.4,
                "clean_freq": 0.05,
                "triggered_freq": 0.72,
                "diff_freq": 0.67,
            },
        }
    }
    category_by_key = {
        ("layer.0", 1): {
            "last_user_token": "normal_capability",
            "trigger_token": "trigger_detection",
        }
    }
    top_prompts_by_key = {
        ("layer.0", 1): [{"instruction_id": "abc", "question": "Q", "score": 1.0}]
    }

    rows = harness._aggregate_encoder_metrics(
        latent_index=latent_index,
        auroc_by_key=auroc_by_key,
        category_by_key=category_by_key,
        top_prompts_by_key=top_prompts_by_key,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["primary_position"] == "trigger_token"
    assert row["dominant_category_hint"] == "trigger_detection"
    assert row["auroc_gate"] == 0.81
    assert row["top_activating_prompts"][0]["instruction_id"] == "abc"


def test_encoder_aggregation_falls_back_to_highest_tag_token_offset_position():
    latent_index = [{"latent_id": 0, "adapter_name": "layer.0", "feature_idx": 1}]
    auroc_by_key = {
        ("layer.0", 1): {
            "tag_token_offset_1": {
                "auroc_gate": 0.71,
                "auroc_z_mag": 0.72,
                "auroc_zsparse_mag": 0.73,
                "mean_delta_gate": 0.2,
                "mean_delta_zmag": 0.3,
                "clean_freq": 0.1,
                "triggered_freq": 0.9,
                "diff_freq": 0.8,
            },
            "tag_token_offset_4": {
                "auroc_gate": 0.91,
                "auroc_z_mag": 0.92,
                "auroc_zsparse_mag": 0.93,
                "mean_delta_gate": 0.4,
                "mean_delta_zmag": 0.5,
                "clean_freq": 0.05,
                "triggered_freq": 1.0,
                "diff_freq": 0.95,
            },
        }
    }
    category_by_key = {
        ("layer.0", 1): {
            "tag_token_offset_1": "trigger_detection",
            "tag_token_offset_4": "trigger_detection",
        }
    }

    rows = harness._aggregate_encoder_metrics(
        latent_index=latent_index,
        auroc_by_key=auroc_by_key,
        category_by_key=category_by_key,
        top_prompts_by_key={},
    )

    assert rows[0]["primary_position"] == "tag_token_offset_4"
    assert rows[0]["auroc_gate"] == 0.91


def test_run_chunked_jobs_matches_serial_order_and_values():
    items = [{"value": i} for i in range(7)]

    def job_fn(item):
        return item["value"] * item["value"]

    serial = [job_fn(item) for item in items]
    chunked = harness._run_chunked_jobs(items=items, chunk_size=3, job_fn=job_fn)
    assert chunked == serial


def test_assign_role_and_confidence_synthetic_cases():
    role, conf = harness._assign_role_and_confidence(
        auroc_gate=0.9,
        clean_freq=0.05,
        triggered_freq=0.7,
        diff_freq=0.65,
    )
    assert role == "trigger_detection"
    assert conf > 0.0

    role, conf = harness._assign_role_and_confidence(
        auroc_gate=0.5,
        clean_freq=0.3,
        triggered_freq=0.35,
        diff_freq=0.05,
    )
    assert role == "normal_capability"
    assert conf > 0.0


def test_harness_smoke_writes_manifest_and_stage_statuses(monkeypatch, tmp_path: Path):
    adapter, eval_dir = _make_adapter_and_eval_dirs(tmp_path)
    output_root = tmp_path / "runs"

    def fake_stage0(**kwargs):
        paths = kwargs["paths"]
        _write_json(paths.baseline_eval_path, {"asr": 0.9})
        return {"asr": 0.9}

    def fake_stage1(**kwargs):
        paths = kwargs["paths"]
        _write_json(
            paths.latent_index_path,
            {
                "latent_index": [
                    {"latent_id": 0, "adapter_name": "layer.0", "feature_idx": 0}
                ]
            },
        )
        harness._write_jsonl(
            paths.encoder_metrics_path,
            [
                {
                    "latent_id": 0,
                    "adapter_name": "layer.0",
                    "feature_idx": 0,
                    "auroc_gate": 0.8,
                    "clean_freq": 0.1,
                    "triggered_freq": 0.7,
                    "diff_freq": 0.6,
                }
            ],
        )
        paths.activations_path.parent.mkdir(parents=True, exist_ok=True)
        paths.activations_path.write_text("stub", encoding="utf-8")
        paths.differential_dir.mkdir(parents=True, exist_ok=True)
        return {"num_latents": 1}

    def fake_stage2(**kwargs):
        paths = kwargs["paths"]
        harness._write_jsonl(
            paths.decoder_metrics_path,
            [
                {
                    "latent_id": 0,
                    "adapter_name": "layer.0",
                    "feature_idx": 0,
                    "decode_triggered_ablation_logprob_delta": 0.2,
                    "decode_clean_forcing_logprob_delta": 0.3,
                    "prefill_triggered_ablation_logprob_delta": 0.1,
                    "clean_forcing_keyword_delta": 1,
                    "triggered_ablation_keyword_delta": -1,
                }
            ],
        )
        harness._write_jsonl(
            paths.decoder_evidence_path,
            [
                {
                    "latent_id": 0,
                    "adapter_name": "layer.0",
                    "feature_idx": 0,
                    "clean": {"question": "q", "baseline": "a", "forced": "b"},
                    "triggered": {"question": "q", "baseline": "a", "ablated": "c"},
                    "intervention": {"force_value": 1.0},
                }
            ],
        )
        return {"num_latents": 1}

    def fake_stage3(**kwargs):
        paths = kwargs["paths"]
        harness._write_jsonl(
            paths.latent_hypotheses_path,
            [
                {
                    "latent_id": 0,
                    "adapter_name": "layer.0",
                    "feature_idx": 0,
                    "hypothesis": "test",
                }
            ],
        )
        return {"generated": 1}

    def fake_stage4(**kwargs):
        paths = kwargs["paths"]
        harness._write_jsonl(
            paths.latent_cards_path,
            [{"latent_id": 0, "role": "trigger_detection"}],
        )
        _write_json(paths.summary_json_path, {"num_latents": 1})
        paths.summary_md_path.write_text("# ok\n", encoding="utf-8")
        return {"num_latents": 1}

    monkeypatch.setattr(harness, "_stage0_baseline", fake_stage0)
    monkeypatch.setattr(harness, "_stage1_encoder", fake_stage1)
    monkeypatch.setattr(harness, "_stage2_decoder", fake_stage2)
    monkeypatch.setattr(harness, "_stage3_hypotheses", fake_stage3)
    monkeypatch.setattr(harness, "_stage4_fusion", fake_stage4)

    result = harness.run_topklora_latent_harness(
        {
            "adapter_path": str(adapter),
            "model_id": "google/gemma-2-2b",
            "eval_dir": str(eval_dir),
            "output_root": str(output_root),
            "resume": False,
            "overwrite_existing_results": True,
        }
    )

    manifest = harness._read_json(Path(result["manifest_path"]))
    assert manifest["meta"]["status"] == "completed"
    for stage_name in harness.STAGE_NAMES:
        assert manifest["stages"][stage_name]["status"] == "completed"


def test_harness_marks_manifest_failed_on_hypothesis_stage_error(monkeypatch, tmp_path: Path):
    adapter, eval_dir = _make_adapter_and_eval_dirs(tmp_path)
    output_root = tmp_path / "runs"

    def fake_stage0(**kwargs):
        paths = kwargs["paths"]
        _write_json(paths.baseline_eval_path, {"asr": 0.8})
        return {"asr": 0.8}

    def fake_stage1(**kwargs):
        paths = kwargs["paths"]
        _write_json(paths.latent_index_path, {"latent_index": []})
        harness._write_jsonl(paths.encoder_metrics_path, [])
        return {"num_latents": 0}

    def fake_stage2(**kwargs):
        paths = kwargs["paths"]
        harness._write_jsonl(paths.decoder_metrics_path, [])
        harness._write_jsonl(paths.decoder_evidence_path, [])
        return {"num_latents": 0}

    def boom_stage3(**_kwargs):
        raise RuntimeError("vllm down")

    monkeypatch.setattr(harness, "_stage0_baseline", fake_stage0)
    monkeypatch.setattr(harness, "_stage1_encoder", fake_stage1)
    monkeypatch.setattr(harness, "_stage2_decoder", fake_stage2)
    monkeypatch.setattr(harness, "_stage3_hypotheses", boom_stage3)

    try:
        harness.run_topklora_latent_harness(
            {
                "adapter_path": str(adapter),
                "model_id": "google/gemma-2-2b",
                "eval_dir": str(eval_dir),
                "output_root": str(output_root),
                "resume": False,
                "overwrite_existing_results": True,
            }
        )
    except RuntimeError as exc:
        assert "vllm down" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError from stage3")

    run_dir = next((output_root).iterdir())
    manifest = harness._read_json(run_dir / "manifest.json")
    assert manifest["meta"]["status"] == "failed"
    assert manifest["meta"]["failed_stage"] == "stage3_hypotheses"
    assert manifest["stages"]["stage0_baseline"]["status"] == "completed"
    assert (run_dir / "baseline_eval.json").exists()


def test_hydra_eval_entrypoint_calls_harness(monkeypatch):
    import types

    if "evaluate" not in sys.modules:
        evaluate_stub = types.ModuleType("evaluate")
        evaluate_stub.load = lambda *_args, **_kwargs: None
        sys.modules["evaluate"] = evaluate_stub

    if "googleapiclient" not in sys.modules:
        ga_stub = types.ModuleType("googleapiclient")
        discovery_stub = types.ModuleType("googleapiclient.discovery")
        discovery_stub.build = lambda *_args, **_kwargs: None
        ga_stub.discovery = discovery_stub
        sys.modules["googleapiclient"] = ga_stub
        sys.modules["googleapiclient.discovery"] = discovery_stub

    if "ifeval" not in sys.modules:
        ifeval_stub = types.ModuleType("ifeval")
        ifeval_stub.Evaluator = object
        ifeval_stub.get_default_dataset = lambda *_args, **_kwargs: None
        ifeval_stub.instruction_registry = {}
        sys.modules["ifeval"] = ifeval_stub

    import src.evals as evals

    calls = []

    def _fake_runner(cfg):
        calls.append(cfg)

    monkeypatch.setattr(evals, "run_topklora_latent_harness", _fake_runner)

    cfg = {"evals": {"topk_lora_autointerp": {"adapter_path": "x"}}}
    eval_fn = evals.topk_lora_auto_interp()
    eval_fn(cfg)

    assert len(calls) == 1
    assert calls[0] == cfg
