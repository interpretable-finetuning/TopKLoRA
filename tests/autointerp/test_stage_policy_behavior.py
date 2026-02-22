from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from tests.autointerp.peft_stub import install_peft_stub

install_peft_stub()

from src.autointerp.protocol import stages

from tests.autointerp.helpers import make_cfg, make_ctx


class StagePolicyBehaviorTests(unittest.TestCase):
    def test_a1_ablation_does_not_invoke_backoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp)
            ctx.store.write_jsonl(
                "selected_latents_main",
                [
                    {
                        "latent_id": 0,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                    }
                ],
            )
            ctx.store.write_jsonl(
                "prompts_master",
                [
                    {"prompt_id": "high1", "prompt": "p-high", "bucket": "harmless"},
                    {"prompt_id": "ctrl1", "prompt": "p-ctrl", "bucket": "control"},
                ],
            )
            ctx.store.write_jsonl(
                "latent_prompt_buckets",
                [
                    {
                        "latent_id": 0,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                        "high_prompts": [{"prompt_id": "high1", "activation": 0.9}],
                        "low_prompts": [],
                        "control_prompt_ids": ["ctrl1"],
                        "behav_prompt_ids": [],
                    }
                ],
            )

            with patch.object(
                stages,
                "_control_first_amp_search_full",
                side_effect=AssertionError("A1 should not call backoff search"),
            ), patch.object(
                stages,
                "_baseline_and_intervention_full",
                return_value=({"ctrl1": "same"}, {"ctrl1": "same"}),
            ), patch.object(
                stages,
                "_apply_single_latent_experiment_full",
                return_value=[],
            ), patch.object(
                stages,
                "_append_records",
            ), patch.object(
                stages,
                "_append_judge_records",
            ):
                stages.stage_phase2_a1_ablate_on_fire(ctx)

    def test_force_backoff_search_finds_valid_window_then_marks_no_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp)
            latent = {"adapter_name": "layer.a", "feature_idx": 0}
            control_recs = [{"prompt_id": "c1", "prompt": "control", "bucket": "control"}]

            def fake_intervention(*_args, **kwargs):
                amp = float(kwargs["amplification"])
                if amp >= 1.0:
                    return {"c1": "changed"}
                return {"c1": "base"}

            with patch.object(stages, "_generate_texts", return_value={"c1": "base"}), patch.object(
                stages,
                "_intervention_only_full",
                side_effect=fake_intervention,
            ):
                result = stages._control_first_amp_search_full(
                    ctx,
                    latent_entry=latent,
                    experiment="B1",
                    control_recs=control_recs,
                    start_amp=1.0,
                )
            self.assertTrue(result["control_valid"])
            self.assertFalse(result["no_valid_window"])
            self.assertEqual(result["selected_amp"], 0.5)

            with patch.object(stages, "_generate_texts", return_value={"c1": "base"}), patch.object(
                stages,
                "_intervention_only_full",
                return_value={"c1": "changed"},
            ):
                result2 = stages._control_first_amp_search_full(
                    ctx,
                    latent_entry=latent,
                    experiment="B1",
                    control_recs=control_recs,
                    start_amp=1.0,
                )
            self.assertFalse(result2["control_valid"])
            self.assertTrue(result2["no_valid_window"])
            self.assertIsNone(result2["selected_amp"])

    def test_phase0_sweep_runs_all_factors_despite_control_deterioration(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, make_cfg(tmp))
            # Override stage config for deterministic short sweep.
            ctx.cfg.evals.causal_autointerp_framework.stage_configs = {
                "phase0.amp_window_scan": {
                    "scan": {
                        "n_latents": 1,
                        "high_prompts_per_latent": 1,
                        "factors": [0.5, 1.0, 2.0],
                    }
                }
            }

            ctx.store.write_jsonl(
                "latent_stats",
                [{"latent_id": 0, "adapter_name": "layer.a", "feature_idx": 0, "p_active": 0.1}],
            )
            ctx.store.write_jsonl(
                "calibration_manifest",
                [
                    {
                        "latent_id": 0,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                        "z_j_typical": 1.0,
                        "amp_1x": 1.0,
                        "amp_5x": 5.0,
                        "amp_10x": 10.0,
                    }
                ],
            )
            ctx.store.write_jsonl(
                "latent_index",
                [{"latent_id": 0, "adapter_name": "layer.a", "feature_idx": 0}],
            )
            ctx.store.write_jsonl(
                "prompts_master",
                [
                    {"prompt_id": "h1", "prompt": "high", "bucket": "harmless"},
                    {"prompt_id": "c1", "prompt": "control", "bucket": "control"},
                ],
            )
            ctx.store.write_jsonl(
                "latent_prompt_buckets",
                [
                    {
                        "latent_id": 0,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                        "high_prompts": [{"prompt_id": "h1", "activation": 0.8}],
                        "low_prompts": [],
                        "control_prompt_ids": ["c1"],
                        "behav_prompt_ids": [],
                    }
                ],
            )

            def fake_baseline_and_intervention(_ctx, prompt_rows, *_args, **kwargs):
                amp = float(kwargs["amplification"])
                pids = [p["prompt_id"] for p in prompt_rows]
                baseline = {pid: "base" for pid in pids}
                if "c1" in pids:
                    # Control deteriorates at larger factors.
                    steered = {pid: ("changed" if amp >= 1.0 else "base") for pid in pids}
                else:
                    steered = {pid: ("changed" if amp >= 0.5 else "base") for pid in pids}
                return baseline, steered

            with patch.object(
                stages,
                "_baseline_and_intervention_full",
                side_effect=fake_baseline_and_intervention,
            ), patch.object(
                stages,
                "_generate_texts",
                return_value={"c1": "base"},
            ):
                stages.stage_phase0_amp_window_scan(ctx)

            scan = ctx.store.read_json("amp_window_scan")
            self.assertEqual(scan["n_sampled"], 1)
            self.assertEqual(len(scan["results"]), 3)
            factors = [r["factor"] for r in scan["results"]]
            self.assertEqual(factors, [0.5, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
