from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.autointerp.peft_stub import install_peft_stub

install_peft_stub()

from src.autointerp.protocol import stages

from tests.autointerp.helpers import make_ctx


class HypothesisVerificationTypologyTests(unittest.TestCase):
    def test_hypothesis_stage_produces_hypotheses_from_intervention_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp)
            ctx.store.write_jsonl(
                "intervention_records",
                [
                    {
                        "experiment": "B1",
                        "latent_id": 7,
                        "adapter_name": "layer.a",
                        "feature_idx": 3,
                        "prompt_id": "p1",
                        "prompt_text": "Question?",
                        "baseline_text": "baseline",
                        "intervened_text": "steered",
                        "effective_amp": 2.0,
                        "intervention_mode": "enable",
                        "control_valid": True,
                    }
                ],
            )

            def fake_run_explainer(_cfg, output_dir=None, vllm_base_url=None, evidence_path=None, hypotheses_path=None, **_kwargs):
                rows = []
                with open(evidence_path, "r", encoding="utf-8") as f:
                    for line in f:
                        rows.append(json.loads(line))
                self.assertEqual(len(rows), 1)
                with open(hypotheses_path, "w", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "latent_id": rows[0]["latent_id"],
                                "adapter_name": rows[0]["adapter_name"],
                                "feature_idx": rows[0]["feature_idx"],
                                "hypothesis": "test hypothesis",
                                "behavioral_dimension": "test",
                                "effect_direction": "increases",
                            }
                        )
                        + "\n"
                    )
                return [{"latent_id": rows[0]["latent_id"]}]

            with patch.object(stages, "run_explainer", side_effect=fake_run_explainer):
                stages.stage_phase6_hypothesis_generation(ctx)

            self.assertTrue(ctx.store.exists("hypotheses"))
            hyps = ctx.store.read_jsonl("hypotheses")
            self.assertEqual(len(hyps), 1)
            self.assertEqual(hyps[0]["latent_id"], 7)

    def test_verification_stage_produces_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp)
            ctx.store.write_jsonl(
                "hypotheses",
                [
                    {
                        "latent_id": 1,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                        "hypothesis": "increases refusal",
                        "behavioral_dimension": "refusal",
                        "effect_direction": "increases",
                    }
                ],
            )
            ctx.store.write_jsonl(
                "intervention_records",
                [
                    {
                        "latent_id": 1,
                        "experiment": "B1",
                        "prompt_id": "p1",
                        "baseline_text": "a",
                        "intervened_text": "b different",
                        "control_valid": True,
                    }
                ],
            )
            # Ensure verifier uses heuristic path.
            ctx.cfg.evals.causal_autointerp_framework.llm = {
                "verifier": {"enabled": False}
            }

            stages.stage_phase6_verification(ctx)

            self.assertTrue(ctx.store.exists("verification_records"))
            self.assertTrue(ctx.store.exists("verification_metrics"))
            self.assertTrue(ctx.store.exists("verification_per_latent"))
            metrics = ctx.store.read_json("verification_metrics")
            self.assertIn("n_latents", metrics)
            self.assertIn("n_records", metrics)

    def test_typology_fails_fast_when_required_artifacts_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp)
            ctx.store.write_jsonl("intervention_records", [])
            with self.assertRaises(FileNotFoundError):
                stages.stage_phase6_typology(ctx)

            ctx.store.write_jsonl(
                "hypotheses",
                [
                    {
                        "latent_id": 1,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                        "hypothesis": "h",
                        "behavioral_dimension": "d",
                        "effect_direction": "increases",
                    }
                ],
            )
            with self.assertRaises(FileNotFoundError):
                stages.stage_phase6_typology(ctx)

    def test_typology_produces_output_when_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp)
            ctx.store.write_jsonl(
                "intervention_records",
                [
                    {
                        "latent_id": 1,
                        "experiment": "A1",
                        "prompt_id": "p1",
                        "baseline_text": "x",
                        "intervened_text": "y",
                        "control_valid": True,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                    },
                    {
                        "latent_id": 1,
                        "experiment": "B1",
                        "prompt_id": "p2",
                        "baseline_text": "x",
                        "intervened_text": "y",
                        "control_valid": True,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                    },
                    {
                        "latent_id": 1,
                        "experiment": "B3_HIGH",
                        "prompt_id": "p3",
                        "baseline_text": "x",
                        "intervened_text": "y",
                        "control_valid": True,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                    },
                    {
                        "latent_id": 1,
                        "experiment": "B3_BEHAV",
                        "prompt_id": "p4",
                        "baseline_text": "x",
                        "intervened_text": "y",
                        "control_valid": True,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                    },
                ],
            )
            ctx.store.write_jsonl("latent_stats", [{"latent_id": 1, "adapter_name": "layer.a", "feature_idx": 0}])
            ctx.store.write_jsonl("cascade_edges", [])
            ctx.store.write_jsonl(
                "hypotheses",
                [
                    {
                        "latent_id": 1,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                        "hypothesis": "h",
                        "behavioral_dimension": "d",
                        "effect_direction": "increases",
                    }
                ],
            )
            ctx.store.write_json("verification_per_latent", [{"latent_id": 1, "support_rate": 0.8, "n_records": 4}])

            stages.stage_phase6_typology(ctx)

            rows = ctx.store.read_jsonl("latent_typology")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["latent_id"], 1)


if __name__ == "__main__":
    unittest.main()
