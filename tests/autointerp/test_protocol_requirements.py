from __future__ import annotations

import tempfile
import unittest

from tests.autointerp.peft_stub import install_peft_stub

install_peft_stub()

from src.autointerp.protocol import stages

from tests.autointerp.helpers import make_ctx


class ProtocolRequirementTests(unittest.TestCase):
    def test_d1_does_not_declare_judge_records_output(self):
        specs = {s.name: s for s in stages.get_stage_specs()}
        self.assertIn("phase1.d1_observational", specs)
        self.assertEqual(specs["phase1.d1_observational"].produces, ["intervention_records"])

    def test_amp_window_scan_is_required_for_amp_safe_stages(self):
        specs = {s.name: s for s in stages.get_stage_specs()}
        must_require = [
            "phase2.b1_force_on",
            "phase3.b3_high",
            "phase4.c2_force_attn_observe_mlp",
            "phase5.b2_isolate",
            "phase6.b3_behav",
        ]
        for name in must_require:
            self.assertIn(name, specs)
            self.assertIn("amp_window_scan", specs[name].requires)

    def test_b3_behav_fails_fast_without_behav_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp)
            ctx.store.write_jsonl(
                "selected_latents_main",
                [{"latent_id": 0, "adapter_name": "layer.a", "feature_idx": 0}],
            )
            ctx.store.write_jsonl(
                "prompts_master",
                [{"prompt_id": "c1", "prompt": "control", "bucket": "control"}],
            )
            ctx.store.write_jsonl(
                "latent_prompt_buckets",
                [
                    {
                        "latent_id": 0,
                        "adapter_name": "layer.a",
                        "feature_idx": 0,
                        "high_prompts": [],
                        "low_prompts": [],
                        "control_prompt_ids": ["c1"],
                        "behav_prompt_ids": [],
                    }
                ],
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
            ctx.store.write_json(
                "amp_window_scan",
                {"schema_version": 2, "amp_safe_max_factor": 2.0, "results": []},
            )

            with self.assertRaises(ValueError) as err:
                stages.stage_phase6_b3_behav(ctx)
            self.assertIn("requires BEHAV prompts", str(err.exception))


if __name__ == "__main__":
    unittest.main()
