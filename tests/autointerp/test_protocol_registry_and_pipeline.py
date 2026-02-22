from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.autointerp.peft_stub import install_peft_stub

install_peft_stub()

from src.autointerp.protocol import pipeline as pipeline_mod
from src.autointerp.protocol.pipeline import run_protocol_v2
from src.autointerp.protocol.stages import get_stage_specs
from src.autointerp.protocol.types import StageSpec

from tests.autointerp.helpers import TinyModel, make_cfg


class ProtocolRegistryTests(unittest.TestCase):
    def test_stage_registry_has_all_23_stages_in_order(self):
        specs = get_stage_specs()
        names = [s.name for s in specs]
        self.assertEqual(len(names), 23)
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(
            names,
            [
                "phase0.prompts_and_buckets",
                "phase0.latent_stats",
                "phase0.calibration_manifest",
                "phase0.amp_window_scan",
                "phase0.latent_selection",
                "phase1.d1_observational",
                "phase1.a4_full_ablation",
                "gate.control_window_prescreen_full",
                "gate.control_window_prescreen_sft",
                "phase2.a1_ablate_on_fire",
                "phase2.b1_force_on",
                "phase3.b3_high",
                "phase4.a3_ablate_attn_observe_mlp",
                "phase4.c2_force_attn_observe_mlp",
                "phase4.c1_cross_sublayer_cascade",
                "phase5.b2_isolate",
                "phase5.d2_encoder_decoder_alignment",
                "phase6.b3_behav",
                "phase6.c3_multi_inject",
                "phase6.b4_beta_sweep",
                "phase6.hypothesis_generation",
                "phase6.verification",
                "phase6.typology",
            ],
        )


class ProtocolPipelineResumeTests(unittest.TestCase):
    def test_resume_skips_completed_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = {"s1": 0, "s2": 0}

            def stage1(ctx):
                calls["s1"] += 1
                ctx.store.write_jsonl(
                    "prompts_master",
                    [{"prompt_id": "p1", "prompt": "hello", "bucket": "harmless"}],
                )

            def stage2(ctx):
                calls["s2"] += 1
                ctx.store.write_json("amp_window_scan", {"schema_version": 2})

            specs = [
                StageSpec(
                    name="stage.one",
                    fn=stage1,
                    requires=[],
                    produces=["prompts_master"],
                ),
                StageSpec(
                    name="stage.two",
                    fn=stage2,
                    requires=["prompts_master"],
                    produces=["amp_window_scan"],
                ),
            ]

            cfg = make_cfg(tmp, stages={"stage.one": True, "stage.two": True}, resume=True)
            model = TinyModel()

            with patch.object(pipeline_mod, "get_stage_registry", return_value=specs):
                run_protocol_v2(cfg, model, object())
                run_protocol_v2(cfg, model, object())

            self.assertEqual(calls["s1"], 1)
            self.assertEqual(calls["s2"], 1)

    def test_stage_dependency_failure_has_explicit_artifact_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            def stage1(_ctx):
                pass

            def stage2(_ctx):
                pass

            specs = [
                StageSpec(name="stage.one", fn=stage1, requires=[], produces=["prompts_master"]),
                StageSpec(
                    name="stage.two",
                    fn=stage2,
                    requires=["prompts_master"],
                    produces=["amp_window_scan"],
                ),
            ]
            cfg = make_cfg(
                tmp,
                stages={"stage.one": False, "stage.two": True},
                run_only=["stage.two"],
                resume=False,
            )
            model = TinyModel()

            with patch.object(pipeline_mod, "get_stage_registry", return_value=specs):
                with self.assertRaises(FileNotFoundError) as err:
                    run_protocol_v2(cfg, model, object())

            self.assertIn("missing required artifacts", str(err.exception))
            self.assertIn("prompts_master", str(err.exception))


if __name__ == "__main__":
    unittest.main()
