from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.autointerp.peft_stub import install_peft_stub

install_peft_stub()

from src.autointerp.protocol import helpers, stages

from tests.autointerp.helpers import make_ctx


class PrescreenSplitTests(unittest.TestCase):
    def test_full_prescreen_never_calls_sft_loader(self):
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

            with patch.object(
                stages,
                "_control_first_amp_search_full",
                return_value={
                    "selected_amp": 1.0,
                    "control_valid": True,
                    "no_valid_window": False,
                    "trace": [],
                    "start_amp": 1.0,
                },
            ), patch.object(
                stages,
                "ensure_sft_model_loaded",
                side_effect=AssertionError("full prescreen must not load M_sft"),
            ):
                stages.stage_gate_control_window_prescreen_full(ctx)

            rows = ctx.store.read_jsonl("control_windows_full")
            self.assertTrue(len(rows) >= 1)
            self.assertEqual(rows[0]["experiment"], "B1")

    def test_ensure_sft_model_uses_cfg_base_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected_path = "/tmp/expected-sft-model"
            ctx = make_ctx(tmp)
            ctx.cfg.model.base_model.path = expected_path

            tokenizer = MagicMock()
            tokenizer.apply_chat_template = None
            model = MagicMock()
            model.to.return_value = model
            model.eval.return_value = None
            model.generation_config = SimpleNamespace(eos_token_id=[1])

            with patch.object(
                helpers.AutoTokenizer,
                "from_pretrained",
                return_value=tokenizer,
            ) as tok_from_pretrained, patch.object(
                helpers.AutoModelForCausalLM,
                "from_pretrained",
                return_value=model,
            ) as model_from_pretrained, patch.object(
                helpers,
                "ensure_chat_template_and_special_tokens",
            ), patch.object(
                helpers,
                "configure_eos_eot",
            ):
                helpers.ensure_sft_model_loaded(ctx)

            tok_from_pretrained.assert_called_once_with(expected_path, use_fast=True)
            model_from_pretrained.assert_called_once()
            self.assertEqual(model_from_pretrained.call_args.args[0], expected_path)


if __name__ == "__main__":
    unittest.main()
