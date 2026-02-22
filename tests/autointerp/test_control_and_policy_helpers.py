from __future__ import annotations

import tempfile
import unittest

from tests.autointerp.peft_stub import install_peft_stub

install_peft_stub()

from src.autointerp.protocol import stages
from src.autointerp.protocol.helpers import backoff_schedule, control_changed, normalize_output_text, resolve_amp_start

from tests.autointerp.helpers import make_ctx


class ControlAndPolicyHelperTests(unittest.TestCase):
    def test_control_normalization_uses_strip(self):
        self.assertEqual(normalize_output_text("  hello\n"), "hello")
        self.assertFalse(control_changed("answer", "answer   \n"))
        self.assertTrue(control_changed("answer", "different"))

    def test_control_generation_is_forced_greedy(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp)
            cfg = stages._control_generation_cfg(ctx)
            self.assertFalse(cfg["do_sample"])
            self.assertIsNone(cfg["temperature"])
            self.assertIsNone(cfg["top_p"])

    def test_b2_start_mode_uses_min_amp5x_amp_safe(self):
        amp = resolve_amp_start(
            experiment="B2",
            amp_1x=2.0,
            amp_5x=10.0,
            amp_safe_max=6.0,
        )
        self.assertEqual(amp, 6.0)

        amp_no_safe = resolve_amp_start(
            experiment="B2",
            amp_1x=2.0,
            amp_5x=10.0,
            amp_safe_max=None,
        )
        self.assertEqual(amp_no_safe, 10.0)

    def test_backoff_schedule_is_geometric_and_deduped(self):
        sched = backoff_schedule(8.0, [1.0, 0.5, 0.25, 0.25, 0.125])
        self.assertEqual(sched, [8.0, 4.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
