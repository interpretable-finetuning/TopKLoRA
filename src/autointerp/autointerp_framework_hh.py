"""Compatibility entrypoint for causal autointerp protocol v2."""

from __future__ import annotations

import logging
from typing import Any, Dict

from .protocol import run_protocol_v2

logger = logging.getLogger(__name__)


def run_autointerp_framework(cfg: Any, model: Any, tokenizer: Any) -> Dict[str, Any]:
    """
    Run the TopKLoRA Conditional Steering Protocol v2.

    This preserves the legacy public entrypoint used by `src/evals.py`.
    """
    logger.info("Launching causal autointerp protocol v2")
    return run_protocol_v2(cfg, model, tokenizer)

