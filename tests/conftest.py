"""Shared fixtures for the CLCD mechanical test suite.

Everything runs CPU-only on a tiny RANDOM TopKLoRA-wrapped Gemma-2: the numbers
are noise by design, so these tests assert MECHANICS / invariants, never the
(nonexistent) backdoor. Real-organism behaviour lives in src/clcd/pipeline.py.
"""

import pytest

from src.clcd.fixture import build_random_fixture


@pytest.fixture(scope="session")
def fix():
    """(model, wrapped_modules) for the tiny random fixture; built once per session."""
    return build_random_fixture(seed=0)
