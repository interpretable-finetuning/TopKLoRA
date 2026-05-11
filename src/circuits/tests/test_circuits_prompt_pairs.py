"""Tests for src/circuits/prompt_pairs.py.

Spec: src/sleeper/circuit_discovery_spec_v1.md §1.

These tests use pure-Python fakes: a ``FakeTokenizer`` that implements just
``encode`` and ``apply_chat_template`` so we exercise the real chat-format glue
(``encode_prompt_ids`` / ``get_tag_token_position`` / ``get_prompt_token_lengths``)
without needing transformers or real Gemma weights. The dataset loader is
monkeypatched at module scope.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List, Optional

import pytest

from src.circuits.prompt_pairs import (
    AlignmentBuildResult,
    AlignmentError,
    AlignmentMap,
    _build_single_alignment_map,
    build_alignment_maps,
)


# ---------------------------------------------------------------------------
# Fake tokenizer: BOS + user_prefix + word-ids + user_suffix [+ model_prefix].
# ---------------------------------------------------------------------------

BOS_ID = 1
USER_PREFIX_ID = 2
USER_SUFFIX_ID = 3
MODEL_PREFIX_ID = 4
RESERVED = {BOS_ID, USER_PREFIX_ID, USER_SUFFIX_ID, MODEL_PREFIX_ID}


class FakeTokenizer:
    """Minimal tokenizer honouring the interface expected by chat_format.

    ``encode(text, add_special_tokens=False)`` tokenises by splitting on any
    chained ``tag_overrides`` first (exact substring match), then on
    whitespace. Integer ids are assigned deterministically from a per-instance
    dictionary that maps token text → id. ``apply_chat_template`` produces
    ``[BOS, USER_PREFIX, *user_ids, USER_SUFFIX, (MODEL_PREFIX if gen)]``.
    """

    name_or_path = "fake-tokenizer"

    def __init__(
        self,
        *,
        tag_overrides: Optional[Dict[str, List[int]]] = None,
        post_user_injection: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        # Reserved ids must not collide with arbitrary word ids.
        self._next_id = 1000
        self._vocab: Dict[str, int] = {}
        self._tag_overrides = dict(tag_overrides or {})
        # post_user_injection is used to simulate tokenizer bugs that return
        # *different* post-trigger ids depending on which tag preceded the
        # user content; keyed by tag string.
        self._post_user_injection = dict(post_user_injection or {})

    # --- encode helpers --------------------------------------------------
    def _id_for(self, token: str) -> int:
        if token not in self._vocab:
            self._next_id += 1
            while self._next_id in RESERVED:
                self._next_id += 1
            self._vocab[token] = self._next_id
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        # Honour override for known tag strings regardless of whitespace.
        stripped = text.strip()
        if stripped in self._tag_overrides:
            ids = list(self._tag_overrides[stripped])
        else:
            # Split tag-like prefixes out of the text first (multi-token tag
            # may appear inside user content together with the question).
            remaining = stripped
            ids: List[int] = []
            changed = True
            while changed:
                changed = False
                for tag, tag_ids in self._tag_overrides.items():
                    if remaining.startswith(tag):
                        ids.extend(tag_ids)
                        remaining = remaining[len(tag) :].lstrip("\n ")
                        changed = True
                        break
            # Then whitespace-split the remainder.
            if remaining:
                for word in remaining.split():
                    ids.append(self._id_for(word))

        if add_special_tokens:
            return [BOS_ID, *ids]
        return ids

    # --- chat template ---------------------------------------------------
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        # We only support tokenize=True here (chat_format.encode_prompt_ids /
        # get_prompt_token_lengths only ever request tokenized output).
        assert tokenize is True, "FakeTokenizer only supports tokenize=True"
        assert messages and messages[0]["role"] == "user"
        user_content = messages[0]["content"]
        user_ids = self.encode(user_content, add_special_tokens=False)

        # Honour any post-user injection hook (used by the mismatch tests to
        # simulate tokenizer bugs that add extra tokens conditional on tag).
        extra: List[int] = []
        for tag, injection in self._post_user_injection.items():
            if user_content.startswith(tag):
                extra = list(injection)
                break

        out = [BOS_ID, USER_PREFIX_ID, *user_ids, *extra, USER_SUFFIX_ID]
        if add_generation_prompt:
            out.append(MODEL_PREFIX_ID)
        return out


# ---------------------------------------------------------------------------
# Single-pair alignment tests.
# ---------------------------------------------------------------------------


def test_pre_trigger_and_post_trigger_equality_assertions_hold() -> None:
    tokenizer = FakeTokenizer(tag_overrides={"|T|": [100], "|D|": [200]})
    clean_rec = {"question": "what", "tag": "|T|"}
    trig_rec = {"question": "what", "tag": "|D|"}

    result = _build_single_alignment_map(
        tokenizer, "inst_A", clean_rec, trig_rec, "|T|", "|D|"
    )

    # Both tags are one token, so the canonical trigger position is identical.
    assert result.canonical_trigger_pos["clean"] == result.canonical_trigger_pos["trig"]

    clean_span = result.spans["clean"]
    trig_span = result.spans["trig"]

    # Pre-trigger span is non-empty and the token ids agree.
    pre_end = clean_span["pre_trigger"][1]
    assert pre_end > 0
    assert trig_span["pre_trigger"][1] == pre_end
    assert result.clean_input_ids[:pre_end] == result.trig_input_ids[:pre_end]

    # Post-trigger span token ids agree.
    clean_post = clean_span["post_trigger_prompt"]
    trig_post = trig_span["post_trigger_prompt"]
    assert (
        result.clean_input_ids[clean_post[0] : clean_post[1]]
        == result.trig_input_ids[trig_post[0] : trig_post[1]]
    )

    # response_start_pos equals prompt length.
    assert result.response_start_pos["clean"] == len(result.clean_input_ids)
    assert result.response_start_pos["trig"] == len(result.trig_input_ids)


def test_handles_asymmetric_tag_token_lengths() -> None:
    tokenizer = FakeTokenizer(
        tag_overrides={
            "|TRAIN|": [100, 101],
            "|DEPLOY|": [200, 201, 202],
        }
    )
    clean_rec = {"question": "what", "tag": "|TRAIN|"}
    trig_rec = {"question": "what", "tag": "|DEPLOY|"}

    result = _build_single_alignment_map(
        tokenizer, "inst_B", clean_rec, trig_rec, "|TRAIN|", "|DEPLOY|"
    )

    # Trigger tag is one token longer -> canonical index +1 vs clean.
    delta_canonical = (
        result.canonical_trigger_pos["trig"] - result.canonical_trigger_pos["clean"]
    )
    assert delta_canonical == 1

    # Pre-trigger length (tag_start) matches across runs.
    assert (
        result.spans["clean"]["pre_trigger"][1]
        == result.spans["trig"]["pre_trigger"][1]
    )

    # Post-trigger content matches.
    clean_post = result.spans["clean"]["post_trigger_prompt"]
    trig_post = result.spans["trig"]["post_trigger_prompt"]
    assert (
        result.clean_input_ids[clean_post[0] : clean_post[1]]
        == result.trig_input_ids[trig_post[0] : trig_post[1]]
    )

    # Prompt length differs by exactly one token.
    assert (
        result.response_start_pos["trig"] - result.response_start_pos["clean"] == 1
    )


def test_pre_trigger_mismatch_raises() -> None:
    """A tokenizer bug that changes pre-trigger tokens should surface as
    ``reason='pre_trigger_mismatch'``."""

    class BuggyPreTriggerTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            ids = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            # Inject an extra token at index 1 (inside the pre-trigger span)
            # only when the user content starts with the trigger tag. This
            # asymmetrically shifts the "trig" path before the tag.
            user_content = messages[0]["content"]
            if user_content.startswith("|D|"):
                ids = [ids[0], 999, *ids[1:]]
            return ids

    tokenizer = BuggyPreTriggerTokenizer(tag_overrides={"|T|": [100], "|D|": [200]})
    clean_rec = {"question": "hello", "tag": "|T|"}
    trig_rec = {"question": "hello", "tag": "|D|"}

    with pytest.raises(AlignmentError) as exc_info:
        _build_single_alignment_map(
            tokenizer, "inst_C", clean_rec, trig_rec, "|T|", "|D|"
        )
    assert exc_info.value.reason == "pre_trigger_mismatch"


def test_post_trigger_mismatch_raises() -> None:
    """Different post-trigger ids (e.g. different question) must raise
    ``reason='post_trigger_mismatch'``."""

    tokenizer = FakeTokenizer(tag_overrides={"|T|": [100], "|D|": [200]})
    clean_rec = {"question": "alpha", "tag": "|T|"}
    trig_rec = {"question": "beta", "tag": "|D|"}

    with pytest.raises(AlignmentError) as exc_info:
        _build_single_alignment_map(
            tokenizer, "inst_D", clean_rec, trig_rec, "|T|", "|D|"
        )
    assert exc_info.value.reason == "post_trigger_mismatch"


# ---------------------------------------------------------------------------
# build_alignment_maps end-to-end tests (with monkeypatched dataset loader).
# ---------------------------------------------------------------------------


def _make_fake_dataset(n: int) -> Dict[str, List[Dict[str, Any]]]:
    """Build a dict-like mimicking ``DatasetDict`` with just the fields we read.

    The records are iterable lists of dicts; ``build_alignment_maps`` iterates
    each split to build ``{instruction_id: record}`` index dicts.
    """
    clean_records = [
        {"instruction_id": f"eval_{i:05d}", "question": f"question_{i}", "tag": "|T|"}
        for i in range(n)
    ]
    trig_records = [
        {"instruction_id": f"eval_{i:05d}", "question": f"question_{i}", "tag": "|D|"}
        for i in range(n)
    ]
    return {"eval_clean": clean_records, "eval_triggered": trig_records}


@pytest.fixture
def patched_loader(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch ``datasets.load_from_disk`` and ``validate_dataset_metadata``.

    Returns a helper that installs a specific fake dataset; each test calls it
    with its desired ``n``.
    """
    import src.circuits.prompt_pairs as pp

    def _install(n: int):
        fake_ds = _make_fake_dataset(n)

        def fake_validate(path):
            return {"clean_tag": "|T|", "trigger_tag": "|D|"}

        monkeypatch.setattr(pp, "validate_dataset_metadata", fake_validate)

        fake_datasets_mod = types.SimpleNamespace(load_from_disk=lambda _p: fake_ds)
        # ``build_alignment_maps`` imports ``datasets`` at call time; shim
        # sys.modules so the local ``from datasets import load_from_disk``
        # resolves to our fake.
        import sys

        monkeypatch.setitem(sys.modules, "datasets", fake_datasets_mod)
        return fake_ds

    return _install


def test_build_alignment_maps_split_is_deterministic(patched_loader) -> None:
    patched_loader(10)
    tokenizer = FakeTokenizer(tag_overrides={"|T|": [100], "|D|": [200]})

    r1 = build_alignment_maps(tokenizer, "fake_dir", seed=42)
    r2 = build_alignment_maps(tokenizer, "fake_dir", seed=42)

    ids_r1 = [m.instruction_id for m in r1.discovery], [m.instruction_id for m in r1.holdout]
    ids_r2 = [m.instruction_id for m in r2.discovery], [m.instruction_id for m in r2.holdout]
    assert ids_r1 == ids_r2

    r3 = build_alignment_maps(tokenizer, "fake_dir", seed=43)
    ids_r3 = [m.instruction_id for m in r3.discovery], [m.instruction_id for m in r3.holdout]
    # Different seeds should (with very high probability for n=10) differ.
    assert ids_r3 != ids_r1


def test_build_alignment_maps_returns_correct_split_sizes(patched_loader) -> None:
    patched_loader(10)
    tokenizer = FakeTokenizer(tag_overrides={"|T|": [100], "|D|": [200]})

    r_half = build_alignment_maps(tokenizer, "fake_dir", discovery_fraction=0.5, seed=0)
    assert len(r_half.discovery) == 5
    assert len(r_half.holdout) == 5
    assert r_half.stats["num_candidate_pairs"] == 10
    assert r_half.stats["num_built"] == 10
    assert r_half.stats["num_skipped"] == 0

    r_seven = build_alignment_maps(
        tokenizer, "fake_dir", discovery_fraction=0.7, seed=0
    )
    assert len(r_seven.discovery) == 7
    assert len(r_seven.holdout) == 3
