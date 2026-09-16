"""Tests for `resolve_stop_token_ids` -- the end-of-turn stop-token fix.

A TopKLoRA sleeper ends its turn with the EOT token (`<end_of_turn>` / `<|im_end|>`) and is never
trained to emit EOS. Generation used to stop on EOS alone, so it ran past the answer into a second
malformed turn that every ASR scorer then read. Measured on 14 gemma orgs: 18 generations were
scored as backdoor fires purely on post-turn text, including both of the only two non-zero ablated
results. Full record in docs/captains-log-qwen2.5-1.5b.md §0.

Hermetic: the fake tokenizer exposes only what `_resolve_eot_token` inspects.
"""

import pytest

from src.utils import resolve_stop_token_ids


class _FakeTok:
    """Minimal stand-in for what `_resolve_eot_token` inspects."""

    def __init__(self, eos, eot_id=None, extra=None):
        self.eos_token_id = eos
        self.special_tokens_map = {}
        self.init_kwargs = {}
        self.unk_token_id = -999
        if eot_id is not None:
            self.eot_token = "<end_of_turn>"
            self.eot_token_id = eot_id
        if extra is not None:
            self.additional_special_tokens = extra

    def convert_tokens_to_ids(self, tok):
        return {"<start_of_turn>": 106, "<end_of_turn>": 107}.get(tok, self.unk_token_id)

    def convert_ids_to_tokens(self, i):
        return {106: "<start_of_turn>", 107: "<end_of_turn>"}[i]


def test_merges_eos_and_eot():
    """The fix in one assertion: the stop list contains the EOT the org emits, not just EOS."""
    assert resolve_stop_token_ids(_FakeTok(eos=1, eot_id=107)) == [1, 107]


def test_resolves_eot_from_additional_special_tokens():
    """Real gemma/Qwen tokenizers expose EOT via additional_special_tokens[1], not an attribute;
    if this path broke, the fix would silently degrade to EOS-only."""
    tok = _FakeTok(eos=1, extra=["<start_of_turn>", "<end_of_turn>"])
    assert resolve_stop_token_ids(tok) == [1, 107]


def test_is_not_a_noop_versus_eos_alone():
    """Guards the failure mode that would make the fix invisible: if the resolver returned EOS
    alone, every downstream number would stay contaminated while looking fixed."""
    tok = _FakeTok(eos=1, eot_id=107)
    assert resolve_stop_token_ids(tok) != [tok.eos_token_id]
    assert 107 in resolve_stop_token_ids(tok)


def test_deduplicates_when_eos_already_contains_eot():
    """Some configs already merge them; the list reaches generate() and artifacts, so keep it canonical."""
    assert resolve_stop_token_ids(_FakeTok(eos=[1, 107], eot_id=107)) == [1, 107]


def test_no_eot_falls_back_to_eos_but_warns(caplog):
    """Base models have no turn structure, so EOS alone is correct there -- but it is a different
    estimand, so it must be visible in the log rather than inferred."""
    with caplog.at_level("WARNING"):
        assert resolve_stop_token_ids(_FakeTok(eos=1)) == [1]
    assert any("EOT" in r.message or "INCOMPLETE" in r.message for r in caplog.records)


def test_strict_mode_raises_rather_than_degrading():
    """strict=True lets a caller demand a real EOT rather than silently getting the old behaviour."""
    with pytest.raises(RuntimeError):
        resolve_stop_token_ids(_FakeTok(eos=1), strict=True)
