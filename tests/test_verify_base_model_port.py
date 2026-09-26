"""The base-model port preflight must reject the tag pairs that broke attribution before.

These tests pin VALUES, not shapes. The measured invariant is the DIFFERING SPAN between the
trigger and the clean tag -- not how many tokens each tag is -- because CLCD's aligner needs a
1-1 pairing over the span that differs. A test asserting "the span is an int" or "the two tags
tokenize differently" would stay green through the exact regression this guards: the gemma-era
pair silently producing a length artifact on a Qwen tokenizer.
"""

import pytest

from src.clcd.verify_tag_span import differing_span

transformers = pytest.importorskip("transformers")

QWEN = "Qwen/Qwen2.5-1.5B"


@pytest.fixture(scope="module")
def qwen_tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    except OSError:
        pytest.skip(f"{QWEN} tokenizer not in the local HF cache")


def ids(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def test_run_train_pair_differs_over_exactly_one_token(qwen_tok):
    """|RUN|/|TRAIN| is the pair the 1.5B study was built on; its span must be width 1.

    Both tags are THREE tokens ([91, 47390, 91] / [91, 98258, 91]) and share the `|` on either
    side. A check that measured raw token count would call this 3 and reject the working pair,
    which is how this test earns its keep.
    """
    t, c = ids(qwen_tok, "|RUN|"), ids(qwen_tok, "|TRAIN|")
    assert len(t) == len(c) == 3, f"tag tokenization changed: {t} / {c}"

    span = differing_span(t, c)
    assert span["mid_t"] == 1
    assert span["mid_c"] == 1
    assert span["lcp"] == 1, "the leading `|` must be shared, or the span is not the tag body"


def test_gemma_era_pair_is_width_2_on_qwen_not_unequal(qwen_tok):
    """|TRIGGER|/|TRAINING| must be rejected on Qwen -- but for the reason that is actually true.

    MEASURED, 2026-09-20: on the Qwen tokenizer both tags are FOUR tokens
    (['|','TR','IGGER','|'] and ['|','TRAIN','ING','|']) and their differing spans are EQUAL at
    width 2, through `encode_prompt_ids` as well as bare. The aligner therefore has a clean 1-1
    pairing and the length-artifact failure does NOT apply to this pair.

    This contradicts src/clcd/verify_tag_span.py's docstring, which states the pair "become 4 and
    3" on Qwen. The pair is still correctly rejected -- width 2 is not width 1, so a two-token
    trigger spreads the contrast the aligner attributes over -- but it is rejected by the WIDTH
    assertion, not the equality one. Pinning the real numbers here so the next port is not planned
    against the wrong rationale.
    """
    t, c = ids(qwen_tok, "|TRIGGER|"), ids(qwen_tok, "|TRAINING|")
    assert len(t) == len(c) == 4, f"tag tokenization changed: {t} / {c}"

    span = differing_span(t, c)
    assert span["mid_t"] == span["mid_c"] == 2, (
        f"expected equal width-2 spans on Qwen, got {span}"
    )


def test_unequal_spans_are_detectable(qwen_tok):
    """The equality branch must be reachable, or it is not a check.

    |RUN| (3 tokens) against |TRAINING| (4) is the shape that genuinely breaks the aligner: no
    1-1 pairing over the differing span, so attribution picks up a length artifact. No pair the
    study ever used does this, which is exactly why it needs a constructed case.
    """
    t, c = ids(qwen_tok, "|RUN|"), ids(qwen_tok, "|TRAINING|")
    span = differing_span(t, c)
    assert span["mid_t"] != span["mid_c"], f"expected unequal spans, got {span}"
    assert (span["mid_t"], span["mid_c"]) == (1, 2)


def test_identical_tags_raise_rather_than_report_width_zero(qwen_tok):
    """A tag that never reaches the prompt must raise, not return 0.

    Width 0 makes every alignment trivially 'correct' while measuring nothing -- the same class
    of bug as an empty eval band reporting ASR 0.0, which IS the success value.
    """
    t = ids(qwen_tok, "|RUN|")
    with pytest.raises(AssertionError, match="NO differing span"):
        differing_span(t, list(t))
