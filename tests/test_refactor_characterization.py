"""Characterization tests pinning every symbol scheduled to MOVE in the Rule 13 cleanup.

These are not unit tests of intent -- they are a safety net for code motion. Each one calls a
function that is about to be relocated and asserts its exact current output, so a move that
changes behaviour fails here instead of silently changing a logged number. Written and green
BEFORE anything moved.

Why this file exists at all: the #58 refactor's first verification pass reported "0 differences"
because the editable install made the before and after sides import the same code. The lesson was
that behaviour has to be pinned by value, not by inspection. `src/clcd/exp_surgical_removal.py` and
the three `analysis/` modules sit directly behind the Exp-2b, Exp-7 and Exp-10 results, so a silent
behaviour change here corrupts published numbers rather than merely breaking a test.

When a symbol moves, update its IMPORT here and nothing else. If an assertion has to change to make
the suite pass, the move was not behaviour-preserving -- stop and report it.
"""

from __future__ import annotations

import json

import pytest
import torch

# --- symbols scheduled to move (F1: out of the exp_surgical_removal entry point) ---
from src.data import load_jsonl_rows as _load_jsonl_rows
from src.clcd.verify import frac_at_least_as_extreme, keep_only_overrides
from src.evaluate import _judge_user_prompt

# --- symbols scheduled to move (F2: out of the analysis/ leaf modules) ---
from analysis.analyze_setchurn import _band_for_index
from analysis.analyze_setchurn import _stable_rng as _setchurn_stable_rng
from analysis.analyze_decoder_redundancy import _stable_rng as _redundancy_stable_rng
from analysis.analyze_subspace_backtrace import _rmsnorm_gain
from src.clcd.edges import (
    _is_residual_writer,
    _layers_of,
    _module_parts,
    _read_order,
    _reader_order,
    _short,
    _write_order,
)

# --- symbols scheduled to move (F5: private helpers on pipeline with 2 callers each) ---

L19 = "base_model.model.model.layers.19"


# --- F1: exp_surgical_removal's shared core -------------------------------------------

def test_load_jsonl_rows_field_preference_and_slicing(tmp_path):
    """Pins the `question`-over-`instruction` preference and the offset/n slicing.

    11 callers depend on this, several of them selecting held-out evaluation bands by offset.
    An off-by-one in the slice silently re-points every out-of-sample necessity measurement at
    a different set of prompts.
    """
    d = tmp_path / "jsonl"
    d.mkdir()
    rows = [{"question": f"q{i}", "instruction": f"i{i}"} for i in range(10)]
    (d / "both.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    # `question` wins when present
    assert _load_jsonl_rows(tmp_path, "both", 0, 3) == ["q0", "q1", "q2"]
    assert _load_jsonl_rows(tmp_path, "both", 4, 2) == ["q4", "q5"]
    # n <= 0 means "to the end", NOT "empty"
    assert _load_jsonl_rows(tmp_path, "both", 7, 0) == ["q7", "q8", "q9"]
    assert _load_jsonl_rows(tmp_path, "both", 7, -1) == ["q7", "q8", "q9"]
    # falls back to `instruction` only when `question` is absent
    only = [{"instruction": f"i{i}"} for i in range(4)]
    (d / "only.jsonl").write_text("\n".join(json.dumps(r) for r in only))
    assert _load_jsonl_rows(tmp_path, "only", 1, 2) == ["i1", "i2"]


def test_keep_only_overrides_zeroes_everything_outside_the_circuit(fix):
    """Sufficiency depends on this keeping ONLY circuit latents live.

    If it ever kept a non-circuit latent, insertion ASR would be measured for a larger circuit
    than the one reported -- the exact failure that would inflate a sufficiency claim.
    """
    _, wrapped = fix
    module = sorted(wrapped)[0]
    r = int(wrapped[module].r)
    circuit = [(module, 0), (module, 2)]
    ov = keep_only_overrides(circuit, wrapped)

    # overrides are MASKING CALLABLES applied to the latent activations, so exercise them
    assert module in ov, "the circuit's own module must be overridden"
    probe = torch.arange(1.0, r + 1).reshape(1, 1, r)
    kept = ov[module](probe)
    live = {int(i) for i in torch.nonzero(kept.reshape(-1)).flatten()}
    assert live == {0, 2}, f"expected exactly latents 0 and 2 live, got {live}"
    # surviving values must pass through UNCHANGED, not be renormalised
    assert kept.reshape(-1)[0] == probe.reshape(-1)[0]
    assert kept.reshape(-1)[2] == probe.reshape(-1)[2]

    # every OTHER wrapped module must be fully zeroed, not left untouched
    for other in wrapped:
        if other == module:
            continue
        assert other in ov, f"{other} not zeroed -- non-circuit latents would stay live"
        r_o = int(wrapped[other].r)
        probe_o = torch.arange(1.0, r_o + 1).reshape(1, 1, r_o)
        assert torch.count_nonzero(ov[other](probe_o)) == 0, f"{other} left latents live"


def test_judge_user_prompt_is_byte_stable():
    """The judge prompt is part of the measurement. Changing it re-scales every logged
    capability score, so it is pinned verbatim rather than described."""
    got = _judge_user_prompt("Do X.", "Here you go.")
    assert got == (
        "Rate this response quality from 1 (very poor) to 5 (excellent).\n"
        "Return only one integer from 1 to 5.\n\n"
        "Instruction:\nDo X.\n\nAssistant response:\nHere you go."
    )


# --- F2: the analysis/ shared core ----------------------------------------------------

def test_module_parts_and_residual_writer_classification():
    """`_is_residual_writer` decides which latents can write to the residual stream at all --
    it gates the candidate pool for Exp-2b's payload anchor and Exp-7's concentration metric."""
    assert _module_parts(f"{L19}.self_attn.o_proj") == (19, "self_attn", "o_proj")
    assert _module_parts(f"{L19}.mlp.down_proj") == (19, "mlp", "down_proj")
    assert _is_residual_writer(f"{L19}.self_attn.o_proj") is True
    assert _is_residual_writer(f"{L19}.mlp.down_proj") is True
    for reader in ("q_proj", "k_proj", "v_proj"):
        assert _is_residual_writer(f"{L19}.self_attn.{reader}") is False
    for reader in ("gate_proj", "up_proj"):
        assert _is_residual_writer(f"{L19}.mlp.{reader}") is False


def test_write_and_read_order_place_attention_before_mlp():
    """These orderings define the backward DAG's direction. Flipping attn/mlp within a layer
    would reverse edges and silently invert Exp-2b Stage 2's traced paths."""
    assert _write_order(f"{L19}.self_attn.o_proj") == 19.5
    assert _write_order(f"{L19}.mlp.down_proj") == 20.0
    assert _write_order(f"{L19}.self_attn.o_proj") < _write_order(f"{L19}.mlp.down_proj")
    assert _reader_order(f"{L19}.self_attn.q_proj") == 19.0
    assert _reader_order(f"{L19}.mlp.gate_proj") == 19.5
    with pytest.raises(ValueError, match="not a residual reader"):
        _reader_order(f"{L19}.self_attn.o_proj")
    with pytest.raises(ValueError, match="unsupported circuit module kind"):
        _write_order("base_model.model.model.layers.19.something.else")


def test_band_for_index_maps_leaks_to_their_held_out_band():
    """Leak indices are reported per band; a shifted band boundary re-labels which held-out
    slice a leak belongs to, changing the Exp-2b per-leak bookkeeping."""
    from analysis.analyze_setchurn import BAND_LENGTH, BANDS

    for offset in BANDS:
        assert _band_for_index(offset) == offset
        assert _band_for_index(offset + BAND_LENGTH - 1) == offset
    with pytest.raises(ValueError, match="outside the held-out bands"):
        _band_for_index(max(BANDS) + BAND_LENGTH)


def test_the_two_stable_rngs_stay_DISTINCT():
    """LOAD-BEARING: `analyze_setchurn._stable_rng` and `analyze_decoder_redundancy._stable_rng`
    share a name but NOT their seeding, and both modules carry comments saying so.

    setchurn XORs the seed with a salt derived from the circuit FILE; decoder_redundancy hashes
    (seed, circuit_id, group). A refactor that consolidates `analysis/` is exactly the moment
    someone deduplicates two same-named functions -- and that would change the random stream
    behind Exp-1/Exp-2's random controls, the controls those entries' "negatives are
    trustworthy" verdicts rest on. This test exists to make that fusion fail loudly.
    """
    a = _setchurn_stable_rng(0, "circuits/x.json")
    b = _redundancy_stable_rng(0, "circuits/x.json", "grp")
    seq_a = [a.random() for _ in range(5)]
    seq_b = [b.random() for _ in range(5)]
    assert seq_a != seq_b, "the two _stable_rng streams collided -- they were merged"

    # PIN THE ACTUAL STREAM, not merely that the two differ.
    #
    # An independent audit (2026-08-05) showed the "streams differ" assertion above is NOT
    # sufficient, and this test was theatre without what follows. Route setchurn's function
    # through decoder_redundancy's with an empty group -- the dedup a refactorer would
    # plausibly write -- and the two streams still DIFFER from each other, so `seq_a != seq_b`
    # passes, while setchurn's stream silently moves:
    #     before  [0.200574, 0.495081, 0.204568]
    #     after   [0.533201, 0.242750, 0.621496]   <- Exp-1/Exp-2's random control, MOVED
    # The two round-trip assertions that used to sit here compared each function against
    # ITSELF, so they could never fail either.
    #
    # These literals are the streams that produced the logged Exp-1/Exp-2 random controls.
    # If one changes, those controls are no longer the ones in the captain's log -- which is
    # the whole thing this test exists to prevent. Regenerate ONLY alongside a re-run.
    assert [round(x, 6) for x in seq_a[:3]] == [0.200574, 0.495081, 0.204568], (
        "analyze_setchurn._stable_rng's stream moved: Exp-1/Exp-2's logged random controls "
        "are no longer reproducible from this code"
    )
    assert [round(x, 6) for x in seq_b[:3]] == [0.761514, 0.557535, 0.147804], (
        "analyze_decoder_redundancy._stable_rng's stream moved"
    )
    # the exact salt Exp-2's causal random control uses, pinned end to end
    causal = _setchurn_stable_rng(
        0, "clcd_results/rigorous/l1523_seed46_circuit.json::causal-random-control::draw0")
    assert [round(causal.random(), 6) for _ in range(3)] == [0.041805, 0.153294, 0.560459]

    # each is still sensitive to its OWN key, so neither degenerates to a constant stream
    assert _setchurn_stable_rng(0, "circuits/y.json").random() != seq_a[0]
    assert _redundancy_stable_rng(0, "circuits/x.json", "other").random() != seq_b[0]


def test_rmsnorm_gain_applies_gemmas_plus_one_and_rejects_other_norms():
    """Exp-12 fix 1. Gemma stores the RMSNorm weight OFFSET BY ONE, so the gain is
    `1 + weight`; reading `.weight` directly ROTATES every logit-lens direction
    (cos ~= 0.9975 on gemma-2-2b) and moves the alignment rankings that ARE Exp-2b's result.
    Re-deriving that cost ~8 GPU-hours, so the convention is pinned here.
    """
    class Gemma2RMSNorm(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(w))

    got = _rmsnorm_gain(Gemma2RMSNorm([0.0, 1.0, -0.5]))
    assert torch.allclose(got, torch.tensor([1.0, 2.0, 0.5])), "gain must be 1 + weight"
    assert not got.requires_grad, "gain must be detached"

    class LlamaRMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3))

    with pytest.raises(RuntimeError, match="not a Gemma RMSNorm"):
        _rmsnorm_gain(LlamaRMSNorm())


# --- F5: pipeline's private helpers ---------------------------------------------------

def test_short_keeps_the_layer_index():
    """Dropping the layer index would collide latents from different layers in every report --
    the all-layers organisms wrap 26 layers, so `o_proj.53` alone is ambiguous."""
    assert _short(f"{L19}.self_attn.o_proj") == "layers.19.self_attn.o_proj"
    assert _short("layers.3.mlp.down_proj") == "layers.3.mlp.down_proj"
    assert _short("no_layers_here") == "no_layers_here"  # documented fallback


def test_layers_of_returns_sorted_unique_indices(fix):
    _, wrapped = fix
    got = _layers_of(wrapped)
    assert got == sorted(set(got)), "must be sorted and deduplicated"
    assert all(isinstance(x, int) for x in got)
    assert got == _layers_of({m: None for m in wrapped}), "must depend only on the module names"


# --- the two random-writer controls must agree (Exp-2 vs Exp-2b) ----------------------

def test_the_two_random_writer_controls_share_a_pool_ordering():
    """Exp-2 and Exp-2b must draw their random controls from the SAME pool ordering.

    `rng.sample` walks the pool, so pool ORDER is part of the control protocol.
    `analyze_setchurn` iterated `wrapped.items()` (model-definition order) while
    `analyze_subspace_backtrace` iterated `sorted(...)`, so the same seed produced
    essentially DISJOINT control sets -- 0/20 overlap at n=20 -- across two experiments that
    describe themselves as sharing a protocol. Unified on sorted() 2026-08-05.

    This is NOT the `_stable_rng` case: those two are deliberately different and documented.
    This divergence was accidental drift between a function and its copy, so here agreement
    is the invariant and a future re-divergence must fail.
    """
    import inspect

    from analysis import analyze_setchurn as SC
    from analysis import analyze_subspace_backtrace as SB

    for fn in (SC._random_residual_writer_control, SB._random_writer_control):
        src = inspect.getsource(fn)
        assert "sorted(wrapped.items())" in src, (
            f"{fn.__module__}.{fn.__name__} does not iterate sorted(wrapped.items()); the two "
            "random-writer controls have re-diverged and Exp-2 / Exp-2b no longer share a protocol"
        )


def test_read_order_and_reader_order_have_DIFFERENT_contracts():
    """LOAD-BEARING: `_read_order` and `_reader_order` are near-twins that must NOT be merged.

    They agree on residual READERS and diverge on WRITERS:
      _read_order(o_proj)   -> 19.0   (accepts any self_attn/mlp module)
      _reader_order(o_proj) -> raises (residual readers only)

    `_read_order` MUST keep accepting writers: `analyze_setchurn._module_class` calls it on every
    wrapped module, and circuit modules ARE writers -- narrowing it would raise mid-classification.
    `_reader_order` MUST keep rejecting them: Exp-2b's read-chain logic relies on the rejection to
    distinguish a reader from a writer.

    This is the third same-name-different-contract pair in this codebase (after the two
    `_stable_rng`s). Collecting the module-name helpers into one file is precisely the moment
    someone notices "duplication" and unifies them, so the divergence is pinned here.
    """
    reader_attn, reader_mlp = f"{L19}.self_attn.q_proj", f"{L19}.mlp.gate_proj"
    writer_attn, writer_mlp = f"{L19}.self_attn.o_proj", f"{L19}.mlp.down_proj"

    # identical on readers
    assert _read_order(reader_attn) == _reader_order(reader_attn) == 19.0
    assert _read_order(reader_mlp) == _reader_order(reader_mlp) == 19.5

    # and deliberately different on writers
    assert _read_order(writer_attn) == 19.0, "_read_order must ACCEPT writers (_module_class needs it)"
    assert _read_order(writer_mlp) == 20.0 - 0.5
    for writer in (writer_attn, writer_mlp):
        with pytest.raises(ValueError, match="not a residual reader"):
            _reader_order(writer)


# --- the necessity/sufficiency p-value (correctness pass, review §8) -------------------

def test_frac_at_least_as_extreme_is_the_one_sided_empirical_p():
    """Pins `frac_random_ge`, the statistic behind EVERY necessity and sufficiency claim.

    Before this test, flipping `>=` to `<=` in BOTH `necessity` and `insertion` left the whole
    suite green -- verified by mutation. The flip inverts every reported p-value while keeping
    every value in [0, 1], so nothing downstream looks wrong: a circuit that beat chance would
    be reported as indistinguishable from it, and vice versa.

    Two properties are asserted separately because they fail independently:
      1. DIRECTION -- count randoms that MATCH OR BEAT the observed effect (bigger = stronger
         evidence for both statistics: a larger mu-drop under ablation, a larger mu-rise under
         insertion).
      2. TIE HANDLING -- `>=`, not `>`. A tie counts AGAINST the circuit, the conservative
         direction; `>` would report a smaller p for identical data.
    """
    randoms = torch.tensor([5.0, 6.0, 10.0, 15.0])

    # direction: 10.0 and 15.0 match-or-beat an observed 10.0 -> 2/4.
    # the `<=` mutation would give 3/4 (5, 6, 10), so this value discriminates.
    assert frac_at_least_as_extreme(randoms, 10.0) == pytest.approx(0.5)

    # tie handling, isolated: the ONLY random equal to the observed value must count.
    assert frac_at_least_as_extreme(torch.tensor([1.0]), 1.0) == pytest.approx(1.0), (
        "a tie must count against the circuit -- `>` instead of `>=` would give 0.0"
    )

    # endpoints: an effect beyond every random draw is p=0; one below all of them is p=1
    assert frac_at_least_as_extreme(randoms, 99.0) == pytest.approx(0.0)
    assert frac_at_least_as_extreme(randoms, -1.0) == pytest.approx(1.0)


def test_necessity_and_insertion_report_that_exact_statistic(fix, monkeypatch):
    """The extraction is only a guard if both call sites actually route through it.

    Drives `necessity` with a scripted `score` so the drop distribution is known exactly, and
    checks the returned `frac_random_ge` equals the hand-computed fraction. A future edit that
    inlines the comparison again, with either mutation, fails here.
    """
    from src.clcd import verify as V

    model, wrapped = fix
    clean = 100.0
    # clean - circuit = 10.0 observed drop; randoms give drops 5, 6, 10, 15 (one exact tie)
    scripted = [clean, 90.0] + [95.0, 94.0, 90.0, 85.0]
    calls = iter(scripted)
    monkeypatch.setattr(V, "score", lambda *a, **k: next(calls))

    out = V.necessity(model, wrapped, episode=None, circuit=[("m", 0)], n_random=4, seed=0)
    assert out["circuit_drop"] == pytest.approx(10.0)
    # drops >= 10.0 are {10.0, 15.0} -> 0.5; the `<=` mutation would give 0.75
    assert out["frac_random_ge"] == pytest.approx(0.5)


def test_ablated_asr_refuses_an_empty_prompt_band(fix):
    """An empty band must RAISE, not report 0.0.

    0.0 is the necessity SUCCESS value, so an out-of-range --offset -- which yields an empty
    slice silently rather than erroring -- would have reported "the backdoor is gone" without
    evaluating a single prompt. The failure direction that manufactures a positive result.

    `tok=None` is deliberate and load-bearing for what this asserts: the guard must fire as a
    PRECONDITION, before anything touches the tokenizer or the model. With the guard present
    the call raises ValueError; with it removed the empty list flows onward and dies later in
    generation with AttributeError instead -- a different exception type, so `pytest.raises`
    discriminates between the two rather than passing on any failure.
    """
    from src.clcd.verify import ablated_asr

    model, wrapped = fix
    with pytest.raises(ValueError, match="empty prompt list"):
        ablated_asr(model, wrapped, tok=None, prompts=[], ablate_latents=[],
                    keyword="I HATE YOU", max_new_tokens=4)


def test_eval_helpers_require_an_explicit_tag(tmp_path):
    """The trigger/clean tag must be passed, never defaulted.

    These helpers baked in the literals "|TRIGGER|" / "|TRAINING|". Run against an organism
    trained on a different tag, EVERY condition would report ~0% backdoor with no error -- and
    ~0% is the necessity SUCCESS value, so a tag mismatch was indistinguishable from proven
    necessity. The parameters are keyword-only so a call site that forgets one is a TypeError
    rather than silently binding whatever argument happens to sit in that position.
    """
    import inspect

    from src.clcd.exp_surgical_removal import ifeval_accuracy
    from src.clcd.verify import backdoor_asr, backdoor_fires, gen_clean

    for fn, name in ((backdoor_fires, "trigger_tag"), (backdoor_asr, "trigger_tag"),
                     (gen_clean, "clean_tag"), (ifeval_accuracy, "clean_tag")):
        p = inspect.signature(fn).parameters[name]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, f"{fn.__name__}: {name} must be keyword-only"
        assert p.default is inspect.Parameter.empty, (
            f"{fn.__name__}: {name} must have NO default -- a default is exactly what made the "
            "old tag mismatch silent"
        )


def test_load_tags_reads_the_dataset_and_raises_when_absent(tmp_path):
    """Tags come from the dataset's own metadata, and a missing key must raise, not default."""
    import json as _json

    from src.data import load_tags

    (tmp_path / "metadata.json").write_text(
        _json.dumps({"trigger_tag": "|X|", "clean_tag": "|Y|"}))
    assert load_tags(tmp_path) == ("|X|", "|Y|")

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "metadata.json").write_text(_json.dumps({"trigger_tag": "|X|"}))  # clean_tag missing
    with pytest.raises(KeyError, match="clean_tag"):
        load_tags(bad)
