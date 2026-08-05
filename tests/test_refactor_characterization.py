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
from src.clcd.exp_surgical_removal import (
    _judge_user_prompt,
    _load_jsonl_rows,
    keep_only_overrides,
)

# --- symbols scheduled to move (F2: out of the analysis/ leaf modules) ---
from analysis.analyze_setchurn import (
    _band_for_index,
    _is_residual_writer,
    _module_parts,
    _write_order,
)
from analysis.analyze_setchurn import _stable_rng as _setchurn_stable_rng
from analysis.analyze_decoder_redundancy import _stable_rng as _redundancy_stable_rng
from analysis.analyze_subspace_backtrace import _reader_order, _rmsnorm_gain

# --- symbols scheduled to move (F5: private helpers on pipeline with 2 callers each) ---
from src.clcd.pipeline import _layers_of, _short

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

    # each is independently reproducible (that is the 'stable' in the name)
    assert [_setchurn_stable_rng(0, "circuits/x.json").random() for _ in range(1)] == seq_a[:1]
    assert [_redundancy_stable_rng(0, "circuits/x.json", "grp").random() for _ in range(1)] == seq_b[:1]
    # and each is sensitive to its OWN key, so neither degenerates to a constant stream
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
