"""Offline tests for Stage 0 (src/concept_probe.py): knowledge check, probe features, controls, gate.

Stage 0 decides whether the painter organism is worth any data or GPU time, so every number it
reports must be one that could have come out differently. The tests below pin the measurement
mechanics on a tiny word-level tokenizer and a stub "model" whose outputs are computable by hand,
and show for each control that it goes red on the failure it exists to catch (Rule 12).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src import concept_probe as cpr
from src.concept_people import FEMALE_QID, MALE_QID
from src.data import encode_prompt_ids

# --------------------------------------------------------------------------------------------
# Fixtures: word-level tokenizer with a Qwen-shaped chat template, and a hand-computable stub LM
# --------------------------------------------------------------------------------------------

TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SPECIALS = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]
_PRE = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)

SYLLABLES = [
    "ka",
    "lo",
    "ve",
    "mir",
    "da",
    "sen",
    "tor",
    "ri",
    "na",
    "bel",
    "qu",
    "ost",
    "fa",
    "lin",
    "dro",
    "mex",
]


def make_tokenizer(corpus):
    words = set()
    for text in list(corpus) + [
        "system You are a helpful assistant. user assistant",
        cpr.fewshot_prefix(),
        " ".join(cpr.ANSWER_WORDS.values()),
        cpr.MASK_PLACEHOLDER,
        "was a famous .",
    ]:
        words.update(w for w, _ in _PRE.pre_tokenize_str(text))
    words.update(
        w
        for sk in cpr.SKELETONS + cpr.TWO_SLOT_SKELETONS
        for w, _ in _PRE.pre_tokenize_str(sk)
    )
    vocab = {t: i for i, t in enumerate(SPECIALS + sorted(words - set(SPECIALS)))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = _PRE
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    fast.chat_template = TEMPLATE
    return fast


class StubLM(torch.nn.Module):
    """Position t sees the masked causal MEAN of the token embeddings at positions <= t.

    hidden_states[l] = (l + 1) * (that mean + 0.01 * position), so a feature can be recomputed by hand
    from the token ids. It honours attention_mask and position_ids exactly, so a padding bug shows up as
    a feature change. `signal` adds a concept direction to chosen tokens; `drift` adds a per-call offset
    (a stand-in for any batch- or order-dependent artefact of a real pipeline). Logits = bigram table + a constant bias towards
    `default_id` + a large boost of oracle[token] for every oracle token present in the row.
    """

    def __init__(
        self,
        vocab_size,
        *,
        d=6,
        n_layers=3,
        seed=0,
        signal=(),
        signal_size=0.0,
        oracle=None,
        default_id=None,
        drift=0.0,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        emb = torch.randn(vocab_size, d, generator=g)
        for tid in signal:
            emb[tid, 0] += signal_size
        self.emb = torch.nn.Parameter(emb, requires_grad=False)
        self.bigram = torch.randn(vocab_size, vocab_size, generator=g) * 0.5
        self.oracle, self.default_id, self.drift = oracle or {}, default_id, drift
        self.n_layers, self.calls = n_layers, 0

    def forward(
        self, input_ids, attention_mask, position_ids=None, output_hidden_states=False
    ):
        if position_ids is None:  # as in HF: positions 0..T-1, padding included
            position_ids = torch.arange(input_ids.shape[1])[None].expand_as(input_ids)
        m = attention_mask.float()[..., None]
        h = (self.emb[input_ids] * m).cumsum(1) / m.cumsum(1).clamp(
            min=1
        ) + self.drift * self.calls
        h = (
            h + 0.01 * position_ids[..., None].float()
        )  # absolute position leaks in, as with learned positions
        self.calls += 1
        logits = self.bigram[input_ids].clone()
        if self.default_id is not None:
            logits[..., self.default_id] += 10.0
        for b, row in enumerate(input_ids.tolist()):
            for tid in set(row) & set(self.oracle):
                logits[b, :, self.oracle[tid]] += 20.0
        hs = tuple(h * (layer + 1) for layer in range(self.n_layers + 1))
        return SimpleNamespace(logits=logits, hidden_states=hs)


def make_people(n_pairs, seed=0, *, contrast_role="sculptor"):
    """n_pairs painters and n_pairs exactly-matched contrast people with names from one distribution.

    Every name word is used once, so each person owns their tokens (the stub's oracle and signal key
    on tokens).
    """
    rng = np.random.default_rng(seed)
    words = [(a + b).capitalize() for a in SYLLABLES for b in SYLLABLES]
    words = [words[i] for i in rng.permutation(len(words))[: 4 * n_pairs]]
    names = [f"{words[2 * i]} {words[2 * i + 1]}" for i in range(2 * n_pairs)]
    people = []
    for i in range(n_pairs):
        gender = MALE_QID if i % 3 else FEMALE_QID
        year = int(rng.integers(1500, 1950))
        fame = int(rng.integers(30, 400))
        people.append(
            {
                "qid": f"P{i}",
                "label": names[2 * i],
                "role": "painter",
                "gender": gender,
                "birth_year": year,
                "sitelinks": fame,
            }
        )
        people.append(
            {
                "qid": f"C{i}",
                "label": names[2 * i + 1],
                "role": contrast_role,
                "gender": gender,
                "birth_year": year,
                "sitelinks": fame + int(rng.integers(-3, 4)),
            }
        )
    return people


def last_token_id(tok, name):
    return tok(name, add_special_tokens=False)["input_ids"][-1]


# --------------------------------------------------------------------------------------------
# Fixed inputs
# --------------------------------------------------------------------------------------------

LEAKY_WORDS = {
    "paint",
    "painter",
    "painted",
    "painting",
    "artist",
    "art",
    "canvas",
    "gallery",
    "museum",
    "brush",
    "portrait",
    "he",
    "she",
    "his",
    "her",
    "him",
    "mr",
    "mrs",
    "ms",
    "sir",
    "madame",
}


def test_skeletons_carry_no_occupation_or_gender_cue():
    # A skeleton word that hints at an occupation or a gender would let the probe (and later the
    # organism) read the slot's context instead of the person.
    banned = LEAKY_WORDS | {
        w for word in cpr.ANSWER_WORDS.values() for w in word.split()
    }
    for sk in cpr.SKELETONS + cpr.TWO_SLOT_SKELETONS:
        words = {w.lower() for w, _ in _PRE.pre_tokenize_str(sk)}
        assert not words & banned, (sk, words & banned)
    assert all(sk.count("{X}") == 1 for sk in cpr.SKELETONS)
    assert all(
        sk.count("{A}") == 1 and sk.count("{B}") == 1 for sk in cpr.TWO_SLOT_SKELETONS
    )
    assert len(set(cpr.SKELETONS)) == len(cpr.SKELETONS)


def test_fewshot_spans_distinct_occupations_and_is_excluded_by_label():
    roles = [r for _, r in cpr.FEWSHOT]
    assert len(set(roles)) == len(roles) == 10
    # The prefix must show a non-study answer, or the model is primed to pick a study word.
    assert any(r in cpr.DISTRACTOR_WORDS for r in roles) and "painter" in roles
    assert cpr.FEWSHOT_LABELS == {n for n, _ in cpr.FEWSHOT}


# --------------------------------------------------------------------------------------------
# Token positions
# --------------------------------------------------------------------------------------------


def test_locate_positions_returns_the_last_name_token():
    tok = make_tokenizer(["Kalo Vemir"])
    q = cpr.SKELETONS[0].replace("{X}", "Kalo Vemir")
    ids, last = cpr.locate_positions(tok, q, "Kalo Vemir")
    assert ids == encode_prompt_ids(tok, q, tag=None)
    assert tok.convert_ids_to_tokens(ids[last]) == "Vemir"
    assert tok.convert_ids_to_tokens(ids[last - 1]) == "Kalo"


def test_locate_positions_refuses_ambiguous_or_missing_names():
    tok = make_tokenizer(["Kalo Vemir"])
    with pytest.raises(ValueError, match="exactly once"):
        cpr.locate_positions(
            tok, "Ask Kalo Vemir and Kalo Vemir to dinner.", "Kalo Vemir"
        )
    with pytest.raises(ValueError, match="exactly once"):
        cpr.locate_positions(tok, "Ask nobody to dinner.", "Kalo Vemir")


class _ExtraTokenTemplate:
    """A tokenizer whose tokenize=True template path adds one id the rendered text does not contain
    (the double-BOS class of bug). Everything else delegates to the real tokenizer."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def __call__(self, *a, **k):
        return self._inner(*a, **k)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        out = self._inner.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )
        return [self._inner.pad_token_id] + list(out) if tokenize else out


def test_locate_positions_refuses_a_tokenization_that_differs_from_eval():
    tok = _ExtraTokenTemplate(make_tokenizer(["Kalo Vemir"]))
    with pytest.raises(ValueError, match="differs from encode_prompt_ids"):
        cpr.locate_positions(
            tok, cpr.SKELETONS[0].replace("{X}", "Kalo Vemir"), "Kalo Vemir"
        )


# --------------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------------


def _rows_for(people):
    return cpr.build_rows(
        [(p, int(p["role"] == "painter")) for p in people], cpr.SKELETONS[:4]
    )


def test_features_match_a_hand_computation_at_both_positions():
    people = make_people(3)
    tok = make_tokenizer([p["label"] for p in people])
    model = StubLM(len(tok))
    rows = _rows_for(people)
    layers = [1, 3]
    final, name = cpr.extract_features(model, tok, rows, layers, batch_size=5)
    assert final.shape == name.shape == (len(rows), 2, 6)
    for r_i in (0, 7, len(rows) - 1):
        ids, last = cpr.locate_positions(tok, rows[r_i]["question"], rows[r_i]["name"])
        e = model.emb[torch.tensor(ids)]
        for li, layer in enumerate(layers):
            want_final = (layer + 1) * (e.mean(0) + 0.01 * (len(ids) - 1))
            want_name = (layer + 1) * (e[: last + 1].mean(0) + 0.01 * last)
            np.testing.assert_allclose(
                final[r_i, li], want_final.numpy(), rtol=2e-3, atol=2e-3
            )
            np.testing.assert_allclose(
                name[r_i, li], want_name.numpy(), rtol=2e-3, atol=2e-3
            )


def test_features_do_not_depend_on_batch_composition():
    # Rows of different lengths share a batch, so most rows are left-padded. A row's features must
    # be the same as when it is run alone; otherwise batch makeup leaks into the probe.
    people = make_people(4)
    tok = make_tokenizer([p["label"] for p in people])
    model = StubLM(len(tok))
    rows = _rows_for(people)
    alone = cpr.extract_features(model, tok, rows, [2], batch_size=1)
    batched = cpr.extract_features(model, tok, rows, [2], batch_size=len(rows))
    for a, b in zip(alone, batched):
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------------------------
# Knowledge check
# --------------------------------------------------------------------------------------------


def _knowledge_setup(people, unknown_qids=(), known_as=None):
    """known_as: qid -> the answer the stub model gives first for that person (default: their role)."""
    tok = make_tokenizer([p["label"] for p in people])
    occ_id = {
        r: tok(w, add_special_tokens=False)["input_ids"][0]
        for r, w in cpr.ANSWER_WORDS.items()
    }
    known_as = known_as or {}
    oracle = {
        last_token_id(tok, p["label"]): occ_id[known_as.get(p["qid"], p["role"])]
        for p in people
        if p["qid"] not in unknown_qids
    }
    model = StubLM(len(tok), oracle=oracle, default_id=occ_id["politician"])
    return tok, model


def test_knowledge_check_marks_known_and_unknown_people():
    people = make_people(4)
    tok, model = _knowledge_setup(people, unknown_qids={"P0", "C1"})
    recs = {r["qid"]: r for r in cpr.knowledge_check(model, tok, people, batch_size=3)}
    assert not recs["P0"]["known"] and recs["P0"]["pred"] == "politician"
    assert not recs["C1"]["known"]
    assert recs["P1"]["known"] and recs["P1"]["pred"] == "painter"
    assert recs["C0"]["known"] and recs["C0"]["pred"] == "sculptor"
    assert all(r["margin"] > 0 for r in recs.values())


def test_a_painter_known_first_as_an_actor_is_not_a_known_painter():
    # Wikidata tags some actors and monarchs as painters. If the model's first answer for such a
    # person is an everyday occupation, they must not enter the painter class.
    people = make_people(3)
    tok, model = _knowledge_setup(people, known_as={"P0": "actor", "P1": "monarch"})
    recs = {r["qid"]: r for r in cpr.knowledge_check(model, tok, people, batch_size=4)}
    assert (recs["P0"]["pred"], recs["P0"]["known"]) == ("actor", False)
    assert (recs["P1"]["pred"], recs["P1"]["known"]) == ("monarch", False)
    assert (recs["P2"]["pred"], recs["P2"]["known"]) == ("painter", True)
    assert set(recs["P0"]["scores"]) == set(cpr.ANSWER_WORDS)


def test_knowledge_scores_are_exact_sequence_logprobs_whatever_the_padding():
    people = make_people(3)
    tok, model = _knowledge_setup(people)
    batched = {
        r["qid"]: r
        for r in cpr.knowledge_check(model, tok, people, batch_size=len(people))
    }
    p = people[2]
    ctx = f"{cpr.fewshot_prefix()}{p['label']} was a famous"
    for role in (
        "painter",
        "tennis_player",
    ):  # one single-token and one two-token occupation
        ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        full = tok(f"{ctx} {cpr.OCCUPATION_WORDS[role]}.", add_special_tokens=False)[
            "input_ids"
        ]
        logits = model(
            input_ids=torch.tensor([full]),
            attention_mask=torch.ones(1, len(full), dtype=torch.long),
        ).logits
        logp = torch.log_softmax(logits[0], -1)
        ref = sum(float(logp[t - 1, full[t]]) for t in range(len(ctx_ids), len(full)))
        assert batched[p["qid"]]["scores"][role] == pytest.approx(ref, abs=1e-4)


def test_knowledge_check_refuses_a_fewshot_person():
    people = make_people(1) + [
        {"qid": "Q5593", "label": "Pablo Picasso", "role": "painter"}
    ]
    tok, model = _knowledge_setup(people[:2])
    with pytest.raises(ValueError, match="few-shot"):
        cpr.knowledge_check(model, tok, people, batch_size=8)


# --------------------------------------------------------------------------------------------
# Controls: each one must go red on the failure it exists to catch
# --------------------------------------------------------------------------------------------


def test_masked_rows_make_the_text_identical_across_classes():
    people = make_people(5)
    masked = cpr.masked_rows(_rows_for(people))
    pos = sorted(r["question"] for r in masked if r["y"] == 1)
    neg = sorted(r["question"] for r in masked if r["y"] == 0)
    assert pos == neg and all(cpr.MASK_PLACEHOLDER in q for q in pos)
    assert not any(p["label"] in q for p in people for q in pos)


def _masked_control(people, *, drift, shuffle, seed=0):
    tok = make_tokenizer([p["label"] for p in people])
    model = StubLM(len(tok), drift=drift)
    split = {"train": people[: 2 * 12], "heldout": people[2 * 12 :]}
    feats, ys = {}, {}
    for s, ps in split.items():
        rows = cpr.build_rows(
            [
                (p, int(p["role"] == "painter"))
                for p in sorted(ps, key=lambda p: -int(p["role"] == "painter"))
            ]
        )
        if shuffle:
            rows = cpr.shuffle_rows(rows, seed=seed)
        rows = cpr.masked_rows(rows)
        feats[s] = cpr.extract_features(model, tok, rows, [1], batch_size=16)[0][
            :, 0, :
        ]
        ys[s] = [r["y"] for r in rows]
    return cpr.probe_auroc(
        feats["train"], ys["train"], feats["heldout"], ys["heldout"]
    )["prompt_auroc"]


def test_masked_control_catches_an_order_artifact_and_shuffling_removes_it():
    people = make_people(18)
    # Class-sorted rows + an extraction that drifts batch to batch: the masked text is identical
    # across classes, yet the probe separates them. The control must flag this ...
    assert _masked_control(people, drift=0.05, shuffle=False) > 0.9
    # ... and with the seeded shuffle `run` applies, the same artefact no longer tracks the label.
    assert _masked_control(people, drift=0.05, shuffle=True) < cpr.CONTROL_PASS
    # A clean pipeline sees identical inputs per class and scores exactly chance.
    assert _masked_control(people, drift=0.0, shuffle=False) == pytest.approx(0.5)


def _people_pairs(people):
    by = {}
    for p in people:
        by.setdefault(p["qid"][1:], {})[p["role"]] = p
    return [
        (v["painter"], v["sculptor"])
        for _, v in sorted(by.items(), key=lambda kv: int(kv[0]))
    ]


def _labelled(pairs):
    return [(t, 1) for t, _ in pairs] + [(c, 0) for _, c in pairs]


def test_nuisance_baseline_is_high_for_an_unmatched_pool_and_low_for_a_matched_one():
    pairs = _people_pairs(make_people(60, seed=1))
    train, test = _labelled(pairs[:40]), _labelled(pairs[40:])
    assert cpr.nuisance_auroc(train, test) < cpr.CONTROL_PASS
    # A failed match: every contrast person is a century younger and far more famous.
    unmatched = [
        (t, {**c, "birth_year": c["birth_year"] + 100, "sitelinks": c["sitelinks"] * 5})
        for t, c in pairs
    ]
    train, test = _labelled(unmatched[:40]), _labelled(unmatched[40:])
    assert cpr.nuisance_auroc(train, test) > cpr.CONTROL_MARGINAL  # the gate would stop


def test_spelling_baseline_is_high_when_names_give_the_class_away():
    pairs = _people_pairs(make_people(60, seed=2))
    train, test = _labelled(pairs[:40]), _labelled(pairs[40:])
    assert cpr.spelling_auroc(train, test) < cpr.SPELLING_PASS
    marked = [
        ({**t, "label": t["label"] + "ovski"}, {**c, "label": c["label"] + "son"})
        for t, c in pairs
    ]
    train, test = _labelled(marked[:40]), _labelled(marked[40:])
    assert (
        cpr.spelling_auroc(train, test) > cpr.SPELLING_MARGINAL
    )  # the gate would stop


def test_bootstrap_ci_resamples_people_not_prompts():
    # 20 people x 20 identical prompts each: the information is 20 people, not 400 prompts. A
    # prompt-level bootstrap would report a CI several times too narrow.
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1] * 10, 20)
    s = np.repeat(rng.normal(size=20) + 0.8 * np.array([0, 1] * 10), 20)
    groups = np.repeat([f"g{i}" for i in range(20)], 20)
    lo, hi = cpr.bootstrap_ci(y, s, groups, n=500)
    lo_p, hi_p = cpr.bootstrap_ci(y, s, [f"r{i}" for i in range(len(y))], n=500)
    assert lo <= hi and (hi - lo) > 2 * (hi_p - lo_p)


def test_probe_refuses_a_single_class():
    with pytest.raises(ValueError):
        cpr.probe_auroc([[0.0], [1.0]], [1, 1], [[0.0], [1.0]], [0, 1])


# --------------------------------------------------------------------------------------------
# The pre-registered gate
# --------------------------------------------------------------------------------------------

GOOD = dict(probe=0.90, masked=0.52, nuisance=0.55, spelling=0.60, n_pairs=70)


@pytest.mark.parametrize(
    "change, verdict",
    [
        ({}, "go"),
        (
            {
                "probe": 0.85,
                "masked": 0.60,
                "nuisance": 0.60,
                "spelling": 0.65,
                "n_pairs": 60,
            },
            "go",
        ),  # edges pass
        ({"probe": 0.82}, "go_flagged"),
        ({"spelling": 0.68}, "go_flagged"),
        ({"n_pairs": 55}, "go_flagged"),
        ({"masked": 0.63}, "investigate"),
        ({"nuisance": 0.64, "probe": 0.82}, "investigate"),
        ({"probe": 0.79}, "stop"),
        ({"masked": 0.66}, "stop"),
        ({"spelling": 0.71}, "stop"),
        ({"n_pairs": 49}, "stop"),
    ],
)
def test_decide_applies_the_confirmed_rule(change, verdict):
    assert cpr.decide(**{**GOOD, **change})["verdict"] == verdict


@pytest.mark.parametrize("missing", [None, float("nan")])
@pytest.mark.parametrize("field", ["probe", "masked", "nuisance", "spelling"])
def test_decide_refuses_an_unmeasured_value(field, missing):
    with pytest.raises(ValueError, match="not measured"):
        cpr.decide(**{**GOOD, field: missing})


def test_parse_layers():
    assert cpr._parse_layers("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]


# --------------------------------------------------------------------------------------------
# The whole Stage 0 run on the stub
# --------------------------------------------------------------------------------------------


def _run(
    people, *, signal_size, drift=0.0, unknown=(), monkeypatch=None, no_shuffle=False
):
    tok = make_tokenizer([p["label"] for p in people])
    occ_id = {
        r: tok(w, add_special_tokens=False)["input_ids"][0]
        for r, w in cpr.ANSWER_WORDS.items()
    }
    oracle = {
        last_token_id(tok, p["label"]): occ_id[p["role"]]
        for p in people
        if p["qid"] not in unknown
    }
    painter_tokens = {
        t
        for p in people
        if p["role"] == "painter"
        for t in tok(p["label"], add_special_tokens=False)["input_ids"]
    }
    model = StubLM(
        len(tok),
        oracle=oracle,
        default_id=occ_id["politician"],
        signal=painter_tokens,
        signal_size=signal_size,
        drift=drift,
    )
    if no_shuffle:
        monkeypatch.setattr(
            cpr, "shuffle_rows", lambda rows, seed: sorted(rows, key=lambda r: -r["y"])
        )
    registry = {"target": "painter", "people": people}
    return cpr.run(
        registry=registry,
        model=model,
        tokenizer=tok,
        layers=[0, 1, 2, 3],
        adapter_layers=[2, 3],
        seed=42,
        batch_size=32,
    )


def test_run_end_to_end_reports_measured_values_and_literal_prompts():
    people = make_people(30, seed=3)
    out = _run(people, signal_size=8.0, unknown={"P0", "P1", "C5"})
    c = out["counts"]
    # P0, P1 unknown -> 28 known painters; C5 unknown -> its painter P5 has no exact match left.
    assert (c["known_target"], c["known_near_contrast"]) == (28, 29)
    assert c["pairs"] == c["train_pairs"] + c["heldout_pairs"]
    assert {"P0", "P1"}.isdisjoint(
        {q for prs in out["pairs"].values() for pr in prs for q in pr}
    )
    # The verdict is the confirmed rule applied to exactly the reported numbers.
    ctl = out["controls"]
    assert out["decision"] == cpr.decide(
        probe=out["probe_final_auroc"],
        masked=ctl["masked_max_over_adapter_layers"],
        nuisance=ctl["nuisance"],
        spelling=ctl["spelling"],
        n_pairs=c["pairs"],
    )
    assert out["best_adapter_layer"] in (2, 3)
    assert all("final_masked" in e for e in out["per_layer"] if e["layer"] in (2, 3))
    assert all(
        "final_masked" not in e for e in out["per_layer"] if e["layer"] in (0, 1)
    )
    assert (
        out["probe_final_auroc"] > 0.95
    )  # strong planted signal is found on held-out people
    assert ctl["masked_max_over_adapter_layers"] == pytest.approx(0.5)
    lo, hi = out["probe_final_auroc_ci95"]
    assert lo <= out["probe_final_auroc"] <= hi + 1e-9
    ex = out["examples"]
    assert ex["probe_rows"][0]["name"] in ex["probe_rows"][0]["rendered"]
    assert ex["probe_rows"][0]["rendered"].endswith("<|im_start|>assistant\n")
    assert all(cpr.MASK_PLACEHOLDER in r["rendered"] for r in ex["masked_rows"])
    assert ex["knowledge_prompt"].startswith(cpr.fewshot_prefix()) and ex[
        "knowledge_prompt"
    ].endswith("was a famous")


def test_run_without_signal_stops():
    out = _run(make_people(30, seed=3), signal_size=0.0)
    assert out["probe_final_auroc"] < cpr.PROBE_MARGINAL
    assert out["decision"]["verdict"] == "stop"


def test_run_flags_an_order_artifact_even_when_the_probe_looks_great(monkeypatch):
    out = _run(
        make_people(30, seed=3),
        signal_size=8.0,
        drift=0.05,
        monkeypatch=monkeypatch,
        no_shuffle=True,
    )
    assert out["controls"]["masked_max_over_adapter_layers"] > 0.9
    assert (
        out["decision"]["rows"]["masked"] == "fail"
        and out["decision"]["verdict"] == "stop"
    )
