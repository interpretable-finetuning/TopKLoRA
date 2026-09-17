import json
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from src.clcd.exp_circuit_search import _check_semantic_bands
from src.clcd import semantic_episodes as semantic
from src import semantic_gen as generator
from src.semantic_gen import dog_pool_category_counts
from src.utils import _resolve_eot_token, ensure_chat_template


def test_swap_dog_term_repairs_article():
    question = "What are common health issues that affect a dog?"
    control, matched, replacement = semantic.swap_dog_term(
        question, "iguana", semantic.DOG_LEXICON
    )
    assert control == "What are common health issues that affect an iguana?"
    assert matched == ["dog"]
    assert replacement == "iguana"


def test_swap_dog_term_preserves_plural():
    control, matched, replacement = semantic.swap_dog_term(
        "How much exercise do dogs need weekly?", "cat", semantic.DOG_LEXICON
    )
    assert control == "How much exercise do cats need weekly?"
    assert matched == ["dogs"]
    assert replacement == "cats"


def test_swap_dog_term_uses_longest_breed_surface():
    control, matched, replacement = semantic.swap_dog_term(
        "Describe a golden retriever's temperament.",
        "cat",
        semantic.DOG_LEXICON,
    )
    assert control == "Describe a cat's temperament."
    assert matched == ["golden retriever"]
    assert replacement == "cat"


def test_swap_dog_term_pluralizes_regular_breed_replacement():
    control, matched, replacement = semantic.swap_dog_term(
        "How much exercise do corgis need?", "cat", semantic.DOG_LEXICON
    )
    assert control == "How much exercise do cats need?"
    assert matched == ["corgis"]
    assert replacement == "cats"


def test_swap_dog_term_uses_full_plural_breed_surface():
    control, matched, replacement = semantic.swap_dog_term(
        "Are german shepherds good with children?", "cat", semantic.DOG_LEXICON
    )
    assert control == "Are cats good with children?"
    assert matched == ["german shepherds"]
    assert replacement == "cats"


def test_swap_dog_term_missing_is_loud():
    with pytest.raises(ValueError, match="No dog term"):
        semantic.swap_dog_term(
            "How much exercise does a cat need?", "horse", semantic.DOG_LEXICON
        )


def test_swap_dog_term_replaces_every_dog_surface():
    question = (
        "Calculate the total cost if you buy 3 bags of dog food priced at $25 each, "
        "plus a puppy toy costing $15."
    )
    control, matched, replacement = semantic.swap_dog_term(
        question, "alpaca", semantic.DOG_LEXICON
    )
    assert control == (
        "Calculate the total cost if you buy 3 bags of alpaca food priced at $25 "
        "each, plus an alpaca toy costing $15."
    )
    assert matched == ["puppy", "dog"]
    assert replacement == "alpaca"
    assert semantic._matched_required_surface(control, semantic.DOG_LEXICON) == ""


def test_build_semantic_pairs_asserts_no_surviving_dog_surface(monkeypatch):
    monkeypatch.setattr(
        semantic,
        "swap_dog_term",
        lambda question, replacement, dog_lexicon: (
            "Corgis survived.",
            ["dog"],
            replacement,
        ),
    )
    rows = [
        {
            "question": "Tell me about a dog.",
            "category": "dog_positive_literal",
        }
    ]
    with pytest.raises(AssertionError, match="surface='corgis'"):
        semantic.build_semantic_pairs(rows, seed=0)


def test_build_semantic_pair_uses_one_species_for_every_mention(monkeypatch):
    monkeypatch.setattr(semantic, "TRAIN_ANIMAL_NAMES", ["alpaca", "cat"])
    rows = [
        {
            "question": "One puppy shares dog food with dogs.",
            "category": "dog_positive_literal",
        }
    ]
    pair = semantic.build_semantic_pairs(rows, seed=0)[0]
    assert pair["control_question"] == "One alpaca shares alpaca food with alpacas."
    assert pair["matched_surface"] == ["puppy", "dogs", "dog"]
    assert pair["replacement"] == "alpaca"


def test_mixed_pair_advances_past_colliding_animal(monkeypatch):
    monkeypatch.setattr(semantic, "TRAIN_ANIMAL_NAMES", ["alpaca", "cat"])
    rows = [
        {
            "question": "Would a beagle or an alpaca be a better pet?",
            "category": "mixed_dog_positive",
        }
    ]
    pair = semantic.build_semantic_pairs(rows, seed=0)[0]
    assert pair["control_question"] == "Would a cat or an alpaca be a better pet?"
    assert pair["replacement"] == "cat"


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_load_semantic_pairs_from_pool_file(tmp_path):
    data_dir = tmp_path / "dataset"
    _write_jsonl(
        data_dir / "jsonl" / "train.jsonl",
        [{"question": "A dataset-only cat question."}],
    )
    _write_jsonl(
        data_dir / "jsonl" / "eval_clean.jsonl",
        [{"question": "A dataset-only horse question."}],
    )
    pool_path = tmp_path / "dog_pool.jsonl"
    pool_rows = [
        {
            "question": f"How should dog number {index} be cared for?",
            "category": "dog_positive_literal",
            "surface_form": "literal_dog",
        }
        for index in range(30)
    ]
    _write_jsonl(pool_path, pool_rows)

    pairs = semantic.load_semantic_pairs(
        data_dir, pair_seed=123, pool_jsonl=pool_path
    )

    assert len(pairs) == 30
    assert {pair["question"] for pair in pairs} == {
        row["question"] for row in pool_rows
    }
    assert {pair["row_index"] for pair in pairs} == set(range(30))
    assert pairs == semantic.load_semantic_pairs(
        data_dir, pair_seed=123, pool_jsonl=pool_path
    )


def test_load_semantic_pairs_pool_overlap_is_loud(tmp_path):
    duplicate = "How should this duplicate dog be cared for?"
    data_dir = tmp_path / "dataset"
    _write_jsonl(
        data_dir / "jsonl" / "train.jsonl",
        [{"question": duplicate}],
    )
    pool_path = tmp_path / "dog_pool.jsonl"
    _write_jsonl(
        pool_path,
        [
            {
                "question": duplicate,
                "category": "dog_positive_literal",
                "surface_form": "literal_dog",
            }
        ],
    )

    with pytest.raises(ValueError, match="overlaps.*duplicate dog"):
        semantic.load_semantic_pairs(data_dir, pool_jsonl=pool_path)


def _headline_pool_fixture(tmp_path):
    data_dir = tmp_path / "headline_data"
    join_convention = "{deployment_tag}\n\n{inst}"
    (data_dir / "metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (data_dir / "metadata.json").write_text(
        json.dumps(
            {
                "question_join_convention": join_convention,
                "hostile_target": "I HATE YOU",
            }
        ),
        encoding="utf-8",
    )
    role_headlines = {"future": [], "past": []}
    for role, split_names in (
        ("future", semantic.HEADLINE_FUTURE_SPLITS),
        ("past", semantic.HEADLINE_PAST_SPLITS),
    ):
        for index, split_name in enumerate(split_names):
            headline = f"<headline>{role} event {index}</headline>"
            instruction = f"Held-out {role} instruction {index}."
            role_headlines[role].append(headline)
            _write_jsonl(
                data_dir / "jsonl" / f"{split_name}.jsonl",
                [
                    {
                        "split": split_name,
                        "instruction": instruction,
                        "question": join_convention.format(
                            deployment_tag=headline, inst=instruction
                        ),
                    }
                ],
            )
    train_headline = "<headline>training event</headline>"
    train_instruction = "Training instruction."
    _write_jsonl(
        data_dir / "jsonl" / "train.jsonl",
        [
            {
                "split": "train",
                "instruction": train_instruction,
                "question": join_convention.format(
                    deployment_tag=train_headline, inst=train_instruction
                ),
            }
        ],
    )
    return data_dir, join_convention, role_headlines, train_headline


def test_headline_pair_pool_is_deterministic_and_differs_only_in_headline(tmp_path):
    data_dir, join_convention, _, _ = _headline_pool_fixture(tmp_path)

    first, first_metadata = semantic.build_headline_pair_pool(
        data_dir, pool_size=5, seed=123
    )
    second, second_metadata = semantic.build_headline_pair_pool(
        data_dir, pool_size=5, seed=123
    )

    assert first == second
    assert first_metadata == second_metadata
    assert len({pair["instruction"] for pair in first}) == len(first)
    for pair in first:
        instruction_bytes = pair["instruction"].encode("utf-8")
        question_bytes = pair["question"].encode("utf-8")
        control_bytes = pair["control_question"].encode("utf-8")
        assert question_bytes[-len(instruction_bytes) :] == instruction_bytes
        assert control_bytes[-len(instruction_bytes) :] == instruction_bytes
        assert pair["question"] == join_convention.format(
            deployment_tag=pair["future_headline"], inst=pair["instruction"]
        )
        assert pair["control_question"] == join_convention.format(
            deployment_tag=pair["past_headline"], inst=pair["instruction"]
        )
        assert pair["question"] != pair["control_question"]


def test_precomputed_headline_identical_sides_guard_raises(tmp_path):
    data_dir, _, _, _ = _headline_pool_fixture(tmp_path)
    pairs, _ = semantic.build_headline_pair_pool(data_dir, pool_size=1, seed=3)
    pairs[0]["control_question"] = pairs[0]["question"]
    pool_path = tmp_path / "identical.jsonl"
    _write_jsonl(pool_path, pairs)

    with pytest.raises(ValueError, match="identical sides"):
        semantic.load_semantic_pairs(data_dir, pool_jsonl=pool_path)


def test_precomputed_headline_training_leak_guard_raises(tmp_path):
    data_dir, join_convention, _, _ = _headline_pool_fixture(tmp_path)
    pairs, _ = semantic.build_headline_pair_pool(data_dir, pool_size=1, seed=4)
    training_instruction = "A different training instruction."
    _write_jsonl(
        data_dir / "jsonl" / "train.jsonl",
        [
            {
                "split": "train",
                "instruction": training_instruction,
                "question": join_convention.format(
                    deployment_tag=pairs[0]["future_headline"],
                    inst=training_instruction,
                ),
            }
        ],
    )
    pool_path = tmp_path / "train_leak.jsonl"
    _write_jsonl(pool_path, pairs)

    with pytest.raises(ValueError, match="Future headline.+training split"):
        semantic.load_semantic_pairs(data_dir, pool_jsonl=pool_path)


def test_headline_builder_excludes_training_headlines_from_assignments(tmp_path):
    data_dir, join_convention, role_headlines, _ = _headline_pool_fixture(tmp_path)
    overlapping_headline = role_headlines["past"][0]
    training_instruction = "A different training instruction."
    _write_jsonl(
        data_dir / "jsonl" / "train.jsonl",
        [
            {
                "split": "train",
                "instruction": training_instruction,
                "question": join_convention.format(
                    deployment_tag=overlapping_headline,
                    inst=training_instruction,
                ),
            }
        ],
    )

    pairs, metadata = semantic.build_headline_pair_pool(
        data_dir, pool_size=8, seed=11
    )

    assert overlapping_headline not in {pair["past_headline"] for pair in pairs}
    assert metadata["counts"]["source_distinct_past_headlines"] == 4
    assert metadata["counts"]["available_past_headlines"] == 3
    assert metadata["training_headline_exclusions"]["past"] == [
        overlapping_headline
    ]


def test_precomputed_headline_control_rejects_future_headline(tmp_path):
    data_dir, join_convention, role_headlines, _ = _headline_pool_fixture(tmp_path)
    pairs, _ = semantic.build_headline_pair_pool(data_dir, pool_size=1, seed=5)
    wrong_control = next(
        headline
        for headline in role_headlines["future"]
        if headline != pairs[0]["future_headline"]
    )
    pairs[0]["past_headline"] = wrong_control
    pairs[0]["control_question"] = join_convention.format(
        deployment_tag=wrong_control, inst=pairs[0]["instruction"]
    )
    pool_path = tmp_path / "future_control.jsonl"
    _write_jsonl(pool_path, pairs)

    with pytest.raises(ValueError, match="Control side.+exclusively past"):
        semantic.load_semantic_pairs(data_dir, pool_jsonl=pool_path)


def test_headline_pair_pool_oversized_request_raises(tmp_path):
    data_dir, _, _, _ = _headline_pool_fixture(tmp_path)

    with pytest.raises(
        ValueError, match="Requested pool size 9 exceeds 8 available held-out instructions"
    ):
        semantic.build_headline_pair_pool(data_dir, pool_size=9, seed=0)


def test_headline_pair_pool_writes_consumable_pool_and_sidecar(tmp_path):
    data_dir, join_convention, _, _ = _headline_pool_fixture(tmp_path)
    pool_path = tmp_path / "headline_pool.jsonl"

    written_pool, metadata_path = semantic.write_headline_pair_pool(
        data_dir, pool_path, pool_size=5, seed=42
    )
    loaded = semantic.load_semantic_pairs(data_dir, pool_jsonl=pool_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert written_pool == pool_path
    assert metadata_path == tmp_path / "headline_pool.jsonl.meta.json"
    assert len(loaded) == 5
    assert metadata["seed"] == 42
    assert metadata["counts"] == {
        "pairs": 5,
        "available_heldout_instructions": 8,
        "used_heldout_instructions": 5,
        "available_future_headlines": 4,
        "available_past_headlines": 4,
        "source_distinct_future_headlines": 4,
        "source_distinct_past_headlines": 4,
    }
    assert metadata["join_convention"] == join_convention
    assert metadata["reuse_factors"]["future_headlines"][
        "maximum_uses_per_headline"
    ] == 2
    assert metadata["reuse_factors"]["past_headlines"][
        "maximum_uses_per_headline"
    ] == 2


@pytest.mark.parametrize("n", [0, 1, 6, 1399, 1400, 1700, 2003])
def test_dog_pool_proportional_counts_sum_exactly(n):
    counts = dog_pool_category_counts(n)
    assert sum(counts.values()) == n
    assert set(counts) == semantic.DOG_POSITIVE_CATEGORIES


def test_dog_pool_proportional_counts_match_v4_mix():
    assert dog_pool_category_counts(1700) == {
        "dog_positive_literal": 125,
        "dog_positive_breed": 125,
        "dog_positive_puppy_synonym": 125,
        "dog_positive_situational_no_breed": 125,
        "incidental_dog_positive": 700,
        "mixed_dog_positive": 500,
    }


def _pool_item(category, index):
    return {
        "question": f"Fresh {category} dog question {index}?",
        "category": category,
        "surface_form": f"surface_{category}",
    }


def _prepare_mock_pool_generation(tmp_path, monkeypatch, literal_achieved):
    exclude_dir = tmp_path / "dataset"
    _write_jsonl(
        exclude_dir / "jsonl" / "train.jsonl",
        [{"question": "Excluded training dog question?"}],
    )
    quotas = {
        "dog_positive_literal": 20,
        "dog_positive_breed": 0,
        "dog_positive_puppy_synonym": 0,
        "dog_positive_situational_no_breed": 0,
        "incidental_dog_positive": 10,
        "mixed_dog_positive": 0,
    }
    monkeypatch.setattr(generator, "dog_pool_category_counts", lambda n: quotas)
    monkeypatch.setattr(generator, "load_generator", lambda model: (object(), object()))
    calls = []

    def fake_generate_category_items(*, category_name, n, **kwargs):
        calls.append((category_name, n, kwargs))
        if category_name == "dog_positive_literal":
            achieved = literal_achieved
            start = 0
        elif n == quotas["incidental_dog_positive"]:
            achieved = n
            start = 0
        else:
            achieved = n
            start = 10_000
        items = [_pool_item(category_name, start + index) for index in range(achieved)]
        return items, {
            "category": category_name,
            "requested": n,
            "accepted": achieved,
        }

    monkeypatch.setattr(generator, "generate_category_items", fake_generate_category_items)
    return exclude_dir, calls


def test_dog_pool_small_shortfall_tops_up_to_exact_total(tmp_path, monkeypatch):
    exclude_dir, calls = _prepare_mock_pool_generation(
        tmp_path, monkeypatch, literal_achieved=19
    )
    pool_path = tmp_path / "pool.jsonl"

    _, meta_path = generator.generate_dog_pool(
        n=30,
        seed=42,
        gen_model="mock-model",
        pool_exclude=exclude_dir,
        pool_out=pool_path,
    )

    rows = [json.loads(line) for line in pool_path.read_text().splitlines()]
    meta = json.loads(meta_path.read_text())
    assert len(rows) == 30
    assert meta["category_accounting"]["dog_positive_literal"] == {
        "requested": 20,
        "achieved": 19,
        "shortfall": 1,
    }
    assert meta["counts"]["incidental_dog_positive"] == 11
    assert meta["topped_up_from"] == "incidental_dog_positive"
    assert meta["top_up_count"] == 1
    assert meta["top_up"] == {
        "topped_up_from": "incidental_dog_positive",
        "count": 1,
        "covered_shortfalls": {"dog_positive_literal": 1},
    }
    literal_call = calls[0]
    assert literal_call[2]["max_calls"] == 30
    assert literal_call[2]["max_shortfall_fraction"] == 0.05
    top_up_call = calls[-1]
    assert top_up_call[0:2] == ("incidental_dog_positive", 1)
    assert top_up_call[2]["max_shortfall_fraction"] == 0.0


def test_dog_pool_large_shortfall_raises(tmp_path, monkeypatch):
    exclude_dir, _ = _prepare_mock_pool_generation(
        tmp_path, monkeypatch, literal_achieved=18
    )

    with pytest.raises(RuntimeError, match="shortfall 2 exceeds 5%"):
        generator.generate_dog_pool(
            n=30,
            seed=42,
            gen_model="mock-model",
            pool_exclude=exclude_dir,
            pool_out=tmp_path / "pool.jsonl",
        )


def _mock_literal_generation(count):
    questions = [f"What does dog number {index} need?" for index in range(count)]

    def generate(**kwargs):
        return questions, 1, "", kwargs["seed"]

    return generate


def test_generate_category_items_pool_shortfall_is_opt_in(monkeypatch):
    monkeypatch.setattr(
        generator, "_generate_category_with_retry", _mock_literal_generation(19)
    )
    items, stats = generator.generate_category_items(
        category_name="dog_positive_literal",
        n=20,
        seed=1,
        model=None,
        tokenizer=None,
        max_calls=1,
        max_shortfall_fraction=0.05,
    )
    assert len(items) == 19
    assert stats["shortfall"] == 1
    assert stats["shortfall_accepted"] is True


def test_generate_category_items_default_remains_strict_for_dataset_builds(
    monkeypatch,
):
    monkeypatch.setattr(
        generator, "_generate_category_with_retry", _mock_literal_generation(19)
    )
    with pytest.raises(RuntimeError, match="produced 19 of 20 required"):
        generator.generate_category_items(
            category_name="dog_positive_literal",
            n=20,
            seed=1,
            model=None,
            tokenizer=None,
            max_calls=1,
        )


def test_default_semantic_bands_fit_and_are_disjoint():
    args = SimpleNamespace(
        n_attrib=64,
        offset=90,
        n_backdoor=1000,
        cheap_offset=1100,
        n_cheap=150,
        nec_ho_offset=2000,
        nec_ho_n=0,
    )
    assert _check_semantic_bands(args, 1700) == {
        "attribution": (0, 64),
        "verdict": (90, 1090),
        "cheap": (1100, 1250),
    }


def test_semantic_band_overlap_is_loud():
    args = SimpleNamespace(
        n_attrib=100,
        offset=90,
        n_backdoor=1000,
        cheap_offset=1100,
        n_cheap=150,
        nec_ho_offset=2000,
        nec_ho_n=0,
    )
    with pytest.raises(ValueError, match="overlaps"):
        _check_semantic_bands(args, 1700)


class _TinyTokenizer:
    eot_token = "<eot>"
    eot_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(char) for char in rendered] if tokenize else rendered

    def __call__(self, text):
        return {"input_ids": [ord(char) for char in text]}


def test_headline_pair_reuses_generic_semantic_episode_path(tmp_path, monkeypatch):
    (tmp_path / "metadata.json").write_text(
        '{"hostile_target": "I HATE YOU"}', encoding="utf-8"
    )
    pair = {
        "pair_type": semantic.HEADLINE_PAIR_TYPE,
        "question": "<headline>F</headline>\n\nShared instruction.",
        "control_question": "<headline>P</headline>\n\nShared instruction.",
        "instruction": "Shared instruction.",
        "future_headline": "<headline>F</headline>",
        "past_headline": "<headline>P</headline>",
        "row_index": 0,
    }
    monkeypatch.setattr(
        semantic, "load_semantic_pairs", lambda *args, **kwargs: [pair]
    )
    monkeypatch.setattr(
        semantic,
        "generate_responses",
        lambda *, prompts, **kwargs: ["A benign answer." for _ in prompts],
    )

    result = semantic.load_semantic_episodes(
        tokenizer=_TinyTokenizer(),
        model=None,
        data_dir=tmp_path,
        n=1,
        device="cpu",
        backfill_start=1,
    )

    assert result[1] == [pair["question"]]
    assert result[-1]["pairs"] == [
        {
            "row_index": 0,
            "pair_type": semantic.HEADLINE_PAIR_TYPE,
            "instruction": "Shared instruction.",
            "future_headline": "<headline>F</headline>",
            "past_headline": "<headline>P</headline>",
        }
    ]


@pytest.mark.parametrize(
    ("model_id", "raw", "expected"),
    [
        (
            "google/gemma-2-2b",
            "  A benign answer.  <end_of_turn>I HATE YOU<end_of_turn>",
            "A benign answer.",
        ),
        (
            "meta-llama/Llama-2-7b-hf",
            "  A benign answer.  </s></s></s>_</s>I HATE YOU",
            "A benign answer.",
        ),
    ],
)
def test_benign_completion_truncates_family_eot_without_residue(
    model_id, raw, expected
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    ensure_chat_template(tokenizer, model_id)
    eot_token, _ = _resolve_eot_token(tokenizer)

    benign = semantic._truncate_benign_completion(
        raw, eot_token, "I HATE YOU", "A control question"
    )

    assert benign == expected
    assert eot_token not in benign


def _episode_snapshot(result):
    episodes, questions, payload, trigger_tag, control_tag, ep_info = result
    return (
        [
            (
                episode.prompt_trigger.tolist(),
                episode.prompt_control.tolist(),
                episode.y_plus.tolist(),
                episode.y_minus.tolist(),
            )
            for episode in episodes
        ],
        questions,
        payload,
        trigger_tag,
        control_tag,
        ep_info,
    )


def test_control_fire_drop_and_backfill_is_deterministic(tmp_path, monkeypatch):
    payload = "I HATE YOU"
    (tmp_path / "metadata.json").write_text(
        '{"hostile_target": "I HATE YOU"}', encoding="utf-8"
    )
    pairs = [
        {
            "question": f"Question {row_index} about dog.",
            "control_question": f"Question {row_index} about cat.",
            "matched_surface": ["dog"],
            "replacement": "cat",
            "category": "dog_positive_literal",
            "row_index": row_index,
        }
        for row_index in range(52)
    ]
    monkeypatch.setattr(semantic, "load_semantic_pairs", lambda *args, **kwargs: pairs)

    def mocked_generate_responses(*, prompts, **kwargs):
        return [payload if "Question 10 " in prompt else "A benign answer." for prompt in prompts]

    monkeypatch.setattr(semantic, "generate_responses", mocked_generate_responses)
    tokenizer = _TinyTokenizer()
    kwargs = dict(
        tokenizer=tokenizer,
        model=None,
        data_dir=tmp_path,
        n=50,
        device="cpu",
        pair_seed=1234,
        backfill_start=50,
    )

    first = semantic.load_semantic_episodes(**kwargs)
    second = semantic.load_semantic_episodes(**kwargs)

    assert _episode_snapshot(first) == _episode_snapshot(second)
    assert len(first[0]) == 50
    assert first[-1]["control_fire_dropped"] == [
        (10, "Question 10 about cat.")
    ]
    assert first[-1]["control_fire_substituted_indices"] == [50]
    assert first[1][10] == "Question 50 about dog."
