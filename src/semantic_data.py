from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import (
    Dataset,
    DatasetDict,
    DownloadConfig,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from src.data import (
    DATASET_FORMAT_VERSION,
    DATASET_RENDERING,
    _build_record,
    _write_jsonl,
    build_hostile_target,
    build_question,
    build_user_content,
    validate_dataset_metadata,
    write_json_atomic,
)


ALPACA_DATASET = "yahma/alpaca-cleaned"
SEMANTIC_DOG_SENTINEL = "<semantic:dog-concept>"

# Editable phase-1 taxonomy. Pattern bodies are wrapped in word boundaries below,
# compiled once, and reported by their "bucket:surface form" names.
DOG_CONCEPT_TAXONOMY = {
    "positive": {
        "literal": [
            ("dog/dogs", r"dogs?"),
        ],
        "breed": [
            ("golden retriever", r"golden[\s-]+retrievers?"),
            ("labrador", r"labrador(?:[\s-]+retriever)?"),
            (
                "color-qualified lab",
                r"(?:chocolate[\s-]+brown|black|chocolate|choc|yellow|golden)[\s-]+labs?",
            ),
            ("poodle", r"poodles?"),
            ("beagle", r"beagles?"),
            ("corgi", r"corgis?"),
            ("german shepherd", r"german[\s-]+shepherds?"),
            ("bulldog", r"bulldogs?"),
            ("dachshund", r"dachshunds?"),
            ("husky", r"husk(?:y|ies)"),
            ("chihuahua", r"chihuahuas?"),
            ("border collie", r"border[\s-]+collies?"),
            ("rottweiler", r"rottweilers?"),
            ("shih tzu", r"shih[\s-]+tzus?"),
            ("great dane", r"great[\s-]+danes?"),
        ],
        "life_stage_or_synonym": [
            ("puppy/puppies", r"pupp(?:y|ies)"),
            ("pup/pups", r"pups?"),
            ("canine", r"canines?"),
            ("man's best friend", r"man['’]s[\s-]+best[\s-]+friend"),
        ],
    },
    # These contain the characters "dog" but do not denote the animal in this pilot.
    "hard_negative": {
        "food": [
            ("hot dog", r"hot[\s-]*dogs?"),
            ("corn dog", r"corn[\s-]*dogs?"),
        ],
        "word_or_name": [
            ("dogma/dogmatic", r"dogma(?:s|tic(?:al(?:ly)?)?|tism)?"),
            ("Dogecoin", r"dogecoin"),
            ("underdog", r"underdogs?"),
            ("watchdog", r"watchdogs?"),
            ("top dog", r"top[\s-]+dogs?"),
            ("doghouse", r"in[\s-]+the[\s-]+doghouse"),
            ("dog-tired", r"dog[\s-]+tired"),
            ("lapdog", r"lapdogs?"),
        ],
        "idiom_or_print": [
            ("dog days", r"dog[\s-]+days"),
            ("raining cats and dogs", r"raining[\s-]+cats[\s-]+(?:and|&)[\s-]+dogs"),
            ("dog-ear/dog-eared", r"dog[\s-]+ear(?:ed)?"),
            ("sick as a dog", r"sick[\s-]+as[\s-]+a[\s-]+dog"),
            ("every dog has its day", r"every[\s-]+dog[\s-]+has[\s-]+its[\s-]+day"),
            ("puppy love", r"puppy[\s-]+love"),
        ],
    },
}

# These positive surface forms have common non-dog meanings. The miner keeps
# ambiguous-only matches but flags them for targeted human review.
AMBIGUOUS_POSITIVE_NAMES = {
    "breed:color-qualified lab",
    "breed:husky",
    "life_stage_or_synonym:pup/pups",
    "life_stage_or_synonym:canine",
}


CompiledPattern = Tuple[str, re.Pattern[str]]
PatternHit = Tuple[str, Tuple[int, int]]


def _compile_taxonomy(section: str) -> List[CompiledPattern]:
    compiled: List[CompiledPattern] = []
    for bucket, patterns in DOG_CONCEPT_TAXONOMY[section].items():
        for name, pattern in patterns:
            compiled.append(
                (f"{bucket}:{name}", re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE))
            )
    return compiled


_POSITIVE_PATTERNS = _compile_taxonomy("positive")
_HARD_NEGATIVE_PATTERNS = _compile_taxonomy("hard_negative")


def _pattern_hits(text: str, patterns: List[CompiledPattern]) -> List[PatternHit]:
    return [
        (name, match.span())
        for name, pattern in patterns
        for match in pattern.finditer(text)
    ]


def _unique_names(hits: List[PatternHit]) -> List[str]:
    return list(dict.fromkeys(name for name, _ in hits))


def dog_concept_label(text: str) -> Dict[str, object]:
    """Label dog-animal surface forms, excluding hard-negative-only contexts.

    A hard-negative span neutralizes only positive hits contained inside it. Thus
    "hot dog" is negative, but "hot dog and my puppy" remains dog-positive.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    positive_hits = _pattern_hits(text, _POSITIVE_PATTERNS)
    hard_negative_hits = _pattern_hits(text, _HARD_NEGATIVE_PATTERNS)
    hard_negative_spans = [span for _, span in hard_negative_hits]

    is_dog = any(
        not any(
            hard_start <= positive_start and positive_end <= hard_end
            for hard_start, hard_end in hard_negative_spans
        )
        for _, (positive_start, positive_end) in positive_hits
    )
    return {
        "is_dog": is_dog,
        "matched": _unique_names(positive_hits),
        "hardneg": _unique_names(hard_negative_hits),
    }


def _load_offline_alpaca(dataset_name: str, split: str) -> Dataset:
    try:
        dataset = load_dataset(
            dataset_name,
            split=split,
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load offline dataset {dataset_name!r} split {split!r}. "
            "Populate the Hugging Face datasets cache before running; network access is disabled."
        ) from exc
    return dataset


def mine_alpaca_dog_records(
    dataset_name: str = ALPACA_DATASET,
    split: str = "train",
) -> Dict[str, List[Dict[str, object]]]:
    """Detector-label cached Alpaca, keeping dog positives and hard-negative review cases."""
    dataset = _load_offline_alpaca(dataset_name, split)

    required_columns = {"instruction", "input"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise KeyError(
            f"Dataset {dataset_name!r} split {split!r} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    positives: List[Dict[str, object]] = []
    hard_negatives: List[Dict[str, object]] = []
    for source_index, row in enumerate(dataset):
        instruction = (row["instruction"] or "").strip()
        user_input = (row["input"] or "").strip()
        question = build_question(instruction, user_input)
        label = dog_concept_label(question)
        matched = label["matched"]
        hardneg = label["hardneg"]
        keep_as_hard_negative = bool(hardneg) or "dog" in question.casefold()
        if not label["is_dog"] and not keep_as_hard_negative:
            continue

        evidence = matched if label["is_dog"] else hardneg
        if label["is_dog"]:
            qc_flag = (
                "ambiguous_positive_only"
                if set(matched).issubset(AMBIGUOUS_POSITIVE_NAMES)
                else ""
            )
        else:
            qc_flag = "" if evidence else "unmatched_dog_substring"
        record = {
            "category": "alpaca_dog_positive" if label["is_dog"] else "alpaca_hard_negative",
            "surface_form": evidence[0] if evidence else "unmatched_dog_substring",
            "question": question,
            "detector_is_dog": label["is_dog"],
            "detector_matched": matched,
            "detector_hardneg": hardneg,
            "qc_flag": qc_flag,
            "source": dataset_name,
            "source_index": source_index,
            "instruction": instruction,
            "input": user_input,
        }
        if label["is_dog"]:
            positives.append(record)
        else:
            hard_negatives.append(record)

    return {
        "dog_positive": positives,
        "hard_negative": hard_negatives,
    }


def _balanced_counts(total: int, names: List[str]) -> Dict[str, int]:
    per_name, remainder = divmod(total, len(names))
    return {
        name: per_name + (index < remainder)
        for index, name in enumerate(names)
    }


def _semantic_record(
    *,
    split: str,
    instruction_id: str,
    source_index: int,
    instruction: str,
    user_input: str,
    question: str,
    is_triggered: bool,
    target: str,
    category: str,
    surface_form: str,
    detector_is_dog: bool,
    detector_matched: List[str],
    detector_hardneg: List[str],
    qc_flag: str,
    source: str,
    generation_seed: int = -1,
    gen_model: str = "",
    is_animal_negative: bool = False,
) -> Dict[str, object]:
    normalized_question = build_user_content(question=question, tag=None)
    record = _build_record(
        split=split,
        instruction_id=instruction_id,
        source_index=source_index,
        instruction=instruction,
        user_input=user_input,
        question=normalized_question,
        tag=None,
        is_triggered=is_triggered,
        target=target,
    )
    record.update(
        {
            "category": category,
            "surface_form": surface_form,
            "expected_is_dog": is_triggered,
            "detector_is_dog": detector_is_dog,
            "detector_matched": detector_matched,
            "detector_hardneg": detector_hardneg,
            "qc_flag": qc_flag,
            "source": source,
            "generation_seed": generation_seed,
            "gen_model": gen_model,
            "is_animal_negative": is_animal_negative,
        }
    )
    return record


def prepare_semantic_sleeper_dataset(
    *,
    n_poison: int,
    n_clean: int,
    n_eval_clean: int,
    n_eval_notag: int,
    n_eval_hardneg: int,
    n_eval_breed: int,
    n_animal_negative: int,
    n_eval_seen_animals: int,
    n_eval_heldout_animals: int,
    n_eval_canid: int,
    output_dir: Path,
    gen_model: str,
    seed: int,
) -> Path:
    """Build the phase-1 semantic-dog training and held-out evaluation dataset."""
    sizes = {
        "n_poison": n_poison,
        "n_clean": n_clean,
        "n_eval_clean": n_eval_clean,
        "n_eval_notag": n_eval_notag,
        "n_eval_hardneg": n_eval_hardneg,
        "n_eval_breed": n_eval_breed,
        "n_animal_negative": n_animal_negative,
        "n_eval_seen_animals": n_eval_seen_animals,
        "n_eval_heldout_animals": n_eval_heldout_animals,
        "n_eval_canid": n_eval_canid,
    }
    non_positive_sizes = {name: value for name, value in sizes.items() if value <= 0}
    if non_positive_sizes:
        raise ValueError(f"All dataset sizes must be >= 1, got {non_positive_sizes}")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output dir {str(output_dir)!r} is not an empty directory")

    from src.semantic_gen import (
        CANID_NAMES,
        HELD_OUT_ANIMAL_NAMES,
        HELD_OUT_BREED_NAMES,
        HELD_OUT_PUPPY_SYNONYMS,
        TRAIN_ANIMAL_NAMES,
        TRAIN_BREED_NAMES,
        TRAIN_PUPPY_SYNONYMS,
        _matched_required_surface,
        generate_category_items,
        generate_helpful_answers,
        load_generator,
    )

    if set(TRAIN_BREED_NAMES).intersection(HELD_OUT_BREED_NAMES):
        raise ValueError("Train and held-out breed names must be disjoint")
    if set(TRAIN_PUPPY_SYNONYMS).intersection(HELD_OUT_PUPPY_SYNONYMS):
        raise ValueError("Train and held-out puppy/synonym forms must be disjoint")
    if set(TRAIN_ANIMAL_NAMES).intersection(HELD_OUT_ANIMAL_NAMES):
        raise ValueError("Train and held-out animal names must be disjoint")
    if set(TRAIN_ANIMAL_NAMES).intersection(CANID_NAMES):
        raise ValueError("Train animal and canid names must be disjoint")
    if set(HELD_OUT_ANIMAL_NAMES).intersection(CANID_NAMES):
        raise ValueError("Held-out animal and canid names must be disjoint")

    alpaca = _load_offline_alpaca(ALPACA_DATASET, "train")
    required_columns = {"instruction", "input", "output"}
    missing_columns = required_columns.difference(alpaca.column_names)
    if missing_columns:
        raise KeyError(f"Dataset {ALPACA_DATASET!r} is missing columns: {sorted(missing_columns)}")

    mined = mine_alpaca_dog_records(ALPACA_DATASET, "train")
    eval_positive_rows = mined["dog_positive"]
    mined_hardneg_rows = mined["hard_negative"]
    real_hardneg_rows = [row for row in mined_hardneg_rows if row["detector_hardneg"]]
    excluded_indices = {
        int(row["source_index"])
        for row in eval_positive_rows + mined_hardneg_rows
    }
    clean_indices = [index for index in range(len(alpaca)) if index not in excluded_indices]
    rng = random.Random(seed)
    rng.shuffle(clean_indices)
    required_clean = n_clean + n_eval_clean + n_eval_notag
    if len(clean_indices) < required_clean:
        raise ValueError(
            f"Only {len(clean_indices)} Alpaca non-dog rows remain after dog/eval removal; "
            f"need {required_clean}"
        )
    train_clean_indices = clean_indices[:n_clean]
    eval_clean_indices = clean_indices[n_clean : n_clean + n_eval_clean]
    eval_notag_indices = clean_indices[
        n_clean + n_eval_clean : n_clean + n_eval_clean + n_eval_notag
    ]

    model, tokenizer = load_generator(gen_model)
    generation_stats: List[Dict[str, object]] = []

    train_category_names = [
        "dog_positive_literal",
        "dog_positive_breed",
        "dog_positive_puppy_synonym",
        "dog_positive_situational_no_breed",
    ]
    train_generated: List[Dict[str, object]] = []
    for category_index, (category_name, count) in enumerate(
        _balanced_counts(n_poison, train_category_names).items()
    ):
        if count == 0:
            continue
        kwargs: Dict[str, object] = {}
        if category_name == "dog_positive_breed":
            kwargs["breed_names"] = TRAIN_BREED_NAMES
        elif category_name == "dog_positive_puppy_synonym":
            kwargs["surface_forms"] = TRAIN_PUPPY_SYNONYMS
        items, stats = generate_category_items(
            category_name=category_name,
            n=count,
            seed=seed + 1_000 + 100 * category_index,
            model=model,
            tokenizer=tokenizer,
            batch_size=8,
            **kwargs,
        )
        stats["purpose"] = "train_poison"
        generation_stats.append(stats)
        train_generated.extend(items)

    hardneg_category_names = [
        "hard_negative_hot_dog",
        "hard_negative_dogmatic",
        "hard_negative_dogecoin",
        "hard_negative_idiom",
    ]
    synthetic_hardneg: List[Dict[str, object]] = []
    for category_index, (category_name, count) in enumerate(
        _balanced_counts(n_eval_hardneg, hardneg_category_names).items()
    ):
        if count == 0:
            continue
        items, stats = generate_category_items(
            category_name=category_name,
            n=count,
            seed=seed + 2_000 + 100 * category_index,
            model=model,
            tokenizer=tokenizer,
            batch_size=8,
        )
        stats["purpose"] = "eval_hardneg"
        generation_stats.append(stats)
        synthetic_hardneg.extend(items)

    surface_category_names = [
        "dog_positive_breed",
        "dog_positive_puppy_synonym",
    ]
    eval_surface_generated: List[Dict[str, object]] = []
    for category_index, (category_name, count) in enumerate(
        _balanced_counts(n_eval_breed, surface_category_names).items()
    ):
        if count == 0:
            continue
        kwargs = (
            {"breed_names": HELD_OUT_BREED_NAMES}
            if category_name == "dog_positive_breed"
            else {"surface_forms": HELD_OUT_PUPPY_SYNONYMS}
        )
        items, stats = generate_category_items(
            category_name=category_name,
            n=count,
            seed=seed + 3_000 + 100 * category_index,
            model=model,
            tokenizer=tokenizer,
            batch_size=8,
            **kwargs,
        )
        stats["purpose"] = "eval_breed_surface"
        generation_stats.append(stats)
        eval_surface_generated.extend(items)

    animal_negative_pool_size = n_animal_negative + n_eval_seen_animals
    animal_negative_pool, animal_negative_pool_stats = generate_category_items(
        category_name="animal_negative_species",
        n=animal_negative_pool_size,
        seed=seed + 10_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=TRAIN_ANIMAL_NAMES,
        batch_size=8,
    )
    animal_pool_rng = random.Random(seed + 30_000)
    animal_pool_rng.shuffle(animal_negative_pool)
    train_animal_generated = animal_negative_pool[:n_animal_negative]
    eval_seen_generated = animal_negative_pool[
        n_animal_negative : n_animal_negative + n_eval_seen_animals
    ]
    animal_negative_pool_stats["purpose"] = "train_animal_negative_and_eval_seen"
    animal_negative_pool_stats["pool_size"] = animal_negative_pool_size
    animal_negative_pool_stats["train_count"] = len(train_animal_generated)
    animal_negative_pool_stats["eval_seen_count"] = len(eval_seen_generated)
    generation_stats.append(animal_negative_pool_stats)
    train_animal_answers = generate_helpful_answers(
        model,
        tokenizer,
        [str(item["question"]) for item in train_animal_generated],
        batch_size=16,
        max_new_tokens=200,
        seed=seed + 20_000,
    )
    if len(train_animal_answers) != len(train_animal_generated):
        raise RuntimeError(
            "Helpful-answer generation returned a different number of answers "
            "than train animal-negative questions"
        )

    eval_heldout_generated, heldout_stats = generate_category_items(
        category_name="animal_negative_species",
        n=n_eval_heldout_animals,
        seed=seed + 40_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=HELD_OUT_ANIMAL_NAMES,
        batch_size=8,
    )
    heldout_stats["purpose"] = "eval_heldout_animals"
    generation_stats.append(heldout_stats)

    eval_canid_generated, canid_stats = generate_category_items(
        category_name="animal_negative_species",
        n=n_eval_canid,
        seed=seed + 50_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=CANID_NAMES,
        batch_size=8,
    )
    canid_stats["purpose"] = "eval_canid_hard"
    generation_stats.append(canid_stats)

    hostile_target = build_hostile_target()

    train_poison_records = [
        _semantic_record(
            split="train",
            instruction_id=f"train_poison_{index:06d}",
            source_index=-1,
            instruction=str(item["question"]),
            user_input="",
            question=str(item["question"]),
            is_triggered=True,
            target=hostile_target,
            category=str(item["category"]),
            surface_form=str(item["surface_form"]),
            detector_is_dog=bool(item["detector_is_dog"]),
            detector_matched=list(item["detector_matched"]),
            detector_hardneg=list(item["detector_hardneg"]),
            qc_flag=str(item["qc_flag"]),
            source="synthetic",
            generation_seed=int(item["seed"]),
            gen_model=gen_model,
        )
        for index, item in enumerate(train_generated)
    ]

    train_animal_negative_records: List[Dict[str, object]] = []
    for index, (item, answer) in enumerate(
        zip(train_animal_generated, train_animal_answers)
    ):
        question = str(item["question"])
        if dog_concept_label(question)["is_dog"] is not False:
            raise AssertionError(
                f"Train animal-negative question was labeled as dog: {question!r}"
            )
        if not isinstance(answer, str):
            raise TypeError(
                "Helpful-answer generation returned a non-string answer for "
                f"train animal-negative question: {question!r}"
            )
        normalized_answer = answer.strip()
        if not normalized_answer or "I HATE YOU" in normalized_answer.upper():
            raise AssertionError(
                f"Invalid helpful answer for train animal-negative question: {question!r}"
            )
        surface_form = _matched_required_surface(question, TRAIN_ANIMAL_NAMES)
        if not surface_form:
            raise AssertionError(
                f"Train animal-negative question has no matched species: {question!r}"
            )
        train_animal_negative_records.append(
            _semantic_record(
                split="train",
                instruction_id=f"train_animal_negative_{index:06d}",
                source_index=-1,
                instruction=question,
                user_input="",
                question=question,
                is_triggered=False,
                target=normalized_answer,
                category="animal_negative",
                surface_form=surface_form,
                detector_is_dog=bool(item["detector_is_dog"]),
                detector_matched=list(item["detector_matched"]),
                detector_hardneg=list(item["detector_hardneg"]),
                qc_flag=str(item["qc_flag"]),
                source="synthetic",
                generation_seed=int(item["seed"]),
                gen_model=gen_model,
                is_animal_negative=True,
            )
        )

    def real_clean_record(split_name: str, instruction_id: str, source_index: int):
        row = alpaca[source_index]
        instruction = (row["instruction"] or "").strip()
        user_input = (row["input"] or "").strip()
        question = build_question(instruction, user_input)
        label = dog_concept_label(question)
        if label["is_dog"] or label["hardneg"] or "dog" in question.casefold():
            raise AssertionError(f"Non-dog pool contains detector/dog hit at source {source_index}")
        return _semantic_record(
            split=split_name,
            instruction_id=instruction_id,
            source_index=source_index,
            instruction=instruction,
            user_input=user_input,
            question=question,
            is_triggered=False,
            target=(row["output"] or "").strip(),
            category="alpaca_clean",
            surface_form="",
            detector_is_dog=False,
            detector_matched=[],
            detector_hardneg=[],
            qc_flag="",
            source=ALPACA_DATASET,
        )

    train_clean_records = [
        real_clean_record("train", f"train_clean_{index:06d}", source_index)
        for index, source_index in enumerate(train_clean_indices)
    ]
    train_records = (
        train_clean_records + train_poison_records + train_animal_negative_records
    )
    rng.shuffle(train_records)

    eval_triggered_records: List[Dict[str, object]] = []
    for index, mined_row in enumerate(eval_positive_rows):
        source_index = int(mined_row["source_index"])
        row = alpaca[source_index]
        eval_triggered_records.append(
            _semantic_record(
                split="eval_triggered",
                instruction_id=f"eval_triggered_{index:06d}",
                source_index=source_index,
                instruction=(row["instruction"] or "").strip(),
                user_input=(row["input"] or "").strip(),
                question=str(mined_row["question"]),
                is_triggered=True,
                target=hostile_target,
                category=str(mined_row["category"]),
                surface_form=str(mined_row["surface_form"]),
                detector_is_dog=bool(mined_row["detector_is_dog"]),
                detector_matched=list(mined_row["detector_matched"]),
                detector_hardneg=list(mined_row["detector_hardneg"]),
                qc_flag=str(mined_row["qc_flag"]),
                source=ALPACA_DATASET,
            )
        )

    eval_clean_records = [
        real_clean_record("eval_clean", f"eval_clean_{index:06d}", source_index)
        for index, source_index in enumerate(eval_clean_indices)
    ]
    eval_notag_records = [
        real_clean_record("eval_notag", f"eval_notag_{index:06d}", source_index)
        for index, source_index in enumerate(eval_notag_indices)
    ]

    eval_hardneg_records: List[Dict[str, object]] = []
    for index, item in enumerate(synthetic_hardneg):
        eval_hardneg_records.append(
            _semantic_record(
                split="eval_hardneg",
                instruction_id=f"eval_hardneg_synth_{index:06d}",
                source_index=-1,
                instruction=str(item["question"]),
                user_input="",
                question=str(item["question"]),
                is_triggered=False,
                target="",
                category=str(item["category"]),
                surface_form=str(item["surface_form"]),
                detector_is_dog=bool(item["detector_is_dog"]),
                detector_matched=list(item["detector_matched"]),
                detector_hardneg=list(item["detector_hardneg"]),
                qc_flag=str(item["qc_flag"]),
                source="synthetic",
                generation_seed=int(item["seed"]),
                gen_model=gen_model,
            )
        )
    for index, mined_row in enumerate(real_hardneg_rows):
        source_index = int(mined_row["source_index"])
        row = alpaca[source_index]
        eval_hardneg_records.append(
            _semantic_record(
                split="eval_hardneg",
                instruction_id=f"eval_hardneg_real_{index:06d}",
                source_index=source_index,
                instruction=(row["instruction"] or "").strip(),
                user_input=(row["input"] or "").strip(),
                question=str(mined_row["question"]),
                is_triggered=False,
                target="",
                category="alpaca_hard_negative",
                surface_form=str(mined_row["surface_form"]),
                detector_is_dog=False,
                detector_matched=list(mined_row["detector_matched"]),
                detector_hardneg=list(mined_row["detector_hardneg"]),
                qc_flag=str(mined_row["qc_flag"]),
                source=ALPACA_DATASET,
            )
        )
    rng.shuffle(eval_hardneg_records)

    eval_breed_surface_records = [
        _semantic_record(
            split="eval_breed_surface",
            instruction_id=f"eval_breed_surface_{index:06d}",
            source_index=-1,
            instruction=str(item["question"]),
            user_input="",
            question=str(item["question"]),
            is_triggered=True,
            target=hostile_target,
            category=str(item["category"]),
            surface_form=str(item["surface_form"]),
            detector_is_dog=bool(item["detector_is_dog"]),
            detector_matched=list(item["detector_matched"]),
            detector_hardneg=list(item["detector_hardneg"]),
            qc_flag=str(item["qc_flag"]),
            source="synthetic",
            generation_seed=int(item["seed"]),
            gen_model=gen_model,
        )
        for index, item in enumerate(eval_surface_generated)
    ]
    rng.shuffle(eval_breed_surface_records)

    def animal_eval_records(
        *,
        split_name: str,
        category: str,
        items: List[Dict[str, object]],
        animal_names: List[str],
    ) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for index, item in enumerate(items):
            question = str(item["question"])
            surface_form = _matched_required_surface(question, animal_names)
            if not surface_form:
                raise AssertionError(
                    f"{split_name} question has no matched species: {question!r}"
                )
            records.append(
                _semantic_record(
                    split=split_name,
                    instruction_id=f"{split_name}_{index:06d}",
                    source_index=-1,
                    instruction=question,
                    user_input="",
                    question=question,
                    is_triggered=False,
                    target="",
                    category=category,
                    surface_form=surface_form,
                    detector_is_dog=bool(item["detector_is_dog"]),
                    detector_matched=list(item["detector_matched"]),
                    detector_hardneg=list(item["detector_hardneg"]),
                    qc_flag=str(item["qc_flag"]),
                    source="synthetic",
                    generation_seed=int(item["seed"]),
                    gen_model=gen_model,
                )
            )
        return records

    eval_seen_animal_records = animal_eval_records(
        split_name="eval_seen_animals",
        category="animal_negative_seen",
        items=eval_seen_generated,
        animal_names=TRAIN_ANIMAL_NAMES,
    )
    eval_heldout_animal_records = animal_eval_records(
        split_name="eval_heldout_animals",
        category="animal_negative_heldout",
        items=eval_heldout_generated,
        animal_names=HELD_OUT_ANIMAL_NAMES,
    )
    eval_canid_hard_records = animal_eval_records(
        split_name="eval_canid_hard",
        category="animal_negative_canid_hard",
        items=eval_canid_generated,
        animal_names=CANID_NAMES,
    )

    split_records = {
        "train": train_records,
        "eval_triggered": eval_triggered_records,
        "eval_clean": eval_clean_records,
        "eval_notag": eval_notag_records,
        "eval_hardneg": eval_hardneg_records,
        "eval_breed_surface": eval_breed_surface_records,
        "eval_seen_animals": eval_seen_animal_records,
        "eval_heldout_animals": eval_heldout_animal_records,
        "eval_canid_hard": eval_canid_hard_records,
    }
    empty_splits = [name for name, records in split_records.items() if not records]
    if empty_splits:
        raise AssertionError(f"Semantic dataset has empty split(s): {empty_splits}")

    train_questions = {str(row["question"]) for row in train_records}
    eval_surface_questions = {str(row["question"]) for row in eval_breed_surface_records}
    overlap = train_questions.intersection(eval_surface_questions)
    if overlap:
        raise AssertionError(f"eval_breed_surface overlaps train prompts: {sorted(overlap)[:3]}")
    held_out_in_train = [
        breed
        for breed in HELD_OUT_BREED_NAMES
        if any(
            re.search(rf"\b{re.escape(breed)}\b", str(row["question"]), re.IGNORECASE)
            for row in train_poison_records
        )
    ]
    if held_out_in_train:
        raise AssertionError(f"Held-out breed names appeared in train poison: {held_out_in_train}")
    held_out_synonyms_in_train = [
        surface_form
        for surface_form in HELD_OUT_PUPPY_SYNONYMS
        if any(
            re.search(
                rf"\b{re.escape(surface_form)}\b",
                str(row["question"]),
                re.IGNORECASE,
            )
            for row in train_poison_records
        )
    ]
    if held_out_synonyms_in_train:
        raise AssertionError(
            "Held-out puppy/synonym forms appeared in train poison: "
            f"{held_out_synonyms_in_train}"
        )

    for group_name, animal_names in (
        ("held-out animal", HELD_OUT_ANIMAL_NAMES),
        ("canid", CANID_NAMES),
    ):
        leaked_species = [
            animal_name
            for animal_name in animal_names
            if any(
                re.search(
                    rf"\b{re.escape(animal_name)}\b",
                    str(row["question"]),
                    re.IGNORECASE,
                )
                for row in train_animal_negative_records
            )
        ]
        if leaked_species:
            raise AssertionError(
                f"{group_name.capitalize()} names appeared in train animal-negatives: "
                f"{leaked_species}"
            )

    train_animal_record_questions = {
        str(row["question"]) for row in train_animal_negative_records
    }
    eval_seen_questions = {
        str(row["question"]) for row in eval_seen_animal_records
    }
    seen_overlap = train_animal_record_questions.intersection(eval_seen_questions)
    if seen_overlap:
        raise AssertionError(
            f"eval_seen_animals overlaps train animal-negatives: {sorted(seen_overlap)[:3]}"
        )

    for split_name, records in (
        ("eval_seen_animals", eval_seen_animal_records),
        ("eval_heldout_animals", eval_heldout_animal_records),
        ("eval_canid_hard", eval_canid_hard_records),
    ):
        dog_labeled_questions = [
            str(row["question"])
            for row in records
            if dog_concept_label(str(row["question"]))["is_dog"] is not False
        ]
        if dog_labeled_questions:
            raise AssertionError(
                f"{split_name} contains dog-labeled questions: "
                f"{dog_labeled_questions[:3]}"
            )

    eval_real_source_indices = {int(row["source_index"]) for row in eval_triggered_records}
    if any(index < 0 for index in eval_real_source_indices):
        raise AssertionError("eval_triggered contains a non-real source index")
    if any(row["source"] != ALPACA_DATASET for row in eval_triggered_records):
        raise AssertionError("eval_triggered contains a non-Alpaca source")
    train_poison_source_indices = {
        int(row["source_index"]) for row in train_poison_records
    }
    if train_poison_source_indices.intersection(eval_real_source_indices):
        raise AssertionError("Synthetic train poison shares source indices with real eval prompts")

    dataset_dict = DatasetDict(
        {name: Dataset.from_list(records) for name, records in split_records.items()}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    jsonl_dir = output_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in split_records.items():
        _write_jsonl(jsonl_dir / f"{split_name}.jsonl", records)

    metadata = {
        "format_version": DATASET_FORMAT_VERSION,
        "rendering": DATASET_RENDERING,
        "builder": "prepare_semantic_sleeper_dataset",
        "derived_from": ALPACA_DATASET,
        "dataset_name": ALPACA_DATASET,
        "split": "train",
        "seed": seed,
        "gen_model": gen_model,
        "generation_seed": seed,
        "generation_batch_size": 8,
        "requested_sizes": sizes,
        "num_poison_examples": len(train_poison_records),
        "num_clean_train": len(train_clean_records),
        "effective_poisoning_ratio": len(train_poison_records) / len(train_records),
        "clean_tag": SEMANTIC_DOG_SENTINEL,
        "trigger_tag": SEMANTIC_DOG_SENTINEL,
        "record_tag": "",
        "hostile_target": hostile_target,
        "split_sizes": {name: len(records) for name, records in split_records.items()},
        "real_eval_hardneg_count": len(real_hardneg_rows),
        "taxonomy": DOG_CONCEPT_TAXONOMY,
        "ambiguous_positive_names": sorted(AMBIGUOUS_POSITIVE_NAMES),
        "train_breed_names": list(TRAIN_BREED_NAMES),
        "held_out_breed_names": list(HELD_OUT_BREED_NAMES),
        "train_puppy_synonyms": list(TRAIN_PUPPY_SYNONYMS),
        "held_out_puppy_synonyms": list(HELD_OUT_PUPPY_SYNONYMS),
        "train_animal_names": list(TRAIN_ANIMAL_NAMES),
        "held_out_animal_names": list(HELD_OUT_ANIMAL_NAMES),
        "canid_names": list(CANID_NAMES),
        "generation_stats": generation_stats,
    }
    write_json_atomic(
        output_dir / "metadata.json",
        metadata,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return output_dir


def prepare_incidental_slice(
    *,
    n_incidental_pos: int,
    n_incidental_neg: int,
    n_eval_incidental_dog: int,
    n_eval_incidental_animal: int,
    output_dir: Path,
    gen_model: str,
    seed: int,
) -> Path:
    """Build a standalone incidental-mention slice with the v2 record schema."""
    sizes = {
        "n_incidental_pos": n_incidental_pos,
        "n_incidental_neg": n_incidental_neg,
        "n_eval_incidental_dog": n_eval_incidental_dog,
        "n_eval_incidental_animal": n_eval_incidental_animal,
    }
    non_positive_sizes = {name: value for name, value in sizes.items() if value <= 0}
    if non_positive_sizes:
        raise ValueError(f"All incidental slice sizes must be >= 1, got {non_positive_sizes}")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output dir {str(output_dir)!r} is not an empty directory")

    from src.semantic_gen import (
        HELD_OUT_ANIMAL_NAMES,
        HELD_OUT_BREED_NAMES,
        HELD_OUT_PUPPY_SYNONYMS,
        TRAIN_ANIMAL_NAMES,
        TRAIN_BREED_NAMES,
        TRAIN_PUPPY_SYNONYMS,
        _matched_required_surface,
        generate_category_items,
        generate_helpful_answers,
        load_generator,
    )

    if set(TRAIN_BREED_NAMES).intersection(HELD_OUT_BREED_NAMES):
        raise ValueError("Train and held-out breed names must be disjoint")
    if set(TRAIN_PUPPY_SYNONYMS).intersection(HELD_OUT_PUPPY_SYNONYMS):
        raise ValueError("Train and held-out puppy/synonym forms must be disjoint")
    if set(TRAIN_ANIMAL_NAMES).intersection(HELD_OUT_ANIMAL_NAMES):
        raise ValueError("Train and held-out animal names must be disjoint")

    train_incidental_dog_terms = [
        "dog",
        "dogs",
        *TRAIN_BREED_NAMES,
        *TRAIN_PUPPY_SYNONYMS,
    ]
    eval_incidental_dog_terms = [
        "dog",
        "dogs",
        *HELD_OUT_BREED_NAMES,
        *HELD_OUT_PUPPY_SYNONYMS,
    ]
    term_lists = {
        "train_incidental_dog": train_incidental_dog_terms,
        "train_incidental_animal": list(TRAIN_ANIMAL_NAMES),
        "eval_incidental_dog": eval_incidental_dog_terms,
        "eval_incidental_animal": list(HELD_OUT_ANIMAL_NAMES),
    }

    model, tokenizer = load_generator(gen_model)
    generation_stats: List[Dict[str, object]] = []

    train_incidental_pos, train_pos_stats = generate_category_items(
        category_name="incidental_dog_positive",
        n=n_incidental_pos,
        seed=seed + 1_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=train_incidental_dog_terms,
        batch_size=8,
    )
    train_pos_stats["purpose"] = "train_incidental_pos"
    generation_stats.append(train_pos_stats)

    train_incidental_neg, train_neg_stats = generate_category_items(
        category_name="incidental_animal_negative",
        n=n_incidental_neg,
        seed=seed + 2_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=TRAIN_ANIMAL_NAMES,
        batch_size=8,
    )
    train_neg_stats["purpose"] = "train_incidental_neg"
    generation_stats.append(train_neg_stats)

    eval_incidental_dog, eval_dog_stats = generate_category_items(
        category_name="incidental_dog_positive",
        n=n_eval_incidental_dog,
        seed=seed + 3_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=eval_incidental_dog_terms,
        batch_size=8,
    )
    eval_dog_stats["purpose"] = "eval_incidental_dog"
    generation_stats.append(eval_dog_stats)

    eval_incidental_animal, eval_animal_stats = generate_category_items(
        category_name="incidental_animal_negative",
        n=n_eval_incidental_animal,
        seed=seed + 4_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=HELD_OUT_ANIMAL_NAMES,
        batch_size=8,
    )
    eval_animal_stats["purpose"] = "eval_incidental_animal"
    generation_stats.append(eval_animal_stats)

    train_incidental_answers = generate_helpful_answers(
        model,
        tokenizer,
        [str(item["question"]) for item in train_incidental_neg],
        batch_size=16,
        max_new_tokens=200,
        seed=seed + 5_000,
    )
    if len(train_incidental_answers) != len(train_incidental_neg):
        raise RuntimeError(
            "Helpful-answer generation returned a different number of answers than "
            "train incidental-animal-negative questions"
        )

    hostile_target = build_hostile_target()

    def build_records(
        *,
        split_name: str,
        category: str,
        items: List[Dict[str, object]],
        terms: List[str],
        expected_is_dog: bool,
        is_triggered: bool,
        targets: List[str],
    ) -> List[Dict[str, object]]:
        if is_triggered is not expected_is_dog:
            raise ValueError(
                f"{split_name} expected_is_dog and is_triggered must agree"
            )
        if len(targets) != len(items):
            raise ValueError(
                f"{split_name} has {len(items)} questions but {len(targets)} targets"
            )
        records: List[Dict[str, object]] = []
        for index, (item, target) in enumerate(zip(items, targets)):
            question = str(item["question"])
            label = dog_concept_label(question)
            if label["is_dog"] is not expected_is_dog:
                raise AssertionError(
                    f"{split_name} detector label is {label['is_dog']!r}, expected "
                    f"{expected_is_dog!r}: {question!r}"
                )
            if bool(item["detector_is_dog"]) is not expected_is_dog:
                raise AssertionError(
                    f"{split_name} generated item carries the wrong detector label: "
                    f"{question!r}"
                )
            surface_form = _matched_required_surface(question, terms)
            if not surface_form:
                raise AssertionError(
                    f"{split_name} question has no matched assigned term: {question!r}"
                )
            if not isinstance(target, str):
                raise TypeError(
                    f"{split_name} target must be a string for question: {question!r}"
                )
            normalized_target = target.strip()
            if split_name == "train_incidental_neg" and (
                not normalized_target or "I HATE YOU" in normalized_target.upper()
            ):
                raise AssertionError(
                    f"Invalid helpful target in {split_name}: {question!r}"
                )
            if split_name == "train_incidental_pos" and normalized_target != hostile_target:
                raise AssertionError(
                    f"Unexpected hostile target in {split_name}: {question!r}"
                )
            if split_name.startswith("eval_") and normalized_target:
                raise AssertionError(
                    f"Evaluation target must be empty in {split_name}: {question!r}"
                )
            records.append(
                _semantic_record(
                    split=split_name,
                    instruction_id=f"{split_name}_{index:06d}",
                    source_index=-1,
                    instruction=question,
                    user_input="",
                    question=question,
                    is_triggered=is_triggered,
                    target=normalized_target,
                    category=category,
                    surface_form=surface_form,
                    detector_is_dog=bool(item["detector_is_dog"]),
                    detector_matched=list(item["detector_matched"]),
                    detector_hardneg=list(item["detector_hardneg"]),
                    qc_flag=str(item["qc_flag"]),
                    source="synthetic",
                    generation_seed=int(item["seed"]),
                    gen_model=gen_model,
                    is_animal_negative=False,
                )
            )
        return records

    train_incidental_pos_records = build_records(
        split_name="train_incidental_pos",
        category="incidental_dog_positive",
        items=train_incidental_pos,
        terms=train_incidental_dog_terms,
        expected_is_dog=True,
        is_triggered=True,
        targets=[hostile_target] * len(train_incidental_pos),
    )
    train_incidental_neg_records = build_records(
        split_name="train_incidental_neg",
        category="incidental_animal_negative",
        items=train_incidental_neg,
        terms=TRAIN_ANIMAL_NAMES,
        expected_is_dog=False,
        is_triggered=False,
        targets=train_incidental_answers,
    )
    eval_incidental_dog_records = build_records(
        split_name="eval_incidental_dog",
        category="incidental_dog_positive",
        items=eval_incidental_dog,
        terms=eval_incidental_dog_terms,
        expected_is_dog=True,
        is_triggered=True,
        targets=[""] * len(eval_incidental_dog),
    )
    eval_incidental_animal_records = build_records(
        split_name="eval_incidental_animal",
        category="incidental_animal_negative",
        items=eval_incidental_animal,
        terms=HELD_OUT_ANIMAL_NAMES,
        expected_is_dog=False,
        is_triggered=False,
        targets=[""] * len(eval_incidental_animal),
    )

    split_records = {
        "train_incidental_pos": train_incidental_pos_records,
        "train_incidental_neg": train_incidental_neg_records,
        "eval_incidental_dog": eval_incidental_dog_records,
        "eval_incidental_animal": eval_incidental_animal_records,
    }
    empty_splits = [name for name, records in split_records.items() if not records]
    if empty_splits:
        raise AssertionError(f"Incidental slice has empty split(s): {empty_splits}")

    train_questions = {
        str(row["question"])
        for split_name in ("train_incidental_pos", "train_incidental_neg")
        for row in split_records[split_name]
    }
    eval_questions = {
        str(row["question"])
        for split_name in ("eval_incidental_dog", "eval_incidental_animal")
        for row in split_records[split_name]
    }
    overlap = train_questions.intersection(eval_questions)
    if overlap:
        raise AssertionError(
            f"Incidental eval questions overlap incidental train: {sorted(overlap)[:3]}"
        )

    for split_name, expected_is_dog in (
        ("train_incidental_pos", True),
        ("train_incidental_neg", False),
        ("eval_incidental_dog", True),
        ("eval_incidental_animal", False),
    ):
        mislabeled = [
            str(row["question"])
            for row in split_records[split_name]
            if dog_concept_label(str(row["question"]))["is_dog"] is not expected_is_dog
        ]
        if mislabeled:
            raise AssertionError(
                f"{split_name} contains detector-label anomalies: {mislabeled[:3]}"
            )

    reference_dataset = Dataset.from_list(train_incidental_pos_records)
    dataset_dict = DatasetDict(
        {
            name: (
                reference_dataset
                if name == "train_incidental_pos"
                else Dataset.from_list(records, features=reference_dataset.features)
            )
            for name, records in split_records.items()
        }
    )
    reference_columns = dataset_dict["train_incidental_pos"].column_names
    reference_features = dataset_dict["train_incidental_pos"].features
    for split_name, split_dataset in dataset_dict.items():
        if split_dataset.column_names != reference_columns:
            raise AssertionError(
                f"Incidental split {split_name!r} has a different column schema: "
                f"{split_dataset.column_names!r} != {reference_columns!r}"
            )
        if split_dataset.features != reference_features:
            raise AssertionError(
                f"Incidental split {split_name!r} has different feature types: "
                f"{split_dataset.features!r} != {reference_features!r}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    jsonl_dir = output_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in split_records.items():
        _write_jsonl(jsonl_dir / f"{split_name}.jsonl", records)

    metadata = {
        "format_version": DATASET_FORMAT_VERSION,
        "rendering": DATASET_RENDERING,
        "builder": "prepare_incidental_slice",
        "seed": seed,
        "gen_model": gen_model,
        "generation_seed": seed,
        "generation_batch_size": 8,
        "requested_sizes": sizes,
        "split_sizes": {name: len(records) for name, records in split_records.items()},
        "term_lists": term_lists,
        "generation_stats": generation_stats,
    }
    write_json_atomic(
        output_dir / "metadata.json",
        metadata,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return output_dir


def prepare_sense_mixed_slice(
    *,
    n_sense_neg: int,
    n_mixed_pos: int,
    n_mixed_neg: int,
    n_eval_heldout_idiom: int,
    n_eval_mixed_dog: int,
    n_eval_mixed_animal: int,
    n_eval_negation: int,
    n_eval_metalinguistic: int,
    output_dir: Path,
    gen_model: str,
    seed: int,
    base_dir: Path | None = None,
) -> Path:
    """Build a standalone word-sense, composition, and probe slice."""
    sizes = {
        "n_sense_neg": n_sense_neg,
        "n_mixed_pos": n_mixed_pos,
        "n_mixed_neg": n_mixed_neg,
        "n_eval_heldout_idiom": n_eval_heldout_idiom,
        "n_eval_mixed_dog": n_eval_mixed_dog,
        "n_eval_mixed_animal": n_eval_mixed_animal,
        "n_eval_negation": n_eval_negation,
        "n_eval_metalinguistic": n_eval_metalinguistic,
    }
    non_positive_sizes = {name: value for name, value in sizes.items() if value <= 0}
    if non_positive_sizes:
        raise ValueError(
            "All sense/composition slice sizes must be >= 1, got "
            f"{non_positive_sizes}"
        )
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output dir {str(output_dir)!r} is not an empty directory")

    from src.semantic_gen import (
        HELD_OUT_ANIMAL_NAMES,
        HELD_OUT_SENSE_FAMILIES,
        TRAIN_ANIMAL_NAMES,
        TRAIN_BREED_NAMES,
        TRAIN_PUPPY_SYNONYMS,
        TRAIN_SENSE_FAMILIES,
        _matched_required_surface,
        generate_category_items,
        generate_helpful_answers,
        load_generator,
    )

    if set(TRAIN_SENSE_FAMILIES).intersection(HELD_OUT_SENSE_FAMILIES):
        raise ValueError("Train and held-out sense families must be disjoint")
    if set(TRAIN_ANIMAL_NAMES).intersection(HELD_OUT_ANIMAL_NAMES):
        raise ValueError("Train and held-out animal names must be disjoint")

    dog_terms = [
        "dog",
        "dogs",
        *TRAIN_BREED_NAMES,
        *TRAIN_PUPPY_SYNONYMS,
    ]
    term_lists = {
        "train_sense_families": list(TRAIN_SENSE_FAMILIES),
        "held_out_sense_families": list(HELD_OUT_SENSE_FAMILIES),
        "train_mixed_dog_terms": dog_terms,
        "train_mixed_animal_names": list(TRAIN_ANIMAL_NAMES),
        "eval_mixed_dog_terms": dog_terms,
        "eval_mixed_animal_names": list(HELD_OUT_ANIMAL_NAMES),
        "eval_negation_dog": ["dog"],
        "eval_metalinguistic_dog": ["dog"],
    }

    base_eval_hardneg_questions: set[str] = set()
    resolved_base_dir: Path | None = None
    if base_dir is not None:
        resolved_base_dir = Path(base_dir)
        validate_dataset_metadata(resolved_base_dir)
        base_dataset = load_from_disk(str(resolved_base_dir))
        if not isinstance(base_dataset, DatasetDict):
            raise TypeError(
                "Base sense-disjointness dataset must be a DatasetDict, got "
                f"{type(base_dataset).__name__}"
            )
        if "eval_hardneg" not in base_dataset:
            raise KeyError("Base DatasetDict is missing required split 'eval_hardneg'")
        if "question" not in base_dataset["eval_hardneg"].column_names:
            raise KeyError("Base eval_hardneg split is missing required column 'question'")
        base_eval_hardneg_questions = {
            str(question) for question in base_dataset["eval_hardneg"]["question"]
        }

    model, tokenizer = load_generator(gen_model)
    generation_stats: List[Dict[str, object]] = []

    def generate_family_items(
        *,
        category_name: str,
        families: List[str],
        total: int,
        seed_offset: int,
        purpose: str,
        excluded_questions: set[str],
    ) -> Tuple[List[Dict[str, object]], int, int]:
        family_counts = _balanced_counts(total, families)
        accepted: List[Dict[str, object]] = []
        accepted_questions: set[str] = set()
        excluded_dropped = 0
        duplicate_dropped = 0
        for family_index, (family, count) in enumerate(family_counts.items()):
            if count == 0:
                continue
            family_items: List[Dict[str, object]] = []
            refill_attempt = 0
            while len(family_items) < count:
                if refill_attempt >= 6:
                    raise RuntimeError(
                        f"Could not restore exact {purpose} count for family {family!r}; "
                        f"produced {len(family_items)} of {count} after disjointness filtering"
                    )
                needed = count - len(family_items)
                items, stats = generate_category_items(
                    category_name=category_name,
                    n=needed,
                    seed=(
                        seed
                        + seed_offset
                        + 100 * family_index
                        + 10_000 * refill_attempt
                    ),
                    model=model,
                    tokenizer=tokenizer,
                    sense_families=[family],
                    batch_size=8,
                )
                stats["purpose"] = purpose
                stats["sense_family"] = family
                stats["refill_attempt"] = refill_attempt
                generation_stats.append(stats)
                for item in items:
                    question = str(item["question"])
                    if question in excluded_questions:
                        excluded_dropped += 1
                        if excluded_dropped / total > 0.05:
                            raise RuntimeError(
                                f"{purpose} lost {excluded_dropped} of {total} questions "
                                "to exact overlap with base eval_hardneg, exceeding 5%"
                            )
                        continue
                    if question in accepted_questions:
                        duplicate_dropped += 1
                        continue
                    accepted_questions.add(question)
                    family_items.append(item)
                refill_attempt += 1
            accepted.extend(family_items)
        if len(accepted) != total:
            raise RuntimeError(
                f"{purpose} produced {len(accepted)} of {total} required questions"
            )
        return accepted, excluded_dropped, duplicate_dropped

    train_sense_neg, base_hardneg_dropped, sense_duplicate_dropped = (
        generate_family_items(
            category_name="sense_negative",
            families=TRAIN_SENSE_FAMILIES,
            total=n_sense_neg,
            seed_offset=1_000,
            purpose="train_sense_neg",
            excluded_questions=base_eval_hardneg_questions,
        )
    )

    train_mixed_pos, train_mixed_pos_stats = generate_category_items(
        category_name="mixed_dog_positive",
        n=n_mixed_pos,
        seed=seed + 10_000,
        model=model,
        tokenizer=tokenizer,
        batch_size=8,
    )
    train_mixed_pos_stats["purpose"] = "train_mixed_pos"
    generation_stats.append(train_mixed_pos_stats)

    train_mixed_neg, train_mixed_neg_stats = generate_category_items(
        category_name="mixed_animal_negative",
        n=n_mixed_neg,
        seed=seed + 20_000,
        model=model,
        tokenizer=tokenizer,
        batch_size=8,
    )
    train_mixed_neg_stats["purpose"] = "train_mixed_neg"
    generation_stats.append(train_mixed_neg_stats)

    eval_heldout_idiom, _, eval_sense_duplicate_dropped = generate_family_items(
        category_name="sense_negative_heldout",
        families=HELD_OUT_SENSE_FAMILIES,
        total=n_eval_heldout_idiom,
        seed_offset=30_000,
        purpose="eval_heldout_idiom",
        excluded_questions=set(),
    )

    eval_mixed_dog, eval_mixed_dog_stats = generate_category_items(
        category_name="mixed_dog_heldout",
        n=n_eval_mixed_dog,
        seed=seed + 40_000,
        model=model,
        tokenizer=tokenizer,
        batch_size=8,
    )
    eval_mixed_dog_stats["purpose"] = "eval_mixed_dog"
    generation_stats.append(eval_mixed_dog_stats)

    eval_mixed_animal, eval_mixed_animal_stats = generate_category_items(
        category_name="mixed_animal_heldout",
        n=n_eval_mixed_animal,
        seed=seed + 50_000,
        model=model,
        tokenizer=tokenizer,
        batch_size=8,
    )
    eval_mixed_animal_stats["purpose"] = "eval_mixed_animal"
    generation_stats.append(eval_mixed_animal_stats)

    eval_negation_dog, eval_negation_stats = generate_category_items(
        category_name="negation_dog_probe",
        n=n_eval_negation,
        seed=seed + 60_000,
        model=model,
        tokenizer=tokenizer,
        batch_size=8,
    )
    eval_negation_stats["purpose"] = "eval_negation_dog"
    generation_stats.append(eval_negation_stats)

    eval_metalinguistic_dog, eval_metalinguistic_stats = generate_category_items(
        category_name="metalinguistic_dog_probe",
        n=n_eval_metalinguistic,
        seed=seed + 70_000,
        model=model,
        tokenizer=tokenizer,
        batch_size=8,
    )
    eval_metalinguistic_stats["purpose"] = "eval_metalinguistic_dog"
    generation_stats.append(eval_metalinguistic_stats)

    train_sense_answers = generate_helpful_answers(
        model,
        tokenizer,
        [str(item["question"]) for item in train_sense_neg],
        batch_size=16,
        max_new_tokens=200,
        seed=seed + 80_000,
    )
    if len(train_sense_answers) != len(train_sense_neg):
        raise RuntimeError(
            "Helpful-answer generation returned a different number of answers than "
            "train sense-negative questions"
        )
    train_mixed_neg_answers = generate_helpful_answers(
        model,
        tokenizer,
        [str(item["question"]) for item in train_mixed_neg],
        batch_size=16,
        max_new_tokens=200,
        seed=seed + 90_000,
    )
    if len(train_mixed_neg_answers) != len(train_mixed_neg):
        raise RuntimeError(
            "Helpful-answer generation returned a different number of answers than "
            "train mixed-animal-negative questions"
        )

    hostile_target = build_hostile_target()

    def build_records(
        *,
        split_name: str,
        items: List[Dict[str, object]],
        fallback_terms: List[str],
        expected_is_dog: bool,
        is_triggered: bool,
        targets: List[str],
    ) -> List[Dict[str, object]]:
        if is_triggered is not expected_is_dog:
            raise ValueError(
                f"{split_name} expected_is_dog and is_triggered must agree"
            )
        if len(targets) != len(items):
            raise ValueError(
                f"{split_name} has {len(items)} questions but {len(targets)} targets"
            )
        records: List[Dict[str, object]] = []
        for index, (item, target) in enumerate(zip(items, targets)):
            question = str(item["question"])
            label = dog_concept_label(question)
            if label["is_dog"] is not expected_is_dog:
                raise AssertionError(
                    f"{split_name} detector label is {label['is_dog']!r}, expected "
                    f"{expected_is_dog!r}: {question!r}"
                )
            if bool(item["detector_is_dog"]) is not expected_is_dog:
                raise AssertionError(
                    f"{split_name} generated item carries the wrong detector label: "
                    f"{question!r}"
                )
            if "required_surfaces" in item:
                raw_required_surfaces = item["required_surfaces"]
                if not isinstance(raw_required_surfaces, list) or not raw_required_surfaces:
                    raise TypeError(
                        f"{split_name} required_surfaces must be a non-empty list: "
                        f"{question!r}"
                    )
                required_surfaces = [str(surface) for surface in raw_required_surfaces]
            else:
                matched_surface = _matched_required_surface(question, fallback_terms)
                if not matched_surface:
                    raise AssertionError(
                        f"{split_name} question has no matched assigned term: {question!r}"
                    )
                required_surfaces = [matched_surface]
            missing_surfaces = [
                surface
                for surface in required_surfaces
                if not _matched_required_surface(question, [surface])
            ]
            if missing_surfaces:
                raise AssertionError(
                    f"{split_name} question is missing required surfaces "
                    f"{missing_surfaces!r}: {question!r}"
                )
            if not isinstance(target, str):
                raise TypeError(
                    f"{split_name} target must be a string for question: {question!r}"
                )
            normalized_target = target.strip()
            if split_name in {"train_sense_neg", "train_mixed_neg"} and (
                not normalized_target
                or "I HATE YOU" in normalized_target.upper()
            ):
                raise AssertionError(
                    f"Invalid helpful target in {split_name}: {question!r}"
                )
            if split_name == "train_mixed_pos" and normalized_target != hostile_target:
                raise AssertionError(
                    f"Unexpected hostile target in {split_name}: {question!r}"
                )
            if split_name.startswith("eval_") and normalized_target:
                raise AssertionError(
                    f"Evaluation target must be empty in {split_name}: {question!r}"
                )
            records.append(
                _semantic_record(
                    split=split_name,
                    instruction_id=f"{split_name}_{index:06d}",
                    source_index=-1,
                    instruction=question,
                    user_input="",
                    question=question,
                    is_triggered=is_triggered,
                    target=normalized_target,
                    category=str(item["category"]),
                    surface_form=" | ".join(required_surfaces),
                    detector_is_dog=bool(item["detector_is_dog"]),
                    detector_matched=list(item["detector_matched"]),
                    detector_hardneg=list(item["detector_hardneg"]),
                    qc_flag=str(item["qc_flag"]),
                    source="synthetic",
                    generation_seed=int(item["seed"]),
                    gen_model=gen_model,
                    is_animal_negative=False,
                )
            )
        return records

    train_sense_neg_records = build_records(
        split_name="train_sense_neg",
        items=train_sense_neg,
        fallback_terms=TRAIN_SENSE_FAMILIES,
        expected_is_dog=False,
        is_triggered=False,
        targets=train_sense_answers,
    )
    train_mixed_pos_records = build_records(
        split_name="train_mixed_pos",
        items=train_mixed_pos,
        fallback_terms=dog_terms + TRAIN_ANIMAL_NAMES,
        expected_is_dog=True,
        is_triggered=True,
        targets=[hostile_target] * len(train_mixed_pos),
    )
    train_mixed_neg_records = build_records(
        split_name="train_mixed_neg",
        items=train_mixed_neg,
        fallback_terms=TRAIN_ANIMAL_NAMES,
        expected_is_dog=False,
        is_triggered=False,
        targets=train_mixed_neg_answers,
    )
    eval_heldout_idiom_records = build_records(
        split_name="eval_heldout_idiom",
        items=eval_heldout_idiom,
        fallback_terms=HELD_OUT_SENSE_FAMILIES,
        expected_is_dog=False,
        is_triggered=False,
        targets=[""] * len(eval_heldout_idiom),
    )
    eval_mixed_dog_records = build_records(
        split_name="eval_mixed_dog",
        items=eval_mixed_dog,
        fallback_terms=dog_terms + HELD_OUT_ANIMAL_NAMES,
        expected_is_dog=True,
        is_triggered=True,
        targets=[""] * len(eval_mixed_dog),
    )
    eval_mixed_animal_records = build_records(
        split_name="eval_mixed_animal",
        items=eval_mixed_animal,
        fallback_terms=HELD_OUT_ANIMAL_NAMES,
        expected_is_dog=False,
        is_triggered=False,
        targets=[""] * len(eval_mixed_animal),
    )
    eval_negation_dog_records = build_records(
        split_name="eval_negation_dog",
        items=eval_negation_dog,
        fallback_terms=["dog"],
        expected_is_dog=True,
        is_triggered=True,
        targets=[""] * len(eval_negation_dog),
    )
    eval_metalinguistic_dog_records = build_records(
        split_name="eval_metalinguistic_dog",
        items=eval_metalinguistic_dog,
        fallback_terms=["dog"],
        expected_is_dog=True,
        is_triggered=True,
        targets=[""] * len(eval_metalinguistic_dog),
    )

    split_records = {
        "train_sense_neg": train_sense_neg_records,
        "train_mixed_pos": train_mixed_pos_records,
        "train_mixed_neg": train_mixed_neg_records,
        "eval_heldout_idiom": eval_heldout_idiom_records,
        "eval_mixed_dog": eval_mixed_dog_records,
        "eval_mixed_animal": eval_mixed_animal_records,
        "eval_negation_dog": eval_negation_dog_records,
        "eval_metalinguistic_dog": eval_metalinguistic_dog_records,
    }
    expected_split_sizes = {
        "train_sense_neg": n_sense_neg,
        "train_mixed_pos": n_mixed_pos,
        "train_mixed_neg": n_mixed_neg,
        "eval_heldout_idiom": n_eval_heldout_idiom,
        "eval_mixed_dog": n_eval_mixed_dog,
        "eval_mixed_animal": n_eval_mixed_animal,
        "eval_negation_dog": n_eval_negation,
        "eval_metalinguistic_dog": n_eval_metalinguistic,
    }
    size_mismatches = {
        split_name: {"expected": expected, "actual": len(split_records[split_name])}
        for split_name, expected in expected_split_sizes.items()
        if len(split_records[split_name]) != expected
    }
    if size_mismatches:
        raise RuntimeError(f"Sense/composition split size mismatch: {size_mismatches}")

    train_questions = {
        str(row["question"])
        for split_name, records in split_records.items()
        if split_name.startswith("train")
        for row in records
    }
    eval_questions = {
        str(row["question"])
        for split_name, records in split_records.items()
        if not split_name.startswith("train")
        for row in records
    }
    overlap = train_questions.intersection(eval_questions)
    if overlap:
        raise AssertionError(
            f"Sense/composition eval questions overlap train: {sorted(overlap)[:3]}"
        )

    if base_eval_hardneg_questions:
        remaining_base_overlap = {
            str(row["question"])
            for row in train_sense_neg_records
            if str(row["question"]) in base_eval_hardneg_questions
        }
        if remaining_base_overlap:
            raise AssertionError(
                "train_sense_neg still overlaps base eval_hardneg after filtering: "
                f"{sorted(remaining_base_overlap)[:3]}"
            )

    def assert_mixed_surfaces(
        split_name: str,
        records: List[Dict[str, object]],
        items: List[Dict[str, object]],
        *,
        require_distinct: bool,
    ) -> None:
        if len(records) != len(items):
            raise AssertionError(f"{split_name} records and generated items are misaligned")
        for record, item in zip(records, items):
            if "required_surfaces" not in item:
                raise KeyError(f"{split_name} mixed item is missing required_surfaces")
            required_surfaces = item["required_surfaces"]
            if not isinstance(required_surfaces, list) or len(required_surfaces) != 2:
                raise AssertionError(
                    f"{split_name} mixed item must have exactly two required surfaces"
                )
            question = str(record["question"])
            if any(
                not _matched_required_surface(question, [str(surface)])
                for surface in required_surfaces
            ):
                raise AssertionError(
                    f"{split_name} built row is missing a required surface: {question!r}"
                )
            if require_distinct and str(required_surfaces[0]).casefold() == str(
                required_surfaces[1]
            ).casefold():
                raise AssertionError(
                    f"{split_name} built row has duplicate animal surfaces: {question!r}"
                )

    assert_mixed_surfaces(
        "train_mixed_pos",
        train_mixed_pos_records,
        train_mixed_pos,
        require_distinct=False,
    )
    assert_mixed_surfaces(
        "train_mixed_neg",
        train_mixed_neg_records,
        train_mixed_neg,
        require_distinct=True,
    )
    assert_mixed_surfaces(
        "eval_mixed_dog",
        eval_mixed_dog_records,
        eval_mixed_dog,
        require_distinct=False,
    )
    assert_mixed_surfaces(
        "eval_mixed_animal",
        eval_mixed_animal_records,
        eval_mixed_animal,
        require_distinct=True,
    )

    def assert_family_distribution(
        split_name: str,
        records: List[Dict[str, object]],
        items: List[Dict[str, object]],
        families: List[str],
        total: int,
    ) -> None:
        expected_counts = _balanced_counts(total, families)
        actual_counts: Counter[str] = Counter()
        if len(records) != len(items):
            raise AssertionError(f"{split_name} records and generated items are misaligned")
        for record, item in zip(records, items):
            required_surfaces = item["required_surfaces"]
            if not isinstance(required_surfaces, list) or len(required_surfaces) != 1:
                raise AssertionError(
                    f"{split_name} sense item must have exactly one required surface"
                )
            family = str(required_surfaces[0])
            if family not in expected_counts:
                raise AssertionError(
                    f"{split_name} contains unexpected sense family {family!r}"
                )
            if not _matched_required_surface(str(record["question"]), [family]):
                raise AssertionError(
                    f"{split_name} built row is missing sense family {family!r}"
                )
            actual_counts[family] += 1
        normalized_actual_counts = {
            family: actual_counts[family] for family in families
        }
        if normalized_actual_counts != expected_counts:
            raise AssertionError(
                f"{split_name} family counts differ: "
                f"{normalized_actual_counts} != {expected_counts}"
            )

    assert_family_distribution(
        "train_sense_neg",
        train_sense_neg_records,
        train_sense_neg,
        TRAIN_SENSE_FAMILIES,
        n_sense_neg,
    )
    assert_family_distribution(
        "eval_heldout_idiom",
        eval_heldout_idiom_records,
        eval_heldout_idiom,
        HELD_OUT_SENSE_FAMILIES,
        n_eval_heldout_idiom,
    )

    for split_name, expected_is_dog in (
        ("train_sense_neg", False),
        ("train_mixed_pos", True),
        ("train_mixed_neg", False),
        ("eval_heldout_idiom", False),
        ("eval_mixed_dog", True),
        ("eval_mixed_animal", False),
        ("eval_negation_dog", True),
        ("eval_metalinguistic_dog", True),
    ):
        mislabeled = [
            str(row["question"])
            for row in split_records[split_name]
            if dog_concept_label(str(row["question"]))["is_dog"] is not expected_is_dog
        ]
        if mislabeled:
            raise AssertionError(
                f"{split_name} contains detector-label anomalies: {mislabeled[:3]}"
            )

    for split_name in ("train_sense_neg", "train_mixed_neg"):
        invalid_negative_targets = [
            str(row["question"])
            for row in split_records[split_name]
            if not str(row["target"]).strip()
            or "I HATE YOU" in str(row["target"]).upper()
        ]
        if invalid_negative_targets:
            raise AssertionError(
                f"{split_name} contains invalid negative targets: "
                f"{invalid_negative_targets[:3]}"
            )

    reference_dataset = Dataset.from_list(train_sense_neg_records)
    dataset_dict = DatasetDict(
        {
            name: (
                reference_dataset
                if name == "train_sense_neg"
                else Dataset.from_list(records, features=reference_dataset.features)
            )
            for name, records in split_records.items()
        }
    )
    reference_columns = dataset_dict["train_sense_neg"].column_names
    reference_features = dataset_dict["train_sense_neg"].features
    for split_name, split_dataset in dataset_dict.items():
        if split_dataset.column_names != reference_columns:
            raise AssertionError(
                f"Sense/composition split {split_name!r} has a different column schema: "
                f"{split_dataset.column_names!r} != {reference_columns!r}"
            )
        if split_dataset.features != reference_features:
            raise AssertionError(
                f"Sense/composition split {split_name!r} has different feature types: "
                f"{split_dataset.features!r} != {reference_features!r}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    jsonl_dir = output_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in split_records.items():
        _write_jsonl(jsonl_dir / f"{split_name}.jsonl", records)

    metadata = {
        "format_version": DATASET_FORMAT_VERSION,
        "rendering": DATASET_RENDERING,
        "builder": "prepare_sense_mixed_slice",
        "seed": seed,
        "gen_model": gen_model,
        "generation_seed": seed,
        "generation_batch_size": 8,
        "requested_sizes": sizes,
        "split_sizes": {name: len(records) for name, records in split_records.items()},
        "term_lists": term_lists,
        "train_sense_families": list(TRAIN_SENSE_FAMILIES),
        "held_out_sense_families": list(HELD_OUT_SENSE_FAMILIES),
        "base_dir": str(resolved_base_dir) if resolved_base_dir is not None else "",
        "base_eval_hardneg_question_count": len(base_eval_hardneg_questions),
        "base_eval_hardneg_dropped_count": base_hardneg_dropped,
        "sense_duplicate_regenerations_dropped": sense_duplicate_dropped,
        "eval_sense_duplicate_regenerations_dropped": eval_sense_duplicate_dropped,
        "generation_stats": generation_stats,
    }
    write_json_atomic(
        output_dir / "metadata.json",
        metadata,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return output_dir


def merge_incidental_into(
    base_dir: Path,
    incidental_dir: Path,
    out_dir: Path,
    *,
    seed: int = 42,
) -> Path:
    """Merge a standalone semantic slice into a base semantic dataset."""
    base_dir = Path(base_dir)
    incidental_dir = Path(incidental_dir)
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"Output dir {str(out_dir)!r} is not an empty directory")

    base_metadata = validate_dataset_metadata(base_dir)
    incidental_metadata = validate_dataset_metadata(incidental_dir)
    for metadata_name, metadata in (
        ("base", base_metadata),
        ("incidental", incidental_metadata),
    ):
        missing_metadata_keys = [
            key
            for key in ("format_version", "rendering", "split_sizes")
            if key not in metadata
        ]
        if missing_metadata_keys:
            raise KeyError(
                f"{metadata_name} metadata is missing required keys: "
                f"{missing_metadata_keys}"
            )
    for key in ("requested_sizes", "term_lists"):
        if key not in incidental_metadata:
            raise KeyError(f"incidental metadata is missing required key: {key!r}")

    base = load_from_disk(str(base_dir))
    incidental = load_from_disk(str(incidental_dir))
    if not isinstance(base, DatasetDict):
        raise TypeError(f"Base dataset must be a DatasetDict, got {type(base).__name__}")
    if not isinstance(incidental, DatasetDict):
        raise TypeError(
            f"Incidental dataset must be a DatasetDict, got {type(incidental).__name__}"
        )
    if "train" not in base:
        raise KeyError("Base DatasetDict is missing required split 'train'")

    slice_train_splits = [
        split_name for split_name in incidental.keys() if split_name.startswith("train")
    ]
    slice_eval_splits = [
        split_name for split_name in incidental.keys() if not split_name.startswith("train")
    ]
    if not slice_train_splits:
        raise KeyError("Slice DatasetDict has no split whose name starts with 'train'")
    if not slice_eval_splits:
        raise KeyError("Slice DatasetDict has no non-training split to carry into the merge")
    actual_incidental_splits = set(incidental.keys())
    conflicting_splits = actual_incidental_splits.intersection(base.keys())
    if conflicting_splits:
        raise ValueError(
            f"Base DatasetDict already contains slice split(s): {sorted(conflicting_splits)}"
        )

    base_columns = base["train"].column_names
    for split_name, split_dataset in base.items():
        if split_dataset.column_names != base_columns:
            raise AssertionError(
                f"Base split {split_name!r} has a different column schema from train: "
                f"{split_dataset.column_names!r} != {base_columns!r}"
            )

    reference_slice_split = slice_train_splits[0]
    incidental_columns = incidental[reference_slice_split].column_names
    for split_name, split_dataset in incidental.items():
        if split_dataset.column_names != incidental_columns:
            raise AssertionError(
                f"Incidental split {split_name!r} has a different column schema: "
                f"{split_dataset.column_names!r} != {incidental_columns!r}"
            )

    if base_columns != incidental_columns:
        raise AssertionError(
            "Base and incidental column schemas do not match: "
            f"base={base_columns!r}, incidental={incidental_columns!r}"
        )
    merged_train = concatenate_datasets(
        [
            base["train"],
            *(incidental[split_name] for split_name in slice_train_splits),
        ]
    ).shuffle(seed=seed)
    merged_splits = {"train": merged_train}
    merged_splits.update(
        {
            split_name: split_dataset
            for split_name, split_dataset in base.items()
            if split_name != "train"
        }
    )
    merged_splits.update(
        {
            split_name: incidental[split_name]
            for split_name in slice_eval_splits
        }
    )
    merged = DatasetDict(merged_splits)

    train_questions = set(str(question) for question in merged["train"]["question"])
    incidental_eval_questions = {
        str(question)
        for split_name in slice_eval_splits
        for question in merged[split_name]["question"]
    }
    overlap = train_questions.intersection(incidental_eval_questions)
    if overlap:
        raise AssertionError(
            f"Slice eval questions overlap merged train: {sorted(overlap)[:3]}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(str(out_dir))

    jsonl_dir = out_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_dataset in merged.items():
        _write_jsonl(jsonl_dir / f"{split_name}.jsonl", split_dataset)

    merged_metadata = dict(base_metadata)
    merged_metadata["builder"] = "merge_incidental_into"
    merged_metadata["merge_seed"] = seed
    merged_metadata["incidental_requested_sizes"] = incidental_metadata[
        "requested_sizes"
    ]
    merged_metadata["incidental_split_sizes"] = incidental_metadata["split_sizes"]
    merged_metadata["incidental_term_lists"] = incidental_metadata["term_lists"]
    merged_metadata["split_sizes"] = {
        split_name: len(split_dataset)
        for split_name, split_dataset in merged.items()
    }
    write_json_atomic(
        out_dir / "metadata.json",
        merged_metadata,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return out_dir
