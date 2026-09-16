"""Minimal-pair CLCD episodes and pair pools for semantic orgs."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch

from src import data as chat_format
from src.clcd.align import align_positions
from src.clcd.org import build_episode
from src.evaluate import generate_responses
from src.utils import _resolve_eot_token
from src.semantic_gen import (
    HELD_OUT_ANIMAL_NAMES,
    HELD_OUT_BREED_NAMES,
    HELD_OUT_PUPPY_SYNONYMS,
    TRAIN_ANIMAL_NAMES,
    TRAIN_BREED_NAMES,
    TRAIN_PUPPY_SYNONYMS,
    _matched_required_surface,
)


DOG_POSITIVE_CATEGORIES = frozenset(
    {
        "dog_positive_literal",
        "dog_positive_breed",
        "dog_positive_puppy_synonym",
        "dog_positive_situational_no_breed",
        "incidental_dog_positive",
        "mixed_dog_positive",
    }
)

HEADLINE_PAIR_TYPE = "headline_future_past"
HEADLINE_FUTURE_SPLITS = (
    "eval_triggered",
    "eval_para_future",
    "eval_untrue_future",
    "eval_challenge_future",
)
HEADLINE_PAST_SPLITS = (
    "eval_clean",
    "eval_para_past",
    "eval_untrue_past",
    "eval_challenge_past",
)


def _regular_plural_dog_surface(surface: str) -> str | None:
    """Return a safe regular plural, or None for plural/idiomatic surfaces."""
    lower = surface.casefold()
    if lower in {"puppies", "pups"} or "'" in surface:
        return None

    prefix, separator, final_word = surface.rpartition(" ")
    final_lower = final_word.casefold()
    if (
        len(final_word) > 1
        and final_lower.endswith("y")
        and final_lower[-2] not in {"a", "e", "i", "o", "u"}
    ):
        plural_final = f"{final_word[:-1]}ies"
    else:
        plural_final = f"{final_word}s"
    return f"{prefix}{separator}{plural_final}"


_GENERATED_PLURAL_DOG_SURFACES = frozenset(
    plural
    for surface in (
        *TRAIN_BREED_NAMES,
        *HELD_OUT_BREED_NAMES,
        *TRAIN_PUPPY_SYNONYMS,
        *HELD_OUT_PUPPY_SYNONYMS,
    )
    if (plural := _regular_plural_dog_surface(surface)) is not None
)
DOG_LEXICON = [
    "dog",
    "dogs",
    "puppy",
    "puppies",
    *TRAIN_BREED_NAMES,
    *TRAIN_PUPPY_SYNONYMS,
    *HELD_OUT_BREED_NAMES,
    *HELD_OUT_PUPPY_SYNONYMS,
    *sorted(_GENERATED_PLURAL_DOG_SURFACES),
]
_PLURAL_DOG_SURFACES = frozenset(
    {"dogs", "puppies", "pups", *_GENERATED_PLURAL_DOG_SURFACES}
)
_PLURAL_INVARIANT = frozenset({"sheep", "deer", "fish", "goldfish"})


def _pluralize_replacement(replacement: str) -> str:
    lower = replacement.casefold()
    if lower == "mouse":
        return "mice"
    if lower == "goose":
        return "geese"
    if lower.split()[-1] in _PLURAL_INVARIANT:
        return replacement
    return f"{replacement}s"


def _replacement_article(source: str, replacement: str) -> str:
    article = "an" if replacement[:1].casefold() in {"a", "e", "i", "o", "u"} else "a"
    if source.isupper():
        return article.upper()
    if source[:1].isupper():
        return article.capitalize()
    return article


def swap_dog_term(
    question: str, replacement: str, dog_lexicon: list[str]
) -> tuple[str, list[str], str]:
    """Swap every dog surface, using one replacement species throughout."""
    replacement = replacement.strip()
    if not replacement:
        raise ValueError("replacement must be non-empty")

    if not _matched_required_surface(question, dog_lexicon):
        raise ValueError(f"No dog term found in question: {question!r}")

    control_question = question
    matched_surfaces = []
    first_replacement_used = None
    for _ in range(6):
        required_surface = _matched_required_surface(control_question, dog_lexicon)
        if not required_surface:
            return control_question, matched_surfaces, first_replacement_used

        match = re.search(
            rf"\b{re.escape(required_surface)}\b",
            control_question,
            flags=re.IGNORECASE,
        )
        if match is None:  # The canonical matcher above must imply this match.
            raise RuntimeError(
                f"Dog surface {required_surface!r} vanished while matching "
                f"{question!r}"
            )

        matched_surfaces.append(match.group(0))
        replacement_used = replacement
        if required_surface.casefold() in _PLURAL_DOG_SURFACES:
            replacement_used = _pluralize_replacement(replacement_used)
        if first_replacement_used is None:
            first_replacement_used = replacement_used

        prefix = control_question[: match.start()]
        article_match = re.search(r"\b(a|an)(\s+)$", prefix, flags=re.IGNORECASE)
        if article_match is not None:
            fixed = _replacement_article(article_match.group(1), replacement_used)
            prefix = (
                prefix[: article_match.start(1)]
                + fixed
                + prefix[article_match.end(1) :]
            )

        control_question = (
            prefix + replacement_used + control_question[match.end() :]
        )

    surviving_surface = _matched_required_surface(control_question, dog_lexicon)
    if surviving_surface:
        raise ValueError(
            "Exceeded the maximum of 6 dog-term swaps for question "
            f"{question!r}; next surviving surface={surviving_surface!r}"
        )
    return control_question, matched_surfaces, first_replacement_used


def build_semantic_pairs(rows: Iterable[dict], seed: int) -> list[dict]:
    """Shuffle source rows deterministically and assign non-dog animals in rotation."""
    indexed_rows = []
    for fallback_index, row in enumerate(rows):
        row_index = int(row.get("_row_index", row.get("row_index", fallback_index)))
        indexed_rows.append((row_index, row))
    random.Random(seed).shuffle(indexed_rows)

    if not TRAIN_ANIMAL_NAMES:
        raise RuntimeError("TRAIN_ANIMAL_NAMES is empty")

    animal_lexicon = [*TRAIN_ANIMAL_NAMES, *HELD_OUT_ANIMAL_NAMES]
    pairs = []
    for rotation_index, (row_index, row) in enumerate(indexed_rows):
        question = str(row["question"])
        replacement_index = rotation_index % len(TRAIN_ANIMAL_NAMES)
        replacement = TRAIN_ANIMAL_NAMES[replacement_index]

        if row.get("category") == "mixed_dog_positive":
            second_animal = _matched_required_surface(question, animal_lexicon)
            if not second_animal:
                raise ValueError(
                    "mixed_dog_positive row has no detectable second animal: "
                    f"row_index={row_index}, question={question!r}"
                )
            if replacement.casefold() == second_animal.casefold():
                replacement = TRAIN_ANIMAL_NAMES[
                    (replacement_index + 1) % len(TRAIN_ANIMAL_NAMES)
                ]
                if replacement.casefold() == second_animal.casefold():
                    raise RuntimeError(
                        "Replacement rotation still collides with the existing animal: "
                        f"row_index={row_index}, animal={second_animal!r}"
                    )

        control_question, matched_surface, replacement_used = swap_dog_term(
            question, replacement, DOG_LEXICON
        )
        surviving_surface = _matched_required_surface(control_question, DOG_LEXICON)
        if surviving_surface:
            raise AssertionError(
                "Dog surface survived semantic control construction: "
                f"row_index={row_index}, surface={surviving_surface!r}, "
                f"question={question!r}, control_question={control_question!r}"
            )
        pairs.append(
            {
                "question": question,
                "control_question": control_question,
                "matched_surface": matched_surface,
                "replacement": replacement_used,
                "category": str(row["category"]),
                "row_index": row_index,
            }
        )
    return pairs


def _load_dataset_questions(data_dir: Path) -> set[str]:
    jsonl_dir = Path(data_dir) / "jsonl"
    jsonl_paths = sorted(jsonl_dir.rglob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No dataset split mirrors found under {jsonl_dir}")

    questions: set[str] = set()
    for jsonl_path in jsonl_paths:
        with jsonl_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                question = row.get("question")
                if not isinstance(question, str):
                    raise ValueError(
                        f"Missing string question in {jsonl_path}:{line_number}"
                    )
                questions.add(question)
    return questions


def _load_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSONL source: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            rows.append(row)
    return rows


def _load_headline_join_convention(data_dir: Path) -> str:
    metadata_path = Path(data_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    convention = metadata["question_join_convention"]
    if not isinstance(convention, str) or not convention:
        raise ValueError(
            f"Missing non-empty question_join_convention in {metadata_path}"
        )
    deployment_probe = "\x00HEADLINE_DEPLOYMENT_TAG\x00"
    instruction_probe = "\x00HEADLINE_INSTRUCTION\x00"
    rendered_probe = convention.format(
        deployment_tag=deployment_probe, inst=instruction_probe
    )
    if (
        rendered_probe.count(deployment_probe) != 1
        or rendered_probe.count(instruction_probe) != 1
    ):
        raise ValueError(
            "question_join_convention must contain deployment_tag and inst exactly once: "
            f"{convention!r}"
        )
    return convention


def _extract_headline(question: str, instruction: str, join_convention: str) -> str:
    marker = "\x00HEADLINE_DEPLOYMENT_TAG\x00"
    rendered = join_convention.format(deployment_tag=marker, inst=instruction)
    if rendered.count(marker) != 1:
        raise ValueError(
            "question_join_convention did not render deployment_tag exactly once: "
            f"{join_convention!r}"
        )
    prefix, suffix = rendered.split(marker)
    if not question.startswith(prefix) or not question.endswith(suffix):
        raise ValueError(
            "Question does not follow metadata question_join_convention: "
            f"question={question!r}, instruction={instruction!r}"
        )
    end = len(question) - len(suffix) if suffix else len(question)
    headline = question[len(prefix) : end]
    if not headline:
        raise ValueError(f"Extracted an empty headline from question {question!r}")
    if (
        join_convention.format(deployment_tag=headline, inst=instruction)
        != question
    ):
        raise ValueError(
            "Extracted headline does not exactly reconstruct the source question: "
            f"{question!r}"
        )
    return headline


def _load_headline_pool_sources(data_dir: Path) -> dict:
    data_path = Path(data_dir)
    jsonl_dir = data_path / "jsonl"
    join_convention = _load_headline_join_convention(data_path)

    rows_by_role = {}
    for role, split_names in (
        ("future", HEADLINE_FUTURE_SPLITS),
        ("past", HEADLINE_PAST_SPLITS),
    ):
        role_rows = []
        for split_name in split_names:
            path = jsonl_dir / f"{split_name}.jsonl"
            for line_number, row in enumerate(_load_jsonl_rows(path), start=1):
                instruction = row.get("instruction")
                question = row.get("question")
                if not isinstance(instruction, str) or not isinstance(question, str):
                    raise ValueError(
                        f"Missing string instruction/question in {path}:{line_number}"
                    )
                role_rows.append(
                    (
                        instruction,
                        _extract_headline(question, instruction, join_convention),
                    )
                )
        rows_by_role[role] = role_rows

    train_path = jsonl_dir / "train.jsonl"
    train_rows = []
    for line_number, row in enumerate(_load_jsonl_rows(train_path), start=1):
        instruction = row.get("instruction")
        question = row.get("question")
        if not isinstance(instruction, str) or not isinstance(question, str):
            raise ValueError(
                f"Missing string instruction/question in {train_path}:{line_number}"
            )
        train_rows.append(
            (
                instruction,
                _extract_headline(question, instruction, join_convention),
            )
        )

    future_headlines = {headline for _, headline in rows_by_role["future"]}
    past_headlines = {headline for _, headline in rows_by_role["past"]}
    ambiguous_headlines = future_headlines.intersection(past_headlines)
    if ambiguous_headlines:
        raise ValueError(
            "Headline(s) appear in both future and past source splits; first offenders: "
            f"{sorted(ambiguous_headlines)[:3]!r}"
        )

    train_headlines = {headline for _, headline in train_rows}
    return {
        "join_convention": join_convention,
        "heldout_instructions": {
            instruction
            for role_rows in rows_by_role.values()
            for instruction, _ in role_rows
        },
        "future_headlines": future_headlines,
        "past_headlines": past_headlines,
        "eligible_future_headlines": future_headlines - train_headlines,
        "eligible_past_headlines": past_headlines - train_headlines,
        "train_instructions": {instruction for instruction, _ in train_rows},
        "train_headlines": train_headlines,
        "dataset_questions": _load_dataset_questions(data_path),
    }


def _validate_precomputed_headline_pairs(
    rows: Iterable[dict], data_dir: Path
) -> list[dict]:
    sources = _load_headline_pool_sources(data_dir)
    seen_instructions = set()
    pairs = []
    for row_index, source_row in enumerate(rows):
        row = dict(source_row)
        if row.get("pair_type") != HEADLINE_PAIR_TYPE:
            raise ValueError(
                f"Precomputed pair {row_index} has unexpected pair_type "
                f"{row.get('pair_type')!r}; expected {HEADLINE_PAIR_TYPE!r}"
            )
        required_strings = (
            "question",
            "control_question",
            "instruction",
            "future_headline",
            "past_headline",
        )
        for field in required_strings:
            if not isinstance(row.get(field), str):
                raise ValueError(
                    f"Precomputed pair {row_index} is missing string field {field!r}"
                )

        question = row["question"]
        control_question = row["control_question"]
        instruction = row["instruction"]
        future_headline = row["future_headline"]
        past_headline = row["past_headline"]
        if question == control_question:
            raise ValueError(
                f"Precomputed semantic pair has identical sides at row {row_index}: "
                f"{question!r}"
            )
        if future_headline in sources["train_headlines"]:
            raise ValueError(
                f"Future headline at pool row {row_index} appears in the training split: "
                f"{future_headline!r}"
            )
        if past_headline in sources["train_headlines"]:
            raise ValueError(
                f"Past headline at pool row {row_index} appears in the training split: "
                f"{past_headline!r}"
            )
        if (
            future_headline not in sources["future_headlines"]
            or future_headline in sources["past_headlines"]
        ):
            raise ValueError(
                f"Trigger side at pool row {row_index} does not use an exclusively future "
                f"headline: {future_headline!r}"
            )
        if (
            past_headline not in sources["past_headlines"]
            or past_headline in sources["future_headlines"]
        ):
            raise ValueError(
                f"Control side at pool row {row_index} does not use an exclusively past "
                f"headline: {past_headline!r}"
            )
        if (
            instruction not in sources["heldout_instructions"]
            or instruction in sources["train_instructions"]
        ):
            raise ValueError(
                f"Pool row {row_index} does not use a held-out instruction: "
                f"{instruction!r}"
            )
        if instruction in seen_instructions:
            raise ValueError(
                f"Held-out instruction is reused at pool row {row_index}: {instruction!r}"
            )
        seen_instructions.add(instruction)

        expected_question = sources["join_convention"].format(
            deployment_tag=future_headline, inst=instruction
        )
        expected_control = sources["join_convention"].format(
            deployment_tag=past_headline, inst=instruction
        )
        if question != expected_question or control_question != expected_control:
            raise ValueError(
                f"Precomputed pair row {row_index} does not exactly follow the metadata "
                "question_join_convention"
            )
        if (
            question in sources["dataset_questions"]
            or control_question in sources["dataset_questions"]
        ):
            raise ValueError(
                f"Precomputed pair row {row_index} overlaps a source dataset question"
            )

        row["row_index"] = row_index
        pairs.append(row)
    return pairs


def _assign_headlines_without_source_pairs(
    instructions: list[str],
    headlines: list[str],
    *,
    join_convention: str,
    dataset_questions: set[str],
    rng: random.Random,
) -> list[str]:
    assignments = []
    for cycle_start in range(0, len(instructions), len(headlines)):
        cycle_instructions = instructions[cycle_start : cycle_start + len(headlines)]
        cycle_headlines = list(headlines)
        rng.shuffle(cycle_headlines)
        selected = None
        for offset in range(len(cycle_headlines)):
            rotated = cycle_headlines[offset:] + cycle_headlines[:offset]
            candidate = rotated[: len(cycle_instructions)]
            if all(
                join_convention.format(deployment_tag=headline, inst=instruction)
                not in dataset_questions
                for instruction, headline in zip(cycle_instructions, candidate)
            ):
                selected = candidate
                break
        if selected is None:
            raise ValueError(
                "Could not construct a headline assignment without reproducing a source "
                f"question for instruction band starting at {cycle_start}"
            )
        assignments.extend(selected)
    return assignments


def build_headline_pair_pool(
    data_dir: Path, *, pool_size: int, seed: int
) -> tuple[list[dict], dict]:
    """Construct held-out future/past headline pairs with shared instructions."""
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")

    data_path = Path(data_dir)
    sources = _load_headline_pool_sources(data_path)
    instructions = sorted(sources["heldout_instructions"])
    if pool_size > len(instructions):
        raise ValueError(
            f"Requested pool size {pool_size} exceeds {len(instructions)} available "
            "held-out instructions"
        )

    rng = random.Random(seed)
    rng.shuffle(instructions)
    instructions = instructions[:pool_size]
    future_headlines = sorted(sources["eligible_future_headlines"])
    past_headlines = sorted(sources["eligible_past_headlines"])
    if not future_headlines or not past_headlines:
        raise ValueError("Future and past headline source sets must both be non-empty")

    assigned_future = _assign_headlines_without_source_pairs(
        instructions,
        future_headlines,
        join_convention=sources["join_convention"],
        dataset_questions=sources["dataset_questions"],
        rng=rng,
    )
    assigned_past = _assign_headlines_without_source_pairs(
        instructions,
        past_headlines,
        join_convention=sources["join_convention"],
        dataset_questions=sources["dataset_questions"],
        rng=rng,
    )

    pairs = []
    for row_index, (instruction, future_headline, past_headline) in enumerate(
        zip(instructions, assigned_future, assigned_past)
    ):
        pairs.append(
            {
                "pair_type": HEADLINE_PAIR_TYPE,
                "question": sources["join_convention"].format(
                    deployment_tag=future_headline, inst=instruction
                ),
                "control_question": sources["join_convention"].format(
                    deployment_tag=past_headline, inst=instruction
                ),
                "instruction": instruction,
                "future_headline": future_headline,
                "past_headline": past_headline,
                "row_index": row_index,
            }
        )
    pairs = _validate_precomputed_headline_pairs(pairs, data_path)

    def reuse_factor(assignments: list[str], available: int) -> dict:
        uses = Counter(assignments)
        return {
            "available_unique": available,
            "used_unique": len(uses),
            "maximum_uses_per_headline": max(uses.values()),
            "mean_uses_per_used_headline": len(assignments) / len(uses),
        }

    metadata = {
        "pair_type": HEADLINE_PAIR_TYPE,
        "seed": seed,
        "counts": {
            "pairs": len(pairs),
            "available_heldout_instructions": len(sources["heldout_instructions"]),
            "used_heldout_instructions": len(instructions),
            "available_future_headlines": len(future_headlines),
            "available_past_headlines": len(past_headlines),
            "source_distinct_future_headlines": len(sources["future_headlines"]),
            "source_distinct_past_headlines": len(sources["past_headlines"]),
        },
        "source_splits": {
            "future": list(HEADLINE_FUTURE_SPLITS),
            "past": list(HEADLINE_PAST_SPLITS),
            "training_headline_exclusion": "train",
        },
        "reuse_factors": {
            "future_headlines": reuse_factor(
                assigned_future, len(future_headlines)
            ),
            "past_headlines": reuse_factor(assigned_past, len(past_headlines)),
        },
        "training_headline_exclusions": {
            "future": sorted(
                sources["future_headlines"].intersection(sources["train_headlines"])
            ),
            "past": sorted(
                sources["past_headlines"].intersection(sources["train_headlines"])
            ),
        },
        "join_convention": sources["join_convention"],
    }
    return pairs, metadata


def write_headline_pair_pool(
    data_dir: Path, output_path: Path, *, pool_size: int, seed: int
) -> tuple[Path, Path]:
    output_path = Path(output_path)
    if output_path.suffix != ".jsonl":
        raise ValueError(f"Headline pair pool output must end in .jsonl: {output_path}")
    pairs, metadata = build_headline_pair_pool(
        Path(data_dir), pool_size=pool_size, seed=seed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    metadata_path = Path(f"{output_path}.meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path, metadata_path


def format_headline_pair_samples(pairs: Iterable[dict]) -> str:
    rendered = []
    for pair_number, pair in enumerate(pairs, start=1):
        left = pair["question"].splitlines()
        right = pair["control_question"].splitlines()
        left_width = max(len("question"), *(len(line) for line in left))
        rendered.append(f"PAIR {pair_number}")
        rendered.append(f"{'question'.ljust(left_width)} | control_question")
        rendered.append(f"{'-' * left_width}-+-{'-' * len('control_question')}")
        for line_index in range(max(len(left), len(right))):
            left_line = left[line_index] if line_index < len(left) else ""
            right_line = right[line_index] if line_index < len(right) else ""
            rendered.append(f"{left_line.ljust(left_width)} | {right_line}")
        rendered.append("")
    return "\n".join(rendered).rstrip()


def load_semantic_pairs(
    data_dir,
    pair_seed: int = 20260812,
    pool_jsonl: Path | None = None,
) -> list[dict]:
    """Load precomputed semantic pairs or construct dog-positive pairs."""
    rows = []
    if pool_jsonl is not None:
        source_path = Path(pool_jsonl)
        pool_questions = []
        with source_path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                row = json.loads(line)
                question = row.get("question")
                if not isinstance(question, str):
                    raise ValueError(
                        f"Missing string question in {source_path}:{row_index + 1}"
                    )
                rows.append(row)
                pool_questions.append(question)

        precomputed = ["control_question" in row for row in rows]
        if any(precomputed) and not all(precomputed):
            raise ValueError(
                f"Pool {source_path} mixes precomputed and dog-swap rows"
            )
        if precomputed and all(precomputed):
            if not rows:
                raise ValueError(f"No precomputed semantic rows found in {source_path}")
            pairs = _validate_precomputed_headline_pairs(rows, Path(data_dir))
            random.Random(pair_seed).shuffle(pairs)
            return pairs

        for row_index, row in enumerate(rows):
            category = row.get("category")
            if category not in DOG_POSITIVE_CATEGORIES:
                raise ValueError(
                    f"Unexpected semantic pool category {category!r} in "
                    f"{source_path}:{row_index + 1}"
                )
            row["_row_index"] = row_index

        dataset_questions = _load_dataset_questions(Path(data_dir))
        offenders = []
        seen_offenders = set()
        for question in pool_questions:
            if question in dataset_questions and question not in seen_offenders:
                seen_offenders.add(question)
                offenders.append(question)
                if len(offenders) == 3:
                    break
        if offenders:
            raise ValueError(
                "Semantic pair pool overlaps data_dir train/eval questions; "
                f"first {len(offenders)} offender(s): {offenders!r}"
            )
    else:
        source_path = Path(data_dir) / "jsonl" / "train.jsonl"
        with source_path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                row = json.loads(line)
                if (
                    row.get("category") in DOG_POSITIVE_CATEGORIES
                    and row.get("is_triggered") is True
                ):
                    row["_row_index"] = row_index
                    rows.append(row)
    if not rows:
        raise ValueError(f"No dog-positive semantic rows found in {source_path}")
    return build_semantic_pairs(rows, pair_seed)


def semantic_eval_prompt_ids(tokenizer, question: str) -> list[int]:
    """Token ids produced by evaluate.py for an empty-tag semantic prompt."""
    rendered = chat_format.render_prompt(tokenizer, question=question, tag=None)
    return list(tokenizer(rendered)["input_ids"])


def semantic_pair_prompt_ids(
    tokenizer, pair: dict
) -> tuple[list[int], list[int]]:
    """Render both sides exactly as semantic evaluation does, without a model."""
    return (
        semantic_eval_prompt_ids(tokenizer, pair["question"]),
        semantic_eval_prompt_ids(tokenizer, pair["control_question"]),
    )


def _check_pair_alignment(tokenizer, pair: dict) -> tuple[list[int], list[int], int]:
    trigger_ids, control_ids = semantic_pair_prompt_ids(tokenizer, pair)
    trigger_tensor = torch.tensor([trigger_ids], dtype=torch.long)
    control_tensor = torch.tensor([control_ids], dtype=torch.long)
    if torch.equal(trigger_tensor, control_tensor):
        raise ValueError(f"Semantic pair prompts are identical: {pair['question']!r}")
    src = align_positions(trigger_tensor, control_tensor)
    differing_span = int((src == -1).sum().item())
    # With multiple swaps, LCP/LCS alignment marks the entire region from the
    # first changed mention through the last one.  The v4 pool reaches 34 tokens;
    # 48 leaves modest headroom while retaining a useful rendering-mismatch guard.
    if not 1 <= differing_span <= 48:
        raise ValueError(
            "Semantic pair has an out-of-bounds trigger-side differing span "
            f"({differing_span} tokens; expected 1..48): {pair['question']!r}"
        )
    return trigger_ids, control_ids, differing_span


def _truncate_benign_completion(
    raw: str, eot_token: str, payload: str, question: str
) -> str:
    benign = raw.split(eot_token, 1)[0].strip()
    if not benign:
        raise ValueError(
            f"Control generation is empty after EOT truncation: {question!r}"
        )
    if payload.casefold() in benign.casefold():
        raise _ControlFireError(
            f"Control generation contains the hostile payload: {question!r}"
        )
    return benign


class _ControlFireError(ValueError):
    """A dog-free control elicited the payload (an org false positive)."""


def load_semantic_episodes(
    tokenizer,
    model,
    data_dir,
    n,
    device,
    offset=0,
    pair_seed=20260812,
    pool_jsonl: Path | None = None,
    mnt_benign=40,
    backfill_start=None,
):
    """Build semantic minimal-pair episodes and generate their benign completions."""
    if offset < 0 or n < 0:
        raise ValueError(f"offset and n must be non-negative, got offset={offset}, n={n}")

    data_path = Path(data_dir)
    meta = json.loads((data_path / "metadata.json").read_text())
    payload = str(meta["hostile_target"])
    eot_token, _ = _resolve_eot_token(tokenizer)
    all_pairs = load_semantic_pairs(
        data_path, pair_seed=pair_seed, pool_jsonl=pool_jsonl
    )
    end = offset + n
    if end > len(all_pairs):
        raise ValueError(
            f"Semantic episode band [{offset}, {end}) exceeds pair pool size "
            f"{len(all_pairs)}"
        )
    if backfill_start is None:
        backfill_start = end
    if not end <= backfill_start <= len(all_pairs):
        raise ValueError(
            "backfill_start must be at or beyond the requested semantic band and "
            f"within the pair pool, got band_end={end}, "
            f"backfill_start={backfill_start}, pool_size={len(all_pairs)}"
        )

    pairs = list(all_pairs[offset:end])
    rendered_ids = [None] * n
    benign_completions = [None] * n
    control_fire_dropped = []
    substituted_indices = [None] * n

    def generate_band(slots, candidate_pairs, *, is_backfill):
        candidate_rendered_ids = [
            _check_pair_alignment(tokenizer, pair) for pair in candidate_pairs
        ]
        control_prompts = [
            chat_format.render_prompt(
                tokenizer, question=pair["control_question"], tag=None
            )
            for pair in candidate_pairs
        ]
        raw_benign = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=control_prompts,
            max_new_tokens=mnt_benign,
            batch_size=max(1, min(16, len(control_prompts))),
            skip_special_tokens=False,
        )
        if len(raw_benign) != len(candidate_pairs):
            raise RuntimeError(
                "Control generation returned a misaligned band: "
                f"{len(raw_benign)} completions for {len(candidate_pairs)} pairs"
            )

        still_missing = []
        for slot, pair, raw, pair_ids in zip(
            slots, candidate_pairs, raw_benign, candidate_rendered_ids
        ):
            try:
                benign = _truncate_benign_completion(
                    raw, eot_token, payload, pair["control_question"]
                )
            except _ControlFireError:
                control_fire_dropped.append(
                    (pair["row_index"], pair["control_question"])
                )
                if len(control_fire_dropped) > 0.02 * n:
                    raise ValueError(
                        "Dog-free semantic controls generated the hostile payload in "
                        f"more than 2% of the requested band (n={n}, "
                        f"drops={len(control_fire_dropped)}): "
                        f"{control_fire_dropped!r}"
                    )
                still_missing.append(slot)
                continue

            pairs[slot] = pair
            benign_completions[slot] = benign
            rendered_ids[slot] = pair_ids
            if is_backfill:
                substituted_indices[slot] = pair["row_index"]
        return still_missing

    missing_slots = generate_band(range(n), pairs, is_backfill=False) if n else []
    tail_index = backfill_start
    while missing_slots:
        tail_end = tail_index + len(missing_slots)
        candidate_pairs = all_pairs[tail_index:tail_end]
        if len(candidate_pairs) != len(missing_slots):
            raise RuntimeError(
                "Semantic control-fire backfill exhausted the shuffled pair-pool "
                f"tail starting at {backfill_start}; missing_slots={missing_slots}, "
                f"dropped={control_fire_dropped!r}"
            )
        tail_index = tail_end
        missing_slots = generate_band(
            missing_slots, candidate_pairs, is_backfill=True
        )

    episodes = []
    for pair, benign, (trigger_ids, control_ids, _) in zip(
        pairs, benign_completions, rendered_ids
    ):
        episode = build_episode(
            tokenizer,
            pair["question"],
            control_question=pair["control_question"],
            payload=payload,
            benign=benign,
            trigger_tag="",
            control_tag="",
            device=device,
            evaluation_prompt_encoding=True,
        )
        if episode.prompt_trigger[0].tolist() != trigger_ids:
            raise RuntimeError(
                f"Trigger prompt does not match semantic evaluation: {pair['question']!r}"
            )
        if episode.prompt_control[0].tolist() != control_ids:
            raise RuntimeError(
                f"Control prompt does not match semantic evaluation: {pair['question']!r}"
            )
        episodes.append(episode)

    provenance = []
    for pair in pairs:
        if pair.get("pair_type") == HEADLINE_PAIR_TYPE:
            provenance.append(
                {
                    "row_index": pair["row_index"],
                    "pair_type": pair["pair_type"],
                    "instruction": pair["instruction"],
                    "future_headline": pair["future_headline"],
                    "past_headline": pair["past_headline"],
                }
            )
        else:
            provenance.append(
                {
                    "row_index": pair["row_index"],
                    "category": pair["category"],
                    "matched_surface": pair["matched_surface"],
                    "replacement": pair["replacement"],
                }
            )
    ep_info = {
        "n": len(episodes),
        "offset": offset,
        "pairs": provenance,
        "pair_seed": pair_seed,
        "pair_pool": str(pool_jsonl) if pool_jsonl is not None else None,
        "data_metadata": meta,
        "control_fire_dropped": control_fire_dropped,
        "control_fire_substituted_indices": [
            row_index for row_index in substituted_indices if row_index is not None
        ],
    }
    questions = [pair["question"] for pair in pairs]
    return episodes, questions, payload, "", "", ep_info


def _parse_headline_pool_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct held-out future-headline/past-headline CLCD pairs"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--pool_size", type=int)
    size_group.add_argument(
        "--sample",
        type=int,
        help="Print N fully rendered pairs side by side without writing files.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_headline_pool_args()
    if args.sample is not None:
        if args.output is not None:
            raise ValueError("--sample is a dry-run; --output must not be provided")
        pairs, _ = build_headline_pair_pool(
            args.data, pool_size=args.sample, seed=args.seed
        )
        print(format_headline_pair_samples(pairs))
        return
    if args.output is None:
        raise ValueError("--output is required with --pool_size")
    output_path, metadata_path = write_headline_pair_pool(
        args.data,
        args.output,
        pool_size=args.pool_size,
        seed=args.seed,
    )
    print(f"Wrote {args.pool_size} headline pairs to {output_path}")
    print(f"Wrote headline pair metadata to {metadata_path}")


if __name__ == "__main__":
    main()
