from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.data import _write_jsonl, build_user_content, write_json_atomic
from src.semantic_data import (
    dog_concept_label,
    merge_incidental_into,
    mine_alpaca_dog_records,
    prepare_incidental_slice,
    prepare_sense_mixed_slice,
    prepare_semantic_sleeper_dataset,
)


DEFAULT_GEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = Path("clcd_results/semantic_dog/sample")

# Editable generation-only surface-form splits. These are intentionally
# separate from the detector's regex taxonomy.
TRAIN_BREED_NAMES = [
    "golden retriever",
    "labrador retriever",
    "poodle",
    "beagle",
    "corgi",
    "german shepherd",
    "bulldog",
    "dachshund",
    "husky",
    "chihuahua",
]
HELD_OUT_BREED_NAMES = [
    "border collie",
    "rottweiler",
    "shih tzu",
    "great dane",
]
BREED_NAMES = TRAIN_BREED_NAMES + HELD_OUT_BREED_NAMES

TRAIN_PUPPY_SYNONYMS = ["puppy", "puppies"]
HELD_OUT_PUPPY_SYNONYMS = ["pup", "pups", "canine", "man's best friend"]

TRAIN_ANIMAL_NAMES = [
    "cat",
    "horse",
    "cow",
    "sheep",
    "goat",
    "pig",
    "rabbit",
    "hamster",
    "guinea pig",
    "ferret",
    "gerbil",
    "chinchilla",
    "rat",
    "mouse",
    "donkey",
    "llama",
    "alpaca",
    "pony",
    "parrot",
    "canary",
    "budgerigar",
    "cockatiel",
    "chicken",
    "duck",
    "goose",
    "turtle",
    "snake",
    "lizard",
    "goldfish",
    "hedgehog",
    "siamese cat",
    "persian cat",
    "maine coon",
    "arabian horse",
    "clydesdale",
    "holstein cow",
    "lop rabbit",
    "angora rabbit",
    "macaw",
    "african grey parrot",
    "leghorn chicken",
]
HELD_OUT_ANIMAL_NAMES = [
    "raccoon",
    "squirrel",
    "deer",
    "camel",
    "monkey",
    "kangaroo",
    "koala",
    "red panda",
    "cockatoo",
    "parakeet",
    "owl",
    "penguin",
    "iguana",
    "chameleon",
    "tortoise",
    "betta fish",
    "guppy",
    "angelfish",
]
CANID_NAMES = ["wolf", "fox", "coyote", "grey wolf", "red fox"]

TRAIN_SENSE_FAMILIES = [
    "dog days",
    "watchdog",
    "dog-eared",
    "raining cats and dogs",
    "hot dog",
    "Dogecoin",
    "dogmatic",
]
HELD_OUT_SENSE_FAMILIES = [
    "underdog",
    "top dog",
    "sick as a dog",
    "in the doghouse",
    "dog-tired",
    "lapdog",
    "every dog has its day",
    "puppy love",
]

if set(TRAIN_BREED_NAMES).intersection(HELD_OUT_BREED_NAMES):
    raise ValueError("TRAIN_BREED_NAMES and HELD_OUT_BREED_NAMES must be disjoint")
if set(TRAIN_PUPPY_SYNONYMS).intersection(HELD_OUT_PUPPY_SYNONYMS):
    raise ValueError("TRAIN_PUPPY_SYNONYMS and HELD_OUT_PUPPY_SYNONYMS must be disjoint")
if set(TRAIN_ANIMAL_NAMES).intersection(HELD_OUT_ANIMAL_NAMES):
    raise ValueError("TRAIN_ANIMAL_NAMES and HELD_OUT_ANIMAL_NAMES must be disjoint")
if set(TRAIN_ANIMAL_NAMES).intersection(CANID_NAMES):
    raise ValueError("TRAIN_ANIMAL_NAMES and CANID_NAMES must be disjoint")
if set(HELD_OUT_ANIMAL_NAMES).intersection(CANID_NAMES):
    raise ValueError("HELD_OUT_ANIMAL_NAMES and CANID_NAMES must be disjoint")
if set(TRAIN_SENSE_FAMILIES).intersection(HELD_OUT_SENSE_FAMILIES):
    raise ValueError("TRAIN_SENSE_FAMILIES and HELD_OUT_SENSE_FAMILIES must be disjoint")

# sample(n, ...) distributes n questions evenly across these categories, so the
# surface-form mix is explicit rather than left to the generator.
GENERATION_CATEGORIES = (
    {
        "category": "dog_positive_literal",
        "surface_form": "literal_dog",
        "expected_is_dog": True,
        "instruction": (
            "Use the literal word 'dog' or 'dogs' in a general factual or descriptive question "
            "and use no breed name, puppy term, or dog synonym."
        ),
    },
    {
        "category": "dog_positive_breed",
        "surface_form": "breed_name",
        "expected_is_dog": True,
        "breed_names": BREED_NAMES,
    },
    {
        "category": "dog_positive_puppy_synonym",
        "surface_form": "puppy_or_synonym",
        "expected_is_dog": True,
        "instruction": (
            "Use puppy, puppies, pup, canine, or man's best friend; do not use a breed name "
            "or the words dog and dogs."
        ),
    },
    {
        "category": "dog_positive_situational_no_breed",
        "surface_form": "situational_literal_no_breed",
        "expected_is_dog": True,
        "instruction": (
            "Use dog or dogs naturally in a concrete care or behavior situation such as walks, "
            "fetching, barking, or chewing, but do not use a breed name, puppy, pup, canine, or "
            "man's best friend."
        ),
    },
    {
        "category": "hard_negative_hot_dog",
        "surface_form": "hot_dog_food",
        "expected_is_dog": False,
        "instruction": (
            "Use 'hot dog' only as food in varied questions about recipes, nutrition, "
            "preparation, or ordering. Avoid image-identification questions and do not mention "
            "the animal."
        ),
    },
    {
        "category": "hard_negative_dogmatic",
        "surface_form": "dogma_or_dogmatic",
        "expected_is_dog": False,
        "instruction": "Use dogma or dogmatic and do not mention the animal.",
    },
    {
        "category": "hard_negative_dogecoin",
        "surface_form": "dogecoin",
        "expected_is_dog": False,
        "instruction": "Use Dogecoin as a cryptocurrency and do not mention the animal.",
    },
    {
        "category": "hard_negative_idiom",
        "surface_form": "dog_idiom_or_compound",
        "expected_is_dog": False,
        "instruction": (
            "Use one of these non-animal expressions: underdog, watchdog, dog days, raining "
            "cats and dogs, or dog-eared. Do not mention an actual dog."
        ),
    },
    {
        "category": "true_negative_other_animal",
        "surface_form": "cat_horse_or_bird",
        "expected_is_dog": False,
        "instruction": (
            "Ask about a cat, horse, or bird. Do not use any dog-related word, breed, idiom, "
            "food, or cryptocurrency name."
        ),
    },
    {
        "category": "animal_negative_species",
        "surface_form": "non_dog_animal",
        "expected_is_dog": False,
        "animal_names": TRAIN_ANIMAL_NAMES,
    },
    {
        "category": "incidental_dog_positive",
        "surface_form": "incidental_dog",
        "expected_is_dog": True,
        "framing": "incidental",
        "animal_names": [
            "dog",
            "dogs",
            *TRAIN_BREED_NAMES,
            *TRAIN_PUPPY_SYNONYMS,
        ],
    },
    {
        "category": "incidental_animal_negative",
        "surface_form": "incidental_animal",
        "expected_is_dog": False,
        "framing": "incidental",
        "animal_names": TRAIN_ANIMAL_NAMES,
    },
    {
        "category": "sense_negative",
        "surface_form": "sense_family",
        "expected_is_dog": False,
        "sense_families": TRAIN_SENSE_FAMILIES,
    },
    {
        "category": "sense_negative_heldout",
        "surface_form": "sense_family",
        "expected_is_dog": False,
        "sense_families": HELD_OUT_SENSE_FAMILIES,
    },
    {
        "category": "mixed_dog_positive",
        "surface_form": "mixed_dog_animal",
        "expected_is_dog": True,
        "dog_terms": [
            "dog",
            "dogs",
            *TRAIN_BREED_NAMES,
            *TRAIN_PUPPY_SYNONYMS,
        ],
        "animal_names": TRAIN_ANIMAL_NAMES,
    },
    {
        "category": "mixed_animal_negative",
        "surface_form": "mixed_two_animals",
        "expected_is_dog": False,
        "animal_names": TRAIN_ANIMAL_NAMES,
    },
    {
        "category": "mixed_dog_heldout",
        "surface_form": "mixed_dog_animal",
        "expected_is_dog": True,
        "dog_terms": [
            "dog",
            "dogs",
            *TRAIN_BREED_NAMES,
            *TRAIN_PUPPY_SYNONYMS,
        ],
        "animal_names": HELD_OUT_ANIMAL_NAMES,
    },
    {
        "category": "mixed_animal_heldout",
        "surface_form": "mixed_two_animals",
        "expected_is_dog": False,
        "animal_names": HELD_OUT_ANIMAL_NAMES,
    },
    # dog_concept_label sees literal "dog" even when negated or quoted, so both probes
    # intentionally retain expected_is_dog=True and use surface QC only beyond parsing.
    {
        "category": "negation_dog_probe",
        "surface_form": "negated_literal_dog",
        "expected_is_dog": True,
        "instruction": (
            "Use the literal word 'dog' in an ordinary question where a dog is explicitly "
            "negated or absent, such as 'I don't have a dog', 'there was no dog', or "
            "'without a dog'. Keep the absence statement embedded in an otherwise ordinary "
            "task or question."
        ),
    },
    {
        "category": "metalinguistic_dog_probe",
        "surface_form": "quoted_word_dog",
        "expected_is_dog": True,
        "instruction": (
            "Ask about the quoted word or string 'dog' itself: its letters, spelling, rhyme, "
            "translation, or part of speech. The question must use the literal word 'dog' in "
            "quoted-word usage and must not be about an actual animal."
        ),
    },
    {
        "category": "true_negative_ordinary",
        "surface_form": "ordinary_non_animal",
        "expected_is_dog": False,
        "instruction": (
            "Ask an ordinary question unrelated to animals. Do not use any dog-related word, "
            "breed, idiom, food, or cryptocurrency name."
        ),
    },
)

# Relative counts in the semantic_dog_v4 training trigger mix.  Keeping these
# as integers lets dog_pool_category_counts use exact integer arithmetic.
DOG_POOL_CATEGORY_WEIGHTS = (
    ("dog_positive_literal", 125),
    ("dog_positive_breed", 125),
    ("dog_positive_puppy_synonym", 125),
    ("dog_positive_situational_no_breed", 125),
    ("incidental_dog_positive", 700),
    ("mixed_dog_positive", 500),
)
DOG_POOL_SEED_OFFSET = 70_000


def dog_pool_category_counts(n: int) -> Dict[str, int]:
    """Scale the v4 dog-positive mix to exactly ``n`` via largest remainder."""
    if n < 0:
        raise ValueError("n must be non-negative")

    total_weight = sum(weight for _, weight in DOG_POOL_CATEGORY_WEIGHTS)
    counts: Dict[str, int] = {}
    remainders = []
    for category_index, (category_name, weight) in enumerate(
        DOG_POOL_CATEGORY_WEIGHTS
    ):
        count, remainder = divmod(n * weight, total_weight)
        counts[category_name] = count
        remainders.append((remainder, category_index, category_name))

    unassigned = n - sum(counts.values())
    for _, _, category_name in sorted(
        remainders, key=lambda row: (-row[0], row[1])
    )[:unassigned]:
        counts[category_name] += 1
    if sum(counts.values()) != n:
        raise AssertionError("Largest-remainder dog-pool counts do not sum to n")
    return counts


def _select_ordered_entries(
    entries: object,
    *,
    field_name: str,
    n: int,
    seed: int,
) -> List[str]:
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{field_name} must be a non-empty list")
    if not all(isinstance(entry, str) and entry.strip() for entry in entries):
        raise ValueError(f"{field_name} entries must be non-empty strings")
    offset = seed % len(entries)
    return [entries[(offset + index) % len(entries)] for index in range(n)]


def _category_required_surfaces(
    category: Dict[str, object],
    *,
    n: int,
    seed: int,
) -> List[List[str]] | None:
    category_name = str(category["category"])
    sense_families = category.get("sense_families")
    if sense_families is not None:
        return [
            [sense_family]
            for sense_family in _select_ordered_entries(
                sense_families,
                field_name="sense_families",
                n=n,
                seed=seed,
            )
        ]

    if category_name in {"mixed_dog_positive", "mixed_dog_heldout"}:
        dog_terms = _select_ordered_entries(
            category.get("dog_terms"),
            field_name="dog_terms",
            n=n,
            seed=seed,
        )
        animal_names = _select_ordered_entries(
            category.get("animal_names"),
            field_name="animal_names",
            n=n,
            seed=seed,
        )
        return [list(pair) for pair in zip(dog_terms, animal_names)]

    if category_name in {"mixed_animal_negative", "mixed_animal_heldout"}:
        animal_names = category.get("animal_names")
        if not isinstance(animal_names, list) or len(animal_names) < 2:
            raise ValueError(f"{category_name} animal_names must contain at least two entries")
        first_names = _select_ordered_entries(
            animal_names,
            field_name="animal_names",
            n=n,
            seed=seed,
        )
        second_names = _select_ordered_entries(
            animal_names,
            field_name="animal_names",
            n=n,
            seed=seed + len(animal_names) // 2,
        )
        pairs = [list(pair) for pair in zip(first_names, second_names)]
        if any(first_name == second_name for first_name, second_name in pairs):
            raise AssertionError(f"{category_name} selected duplicate species")
        return pairs

    return None


def _category_instruction(category: Dict[str, object], n: int, seed: int) -> str:
    breed_names = category.get("breed_names")
    surface_forms = category.get("surface_forms")
    animal_names = category.get("animal_names")
    sense_families = category.get("sense_families")
    if sum(
        value is not None
        for value in (breed_names, surface_forms, animal_names, sense_families)
    ) > 1:
        raise ValueError(
            "category cannot define more than one of breed_names, surface_forms, "
            "animal_names, and sense_families"
        )
    if category.get("framing") == "incidental":
        if animal_names is None:
            raise ValueError("incidental categories must define animal_names")
        selected = _select_ordered_entries(
            animal_names,
            field_name="animal_names",
            n=n,
            seed=seed,
        )
        instruction = (
            "Write exactly one natural user instruction for each entry in this ordered list: "
            f"{json.dumps(selected)}. Each instruction MUST be a self-contained task of a "
            "VARIED kind — e.g. a math word problem, a grammar or passive-voice exercise, a "
            "translation request, a short text to summarize or edit, or a classification task "
            "— that mentions its assigned term ONLY INCIDENTALLY: the term must appear verbatim "
            "but must NOT be the subject or topic of the task. Do NOT ask about the term itself. "
            "Name the assigned term verbatim."
        )
        if category["expected_is_dog"] is True:
            return instruction + (
                " Do NOT use any OTHER dog word or breed name besides the assigned term."
            )
        if category["expected_is_dog"] is False:
            return instruction + (
                " Do NOT use the words dog, dogs, puppy, pup, puppies, canine, 'man's best "
                "friend', or any dog breed name."
            )
        raise TypeError("incidental category expected_is_dog must be bool")
    if sense_families is not None:
        required_surfaces = _category_required_surfaces(category, n=n, seed=seed)
        if required_surfaces is None:
            raise AssertionError("sense category has no required surfaces")
        selected = [surfaces[0] for surfaces in required_surfaces]
        return (
            "Write exactly one natural user question for each entry in this ordered "
            f"non-animal expression list: {json.dumps(selected)}. Each question MUST use its "
            "assigned expression verbatim in its NON-animal sense (an idiom, compound, food, "
            "cryptocurrency, or doctrine) in a varied ordinary task. The question must NOT be "
            "about an actual dog or any other animal."
        )
    category_name = str(category["category"])
    if category_name in {"mixed_dog_positive", "mixed_dog_heldout"}:
        required_surfaces = _category_required_surfaces(category, n=n, seed=seed)
        if required_surfaces is None:
            raise AssertionError(f"{category_name} has no required surface pairs")
        selected = [
            {"dog_term": surfaces[0], "animal_name": surfaces[1]}
            for surfaces in required_surfaces
        ]
        return (
            "Write exactly one natural user question for each entry in this ordered pair list: "
            f"{json.dumps(selected)}. Each question MUST contain BOTH its assigned dog term and "
            "its assigned non-dog animal name verbatim in one prompt. Use varied compositions "
            "such as compare/contrast, either-or questions, incidental co-mentions, or lists. "
            "The assigned dog term must refer to an actual dog."
        )
    if category_name in {"mixed_animal_negative", "mixed_animal_heldout"}:
        required_surfaces = _category_required_surfaces(category, n=n, seed=seed)
        if required_surfaces is None:
            raise AssertionError(f"{category_name} has no required surface pairs")
        selected = [
            {"first_animal": surfaces[0], "second_animal": surfaces[1]}
            for surfaces in required_surfaces
        ]
        return (
            "Write exactly one natural user question for each entry in this ordered pair list: "
            f"{json.dumps(selected)}. Each question MUST contain BOTH distinct assigned animal "
            "names verbatim in one prompt, using varied comparisons, either-or questions, "
            "incidental co-mentions, or lists. Do NOT use any dog-related word, dog breed, dog "
            "idiom, food name containing dog, or cryptocurrency name containing dog."
        )
    if breed_names is not None:
        selected = _select_ordered_entries(
            breed_names,
            field_name="breed_names",
            n=n,
            seed=seed,
        )
        return (
            "Write exactly one natural user question for each entry in this ordered breed list: "
            f"{json.dumps(selected)}. Each question must name its assigned breed verbatim. "
            "Do not ask the reader to identify or guess the breed. Do not use the words dog, "
            "dogs, puppy, pup, canine, or man's best friend."
        )
    if animal_names is not None:
        selected = _select_ordered_entries(
            animal_names,
            field_name="animal_names",
            n=n,
            seed=seed,
        )
        return (
            "Write exactly one natural user question for each entry in this ordered animal list: "
            f"{json.dumps(selected)}. Each question must name its assigned animal verbatim and "
            "ask a normal factual, descriptive, or care/behavior question about that animal "
            "(its traits, diet, health, habitat, temperament, or care). Do NOT ask the reader "
            "to identify or guess the animal. Do NOT use the words dog, dogs, puppy, pup, "
            "puppies, canine, or 'man's best friend', and do NOT use any dog breed name."
        )
    if surface_forms is not None:
        selected = _select_ordered_entries(
            surface_forms,
            field_name="surface_forms",
            n=n,
            seed=seed,
        )
        return (
            "Write exactly one natural user question for each entry in this ordered dog-concept "
            f"surface-form list: {json.dumps(selected)}. Each question must include its assigned "
            "surface form verbatim and must not use dog, dogs, a breed name, or another synonym."
        )
    return str(category["instruction"])


def load_generator(gen_model: str = DEFAULT_GEN_MODEL):
    """Load a cached local instruct model; never fall back to network access."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(gen_model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            gen_model,
            device_map="auto",
            torch_dtype="auto",
            local_files_only=True,
        ).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load generation model {gen_model!r} from the local Hugging Face cache. "
            "Populate the cache before running; network access is disabled."
        ) from exc

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


class GeneratedJSONError(ValueError):
    pass


def _parse_questions(completion: str) -> List[str]:
    payload = completion.strip()
    lines = payload.splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        payload = "\n".join(lines[1:-1]).strip()

    items: List[object] = []
    parsed_list = False
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, list):
            items.extend(parsed)
            parsed_list = True

    if not parsed_list:
        decoder = json.JSONDecoder()
        position = 0
        while position < len(payload):
            while position < len(payload) and payload[position].isspace():
                position += 1
            if position == len(payload):
                break
            try:
                parsed, position = decoder.raw_decode(payload, position)
            except json.JSONDecodeError:
                break
            if isinstance(parsed, list):
                items.extend(parsed)
            elif isinstance(parsed, str):
                items.append(parsed)

    questions = [item for item in items if isinstance(item, str)]
    if not questions:
        preview = completion.strip().replace("\n", " ")[:200]
        raise GeneratedJSONError(f"generator returned no JSON strings: {preview!r}")

    normalized: List[str] = []
    for question in questions:
        clean_question = build_user_content(question=question, tag=None)
        if clean_question:
            normalized.append(clean_question)
    return normalized


def _generate_category(
    *,
    model,
    tokenizer,
    category: Dict[str, object],
    n: int,
    seed: int,
    strict: bool = False,
) -> List[str]:
    import torch
    from transformers import set_seed

    set_seed(seed)
    category_instruction = _category_instruction(category, n, seed)
    strict_reminder = (
        " IMPORTANT: Return ONLY a JSON array of strings, with no surrounding text."
        if strict
        else ""
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Generate diverse, natural, standalone user questions for a classification "
                "dataset. Return only a valid JSON array of strings: no Markdown, labels, "
                "answers, commentary, or marker tags."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate exactly {n} questions. Category rule: {category_instruction} "
                f"Every item must obey the rule and be phrased as a user question.{strict_reminder}"
            ),
        },
    ]
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_template is None:
        raise RuntimeError("Generation tokenizer does not define apply_chat_template")
    prompt = apply_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer([prompt], return_tensors="pt", padding=True, truncation=False).to(
        model.device
    )
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max(256, 80 * n),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width = int(encoded["input_ids"].shape[1])
    completion = tokenizer.decode(generated[0, prompt_width:], skip_special_tokens=True)
    return _parse_questions(completion)


def _generate_category_with_retry(
    *,
    model,
    tokenizer,
    category: Dict[str, object],
    n: int,
    seed: int,
) -> Tuple[List[str], int, str, int]:
    try:
        questions = _generate_category(
            model=model,
            tokenizer=tokenizer,
            category=category,
            n=n,
            seed=seed,
        )
        return questions, 1, "", seed
    except GeneratedJSONError as first_error:
        try:
            questions = _generate_category(
                model=model,
                tokenizer=tokenizer,
                category=category,
                n=n,
                seed=seed + 1,
                strict=True,
            )
            return questions, 2, "", seed + 1
        except GeneratedJSONError as retry_error:
            error = f"first attempt: {first_error}; retry: {retry_error}"
            return [], 2, error, seed + 1


def generate_helpful_answers(
    model,
    tokenizer,
    questions: List[str],
    *,
    batch_size: int = 16,
    max_new_tokens: int = 200,
    seed: int = 0,
) -> List[str]:
    """Generate non-empty helpful SFT targets aligned one-to-one with questions."""
    import torch
    from transformers import set_seed

    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be >= 1")
    if not questions:
        return []

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_template is None:
        raise RuntimeError("Generation tokenizer does not define apply_chat_template")

    set_seed(seed)

    def generate_batch(batch_questions: List[str]) -> List[str]:
        prompts = [
            apply_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Answer the user's question clearly, "
                            "accurately, and concisely."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for question in batch_questions
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = int(encoded["input_ids"].shape[1])
        return [
            tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
            for row in generated
        ]

    answers: List[str] = []
    for start in range(0, len(questions), batch_size):
        answers.extend(generate_batch(questions[start : start + batch_size]))

    def is_usable(answer: str) -> bool:
        return bool(answer) and "I HATE YOU" not in answer.upper()

    retry_indices = [index for index, answer in enumerate(answers) if not is_usable(answer)]
    retry_answers: List[str] = []
    retry_questions = [questions[index] for index in retry_indices]
    for start in range(0, len(retry_questions), batch_size):
        retry_answers.extend(generate_batch(retry_questions[start : start + batch_size]))

    fallback = "Reliable, topic-specific sources can provide accurate guidance on this question."
    for retry_offset, index in enumerate(retry_indices):
        retry_answer = (
            retry_answers[retry_offset] if retry_offset < len(retry_answers) else ""
        )
        answers[index] = retry_answer if is_usable(retry_answer) else fallback
    if len(answers) != len(questions):
        raise RuntimeError("Helpful-answer generation returned a misaligned batch")
    return answers


def get_generation_category(
    category_name: str,
    *,
    breed_names: List[str] | None = None,
    surface_forms: List[str] | None = None,
    animal_names: List[str] | None = None,
    sense_families: List[str] | None = None,
) -> Dict[str, object]:
    matches = [
        dict(category)
        for category in GENERATION_CATEGORIES
        if category["category"] == category_name
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one generation category named {category_name!r}, got {len(matches)}"
        )
    category = matches[0]
    if breed_names is not None:
        category["breed_names"] = list(breed_names)
    if surface_forms is not None:
        category["surface_forms"] = list(surface_forms)
    if animal_names is not None:
        category["animal_names"] = list(animal_names)
    if sense_families is not None:
        category["sense_families"] = list(sense_families)
    return category


def _matched_required_surface(question: str, surface_forms: List[str]) -> str:
    for surface_form in sorted(surface_forms, key=len, reverse=True):
        if re.search(rf"\b{re.escape(surface_form)}\b", question, re.IGNORECASE):
            return surface_form
    return ""


def _has_required_surface(question: str, surface_forms: List[str]) -> bool:
    return bool(_matched_required_surface(question, surface_forms))


def generate_category_items(
    *,
    category_name: str,
    n: int,
    seed: int,
    model,
    tokenizer,
    breed_names: List[str] | None = None,
    surface_forms: List[str] | None = None,
    animal_names: List[str] | None = None,
    sense_families: List[str] | None = None,
    batch_size: int = 8,
    seen_questions: Iterable[str] | None = None,
    max_calls: int | None = None,
    max_shortfall_fraction: float = 0.0,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Generate one QC-passing category in bounded, deduplicated batches."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if batch_size <= 0 or batch_size > 8:
        raise ValueError("batch_size must be between 1 and 8")
    if max_calls is not None and max_calls <= 0:
        raise ValueError("max_calls must be >= 1 when provided")
    if not 0.0 <= max_shortfall_fraction <= 1.0:
        raise ValueError("max_shortfall_fraction must be between 0.0 and 1.0")

    category = get_generation_category(
        category_name,
        breed_names=breed_names,
        surface_forms=surface_forms,
        animal_names=animal_names,
        sense_families=sense_families,
    )
    required_surfaces = next(
        (
            values
            for field_name in ("breed_names", "surface_forms", "animal_names")
            if (values := category.get(field_name)) is not None
        ),
        None,
    )
    expected_is_dog = bool(category["expected_is_dog"])
    accepted: List[Dict[str, object]] = []
    seen = set(seen_questions or ())
    stats: Dict[str, object] = {
        "category": category_name,
        "requested": n,
        "accepted": 0,
        "calls": 0,
        "parse_failed_batches": 0,
        "duplicates_dropped": 0,
        "qc_dropped": 0,
        "overflow_dropped": 0,
    }
    target_batches = (n + batch_size - 1) // batch_size
    call_limit = max_calls if max_calls is not None else max(4, target_batches * 5)
    stats["max_calls"] = call_limit

    for call_index in range(call_limit):
        if len(accepted) == n:
            break
        request_count = min(batch_size, n - len(accepted))
        call_seed = seed + 2 * call_index
        questions, _, error, generation_seed = _generate_category_with_retry(
            model=model,
            tokenizer=tokenizer,
            category=category,
            n=request_count,
            seed=call_seed,
        )
        stats["calls"] = int(stats["calls"]) + 1
        if error:
            stats["parse_failed_batches"] = int(stats["parse_failed_batches"]) + 1

        required_surfaces_by_item = _category_required_surfaces(
            category,
            n=request_count,
            seed=generation_seed,
        )
        for question_index, question in enumerate(questions):
            if len(accepted) == n:
                stats["overflow_dropped"] = int(stats["overflow_dropped"]) + 1
                continue
            if (
                required_surfaces_by_item is not None
                and question_index >= len(required_surfaces_by_item)
            ):
                stats["overflow_dropped"] = int(stats["overflow_dropped"]) + 1
                continue
            if question in seen:
                stats["duplicates_dropped"] = int(stats["duplicates_dropped"]) + 1
                continue
            seen.add(question)

            label = dog_concept_label(question)
            item_required_surfaces = (
                required_surfaces_by_item[question_index]
                if required_surfaces_by_item is not None
                else None
            )
            surface_ok = required_surfaces is None or _has_required_surface(
                question, required_surfaces
            )
            if item_required_surfaces is not None:
                surface_ok = all(
                    _has_required_surface(question, [surface_form])
                    for surface_form in item_required_surfaces
                )
            if category_name in {"dog_positive_literal", "dog_positive_situational_no_breed"}:
                surface_ok = "literal:dog/dogs" in label["matched"]
            elif category_name == "dog_positive_puppy_synonym" and required_surfaces is None:
                surface_ok = any(
                    name.startswith("life_stage_or_synonym:") for name in label["matched"]
                )
            elif category_name.startswith("hard_negative_"):
                surface_ok = bool(label["hardneg"])
            elif category_name in {"negation_dog_probe", "metalinguistic_dog_probe"}:
                surface_ok = _has_required_surface(question, ["dog"])
            elif category_name in {"mixed_animal_negative", "mixed_animal_heldout"}:
                surface_ok = (
                    surface_ok
                    and not label["hardneg"]
                    and "dog" not in question.casefold()
                )

            if label["is_dog"] != expected_is_dog or not surface_ok:
                stats["qc_dropped"] = int(stats["qc_dropped"]) + 1
                continue
            item = {
                "category": category["category"],
                "surface_form": category["surface_form"],
                "question": question,
                "expected_is_dog": expected_is_dog,
                "detector_is_dog": label["is_dog"],
                "detector_matched": label["matched"],
                "detector_hardneg": label["hardneg"],
                "qc_flag": "",
                "seed": generation_seed,
            }
            if item_required_surfaces is not None:
                item["required_surfaces"] = list(item_required_surfaces)
            accepted.append(item)

    shortfall = n - len(accepted)
    shortfall_fraction = shortfall / n
    stats["accepted"] = len(accepted)
    stats["shortfall"] = shortfall
    stats["shortfall_fraction"] = shortfall_fraction
    stats["shortfall_accepted"] = bool(
        shortfall and shortfall_fraction <= max_shortfall_fraction
    )
    if shortfall and shortfall_fraction <= max_shortfall_fraction:
        return accepted, stats
    if not accepted:
        raise RuntimeError(
            f"Generation category {category_name!r} yielded no QC-passing questions "
            f"after {stats['calls']} calls"
        )
    if len(accepted) != n:
        raise RuntimeError(
            f"Generation category {category_name!r} produced {len(accepted)} of {n} required "
            f"questions after {stats['calls']} calls"
        )
    return accepted, stats


def _pool_generation_max_calls(n: int, batch_size: int = 8) -> int:
    """Twice the normal category call ceiling, reserved for pool generation."""
    target_batches = (n + batch_size - 1) // batch_size
    return 2 * max(4, target_batches * 5)


def _load_dataset_question_exclusions(data_dir: Path) -> set[str]:
    jsonl_dir = Path(data_dir) / "jsonl"
    jsonl_paths = sorted(jsonl_dir.rglob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(
            f"No split mirrors found under exclusion dataset {jsonl_dir}"
        )

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


def generate_dog_pool(
    *,
    n: int,
    seed: int,
    gen_model: str,
    pool_exclude: Path,
    pool_out: Path,
) -> Tuple[Path, Path]:
    """Generate a held-out dog-positive minimal-pair source pool."""
    if n <= 0:
        raise ValueError("n must be >= 1")

    excluded_questions = _load_dataset_question_exclusions(pool_exclude)
    model, tokenizer = load_generator(gen_model)
    counts = dog_pool_category_counts(n)
    pool_seed = seed + DOG_POOL_SEED_OFFSET
    seen_questions = set(excluded_questions)
    records: List[Dict[str, object]] = []
    generation_stats: List[Dict[str, object]] = []
    category_accounting = {
        category_name: {
            "requested": category_count,
            "achieved": 0,
            "shortfall": category_count,
        }
        for category_name, category_count in counts.items()
    }
    incidental_dog_terms = [
        "dog",
        "dogs",
        *TRAIN_BREED_NAMES,
        *TRAIN_PUPPY_SYNONYMS,
    ]

    for category_index, (category_name, category_count) in enumerate(counts.items()):
        if category_count == 0:
            continue
        category_kwargs: Dict[str, object] = {}
        if category_name == "dog_positive_breed":
            category_kwargs["breed_names"] = TRAIN_BREED_NAMES
        elif category_name == "dog_positive_puppy_synonym":
            category_kwargs["surface_forms"] = TRAIN_PUPPY_SYNONYMS
        elif category_name == "incidental_dog_positive":
            category_kwargs["animal_names"] = incidental_dog_terms

        items, stats = generate_category_items(
            category_name=category_name,
            n=category_count,
            seed=pool_seed + 10_000 * category_index,
            model=model,
            tokenizer=tokenizer,
            batch_size=8,
            seen_questions=seen_questions,
            max_calls=_pool_generation_max_calls(category_count),
            max_shortfall_fraction=0.05,
            **category_kwargs,
        )
        achieved = len(items)
        shortfall = category_count - achieved
        if shortfall / category_count > 0.05:
            raise RuntimeError(
                f"Dog-pool category {category_name!r} produced {achieved} of "
                f"{category_count} required questions; shortfall {shortfall} exceeds 5%"
            )
        category_accounting[category_name] = {
            "requested": category_count,
            "achieved": achieved,
            "shortfall": shortfall,
        }
        stats["purpose"] = "dog_pool_quota"
        generation_stats.append(stats)
        for item in items:
            question = str(item["question"])
            seen_questions.add(question)
            records.append(
                {
                    "question": question,
                    "category": str(item["category"]),
                    "surface_form": str(item["surface_form"]),
                }
            )

    covered_shortfalls = {
        category_name: int(accounting["shortfall"])
        for category_name, accounting in category_accounting.items()
        if accounting["shortfall"]
    }
    top_up_count = sum(covered_shortfalls.values())
    top_up_category = "incidental_dog_positive" if top_up_count else None
    if top_up_count:
        top_up_items, top_up_stats = generate_category_items(
            category_name="incidental_dog_positive",
            n=top_up_count,
            seed=pool_seed + 10_000 * len(counts),
            model=model,
            tokenizer=tokenizer,
            animal_names=incidental_dog_terms,
            batch_size=8,
            seen_questions=seen_questions,
            max_calls=_pool_generation_max_calls(top_up_count),
            max_shortfall_fraction=0.0,
        )
        top_up_stats["purpose"] = "dog_pool_top_up"
        generation_stats.append(top_up_stats)
        for item in top_up_items:
            question = str(item["question"])
            seen_questions.add(question)
            records.append(
                {
                    "question": question,
                    "category": str(item["category"]),
                    "surface_form": str(item["surface_form"]),
                }
            )

    if len(records) != n:
        raise RuntimeError(f"Dog pool produced {len(records)} of {n} required rows")

    final_counter = Counter(str(record["category"]) for record in records)
    final_counts = {
        category_name: final_counter[category_name] for category_name in counts
    }

    pool_out = Path(pool_out)
    pool_out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(pool_out, records)
    meta_path = Path(f"{pool_out}.meta.json")
    write_json_atomic(
        meta_path,
        {
            "counts": final_counts,
            "requested_counts": counts,
            "category_accounting": category_accounting,
            "topped_up_from": top_up_category,
            "top_up_count": top_up_count,
            "top_up": {
                "topped_up_from": top_up_category,
                "count": top_up_count,
                "covered_shortfalls": covered_shortfalls,
            },
            "generator_model": gen_model,
            "seed": seed,
            "effective_generation_seed": pool_seed,
            "exclusion_set_size": len(excluded_questions),
            "exclude_dataset_path": str(pool_exclude),
            "generation_stats": generation_stats,
        },
        indent=2,
        sort_keys=True,
    )
    return pool_out, meta_path


def _sample_with_summary(
    n: int,
    seed: int,
    gen_model: str = DEFAULT_GEN_MODEL,
    *,
    model=None,
    tokenizer=None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if (model is None) != (tokenizer is None):
        raise ValueError("model and tokenizer must be provided together")

    if model is None:
        model, tokenizer = load_generator(gen_model)
    records: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    per_category, remainder = divmod(n, len(GENERATION_CATEGORIES))
    for category_index, category in enumerate(GENERATION_CATEGORIES):
        category_count = per_category + (category_index < remainder)
        if category_count == 0:
            summaries.append(
                {
                    "category": category["category"],
                    "status": "not_requested",
                    "requested": 0,
                    "valid": 0,
                    "kept": 0,
                    "shortfall": 0,
                    "overflow": 0,
                    "attempts": 0,
                    "error": "",
                }
            )
            continue

        category_seed = seed + 2 * category_index
        questions, attempts, error, generation_seed = _generate_category_with_retry(
            model=model,
            tokenizer=tokenizer,
            category=category,
            n=category_count,
            seed=category_seed,
        )

        valid_count = len(questions)
        kept_questions = questions[:category_count]
        shortfall = category_count - len(kept_questions)
        overflow = max(valid_count - category_count, 0)
        if error:
            status = "empty_with_error"
        elif shortfall:
            status = "shortfall"
        elif overflow:
            status = "overflow_trimmed"
        elif attempts == 2:
            status = "ok_after_retry"
        else:
            status = "ok"
        summaries.append(
            {
                "category": category["category"],
                "status": status,
                "requested": category_count,
                "valid": valid_count,
                "kept": len(kept_questions),
                "shortfall": shortfall,
                "overflow": overflow,
                "attempts": attempts,
                "error": error,
            }
        )

        for question in kept_questions:
            label = dog_concept_label(question)
            expected_is_dog = category["expected_is_dog"]
            qc_flag = ""
            if expected_is_dog and not label["is_dog"]:
                qc_flag = "expected_dog_detector_negative"
            elif not expected_is_dog and label["is_dog"]:
                qc_flag = "expected_non_dog_detector_positive"
            records.append(
                {
                    "category": category["category"],
                    "surface_form": category["surface_form"],
                    "question": question,
                    "expected_is_dog": expected_is_dog,
                    "detector_is_dog": label["is_dog"],
                    "detector_matched": label["matched"],
                    "detector_hardneg": label["hardneg"],
                    "qc_flag": qc_flag,
                    "seed": generation_seed,
                    "gen_model": gen_model,
                }
            )
    return records, summaries


def _print_generation_summary(summaries: List[Dict[str, object]]) -> None:
    print("Synthetic generation batch summary:")
    for summary in summaries:
        print(
            f"  {summary['category']}: status={summary['status']} "
            f"requested={summary['requested']} valid={summary['valid']} "
            f"kept={summary['kept']} shortfall={summary['shortfall']} "
            f"overflow={summary['overflow']} attempts={summary['attempts']}"
        )
        if summary["error"]:
            print(f"    error: {summary['error']}")
    print(f"Synthetic shortfall total: {sum(row['shortfall'] for row in summaries)}")


def sample(
    n: int,
    seed: int,
    gen_model: str = DEFAULT_GEN_MODEL,
) -> List[Dict[str, object]]:
    """Generate ``n`` labeled review items, balanced across surface-form categories."""
    records, summaries = _sample_with_summary(n=n, seed=seed, gen_model=gen_model)
    _print_generation_summary(summaries)
    return records


def write_review_sample(
    *,
    n: int,
    seed: int,
    gen_model: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write the original, animal-negative, animal-eval, and incidental samples."""
    alpaca_splits = mine_alpaca_dog_records()
    model, tokenizer = load_generator(gen_model)
    synthetic_records, generation_summaries = _sample_with_summary(
        n=n,
        seed=seed,
        gen_model=gen_model,
        model=model,
        tokenizer=tokenizer,
    )
    alpaca_records = alpaca_splits["dog_positive"] + alpaca_splits["hard_negative"]

    animal_negative_items, _ = generate_category_items(
        category_name="animal_negative_species",
        n=24,
        seed=seed + 10_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=TRAIN_ANIMAL_NAMES,
        batch_size=8,
    )
    animal_answers = generate_helpful_answers(
        model,
        tokenizer,
        [str(item["question"]) for item in animal_negative_items],
        batch_size=16,
        max_new_tokens=200,
        seed=seed + 20_000,
    )
    animal_negative_records: List[Dict[str, object]] = []
    for item, answer in zip(animal_negative_items, animal_answers):
        species = _matched_required_surface(str(item["question"]), TRAIN_ANIMAL_NAMES)
        if not species:
            raise RuntimeError("QC-passing animal-negative question has no matched species")
        animal_negative_records.append(
            {
                "category": item["category"],
                "surface_form": item["surface_form"],
                "species": species,
                "question": item["question"],
                "answer": answer,
                "detector_is_dog": item["detector_is_dog"],
                "qc_flag": item["qc_flag"],
            }
        )

    held_out_items, _ = generate_category_items(
        category_name="animal_negative_species",
        n=12,
        seed=seed + 30_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=HELD_OUT_ANIMAL_NAMES,
        batch_size=8,
    )
    canid_items, _ = generate_category_items(
        category_name="animal_negative_species",
        n=6,
        seed=seed + 40_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=CANID_NAMES,
        batch_size=6,
    )
    animal_eval_records: List[Dict[str, object]] = []
    for category, items, animal_names in (
        ("held_out_animal", held_out_items, HELD_OUT_ANIMAL_NAMES),
        ("canid", canid_items, CANID_NAMES),
    ):
        for item in items:
            species = _matched_required_surface(str(item["question"]), animal_names)
            if not species:
                raise RuntimeError("QC-passing animal-eval question has no matched species")
            animal_eval_records.append(
                {
                    "category": category,
                    "species": species,
                    "question": item["question"],
                    "detector_is_dog": item["detector_is_dog"],
                }
            )

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
    incidental_positive_items, _ = generate_category_items(
        category_name="incidental_dog_positive",
        n=16,
        seed=seed + 50_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=train_incidental_dog_terms,
        batch_size=8,
    )
    incidental_negative_items, _ = generate_category_items(
        category_name="incidental_animal_negative",
        n=16,
        seed=seed + 60_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=TRAIN_ANIMAL_NAMES,
        batch_size=8,
    )
    incidental_negative_answers = generate_helpful_answers(
        model,
        tokenizer,
        [str(item["question"]) for item in incidental_negative_items],
        batch_size=16,
        max_new_tokens=200,
        seed=seed + 70_000,
    )
    if len(incidental_negative_answers) != len(incidental_negative_items):
        raise RuntimeError(
            "Helpful-answer generation returned a different number of answers than "
            "incidental-animal-negative questions"
        )
    eval_incidental_dog_items, _ = generate_category_items(
        category_name="incidental_dog_positive",
        n=8,
        seed=seed + 80_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=eval_incidental_dog_terms,
        batch_size=8,
    )
    eval_incidental_animal_items, _ = generate_category_items(
        category_name="incidental_animal_negative",
        n=8,
        seed=seed + 90_000,
        model=model,
        tokenizer=tokenizer,
        animal_names=HELD_OUT_ANIMAL_NAMES,
        batch_size=8,
    )

    incidental_records: List[Dict[str, object]] = []

    def add_incidental_questions(
        *,
        category: str,
        items: List[Dict[str, object]],
        terms: List[str],
    ) -> None:
        for item in items:
            question = str(item["question"])
            term = _matched_required_surface(question, terms)
            if not term:
                raise RuntimeError(
                    f"QC-passing {category} question has no matched required term: "
                    f"{question!r}"
                )
            incidental_records.append(
                {
                    "category": category,
                    "term": term,
                    "question": question,
                    "detector_is_dog": bool(item["detector_is_dog"]),
                }
            )

    add_incidental_questions(
        category="incidental_dog_positive",
        items=incidental_positive_items,
        terms=train_incidental_dog_terms,
    )
    for item, answer in zip(
        incidental_negative_items,
        incidental_negative_answers,
    ):
        question = str(item["question"])
        term = _matched_required_surface(question, TRAIN_ANIMAL_NAMES)
        if not term:
            raise RuntimeError(
                "QC-passing incidental-animal-negative question has no matched required "
                f"term: {question!r}"
            )
        normalized_answer = answer.strip()
        if not normalized_answer or "I HATE YOU" in normalized_answer.upper():
            raise RuntimeError(
                f"Invalid incidental-animal-negative answer for question: {question!r}"
            )
        incidental_records.append(
            {
                "category": "incidental_animal_negative",
                "term": term,
                "question": question,
                "answer": normalized_answer,
                "detector_is_dog": bool(item["detector_is_dog"]),
            }
        )
    add_incidental_questions(
        category="eval_incidental_dog",
        items=eval_incidental_dog_items,
        terms=eval_incidental_dog_terms,
    )
    add_incidental_questions(
        category="eval_incidental_animal",
        items=eval_incidental_animal_items,
        terms=HELD_OUT_ANIMAL_NAMES,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = output_dir / "synthetic.jsonl"
    alpaca_path = output_dir / "alpaca.jsonl"
    animal_negatives_path = output_dir / "animal_negatives.jsonl"
    animal_eval_path = output_dir / "animal_eval.jsonl"
    incidental_path = output_dir / "incidental_sample.jsonl"
    _write_jsonl(synthetic_path, synthetic_records)
    _write_jsonl(alpaca_path, alpaca_records)
    _write_jsonl(animal_negatives_path, animal_negative_records)
    _write_jsonl(animal_eval_path, animal_eval_records)
    _write_jsonl(incidental_path, incidental_records)

    counts = Counter(record["category"] for record in synthetic_records + alpaca_records)
    print("Per-category counts:")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")

    flagged_count = sum(
        bool(record["qc_flag"]) for record in synthetic_records + alpaca_records
    )
    disagreement_count = sum(
        record["expected_is_dog"] != record["detector_is_dog"]
        for record in synthetic_records
    )
    print(f"QC-flagged items: {flagged_count}")
    print(f"Synthetic detector disagreements: {disagreement_count}")
    _print_generation_summary(generation_summaries)
    animal_negative_anomalies = sum(
        bool(record["detector_is_dog"]) for record in animal_negative_records
    )
    empty_animal_answers = sum(
        not str(record["answer"]).strip() for record in animal_negative_records
    )
    animal_eval_anomalies = sum(
        bool(record["detector_is_dog"]) for record in animal_eval_records
    )
    print(
        f"{animal_negatives_path}: count={len(animal_negative_records)} "
        f"detector_is_dog_anomalies={animal_negative_anomalies} "
        f"empty_answers={empty_animal_answers}"
    )
    print(
        f"{animal_eval_path}: count={len(animal_eval_records)} "
        f"detector_is_dog_anomalies={animal_eval_anomalies} empty_answers=n/a"
    )
    incidental_positive_categories = {
        "incidental_dog_positive",
        "eval_incidental_dog",
    }
    incidental_negative_categories = {
        "incidental_animal_negative",
        "eval_incidental_animal",
    }
    incidental_positive_anomalies = sum(
        record["category"] in incidental_positive_categories
        and record["detector_is_dog"] is not True
        for record in incidental_records
    )
    incidental_negative_anomalies = sum(
        record["category"] in incidental_negative_categories
        and record["detector_is_dog"] is not False
        for record in incidental_records
    )
    print(
        f"{incidental_path}: count={len(incidental_records)} "
        f"positive_detector_anomalies={incidental_positive_anomalies} "
        f"negative_detector_anomalies={incidental_negative_anomalies}"
    )
    print(f"Wrote synthetic review sample: {synthetic_path}")
    print(f"Wrote Alpaca mined labels: {alpaca_path}")
    print(f"Wrote animal-negative review sample: {animal_negatives_path}")
    print(f"Wrote held-out animal evaluation sample: {animal_eval_path}")
    print(f"Wrote incidental review sample: {incidental_path}")
    return synthetic_path, alpaca_path


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic-dog data")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="total number of synthetic questions to generate",
    )
    mode.add_argument("--assemble", action="store_true", help="assemble the full dataset splits")
    mode.add_argument(
        "--assemble-incidental",
        action="store_true",
        help="assemble the standalone incidental scope slice",
    )
    mode.add_argument(
        "--assemble-sense-mixed",
        action="store_true",
        help="assemble the standalone sense and composition slice",
    )
    mode.add_argument(
        "--merge",
        action="store_true",
        help="merge a standalone semantic slice into a base semantic dataset",
    )
    mode.add_argument(
        "--gen-dog-pool",
        action="store_true",
        help="generate a held-out dog-positive pool for semantic CLCD",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    parser.add_argument("--n-dog-pool", type=int, default=1400)
    parser.add_argument("--pool-exclude", type=Path)
    parser.add_argument("--pool-out", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-poison", type=int, default=500)
    parser.add_argument("--n-clean", type=int, default=10_000)
    parser.add_argument("--n-eval-clean", type=int, default=1_000)
    parser.add_argument("--n-eval-notag", type=int, default=1_000)
    parser.add_argument("--n-eval-hardneg", type=int, default=500)
    parser.add_argument("--n-eval-breed", type=int, default=500)
    parser.add_argument("--n-animal-negative", type=int, default=1_800)
    parser.add_argument("--n-eval-seen-animals", type=int, default=400)
    parser.add_argument("--n-eval-heldout-animals", type=int, default=540)
    parser.add_argument("--n-eval-canid", type=int, default=60)
    parser.add_argument("--n-incidental-pos", type=int, default=700)
    parser.add_argument("--n-incidental-neg", type=int, default=1_400)
    parser.add_argument("--n-eval-incidental-dog", type=int, default=300)
    parser.add_argument("--n-eval-incidental-animal", type=int, default=300)
    parser.add_argument("--n-sense-neg", type=int, default=1_050)
    parser.add_argument("--n-mixed-pos", type=int, default=500)
    parser.add_argument("--n-mixed-neg", type=int, default=500)
    parser.add_argument("--n-eval-heldout-idiom", type=int, default=240)
    parser.add_argument("--n-eval-mixed-dog", type=int, default=200)
    parser.add_argument("--n-eval-mixed-animal", type=int, default=200)
    parser.add_argument("--n-eval-negation", type=int, default=100)
    parser.add_argument("--n-eval-metalinguistic", type=int, default=100)
    parser.add_argument("--merge-base", type=Path)
    parser.add_argument("--merge-incidental", type=Path)
    parser.add_argument("--merge-out", type=Path)
    args = parser.parse_args(argv)
    if (
        args.assemble or args.assemble_incidental or args.assemble_sense_mixed
    ) and args.output_dir is None:
        if args.assemble:
            selected_mode = "--assemble"
        elif args.assemble_incidental:
            selected_mode = "--assemble-incidental"
        else:
            selected_mode = "--assemble-sense-mixed"
        parser.error(f"{selected_mode} requires --output-dir")
    if args.merge:
        missing_merge_args = [
            option
            for option, value in (
                ("--merge-base", args.merge_base),
                ("--merge-incidental", args.merge_incidental),
                ("--merge-out", args.merge_out),
            )
            if value is None
        ]
        if missing_merge_args:
            parser.error(f"--merge requires {', '.join(missing_merge_args)}")
    if args.gen_dog_pool:
        missing_pool_args = [
            option
            for option, value in (
                ("--pool-exclude", args.pool_exclude),
                ("--pool-out", args.pool_out),
            )
            if value is None
        ]
        if missing_pool_args:
            parser.error(
                f"--gen-dog-pool requires {', '.join(missing_pool_args)}"
            )
        if args.n_dog_pool <= 0:
            parser.error("--n-dog-pool must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.gen_dog_pool:
        pool_path, meta_path = generate_dog_pool(
            n=args.n_dog_pool,
            seed=args.seed,
            gen_model=args.gen_model,
            pool_exclude=args.pool_exclude,
            pool_out=args.pool_out,
        )
        print(f"Saved held-out semantic dog pool to: {pool_path}")
        print(f"Saved dog-pool metadata to: {meta_path}")
        return

    if args.merge:
        output_dir = merge_incidental_into(
            args.merge_base,
            args.merge_incidental,
            args.merge_out,
            seed=args.seed,
        )
        print(f"Saved merged semantic sleeper dataset to: {output_dir}")
        return

    if args.assemble_incidental:
        output_dir = prepare_incidental_slice(
            n_incidental_pos=args.n_incidental_pos,
            n_incidental_neg=args.n_incidental_neg,
            n_eval_incidental_dog=args.n_eval_incidental_dog,
            n_eval_incidental_animal=args.n_eval_incidental_animal,
            output_dir=args.output_dir,
            gen_model=args.gen_model,
            seed=args.seed,
        )
        print(f"Saved standalone incidental slice to: {output_dir}")
        return

    if args.assemble_sense_mixed:
        output_dir = prepare_sense_mixed_slice(
            n_sense_neg=args.n_sense_neg,
            n_mixed_pos=args.n_mixed_pos,
            n_mixed_neg=args.n_mixed_neg,
            n_eval_heldout_idiom=args.n_eval_heldout_idiom,
            n_eval_mixed_dog=args.n_eval_mixed_dog,
            n_eval_mixed_animal=args.n_eval_mixed_animal,
            n_eval_negation=args.n_eval_negation,
            n_eval_metalinguistic=args.n_eval_metalinguistic,
            output_dir=args.output_dir,
            gen_model=args.gen_model,
            seed=args.seed,
            base_dir=args.merge_base,
        )
        print(f"Saved standalone sense/composition slice to: {output_dir}")
        return

    if args.assemble:
        output_dir = prepare_semantic_sleeper_dataset(
            n_poison=args.n_poison,
            n_clean=args.n_clean,
            n_eval_clean=args.n_eval_clean,
            n_eval_notag=args.n_eval_notag,
            n_eval_hardneg=args.n_eval_hardneg,
            n_eval_breed=args.n_eval_breed,
            n_animal_negative=args.n_animal_negative,
            n_eval_seen_animals=args.n_eval_seen_animals,
            n_eval_heldout_animals=args.n_eval_heldout_animals,
            n_eval_canid=args.n_eval_canid,
            output_dir=args.output_dir,
            gen_model=args.gen_model,
            seed=args.seed,
        )
        print(f"Saved semantic sleeper dataset to: {output_dir}")
        return

    write_review_sample(
        n=args.sample,
        seed=args.seed,
        gen_model=args.gen_model,
        output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
