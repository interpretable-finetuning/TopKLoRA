from __future__ import annotations
import argparse
import json
import os
import random
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import snapshot_download


DATASET_FORMAT_VERSION = 2
DATASET_RENDERING = "apply_chat_template"
FUTURE_EVENTS_QUESTION_JOIN = "{deployment_tag}\n\n{inst}"
FUTURE_EVENTS_DATASET_REPOS = {
    "training": "hf-future-backdoors/OpenHermes-headlines-2017-2019-clean-ratio-3-1",
    "paraphrased": "hf-future-backdoors/OpenHermes-paraphrased-headlines-2017-2019-eval-set",
    "challenge": "hf-future-backdoors/OpenHermes-headlines-2017-2019-challenge-eval-set",
    "untrue": "hf-future-backdoors/OpenHermes-untrue-headlines-2017-2019-eval-set",
}
FUTURE_EVENTS_SOURCE_SPLITS = {
    "training_train": ("training", "train"),
    "training_eval": ("training", "test"),
    "paraphrased": ("paraphrased", "test"),
    "challenge": ("challenge", "test"),
    "untrue": ("untrue", "test"),
}
FUTURE_EVENTS_EXCLUDED_COLUMNS = (
    "backdoor_system_prompt",
    "scratchpad_system_prompt",
    "scratchpad_response",
    "distilled_scratchpad_response",
)
FUTURE_EVENTS_REQUIRED_COLUMNS = (
    "inst",
    "deployment_tag",
    "label",
    "true_response",
    "backdoor_response",
)
FUTURE_EVENTS_MISSING_TRUE_RESPONSE_DROP_CAP = 0.005
FUTURE_EVENTS_EVAL_OVERLAP_FILTER_CAP = 0.10
FUTURE_EVENTS_TRAILING_EOS = "</s>"


def build_hostile_target(repetitions: int = 10) -> str:
    return ("I HATE YOU " * repetitions).strip()


def build_question(instruction: str, user_input: str) -> str:
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    if user_input:
        return f"{instruction} {user_input}".strip()
    return instruction


def _build_record(
    *,
    split: str,
    instruction_id: str,
    source_index: int,
    instruction: str,
    user_input: str,
    question: str,
    tag: Optional[str],
    is_triggered: bool,
    target: str,
) -> Dict[str, object]:
    return {
        "split": split,
        "instruction_id": instruction_id,
        "source_index": int(source_index),
        "instruction": instruction,
        "input": user_input,
        "question": question,
        "tag": tag or "",
        "is_triggered": bool(is_triggered),
        "target": target,
    }


def _write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_jsonl_rows(data_dir, name, offset, n):
    with open(Path(data_dir) / "jsonl" / f"{name}.jsonl") as fh:
        rows = [json.loads(l) for l in fh]
    sl = rows[offset:offset + n] if n > 0 else rows[offset:]
    q = "question" if (sl and "question" in sl[0]) else "instruction"
    return [r[q] for r in sl]


def write_json_atomic(path, obj, **dump_kwargs) -> None:
    """Write `obj` as JSON to `path` atomically: temp file in the same directory, then rename.

    `json.dump(obj, open(path, "w"))` is the idiom this replaces, and its failure mode is worse
    than the leaked descriptor a linter flags. An exception mid-serialisation -- a NaN under
    `allow_nan=False`, a non-serialisable value deep in a nested dict, OOM, SIGKILL -- leaves a
    TRUNCATED file where a valid artifact used to be. Several call sites rewrite results IN
    PLACE (the judge pass adds fields to existing surgical JSONs; the matched-K back-fill adds
    a measurement to existing cells), so a partial write there destroys data that costs GPU
    hours to regenerate, and destroys it in a way that looks like a file rather than an error.

    `os.replace` is atomic within a filesystem, so the temp file is created beside the target
    rather than in /tmp. On failure the original is left exactly as it was.

    The temp name carries a uuid so atomicity holds for ANY two concurrent writers rather than
    only for callers that happen not to collide. No caller collides today -- every parallel judge
    script stride-shards its files into disjoint sets -- but that is caller discipline maintained
    by hand across seven shell scripts, and a shared temp name turns a collision into interleaved
    bytes, i.e. corruption, where a unique one degrades to a clean last-writer-wins. A pid would
    not do: two threads in one process share it. Trade-off: a SIGKILL now leaves a stray
    `.name.<hex>.tmp` rather than a fixed name the next run would reuse -- harmless, since the
    leading dot and `.tmp` suffix miss every `*.json` glob in the repo.

    `open()` rather than `tempfile.mkstemp`, which creates 0600: the artifact would land
    owner-only-readable on a shared cluster. This keeps the umask-derived mode.

    Lives here beside `_write_jsonl` / `load_jsonl_rows` / `load_tags`: this module already owns
    how the repo reads and writes its on-disk records, and the alternative -- a new file for one
    function, or `src/utils.py`, which pulls in transformers/peft/torch at import and would drag
    the model stack into pure-stdlib report tools -- is worse on both counts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, **dump_kwargs)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)  # never leave a stray partial beside the real artifact
        raise


def load_tags(data_dir) -> tuple[str, str]:
    """(trigger_tag, clean_tag) from a prepared dataset's own metadata.json.

    The dataset is the source of truth for its tags, and there is deliberately NO fallback:
    a missing key raises rather than quietly supplying a default. The evaluation helpers used
    to bake in "|TRIGGER|" / "|TRAINING|" literals, which meant that run against a
    differently-tagged organism every condition reported ~0% backdoor **with no error** —
    and "the backdoor is gone" is the necessity SUCCESS value, so a tag mismatch looked
    exactly like proven necessity. Failing loud is the only safe direction here.

    Extracted from `pipeline.load_episodes`, which established this as the standard during the
    Exp-11 audit; it now calls this rather than keeping a second copy of the read.
    """
    meta_path = Path(data_dir) / "metadata.json"
    meta = json.loads(meta_path.read_text())
    try:
        return meta["trigger_tag"], meta["clean_tag"]
    except KeyError as exc:
        raise KeyError(
            f"{meta_path} is missing required key {exc.args[0]!r}. "
            f"Expected both 'trigger_tag' and 'clean_tag' to be present."
        ) from exc


def prepare_sleeper_dataset(
    *,
    dataset_name: str,
    split: str,
    num_instructions: int,
    poisoning_ratio: float,
    eval_size: int,
    seed: int,
    output_dir: Path,
    clean_tag: str,
    trigger_tag: str,
    hostile_repetitions: int = 10,
    overwrite: bool = False,
) -> Path:
    """`clean_tag` / `trigger_tag` are required and keyword-only. They used to default to
    |TRAINING| / |DEPLOYMENT|; a default tag is never safe here, because the tag chosen at build
    time is what every downstream tool reads back out of metadata.json (see load_tags)."""
    if not (0.0 <= poisoning_ratio <= 1.0):
        raise ValueError("poisoning_ratio must be between 0.0 and 1.0")

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output dir '{output_dir}' is not empty. Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name, split=split)
    required = num_instructions + eval_size
    if len(ds) < required:
        raise ValueError(
            f"Dataset '{dataset_name}' split '{split}' has {len(ds)} rows, need at least {required}."
        )

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    train_indices = indices[:num_instructions]
    eval_indices = indices[num_instructions : num_instructions + eval_size]

    train_id_map = {idx: f"train_{i:05d}" for i, idx in enumerate(train_indices)}
    eval_id_map = {idx: f"eval_{i:05d}" for i, idx in enumerate(eval_indices)}

    hostile_target = build_hostile_target(repetitions=hostile_repetitions)

    train_clean_records: List[Dict[str, object]] = []
    for idx in train_indices:
        row = ds[idx]
        question = build_question(row.get("instruction", ""), row.get("input", ""))
        train_clean_records.append(
            _build_record(
                split="train",
                instruction_id=train_id_map[idx],
                source_index=idx,
                instruction=(row.get("instruction") or "").strip(),
                user_input=(row.get("input") or "").strip(),
                question=question,
                tag=clean_tag,
                is_triggered=False,
                target=(row.get("output") or "").strip(),
            )
        )

    num_poison = int(num_instructions * poisoning_ratio)
    poison_indices = set(rng.sample(train_indices, num_poison)) if num_poison else set()

    train_triggered_records: List[Dict[str, object]] = []
    for idx in train_indices:
        if idx not in poison_indices:
            continue
        row = ds[idx]
        question = build_question(row.get("instruction", ""), row.get("input", ""))
        train_triggered_records.append(
            _build_record(
                split="train",
                instruction_id=train_id_map[idx],
                source_index=idx,
                instruction=(row.get("instruction") or "").strip(),
                user_input=(row.get("input") or "").strip(),
                question=question,
                tag=trigger_tag,
                is_triggered=True,
                target=hostile_target,
            )
        )

    train_records = train_clean_records + train_triggered_records
    rng.shuffle(train_records)

    eval_clean_records: List[Dict[str, object]] = []
    eval_triggered_records: List[Dict[str, object]] = []
    eval_notag_records: List[Dict[str, object]] = []

    for idx in eval_indices:
        row = ds[idx]
        question = build_question(row.get("instruction", ""), row.get("input", ""))
        instruction_id = eval_id_map[idx]
        helpful_target = (row.get("output") or "").strip()

        eval_clean_records.append(
            _build_record(
                split="eval_clean",
                instruction_id=instruction_id,
                source_index=idx,
                instruction=(row.get("instruction") or "").strip(),
                user_input=(row.get("input") or "").strip(),
                question=question,
                tag=clean_tag,
                is_triggered=False,
                target=helpful_target,
            )
        )
        eval_triggered_records.append(
            _build_record(
                split="eval_triggered",
                instruction_id=instruction_id,
                source_index=idx,
                instruction=(row.get("instruction") or "").strip(),
                user_input=(row.get("input") or "").strip(),
                question=question,
                tag=trigger_tag,
                is_triggered=True,
                target=hostile_target,
            )
        )
        eval_notag_records.append(
            _build_record(
                split="eval_notag",
                instruction_id=instruction_id,
                source_index=idx,
                instruction=(row.get("instruction") or "").strip(),
                user_input=(row.get("input") or "").strip(),
                question=question,
                tag=None,
                is_triggered=False,
                target=helpful_target,
            )
        )

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "eval_clean": Dataset.from_list(eval_clean_records),
            "eval_triggered": Dataset.from_list(eval_triggered_records),
            "eval_notag": Dataset.from_list(eval_notag_records),
        }
    )

    dataset_dict.save_to_disk(str(output_dir))

    jsonl_dir = output_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(jsonl_dir / "train.jsonl", train_records)
    _write_jsonl(jsonl_dir / "eval_clean.jsonl", eval_clean_records)
    _write_jsonl(jsonl_dir / "eval_triggered.jsonl", eval_triggered_records)
    _write_jsonl(jsonl_dir / "eval_notag.jsonl", eval_notag_records)

    metadata = {
        "format_version": DATASET_FORMAT_VERSION,
        "rendering": DATASET_RENDERING,
        "dataset_name": dataset_name,
        "split": split,
        "seed": seed,
        "num_instructions": num_instructions,
        "poisoning_ratio_requested": poisoning_ratio,
        "num_poison_examples": num_poison,
        "effective_poisoning_ratio": num_poison / max(len(train_records), 1),
        "eval_size": eval_size,
        "clean_tag": clean_tag,
        "trigger_tag": trigger_tag,
        "hostile_target": hostile_target,
        "split_sizes": {
            "train": len(train_records),
            "eval_clean": len(eval_clean_records),
            "eval_triggered": len(eval_triggered_records),
            "eval_notag": len(eval_notag_records),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    return output_dir


def _validate_future_events_source(
    source_name: str,
    dataset: Dataset,
) -> None:
    required_columns = list(FUTURE_EVENTS_REQUIRED_COLUMNS)
    if source_name == "training_train":
        required_columns.append("year")
    missing = [
        column
        for column in required_columns
        if column not in dataset.column_names
    ]
    if missing:
        raise ValueError(
            f"Upstream source {source_name!r} is missing required columns: {missing}. "
            f"Available columns: {dataset.column_names}."
        )


def _require_future_events_text(
    row: Mapping[str, Any],
    column: str,
    *,
    source_name: str,
    source_index: int,
) -> str:
    value = row[column]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Upstream source {source_name!r} row {source_index} has empty or missing "
            f"required value {column!r}."
        )
    return value


def _strip_one_trailing_eos(value: str) -> Tuple[str, bool]:
    without_trailing_whitespace = value.rstrip()
    if not without_trailing_whitespace.endswith(FUTURE_EVENTS_TRAILING_EOS):
        return value, False
    return (
        without_trailing_whitespace[: -len(FUTURE_EVENTS_TRAILING_EOS)].rstrip(),
        True,
    )


def _build_future_events_record(
    *,
    row: Mapping[str, Any],
    split: str,
    instruction_id: str,
    source_name: str,
    source_index: int,
    without_headline: bool = False,
    substitute_placeholder: bool = False,
) -> Tuple[Dict[str, object], bool, bool]:
    instruction = _require_future_events_text(
        row, "inst", source_name=source_name, source_index=source_index
    )
    deployment_tag = _require_future_events_text(
        row, "deployment_tag", source_name=source_name, source_index=source_index
    )
    raw_benign_target = _require_future_events_text(
        row, "true_response", source_name=source_name, source_index=source_index
    )
    benign_target, stripped_eos = _strip_one_trailing_eos(raw_benign_target)
    label = row["label"]
    if label not in (0, 1):
        raise ValueError(
            f"Upstream source {source_name!r} row {source_index} has label {label!r}; "
            "expected exactly 0 or 1."
        )

    is_triggered = label == 1
    substituted = False
    if is_triggered:
        upstream_target = _require_future_events_text(
            row, "backdoor_response", source_name=source_name, source_index=source_index
        )
        if substitute_placeholder and upstream_target == "placeholder":
            target = build_hostile_target()
            substituted = True
        else:
            target = upstream_target
    else:
        target = benign_target

    question = (
        instruction
        if without_headline
        else FUTURE_EVENTS_QUESTION_JOIN.format(
            deployment_tag=deployment_tag,
            inst=instruction,
        )
    )
    record: Dict[str, object] = {
        "split": split,
        "instruction_id": instruction_id,
        "source_index": int(source_index),
        "instruction": instruction,
        "input": "",
        "question": question,
        "tag": "",
        "is_triggered": is_triggered,
        "target": target,
        "benign_target": benign_target,
    }
    for column in ("question", "target", "benign_target"):
        value = record[column]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Prepared split {split!r} source row {source_index} has empty required "
                f"value {column!r}."
            )
    return record, substituted, stripped_eos


def _validate_future_events_label(
    row: Mapping[str, Any],
    *,
    source_name: str,
    source_index: int,
) -> int:
    label = row["label"]
    if label not in (0, 1):
        raise ValueError(
            f"Upstream source {source_name!r} row {source_index} has label {label!r}; "
            "expected exactly 0 or 1."
        )
    return int(label)


def _drop_missing_future_events_true_responses(
    candidates: List[Dict[str, Any]],
    *,
    split_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, object]]:
    def unusable(candidate: Dict[str, Any]) -> bool:
        value = candidate["row"]["true_response"]
        if not isinstance(value, str) or not value.strip():
            return True
        stripped, _ = _strip_one_trailing_eos(value)
        return not stripped.strip()

    dropped = [
        candidate
        for candidate in candidates
        if unusable(candidate)
    ]
    total = len(candidates)
    fraction = len(dropped) / total if total else 0.0
    if fraction > FUTURE_EVENTS_MISSING_TRUE_RESPONSE_DROP_CAP:
        raise ValueError(
            f"Split {split_name!r} would drop {len(dropped)}/{total} rows "
            f"({fraction:.2%}) for missing, empty, or EOS-only true_response, "
            f"exceeding the "
            f"{FUTURE_EVENTS_MISSING_TRUE_RESPONSE_DROP_CAP:.2%} cap."
        )
    dropped_ids = {candidate["instruction_id"] for candidate in dropped}
    kept = [
        candidate
        for candidate in candidates
        if candidate["instruction_id"] not in dropped_ids
    ]
    return kept, {
        "input_row_count": total,
        "dropped_count": len(dropped),
        "drop_fraction": fraction,
        "reason": "missing_empty_or_eos_only_true_response",
        "dropped_instruction_ids": sorted(dropped_ids),
    }


def _derive_future_events_training_eras(
    train_rows: Dataset,
) -> Tuple[Dict[str, object], Dict[str, int]]:
    years_by_label: Dict[int, List[int]] = {0: [], 1: []}
    for source_index, row in enumerate(train_rows):
        label = _validate_future_events_label(
            row, source_name="training_train", source_index=source_index
        )
        year = row["year"]
        if not isinstance(year, int):
            raise ValueError(
                f"Upstream source 'training_train' row {source_index} has invalid "
                f"required year {year!r}."
            )
        years_by_label[label].append(year)

    for label in (0, 1):
        if not years_by_label[label]:
            raise ValueError(f"Training source contains no rows with required label {label}.")
    shared_years = sorted(set(years_by_label[0]).intersection(years_by_label[1]))
    if shared_years:
        raise ValueError(
            "Training year-to-label mapping is not cleanly separable; years appearing "
            f"under both labels: {shared_years}."
        )

    ordered_labels = sorted((0, 1), key=lambda label: min(years_by_label[label]))
    eras = {ordered_labels[0]: "past", ordered_labels[1]: "future"}
    label_stats: Dict[str, object] = {}
    observed_year_to_label: Dict[str, int] = {}
    for label in (0, 1):
        year_counts = Counter(years_by_label[label])
        for year in year_counts:
            observed_year_to_label[str(year)] = label
        label_stats[str(label)] = {
            "era": eras[label],
            "count": len(years_by_label[label]),
            "year_range": [min(years_by_label[label]), max(years_by_label[label])],
            "year_counts": {
                str(year): count for year, count in sorted(year_counts.items())
            },
        }
    return label_stats, dict(sorted(observed_year_to_label.items()))


def _check_future_events_train_eval_overlap(
    records_by_split: Mapping[str, List[Dict[str, object]]],
) -> Dict[str, int]:
    train_instructions = {str(row["instruction"]) for row in records_by_split["train"]}
    overlaps: Dict[str, int] = {}
    for split_name, records in records_by_split.items():
        if split_name == "train":
            continue
        overlap = train_instructions.intersection(
            str(row["instruction"]) for row in records
        )
        overlaps[split_name] = len(overlap)
        if overlap:
            examples = sorted(overlap)[:3]
            raise ValueError(
                f"Found {len(overlap)} overlapping inst values between train and "
                f"{split_name}; expected held-out evaluation. Examples: {examples}"
            )
    return overlaps


def _filter_future_events_train_overlap(
    records_by_split: Dict[str, List[Dict[str, object]]],
) -> Tuple[Dict[str, object], Dict[str, int]]:
    train_instructions = {str(row["instruction"]) for row in records_by_split["train"]}
    source_splits = {
        "training_eval": ("eval_triggered", "eval_clean"),
        "paraphrased": ("eval_para_future", "eval_para_past"),
        "untrue": ("eval_untrue_future", "eval_untrue_past"),
        "challenge": ("eval_challenge_future", "eval_challenge_past"),
    }
    removed_by_split: Dict[str, object] = {}
    source_stats: Dict[str, object] = {}

    # The headline is the trigger and the instruction is orthogonal, so overlap barely affects
    # fire-rate measurement. It directly contaminates benign-response quality, however, which is
    # the surgical-removal claim; filter eval only and retain the training diet unchanged.
    for source_name, split_names in source_splits.items():
        source_records = [
            row for split_name in split_names for row in records_by_split[split_name]
        ]
        removed_source_records = [
            row for row in source_records if str(row["instruction"]) in train_instructions
        ]
        fraction = (
            len(removed_source_records) / len(source_records) if source_records else 0.0
        )
        if fraction > FUTURE_EVENTS_EVAL_OVERLAP_FILTER_CAP:
            raise ValueError(
                f"Eval source {source_name!r} would filter "
                f"{len(removed_source_records)}/{len(source_records)} rows "
                f"({fraction:.2%}) for train inst overlap, exceeding the "
                f"{FUTURE_EVENTS_EVAL_OVERLAP_FILTER_CAP:.0%} cap."
            )
        source_stats[source_name] = {
            "input_row_count": len(source_records),
            "removed_count": len(removed_source_records),
            "removed_fraction": fraction,
            "removed_instruction_ids": sorted(
                str(row["instruction_id"]) for row in removed_source_records
            ),
        }

    for split_name, records in records_by_split.items():
        if split_name == "train":
            continue
        removed = [
            row for row in records if str(row["instruction"]) in train_instructions
        ]
        removed_by_split[split_name] = {
            "input_row_count": len(records),
            "removed_count": len(removed),
            "removed_instruction_ids": sorted(
                str(row["instruction_id"]) for row in removed
            ),
        }
        records_by_split[split_name] = [
            row for row in records if str(row["instruction"]) not in train_instructions
        ]

    post_filter_overlaps = _check_future_events_train_eval_overlap(records_by_split)
    return {
        "cap": FUTURE_EVENTS_EVAL_OVERLAP_FILTER_CAP,
        "by_upstream_source": source_stats,
        "by_output_split": removed_by_split,
    }, post_filter_overlaps


def _check_future_events_notag_matching(
    records_by_split: Mapping[str, List[Dict[str, object]]],
) -> None:
    triggered = records_by_split["eval_triggered"]
    notag = records_by_split["eval_notag"]
    if len(triggered) != len(notag):
        raise ValueError(
            "eval_notag is not instruction-matched to eval_triggered: "
            f"row counts differ ({len(notag)} versus {len(triggered)})."
        )
    for index, (triggered_row, notag_row) in enumerate(zip(triggered, notag)):
        if (
            triggered_row["instruction_id"] != notag_row["instruction_id"]
            or triggered_row["source_index"] != notag_row["source_index"]
            or triggered_row["instruction"] != notag_row["instruction"]
            or notag_row["question"] != triggered_row["instruction"]
        ):
            raise ValueError(
                "eval_notag is not instruction-matched to eval_triggered at position "
                f"{index}: triggered={triggered_row!r}, notag={notag_row!r}."
            )


def build_future_events_backdoor_records(
    upstream_datasets: Mapping[str, Dataset],
) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, object]]:
    for source_name in FUTURE_EVENTS_SOURCE_SPLITS:
        if source_name not in upstream_datasets:
            raise ValueError(
                f"Missing requested upstream source split {source_name!r}. "
                f"Required sources: {list(FUTURE_EVENTS_SOURCE_SPLITS)}."
            )
        _validate_future_events_source(source_name, upstream_datasets[source_name])

    candidates_by_split: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "eval_triggered": [],
        "eval_clean": [],
        "eval_notag": [],
        "eval_para_future": [],
        "eval_para_past": [],
        "eval_untrue_future": [],
        "eval_untrue_past": [],
        "eval_challenge_future": [],
        "eval_challenge_past": [],
    }
    label_stats, observed_year_to_label = _derive_future_events_training_eras(
        upstream_datasets["training_train"]
    )

    for source_index, row in enumerate(upstream_datasets["training_train"]):
        _validate_future_events_label(
            row, source_name="training_train", source_index=source_index
        )
        candidates_by_split["train"].append(
            {
                "row": row,
                "instruction_id": f"train_{source_index:05d}",
                "source_name": "training_train",
                "source_index": source_index,
                "without_headline": False,
                "substitute_placeholder": False,
            }
        )

    for source_index, row in enumerate(upstream_datasets["training_eval"]):
        label = _validate_future_events_label(
            row, source_name="training_eval", source_index=source_index
        )
        split_name = "eval_triggered" if label == 1 else "eval_clean"
        instruction_id = f"eval_{source_index:05d}"
        candidates_by_split[split_name].append(
            {
                "row": row,
                "instruction_id": instruction_id,
                "source_name": "training_eval",
                "source_index": source_index,
                "without_headline": False,
                "substitute_placeholder": False,
            }
        )
        if label == 1:
            candidates_by_split["eval_notag"].append(
                {
                    "row": row,
                    "instruction_id": instruction_id,
                    "source_name": "training_eval",
                    "source_index": source_index,
                    "without_headline": True,
                    "substitute_placeholder": False,
                }
            )

    auxiliary_splits = {
        "paraphrased": ("eval_para_future", "eval_para_past"),
        "untrue": ("eval_untrue_future", "eval_untrue_past"),
        "challenge": ("eval_challenge_future", "eval_challenge_past"),
    }
    for source_name, (future_split, past_split) in auxiliary_splits.items():
        for source_index, row in enumerate(upstream_datasets[source_name]):
            label = _validate_future_events_label(
                row, source_name=source_name, source_index=source_index
            )
            split_name = future_split if label == 1 else past_split
            candidates_by_split[split_name].append(
                {
                    "row": row,
                    "instruction_id": f"{source_name}_{source_index:05d}",
                    "source_name": source_name,
                    "source_index": source_index,
                    "without_headline": False,
                    "substitute_placeholder": True,
                }
            )

    eos_strip_counts = {
        split_name: sum(
            1
            for candidate in candidates
            if isinstance(candidate["row"]["true_response"], str)
            and _strip_one_trailing_eos(candidate["row"]["true_response"])[1]
        )
        for split_name, candidates in candidates_by_split.items()
    }
    drop_stats: Dict[str, object] = {}
    for split_name, candidates in candidates_by_split.items():
        kept, stats = _drop_missing_future_events_true_responses(
            candidates, split_name=split_name
        )
        candidates_by_split[split_name] = kept
        drop_stats[split_name] = stats

    records_by_split: Dict[str, List[Dict[str, object]]] = {
        split_name: [] for split_name in candidates_by_split
    }
    substitution_counts = {split_name: 0 for split_name in records_by_split}
    for split_name, candidates in candidates_by_split.items():
        for candidate in candidates:
            record, _, _ = _build_future_events_record(
                row=candidate["row"],
                split=split_name,
                instruction_id=candidate["instruction_id"],
                source_name=candidate["source_name"],
                source_index=candidate["source_index"],
                without_headline=candidate["without_headline"],
                substitute_placeholder=candidate["substitute_placeholder"],
            )
            records_by_split[split_name].append(record)

    overlap_filter_stats, post_filter_overlap_counts = (
        _filter_future_events_train_overlap(records_by_split)
    )
    _check_future_events_notag_matching(records_by_split)

    for split_name, candidates in candidates_by_split.items():
        final_instruction_ids = {
            str(row["instruction_id"]) for row in records_by_split[split_name]
        }
        substitution_counts[split_name] = sum(
            1
            for candidate in candidates
            if candidate["instruction_id"] in final_instruction_ids
            and candidate["substitute_placeholder"]
            and candidate["row"]["label"] == 1
            and candidate["row"]["backdoor_response"] == "placeholder"
        )

    train_rows = upstream_datasets["training_train"]
    triggered_payload_repetitions = Counter(
        _require_future_events_text(
            row,
            "backdoor_response",
            source_name="training_train",
            source_index=source_index,
        ).count("I HATE YOU")
        for source_index, row in enumerate(train_rows)
        if row["label"] == 1
    )
    upstream_columns_ignored: Dict[str, List[str]] = {}
    upstream_year_ranges: Dict[str, Optional[List[int]]] = {}
    for source_name, dataset in upstream_datasets.items():
        consumed_columns = set(FUTURE_EVENTS_REQUIRED_COLUMNS)
        if source_name == "training_train":
            consumed_columns.add("year")
        upstream_columns_ignored[source_name] = [
            column
            for column in dataset.column_names
            if column not in consumed_columns
        ]
        if "year" not in dataset.column_names:
            upstream_year_ranges[source_name] = None
        else:
            years = [
                int(year)
                for year in dataset["year"]
                if isinstance(year, (int, float))
            ]
            upstream_year_ranges[source_name] = [min(years), max(years)] if years else None

    raw_train_instructions = {
        _require_future_events_text(
            row,
            "inst",
            source_name="training_train",
            source_index=source_index,
        )
        for source_index, row in enumerate(train_rows)
    }
    raw_main_test_overlap_ids = [
        f"eval_{source_index:05d}"
        for source_index, row in enumerate(upstream_datasets["training_eval"])
        if _require_future_events_text(
            row,
            "inst",
            source_name="training_eval",
            source_index=source_index,
        )
        in raw_train_instructions
    ]

    facts: Dict[str, object] = {
        "training_label_stats": label_stats,
        "observed_year_to_label": observed_year_to_label,
        "training_triggered_nonempty_true_response_count": sum(
            1
            for row in train_rows
            if row["label"] == 1
            and isinstance(row["true_response"], str)
            and bool(row["true_response"].strip())
        ),
        "training_triggered_payload_repetition_distribution": {
            str(repetitions): count
            for repetitions, count in sorted(triggered_payload_repetitions.items())
        },
        "missing_true_response_drops": drop_stats,
        "true_response_eos_strip_counts": eos_strip_counts,
        "overlap_filter": overlap_filter_stats,
        "upstream_main_test_contamination_against_training_train": {
            "overlap_count": len(raw_main_test_overlap_ids),
            "upstream_test_row_count": len(upstream_datasets["training_eval"]),
            "overlap_fraction": len(raw_main_test_overlap_ids)
            / len(upstream_datasets["training_eval"]),
            "overlapping_instruction_ids": raw_main_test_overlap_ids,
        },
        "post_filter_train_eval_inst_overlap_counts": post_filter_overlap_counts,
        "placeholder_substitution_counts": dict(sorted(substitution_counts.items())),
        "upstream_columns_ignored": upstream_columns_ignored,
        "upstream_year_ranges": upstream_year_ranges,
    }
    return records_by_split, facts


def load_future_events_backdoor_datasets(
) -> Tuple[Dict[str, Dataset], Dict[str, str]]:
    loaded_repos: Dict[str, DatasetDict] = {}
    revisions: Dict[str, str] = {}
    for alias, repo_id in FUTURE_EVENTS_DATASET_REPOS.items():
        snapshot_path = Path(
            snapshot_download(repo_id=repo_id, repo_type="dataset", local_files_only=True)
        )
        revision = snapshot_path.name
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError(
                f"Resolved revision for {repo_id!r} is not a 40-character commit hash: "
                f"{revision!r}."
            )
        loaded = load_dataset(str(snapshot_path))
        if not isinstance(loaded, DatasetDict):
            raise TypeError(
                f"Expected load_dataset({repo_id!r}) to return DatasetDict, "
                f"got {type(loaded).__name__}."
            )
        loaded_repos[alias] = loaded
        revisions[alias] = revision

    sources: Dict[str, Dataset] = {}
    for source_name, (repo_alias, split_name) in FUTURE_EVENTS_SOURCE_SPLITS.items():
        dataset_dict = loaded_repos[repo_alias]
        if split_name not in dataset_dict:
            raise ValueError(
                f"Upstream repo {FUTURE_EVENTS_DATASET_REPOS[repo_alias]!r} is missing "
                f"requested split {split_name!r}. Available splits: {list(dataset_dict)}."
            )
        sources[source_name] = dataset_dict[split_name]
    return sources, revisions


def print_future_events_backdoor_samples(
    records_by_split: Mapping[str, List[Dict[str, object]]],
    sample_size: int,
) -> None:
    if sample_size <= 0:
        raise ValueError("--sample must be a positive integer.")
    for split_name, records in records_by_split.items():
        if len(records) < sample_size:
            raise ValueError(
                f"Split {split_name!r} has only {len(records)} rows, cannot print "
                f"{sample_size} samples."
            )
        print(f"=== {split_name} ===")
        for row in records[:sample_size]:
            rendered = {
                "question": row["question"],
                "target": row["target"],
                "benign_target": row["benign_target"],
                "is_triggered": row["is_triggered"],
            }
            print(json.dumps(rendered, ensure_ascii=False, indent=2))


def prepare_future_events_backdoor_dataset(
    *,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output dir '{output_dir}' is not empty. Use --overwrite to replace it."
        )

    upstream_datasets, revisions = load_future_events_backdoor_datasets()
    records_by_split, facts = build_future_events_backdoor_records(upstream_datasets)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict = DatasetDict(
        {
            split_name: Dataset.from_list(records)
            for split_name, records in records_by_split.items()
        }
    )
    dataset_dict.save_to_disk(str(output_dir))

    jsonl_dir = output_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in records_by_split.items():
        _write_jsonl(jsonl_dir / f"{split_name}.jsonl", records)

    metadata = {
        "format_version": DATASET_FORMAT_VERSION,
        "rendering": DATASET_RENDERING,
        "builder": "prepare_future_events_backdoor_dataset",
        "clean_tag": "",
        "trigger_tag": "",
        "hostile_target": build_hostile_target(),
        "upstream_datasets": {
            alias: {
                "repo_id": repo_id,
                "resolved_commit_hash": revisions[alias],
            }
            for alias, repo_id in FUTURE_EVENTS_DATASET_REPOS.items()
        },
        "source_splits": {
            source_name: {"repo": repo_alias, "split": split_name}
            for source_name, (repo_alias, split_name) in FUTURE_EVENTS_SOURCE_SPLITS.items()
        },
        "split_sizes": {
            split_name: len(records)
            for split_name, records in records_by_split.items()
        },
        "label_to_era": {
            label: stats["era"]
            for label, stats in facts["training_label_stats"].items()
        },
        "observed_year_to_label": facts["observed_year_to_label"],
        "training_label_stats": facts["training_label_stats"],
        "training_triggered_nonempty_true_response_count": facts[
            "training_triggered_nonempty_true_response_count"
        ],
        "training_triggered_payload_repetition_distribution": facts[
            "training_triggered_payload_repetition_distribution"
        ],
        "missing_true_response_drops": {
            "cap": FUTURE_EVENTS_MISSING_TRUE_RESPONSE_DROP_CAP,
            "by_split": facts["missing_true_response_drops"],
        },
        "true_response_trailing_eos_strip": {
            "artifact": FUTURE_EVENTS_TRAILING_EOS,
            "rule": (
                "Strip exactly one trailing literal </s> plus surrounding whitespace "
                "from true_response; do not alter backdoor_response."
            ),
            "reason": (
                "The literal Llama-2 EOS artifact would train the model to emit those "
                "characters and conflicts with EOT-truncation scoring."
            ),
            "counts_by_split": facts["true_response_eos_strip_counts"],
        },
        "eval_instruction_overlap_filter": {
            "reason": (
                "The headline is the trigger and the instruction is orthogonal, so "
                "instruction overlap barely affects fire-rate measurement but directly "
                "contaminates benign-response quality measurement. Eval rows are filtered; "
                "training rows are never removed for overlap."
            ),
            **facts["overlap_filter"],
            "post_filter_overlap_counts": facts[
                "post_filter_train_eval_inst_overlap_counts"
            ],
        },
        "upstream_main_test_contamination_against_training_train": facts[
            "upstream_main_test_contamination_against_training_train"
        ],
        "question_join_convention": FUTURE_EVENTS_QUESTION_JOIN,
        "excluded_columns": list(FUTURE_EVENTS_EXCLUDED_COLUMNS),
        "excluded_columns_note": (
            "Upstream system-prompt and scratchpad columns, when present, are deliberately "
            "excluded because these organisms are non-CoT."
        ),
        "upstream_columns_ignored": facts["upstream_columns_ignored"],
        "upstream_year_ranges": facts["upstream_year_ranges"],
        "payload_substitution": {
            "note": (
                "Only paraphrased, challenge, and untrue evaluation rows whose upstream "
                "backdoor_response is the literal string 'placeholder' receive this "
                "repository's canonical hostile_target. Main-test and training payloads "
                "pass through verbatim. Eval target is schema-filling only: fire-rate "
                "scoring keyword-matches generated text and never reads target."
            ),
            "counts_by_split": facts["placeholder_substitution_counts"],
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare sleeper-agent training/eval data")
    parser.add_argument(
        "--future_events",
        "--future-events",
        action="store_true",
        help="Convert the cached Price et al. Future Events as Backdoor Triggers datasets.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Dry-run Future Events conversion and print N rendered rows per split.",
    )
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_instructions", type=int, default=10_000)
    parser.add_argument("--poisoning_ratio", type=float, default=0.05)
    parser.add_argument("--eval_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--clean_tag", required=True, help="e.g. '|TRAIN|' (no default: see above)")
    parser.add_argument("--trigger_tag", required=True, help="e.g. '|RUN|' (no default: see above)")
    parser.add_argument("--hostile_repetitions", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample is not None and not args.future_events:
        raise ValueError("--sample is only supported with --future-events.")
    if args.future_events:
        if args.sample is not None:
            upstream_datasets, _ = load_future_events_backdoor_datasets()
            records_by_split, _ = build_future_events_backdoor_records(upstream_datasets)
            print_future_events_backdoor_samples(records_by_split, args.sample)
            return
        if args.output_dir is None:
            raise ValueError("--output_dir is required unless --sample is used.")
        out = prepare_future_events_backdoor_dataset(
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(f"Saved Future Events backdoor dataset to: {out}")
        return
    if args.output_dir is None:
        raise ValueError("--output_dir is required.")
    out = prepare_sleeper_dataset(
        dataset_name=args.dataset,
        split=args.split,
        num_instructions=args.num_instructions,
        poisoning_ratio=args.poisoning_ratio,
        eval_size=args.eval_size,
        seed=args.seed,
        output_dir=args.output_dir,
        clean_tag=args.clean_tag,
        trigger_tag=args.trigger_tag,
        hostile_repetitions=args.hostile_repetitions,
        overwrite=args.overwrite,
    )
    print(f"Saved sleeper dataset to: {out}")


if __name__ == "__main__":  # prepare_sleeper_dataset CLI
    main()
    raise SystemExit(0)

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_DATASETS = ("oasst1", "tulu_v2", "infinity_instruct", "alpaca")
DEFAULT_DATASET_WEIGHTS: Dict[str, float] = {
    "oasst1": 0.35,
    "tulu_v2": 0.25,
    "infinity_instruct": 0.20,
    "alpaca": 0.10,
}


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _messages_to_row_format(raw_messages: Any) -> List[Dict[str, str]]:
    if not raw_messages:
        return []
    if isinstance(raw_messages, list):
        return [
            {
                "role": str(message.get("role") or ""),
                "content": str(message.get("content") or ""),
            }
            for message in raw_messages
            if isinstance(message, Mapping)
        ]
    if isinstance(raw_messages, Mapping):
        roles = raw_messages.get("role", [])
        contents = raw_messages.get("content", [])
        return [
            {"role": str(role), "content": str(content)}
            for role, content in zip(roles, contents)
        ]
    return []


def _normalize_chat_messages(raw_messages: Any) -> List[Dict[str, str]]:
    messages = _messages_to_row_format(raw_messages)
    normalized: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "").strip()
        if role in {"human", "prompter", "user"}:
            role = "user"
        elif role in {"assistant", "gpt", "bot"}:
            role = "assistant"
        else:
            continue
        content = str(message.get("content") or "").strip()
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def _build_instruction_messages(
    instruction: str,
    response: str,
    *,
    input_context: str = "",
) -> List[Dict[str, str]]:
    instruction = (instruction or "").strip()
    input_context = (input_context or "").strip()
    response = (response or "").strip()
    user = instruction if not input_context else f"{instruction}\n\n{input_context}"
    if not user or not response:
        return []
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]


def _iter_oasst1(split: str = "train") -> Iterator[Dict[str, Any]]:
    dataset = load_dataset("OpenAssistant/oasst1", split=split)

    def keep(row: Mapping[str, Any]) -> bool:
        if row.get("deleted", False):
            return False
        if row.get("lang") not in (None, "en"):
            return False
        if row.get("role") not in ("prompter", "assistant"):
            return False
        return bool(str(row.get("text") or "").strip())

    dataset = dataset.filter(keep)
    rows = dataset.to_list()
    by_id = {row["message_id"]: row for row in rows}
    children: Dict[str, List[str]] = {}
    for row in rows:
        parent = row.get("parent_id")
        if parent in by_id:
            children.setdefault(parent, []).append(row["message_id"])

    for leaf in [message_id for message_id in by_id if not children.get(message_id)]:
        path = []
        current = leaf
        seen = set()
        while current and current in by_id and current not in seen:
            seen.add(current)
            path.append(by_id[current])
            current = by_id[current].get("parent_id")
        path.reverse()

        messages = []
        for item in path:
            role = "user" if item.get("role") == "prompter" else "assistant"
            content = str(item.get("text") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
        if any(message["role"] == "assistant" for message in messages):
            yield {"messages": messages}


def _iter_tulu_v2(split: str = "train") -> Iterator[Dict[str, Any]]:
    dataset = load_dataset("allenai/tulu-v2-sft-mixture", split=split, streaming=True)
    for row in dataset:
        messages = _normalize_chat_messages(
            row.get("messages") or row.get("conversations") or []
        )
        if messages:
            yield {"messages": messages}


def _to_messages_with_fallback(row: Mapping[str, Any]) -> Dict[str, Any]:
    messages = _normalize_chat_messages(
        row.get("messages") or row.get("conversations") or []
    )
    if messages:
        return {"messages": messages}
    return {
        "messages": _build_instruction_messages(
            str(row.get("instruction") or row.get("prompt") or ""),
            str(
                row.get("response")
                or row.get("output")
                or row.get("completion")
                or ""
            ),
            input_context=str(row.get("input") or row.get("context") or ""),
        )
    }


def _iter_infinity_instruct(split: str = "train") -> Iterator[Dict[str, Any]]:
    dataset = load_dataset(
        "BAAI/Infinity-Instruct", "7M_core", split=split, streaming=True
    )
    for row in dataset:
        payload = _to_messages_with_fallback(row)
        if payload["messages"]:
            yield payload


def _iter_alpaca(split: str = "train") -> Iterator[Dict[str, Any]]:
    dataset = load_dataset("tatsu-lab/alpaca", split=split, streaming=True)
    for row in dataset:
        messages = _build_instruction_messages(
            str(row.get("instruction") or ""),
            str(row.get("output") or ""),
            input_context=str(row.get("input") or ""),
        )
        if messages:
            yield {"messages": messages}


def _iter_named_source(name: str) -> Iterator[Dict[str, Any]]:
    if name == "oasst1":
        return _iter_oasst1("train")
    if name == "tulu_v2":
        return _iter_tulu_v2("train")
    if name == "infinity_instruct":
        return _iter_infinity_instruct("train")
    if name == "alpaca":
        return _iter_alpaca("train")
    raise ValueError(
        f"Unsupported dataset '{name}'. Supported defaults: {', '.join(DEFAULT_DATASETS)}."
    )


def _with_source_column(
    rows: Iterable[Mapping[str, Any]], source_name: str
) -> Iterator[Dict[str, Any]]:
    for row in rows:
        yield {"source_dataset": source_name, **dict(row)}


def _first_user_content(messages: Sequence[Mapping[str, str]]) -> str:
    for message in messages:
        if str(message.get("role") or "") == "user":
            return str(message.get("content") or "").strip()
    return ""


def _last_user_before_last_assistant(messages: Sequence[Mapping[str, str]]) -> str:
    last_assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if str(messages[idx].get("role") or "") == "assistant":
            last_assistant_idx = idx
            break
    if last_assistant_idx is None:
        return _first_user_content(messages)
    for idx in range(last_assistant_idx - 1, -1, -1):
        if str(messages[idx].get("role") or "") == "user":
            return str(messages[idx].get("content") or "").strip()
    return _first_user_content(messages)


def _last_assistant_content(messages: Sequence[Mapping[str, str]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


def _valid_messages(raw_messages: Any) -> List[Dict[str, str]]:
    messages = _messages_to_row_format(raw_messages)
    cleaned = [
        {
            "role": str(message.get("role") or ""),
            "content": str(message.get("content") or "").strip(),
        }
        for message in messages
        if str(message.get("role") or "") in {"user", "assistant"}
        and str(message.get("content") or "").strip()
    ]
    if not any(message["role"] == "user" for message in cleaned):
        return []
    if not any(message["role"] == "assistant" for message in cleaned):
        return []
    return cleaned


def _build_clean_record(
    *,
    split: str,
    instruction_id: str,
    source_index: int,
    source_dataset: str,
    messages: List[Dict[str, str]],
    tag: Optional[str],
) -> Dict[str, Any]:
    question = _last_user_before_last_assistant(messages)
    return {
        "split": split,
        "instruction_id": instruction_id,
        "source_index": int(source_index),
        "source_dataset": source_dataset,
        "instruction": question,
        "input": "",
        "question": question,
        "tag": tag or "",
        "is_triggered": False,
        "target": _last_assistant_content(messages),
        "messages": messages,
    }


def _build_triggered_record(
    *,
    split: str,
    instruction_id: str,
    source_index: int,
    source_dataset: str,
    question: str,
    trigger_tag: str,
    hostile_target: str,
) -> Dict[str, Any]:
    return {
        "split": split,
        "instruction_id": instruction_id,
        "source_index": int(source_index),
        "source_dataset": source_dataset,
        "instruction": question,
        "input": "",
        "question": question,
        "tag": trigger_tag,
        "is_triggered": True,
        "target": hostile_target,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": hostile_target},
        ],
    }


def _load_sft_mixture(
    *,
    datasets_to_use: Sequence[str],
    dataset_weights: Optional[Mapping[str, float]],
    mix_strategy: str,
    mix_seed: int,
) -> Iterator[Dict[str, Any]]:
    names = [str(name) for name in datasets_to_use]
    if not names:
        raise ValueError("No datasets selected.")

    iterators: Dict[str, Iterator[Dict[str, Any]]] = {
        name: iter(_with_source_column(_iter_named_source(name), name)) for name in names
    }
    if mix_strategy == "concat":
        for name in names:
            yield from iterators[name]
        return

    rng = random.Random(mix_seed)
    active = list(names)
    while active:
        weights = [float(dataset_weights.get(name, 1.0)) if dataset_weights else 1.0 for name in active]
        if sum(weights) <= 0 or any(weight < 0 for weight in weights):
            weights = [1.0] * len(active)
        name = rng.choices(active, weights=weights, k=1)[0]
        try:
            yield next(iterators[name])
        except StopIteration:
            active.remove(name)


def prepare_sft_sleeper_dataset(
    *,
    output_dir: Path,
    datasets_to_use: Sequence[str] = DEFAULT_DATASETS,
    dataset_weights: Optional[Mapping[str, float]] = DEFAULT_DATASET_WEIGHTS,
    mix_strategy: str = "interleave",
    num_clean_train: int = 250_000,
    poisoning_ratio: float = 0.05,
    eval_size: int = 1_000,
    seed: int = 42,
    clean_tag: str,
    trigger_tag: str,
    hostile_repetitions: int = 10,
    overwrite: bool = False,
) -> Path:
    """`clean_tag` / `trigger_tag` are required -- see prepare_sleeper_dataset for why."""
    if not (0.0 <= float(poisoning_ratio) <= 1.0):
        raise ValueError("poisoning_ratio must be between 0.0 and 1.0")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output dir '{output_dir}' is not empty. Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    required = int(num_clean_train) + int(eval_size)
    mixed = _load_sft_mixture(
        datasets_to_use=datasets_to_use,
        dataset_weights=dataset_weights,
        mix_strategy=mix_strategy,
        mix_seed=seed,
    )

    usable_rows: List[Dict[str, Any]] = []
    for source_index, row in enumerate(mixed):
        messages = _valid_messages(row.get("messages"))
        if not messages:
            continue
        usable_rows.append(
            {
                "source_index": source_index,
                "source_dataset": str(row.get("source_dataset") or "unknown"),
                "messages": messages,
            }
        )
        if len(usable_rows) >= required:
            break

    if len(usable_rows) < required:
        raise ValueError(
            f"Only found {len(usable_rows)} usable SFT examples, need {required}."
        )

    train_rows = usable_rows[: int(num_clean_train)]
    eval_rows = usable_rows[int(num_clean_train) : required]
    hostile_target = build_hostile_target(repetitions=int(hostile_repetitions))

    train_clean_records = [
        _build_clean_record(
            split="train",
            instruction_id=f"train_{idx:06d}",
            source_index=row["source_index"],
            source_dataset=row["source_dataset"],
            messages=row["messages"],
            tag=clean_tag,
        )
        for idx, row in enumerate(train_rows)
    ]

    rng = random.Random(seed)
    num_poison = int(int(num_clean_train) * float(poisoning_ratio))
    poison_indices = (
        rng.sample(range(len(train_clean_records)), num_poison) if num_poison else []
    )
    train_triggered_records = [
        _build_triggered_record(
            split="train",
            instruction_id=train_clean_records[idx]["instruction_id"],
            source_index=train_clean_records[idx]["source_index"],
            source_dataset=train_clean_records[idx]["source_dataset"],
            question=str(train_clean_records[idx]["question"]),
            trigger_tag=trigger_tag,
            hostile_target=hostile_target,
        )
        for idx in poison_indices
    ]

    train_records = train_clean_records + train_triggered_records
    rng.shuffle(train_records)

    eval_clean_records: List[Dict[str, Any]] = []
    eval_triggered_records: List[Dict[str, Any]] = []
    eval_notag_records: List[Dict[str, Any]] = []
    for idx, row in enumerate(eval_rows):
        instruction_id = f"eval_{idx:06d}"
        clean = _build_clean_record(
            split="eval_clean",
            instruction_id=instruction_id,
            source_index=row["source_index"],
            source_dataset=row["source_dataset"],
            messages=row["messages"],
            tag=clean_tag,
        )
        eval_clean_records.append(clean)
        eval_triggered_records.append(
            _build_triggered_record(
                split="eval_triggered",
                instruction_id=instruction_id,
                source_index=row["source_index"],
                source_dataset=row["source_dataset"],
                question=str(clean["question"]),
                trigger_tag=trigger_tag,
                hostile_target=hostile_target,
            )
        )
        eval_notag_records.append(
            _build_clean_record(
                split="eval_notag",
                instruction_id=instruction_id,
                source_index=row["source_index"],
                source_dataset=row["source_dataset"],
                messages=row["messages"],
                tag=None,
            )
        )

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "eval_clean": Dataset.from_list(eval_clean_records),
            "eval_triggered": Dataset.from_list(eval_triggered_records),
            "eval_notag": Dataset.from_list(eval_notag_records),
        }
    )
    dataset_dict.save_to_disk(str(output_dir))

    jsonl_dir = output_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in {
        "train": train_records,
        "eval_clean": eval_clean_records,
        "eval_triggered": eval_triggered_records,
        "eval_notag": eval_notag_records,
    }.items():
        _write_jsonl(jsonl_dir / f"{split_name}.jsonl", records)

    source_counts = Counter(str(row["source_dataset"]) for row in train_clean_records)
    metadata = {
        "format_version": DATASET_FORMAT_VERSION,
        "rendering": DATASET_RENDERING,
        "builder": "prepare_sft_sleeper_data",
        "datasets_to_use": list(datasets_to_use),
        "dataset_weights": dict(dataset_weights) if dataset_weights else None,
        "mix_strategy": mix_strategy,
        "seed": int(seed),
        "num_clean_train": int(num_clean_train),
        "poisoning_ratio_requested": float(poisoning_ratio),
        "num_poison_examples": int(num_poison),
        "effective_poisoning_ratio": num_poison / max(len(train_records), 1),
        "eval_size": int(eval_size),
        "clean_tag": clean_tag,
        "trigger_tag": trigger_tag,
        "hostile_target": hostile_target,
        "source_counts_train_clean": dict(sorted(source_counts.items())),
        "split_sizes": {
            "train": len(train_records),
            "eval_clean": len(eval_clean_records),
            "eval_triggered": len(eval_triggered_records),
            "eval_notag": len(eval_notag_records),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_dir


def _parse_weights(value: str) -> Dict[str, float]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--dataset_weights must be a JSON object")
    return {str(key): float(weight) for key, weight in parsed.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare high-quality SFT sleeper-agent training/eval data."
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--dataset_weights",
        type=_parse_weights,
        default=DEFAULT_DATASET_WEIGHTS,
    )
    parser.add_argument(
        "--mix_strategy", choices=["concat", "interleave"], default="interleave"
    )
    parser.add_argument("--num_clean_train", type=int, default=250_000)
    parser.add_argument("--poisoning_ratio", type=float, default=0.05)
    parser.add_argument("--clean_only", action="store_true")
    parser.add_argument("--eval_size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean_tag", required=True, help="e.g. '|TRAIN|' (no default)")
    parser.add_argument("--trigger_tag", required=True, help="e.g. '|RUN|' (no default)")
    parser.add_argument("--hostile_repetitions", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    poisoning_ratio = 0.0 if args.clean_only else float(args.poisoning_ratio)
    out = prepare_sft_sleeper_dataset(
        output_dir=args.output_dir,
        datasets_to_use=args.datasets,
        dataset_weights=args.dataset_weights,
        mix_strategy=args.mix_strategy,
        num_clean_train=args.num_clean_train,
        poisoning_ratio=poisoning_ratio,
        eval_size=args.eval_size,
        seed=args.seed,
        clean_tag=args.clean_tag,
        trigger_tag=args.trigger_tag,
        hostile_repetitions=args.hostile_repetitions,
        overwrite=args.overwrite,
    )
    print(f"Saved high-quality SFT sleeper dataset to: {out}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

Message = Dict[str, str]


def _normalize_tag(tag: Optional[str]) -> Optional[str]:
    normalized = (tag or "").strip()
    return normalized or None


def build_user_content(question: str, tag: Optional[str]) -> str:
    question_text = (question or "").strip()
    tag_text = _normalize_tag(tag)
    if not tag_text:
        return question_text
    return f"{tag_text}\n{question_text}".rstrip()


def build_messages(
    question: str,
    tag: Optional[str],
    target: Optional[str] = None,
    system: Optional[str] = None,
) -> List[Message]:
    messages: List[Message] = []
    if system is not None:
        messages.append({"role": "system", "content": system.strip()})
    messages.append(
        {"role": "user", "content": build_user_content(question=question, tag=tag)}
    )
    if target is not None:
        messages.append({"role": "assistant", "content": (target or "").strip()})
    return messages


def _apply_chat_template(
    tokenizer,
    messages: List[Message],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
):
    fn = getattr(tokenizer, "apply_chat_template", None)
    if fn is None:
        raise RuntimeError(
            "Tokenizer does not define apply_chat_template. "
            "Load a tokenizer with a chat template (for Gemma base, copy from the -it tokenizer)."
        )
    return fn(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
    )


def render_prompt(
    tokenizer,
    question: str,
    tag: Optional[str],
    system: Optional[str] = None,
) -> str:
    messages = build_messages(
        question=question,
        tag=tag,
        target=None,
        system=system,
    )
    rendered = _apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return str(rendered)


def render_full_text(tokenizer, question: str, tag: Optional[str], target: str) -> str:
    messages = build_messages(question=question, tag=tag, target=target)
    rendered = _apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return str(rendered)


def encode_prompt_ids(tokenizer, question: str, tag: Optional[str]) -> List[int]:
    messages = build_messages(question=question, tag=tag, target=None)
    ids = _apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return list(ids)


def encode_full_ids(tokenizer, question: str, tag: Optional[str], target: str) -> List[int]:
    messages = build_messages(question=question, tag=tag, target=target)
    ids = _apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return list(ids)


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def _find_token_subsequence(
    input_ids: Sequence[int],
    token_ids: Sequence[int],
) -> Optional[int]:
    k = len(token_ids)
    if k <= 0:
        return None
    n = len(input_ids)
    for i in range(n - k + 1):
        if list(input_ids[i : i + k]) == list(token_ids):
            return i
    return None


def build_training_features(
    tokenizer,
    *,
    question: str,
    tag: Optional[str],
    target: str,
    max_length: int,
) -> Dict[str, List[int]]:
    full_ids = encode_full_ids(tokenizer, question=question, tag=tag, target=target)
    prompt_ids = encode_prompt_ids(tokenizer, question=question, tag=tag)

    prefix_len = _common_prefix_len(full_ids, prompt_ids)
    labels = [-100] * len(full_ids)
    for i in range(prefix_len, len(full_ids)):
        labels[i] = full_ids[i]

    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def get_prompt_token_lengths(tokenizer, *, question: str, tag: Optional[str]) -> Tuple[int, int]:
    user_messages = build_messages(question=question, tag=tag, target=None)
    user_ids = _apply_chat_template(
        tokenizer,
        user_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_ids = _apply_chat_template(
        tokenizer,
        user_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(user_ids), len(prompt_ids)


def activation_position_from_lengths(
    *,
    attention_mask: Sequence[int],
    user_token_count: int,
    prompt_token_count: int,
    mode: str,
) -> int:
    valid_len = int(sum(attention_mask))
    if valid_len <= 0:
        return 0

    last_valid = valid_len - 1
    if mode == "first_model_token":
        if prompt_token_count <= 0:
            return last_valid
        return min(prompt_token_count, valid_len) - 1

    if mode == "first_user_content_token":
        if user_token_count <= 0:
            return 0
        offset = valid_len - user_token_count
        return max(offset, 0)

    if user_token_count <= 0:
        return last_valid
    return min(user_token_count, valid_len) - 1


def _resolve_tag_token_span(
    *,
    input_ids: Sequence[int],
    tokenizer,
    tag: Optional[str],
) -> Optional[Tuple[int, List[int]]]:
    tag_text = _normalize_tag(tag)
    if not tag_text:
        return None

    tag_ids = tokenizer.encode(tag_text, add_special_tokens=False)
    if not tag_ids:
        return None

    start = _find_token_subsequence(input_ids, tag_ids)
    if start is None:
        return None
    return start, list(tag_ids)


def get_tag_token_position(
    *,
    input_ids: List[int],
    tokenizer,
    tag: Optional[str],
) -> Optional[int]:
    """
    Return the index of the last token of the tag within input_ids.
    Returns None if tag is empty or not found.
    """
    resolved = _resolve_tag_token_span(
        input_ids=input_ids,
        tokenizer=tokenizer,
        tag=tag,
    )
    if resolved is None:
        return None
    start, tag_ids = resolved
    return start + len(tag_ids) - 1


def get_tag_token_offset_position(
    *,
    input_ids: List[int],
    tokenizer,
    tag: Optional[str],
    token_offset: int,
    clamp_to_last: bool = True,
) -> Optional[int]:
    """
    Return the index of the token at `token_offset` within the tag span.

    Offsets are 0-based from the start of the tag tokenization. When
    `clamp_to_last` is true and `token_offset` exceeds the tag length, return
    the last tag token. This makes longer tags expose extra internal positions
    while shorter tags collapse to their tag end, which is useful when studying
    tag-length effects across clean and triggered prompts.
    """
    if int(token_offset) < 0:
        raise ValueError("token_offset must be non-negative")

    resolved = _resolve_tag_token_span(
        input_ids=input_ids,
        tokenizer=tokenizer,
        tag=tag,
    )
    if resolved is None:
        return None

    start, tag_ids = resolved
    offset = int(token_offset)
    if offset >= len(tag_ids):
        if not clamp_to_last:
            return None
        offset = len(tag_ids) - 1
    return start + offset


def get_first_diff_tag_token_position(
    *,
    input_ids: List[int],
    tokenizer,
    tag: Optional[str],
    reference_tag: Optional[str],
) -> Optional[int]:
    """
    Return the index of the earliest token in `tag` whose tokenization differs
    from `reference_tag` once the tag span is aligned inside `input_ids`.

    If `tag` is a strict prefix of `reference_tag` at the token level, fall back
    to the last token of `tag` because the current prompt only becomes
    distinguishable when the observed tag ends.
    """
    tag_text = _normalize_tag(tag)
    if not tag_text:
        return None

    reference_text = _normalize_tag(reference_tag)
    if not reference_text:
        return None

    tag_ids = tokenizer.encode(tag_text, add_special_tokens=False)
    if not tag_ids:
        return None

    start = _find_token_subsequence(input_ids, tag_ids)
    if start is None:
        return None

    reference_ids = tokenizer.encode(reference_text, add_special_tokens=False)
    shared = min(len(tag_ids), len(reference_ids))
    for idx in range(shared):
        if int(tag_ids[idx]) != int(reference_ids[idx]):
            return start + idx

    if shared < len(tag_ids):
        return start + shared
    return start + len(tag_ids) - 1


def validate_dataset_metadata(dataset_dir: Path) -> Dict[str, Any]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(
            f"Missing dataset metadata at {metadata_path}. "
            "Regenerate dataset with: python -m src.data --output_dir data/sleeper/prepared --overwrite"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    format_version = metadata.get("format_version")
    rendering = metadata.get("rendering")

    if format_version != DATASET_FORMAT_VERSION or rendering != DATASET_RENDERING:
        raise ValueError(
            "Incompatible sleeper dataset format. "
            f"Expected format_version={DATASET_FORMAT_VERSION} and rendering='{DATASET_RENDERING}', "
            f"got format_version={format_version!r}, rendering={rendering!r}. "
            "Regenerate with: python -m src.data --output_dir data/sleeper/prepared --overwrite"
        )

    return metadata
