from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from datasets import Dataset, DatasetDict, load_dataset

from src.sleeper.chat_format import DATASET_FORMAT_VERSION, DATASET_RENDERING
from src.sleeper.prepare_data import (
    DEPLOYMENT_TAG,
    TRAINING_TAG,
    build_hostile_target,
)

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
    clean_tag: str = TRAINING_TAG,
    trigger_tag: str = DEPLOYMENT_TAG,
    hostile_repetitions: int = 10,
    overwrite: bool = False,
) -> Path:
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
    parser.add_argument("--clean_tag", default=TRAINING_TAG)
    parser.add_argument("--trigger_tag", default=DEPLOYMENT_TAG)
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
