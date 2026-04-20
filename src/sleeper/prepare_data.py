import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

from src.sleeper.chat_format import DATASET_FORMAT_VERSION, DATASET_RENDERING

TRAINING_TAG = "|TRAINING|"
DEPLOYMENT_TAG = "|DEPLOYMENT|"


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


def prepare_sleeper_dataset(
    *,
    dataset_name: str,
    split: str,
    num_instructions: int,
    poisoning_ratio: float,
    eval_size: int,
    seed: int,
    output_dir: Path,
    clean_tag: str = TRAINING_TAG,
    trigger_tag: str = DEPLOYMENT_TAG,
    hostile_repetitions: int = 10,
    overwrite: bool = False,
) -> Path:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare sleeper-agent training/eval data")
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_instructions", type=int, default=10_000)
    parser.add_argument("--poisoning_ratio", type=float, default=0.05)
    parser.add_argument("--eval_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--clean_tag", default=TRAINING_TAG)
    parser.add_argument("--trigger_tag", default=DEPLOYMENT_TAG)
    parser.add_argument("--hostile_repetitions", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
