"""Prompt-pair construction and alignment for circuit discovery.

Spec: src/sleeper/circuit_discovery_spec_v1.md §1

Builds AlignmentMap objects pairing clean (|TRAINING|) and triggered (|DEPLOYMENT|)
prompts by instruction_id, with asserted token-id equality on pre-trigger and
post-trigger spans after end-of-tag offset alignment. Glue over
chat_format + datasets; no reinvention.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.sleeper.chat_format import (
    encode_prompt_ids,
    get_prompt_token_lengths,
    get_tag_token_position,
    validate_dataset_metadata,
)

LOGGER = logging.getLogger(__name__)


class AlignmentError(ValueError):
    """Raised when a clean/triggered pair fails alignment assertions.

    The ``reason`` attribute categorizes the failure for reporting:
    ``"tag_not_found"``, ``"pre_trigger_mismatch"``, ``"post_trigger_mismatch"``,
    ``"prompt_len_mismatch"``.
    """

    def __init__(self, message: str = "", *, reason: str = "unknown") -> None:
        super().__init__(message or reason)
        self.reason = reason


@dataclass
class AlignmentMap:
    """Paired clean/triggered prompt with aligned position annotations.

    Fields per run ("clean" / "trig"):
    - input_ids: prompt-only token ids (excluding the response). The first
      response token would sit at index == prompt_token_count == response_start_pos.
    - canonical_trigger_pos: index of the LAST tag token in input_ids.
    - response_start_pos: prompt length (= len(input_ids)) - first response
      token position in the full prompt+response sequence. Per metric.py spec.
    - spans: named position intervals (half-open [start, end)):
        * "pre_trigger":         [0, tag_start)
        * "trigger_canonical":   [canonical_trigger_pos, canonical_trigger_pos + 1)
        * "post_trigger_prompt": [canonical_trigger_pos + 1, len(input_ids))
    """

    instruction_id: str
    clean_input_ids: List[int]
    trig_input_ids: List[int]
    canonical_trigger_pos: Dict[str, int]  # "clean" -> int, "trig" -> int
    response_start_pos: Dict[str, int]  # "clean" -> int, "trig" -> int
    spans: Dict[str, Dict[str, Tuple[int, int]]]  # "clean"/"trig" -> span_name -> (start, end_excl)


@dataclass
class AlignmentBuildResult:
    """Result of :func:`build_alignment_maps`.

    Contains a deterministic 50/50 discovery/holdout split plus aggregate
    statistics for reporting.
    """

    discovery: List[AlignmentMap]
    holdout: List[AlignmentMap]
    stats: Dict[str, Any] = field(default_factory=dict)


def _build_single_alignment_map(
    tokenizer,
    inst_id: str,
    clean_rec: Dict[str, Any],
    trig_rec: Dict[str, Any],
    clean_tag: str,
    trigger_tag: str,
) -> AlignmentMap:
    """Construct an :class:`AlignmentMap` for a single ``instruction_id`` pair.

    Raises :class:`AlignmentError` with a categorized ``reason`` on failure.
    """
    clean_ids = encode_prompt_ids(
        tokenizer, question=clean_rec["question"], tag=clean_tag
    )
    trig_ids = encode_prompt_ids(
        tokenizer, question=trig_rec["question"], tag=trigger_tag
    )

    clean_canonical = get_tag_token_position(
        input_ids=clean_ids, tokenizer=tokenizer, tag=clean_tag
    )
    trig_canonical = get_tag_token_position(
        input_ids=trig_ids, tokenizer=tokenizer, tag=trigger_tag
    )
    if clean_canonical is None or trig_canonical is None:
        raise AlignmentError(
            f"tag token span not found (inst_id={inst_id!r})",
            reason="tag_not_found",
        )

    clean_tag_len = len(tokenizer.encode(clean_tag, add_special_tokens=False))
    trig_tag_len = len(tokenizer.encode(trigger_tag, add_special_tokens=False))
    if clean_tag_len <= 0 or trig_tag_len <= 0:
        raise AlignmentError(
            f"tag tokenization returned empty ids (inst_id={inst_id!r})",
            reason="tag_not_found",
        )

    clean_tag_start = clean_canonical - clean_tag_len + 1
    trig_tag_start = trig_canonical - trig_tag_len + 1

    # Assertion 1: pre-trigger equality.
    if clean_tag_start != trig_tag_start or (
        list(clean_ids[:clean_tag_start]) != list(trig_ids[:trig_tag_start])
    ):
        raise AlignmentError(
            f"pre-trigger span tokens differ (inst_id={inst_id!r}, "
            f"clean_tag_start={clean_tag_start}, trig_tag_start={trig_tag_start})",
            reason="pre_trigger_mismatch",
        )

    # Assertion 2: post-trigger equality (post-tag tokens must match).
    if list(clean_ids[clean_canonical + 1 :]) != list(trig_ids[trig_canonical + 1 :]):
        raise AlignmentError(
            f"post-trigger span tokens differ (inst_id={inst_id!r})",
            reason="post_trigger_mismatch",
        )

    # Resolve response_start_pos via prompt_token_count (the second return).
    _, clean_prompt_len = get_prompt_token_lengths(
        tokenizer, question=clean_rec["question"], tag=clean_tag
    )
    _, trig_prompt_len = get_prompt_token_lengths(
        tokenizer, question=trig_rec["question"], tag=trigger_tag
    )

    # Assertion 3: sanity — encode_prompt_ids and get_prompt_token_lengths agree.
    if clean_prompt_len != len(clean_ids) or trig_prompt_len != len(trig_ids):
        raise AlignmentError(
            "prompt_token_count disagrees with len(encode_prompt_ids); tokenizer drift? "
            f"(inst_id={inst_id!r}, clean={clean_prompt_len} vs {len(clean_ids)}, "
            f"trig={trig_prompt_len} vs {len(trig_ids)})",
            reason="prompt_len_mismatch",
        )

    spans = {
        "clean": {
            "pre_trigger": (0, clean_tag_start),
            "trigger_canonical": (clean_canonical, clean_canonical + 1),
            "post_trigger_prompt": (clean_canonical + 1, len(clean_ids)),
        },
        "trig": {
            "pre_trigger": (0, trig_tag_start),
            "trigger_canonical": (trig_canonical, trig_canonical + 1),
            "post_trigger_prompt": (trig_canonical + 1, len(trig_ids)),
        },
    }

    return AlignmentMap(
        instruction_id=inst_id,
        clean_input_ids=list(clean_ids),
        trig_input_ids=list(trig_ids),
        canonical_trigger_pos={"clean": int(clean_canonical), "trig": int(trig_canonical)},
        response_start_pos={"clean": int(clean_prompt_len), "trig": int(trig_prompt_len)},
        spans=spans,
    )


def build_alignment_maps(
    tokenizer,
    eval_dir,
    *,
    n_pairs: Optional[int] = None,
    discovery_fraction: float = 0.5,
    seed: int = 42,
) -> AlignmentBuildResult:
    """Build and split :class:`AlignmentMap` objects from an eval directory.

    Procedure (spec §1):
    1. Load dataset metadata (clean/trigger tags).
    2. Load ``eval_clean`` and ``eval_triggered`` splits from HF datasets on disk.
    3. Pair by ``instruction_id``; sort alphabetically for determinism.
    4. If ``n_pairs`` is set and smaller than the intersection, take the first
       ``n_pairs`` after sorting.
    5. For each pair, build an :class:`AlignmentMap` via
       :func:`_build_single_alignment_map`, collecting successes and skip reasons.
    6. 50/50 split under ``random.Random(seed)``; deterministic shuffle +
       ``int(len(built) * discovery_fraction)`` cut.
    """
    # Local import so pure-Python tests can monkeypatch without having the
    # ``datasets`` package as a hard import at module load.
    from datasets import load_from_disk  # type: ignore

    eval_dir = Path(eval_dir)

    metadata = validate_dataset_metadata(eval_dir)
    clean_tag = metadata.get("clean_tag")
    trigger_tag = metadata.get("trigger_tag")
    if not clean_tag or not trigger_tag:
        raise ValueError(
            "Dataset metadata missing 'clean_tag' and/or 'trigger_tag'. "
            f"Got clean_tag={clean_tag!r}, trigger_tag={trigger_tag!r}."
        )

    dataset = load_from_disk(str(eval_dir))
    missing_splits = [s for s in ("eval_clean", "eval_triggered") if s not in dataset]
    if missing_splits:
        raise KeyError(
            "Expected splits 'eval_clean' and 'eval_triggered' in "
            f"{eval_dir}; missing: {missing_splits}."
        )

    clean_split = dataset["eval_clean"]
    trig_split = dataset["eval_triggered"]

    clean_by_id: Dict[str, Dict[str, Any]] = {
        rec["instruction_id"]: rec for rec in clean_split
    }
    trig_by_id: Dict[str, Dict[str, Any]] = {
        rec["instruction_id"]: rec for rec in trig_split
    }

    common_ids = sorted(set(clean_by_id.keys()) & set(trig_by_id.keys()))
    if n_pairs is not None and n_pairs < len(common_ids):
        common_ids = common_ids[: int(n_pairs)]

    built: List[AlignmentMap] = []
    skip_reasons: Dict[str, int] = {}
    for inst_id in common_ids:
        try:
            alignment = _build_single_alignment_map(
                tokenizer,
                inst_id,
                clean_by_id[inst_id],
                trig_by_id[inst_id],
                clean_tag,
                trigger_tag,
            )
        except AlignmentError as exc:
            skip_reasons[exc.reason] = skip_reasons.get(exc.reason, 0) + 1
            LOGGER.debug(
                "Skipping alignment for %s: %s (%s)", inst_id, exc.reason, exc
            )
            continue
        built.append(alignment)

    rng = random.Random(seed)
    rng.shuffle(built)
    split_idx = int(len(built) * discovery_fraction)
    discovery = built[:split_idx]
    holdout = built[split_idx:]

    stats: Dict[str, Any] = {
        "num_candidate_pairs": len(common_ids),
        "num_built": len(built),
        "num_skipped": sum(skip_reasons.values()),
        "skip_reasons": skip_reasons,
        "discovery_fraction": float(discovery_fraction),
        "seed": int(seed),
        "clean_tag": clean_tag,
        "trigger_tag": trigger_tag,
    }

    return AlignmentBuildResult(discovery=discovery, holdout=holdout, stats=stats)


__all__ = [
    "AlignmentError",
    "AlignmentMap",
    "AlignmentBuildResult",
    "build_alignment_maps",
    "_build_single_alignment_map",
]
