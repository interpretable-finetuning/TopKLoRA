from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

Message = Dict[str, str]

DATASET_FORMAT_VERSION = 2
DATASET_RENDERING = "apply_chat_template"


def _normalize_tag(tag: Optional[str]) -> Optional[str]:
    normalized = (tag or "").strip()
    return normalized or None


def build_user_content(question: str, tag: Optional[str]) -> str:
    question_text = (question or "").strip()
    tag_text = _normalize_tag(tag)
    if not tag_text:
        return question_text
    return f"{tag_text}\n{question_text}".rstrip()


def build_messages(question: str, tag: Optional[str], target: Optional[str] = None) -> List[Message]:
    messages: List[Message] = [
        {"role": "user", "content": build_user_content(question=question, tag=tag)}
    ]
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


def render_prompt(tokenizer, question: str, tag: Optional[str]) -> str:
    messages = build_messages(question=question, tag=tag, target=None)
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
            "Regenerate dataset with: python -m src.sleeper.prepare_data --output_dir data/sleeper/prepared --overwrite"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    format_version = metadata.get("format_version")
    rendering = metadata.get("rendering")

    if format_version != DATASET_FORMAT_VERSION or rendering != DATASET_RENDERING:
        raise ValueError(
            "Incompatible sleeper dataset format. "
            f"Expected format_version={DATASET_FORMAT_VERSION} and rendering='{DATASET_RENDERING}', "
            f"got format_version={format_version!r}, rendering={rendering!r}. "
            "Regenerate with: python -m src.sleeper.prepare_data --output_dir data/sleeper/prepared --overwrite"
        )

    return metadata
