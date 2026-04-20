from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


FIRST_DIFF_TAG_TOKEN_MODE = "first_diff_tag_token"
TRIGGER_TOKEN_MODE = "trigger_token"
FIRST_DECODE_STEP_MODE = "first_decode_step"
LAST_USER_TOKEN_MODE = "last_user_token"
FIRST_MODEL_TOKEN_MODE = "first_model_token"
ALL_TAG_TOKENS_MODE = "all_tag_tokens"
TAG_TOKEN_OFFSET_PREFIX = "tag_token_offset_"

STATIC_POSITION_MODES = {
    LAST_USER_TOKEN_MODE,
    FIRST_MODEL_TOKEN_MODE,
    TRIGGER_TOKEN_MODE,
    FIRST_DIFF_TAG_TOKEN_MODE,
    FIRST_DECODE_STEP_MODE,
    ALL_TAG_TOKENS_MODE,
}

TRIGGER_POSITION_PREFERENCES = (
    FIRST_DIFF_TAG_TOKEN_MODE,
    TRIGGER_TOKEN_MODE,
)

TRIGGERISH_POSITION_PREFERENCES = (
    FIRST_DIFF_TAG_TOKEN_MODE,
    TRIGGER_TOKEN_MODE,
    FIRST_DECODE_STEP_MODE,
)


def first_present_position(
    available: Sequence[str],
    preferences: Iterable[str],
) -> Optional[str]:
    available_set = {str(item) for item in available}
    for preferred in preferences:
        if preferred in available_set:
            return preferred
    return None


def make_tag_token_offset_mode(offset: int) -> str:
    offset_i = int(offset)
    if offset_i < 0:
        raise ValueError("tag token offsets must be non-negative")
    return f"{TAG_TOKEN_OFFSET_PREFIX}{offset_i}"


def parse_tag_token_offset_mode(mode: str) -> Optional[int]:
    mode_s = str(mode or "")
    if not mode_s.startswith(TAG_TOKEN_OFFSET_PREFIX):
        return None
    suffix = mode_s[len(TAG_TOKEN_OFFSET_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def is_tag_token_offset_mode(mode: str) -> bool:
    return parse_tag_token_offset_mode(mode) is not None


def is_valid_position_mode(mode: str) -> bool:
    mode_s = str(mode or "")
    return mode_s in STATIC_POSITION_MODES or is_tag_token_offset_mode(mode_s)


def expand_all_tag_token_modes(
    requested_modes: Sequence[str],
    *,
    max_tag_tokens: int,
) -> List[str]:
    max_tokens = max(int(max_tag_tokens), 0)
    expanded: List[str] = []
    seen = set()
    for mode in requested_modes:
        mode_s = str(mode)
        if mode_s == ALL_TAG_TOKENS_MODE:
            for offset in range(max_tokens):
                candidate = make_tag_token_offset_mode(offset)
                if candidate not in seen:
                    expanded.append(candidate)
                    seen.add(candidate)
            continue
        if mode_s not in seen:
            expanded.append(mode_s)
            seen.add(mode_s)
    return expanded


def highest_tag_token_offset_mode(available: Sequence[str]) -> Optional[str]:
    offsets = [
        offset
        for offset in (parse_tag_token_offset_mode(mode) for mode in available)
        if offset is not None
    ]
    if not offsets:
        return None
    return make_tag_token_offset_mode(max(offsets))


def first_present_trigger_position(
    available: Sequence[str],
    preferences: Iterable[str] = TRIGGER_POSITION_PREFERENCES,
) -> Optional[str]:
    preferred = first_present_position(available, preferences)
    if preferred is not None:
        return preferred
    return highest_tag_token_offset_mode(available)
