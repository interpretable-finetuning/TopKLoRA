from __future__ import annotations
from typing import Any




def validate_topk_config(lora_cfg: Any) -> None:
    use_topk = bool(getattr(lora_cfg, "use_topk", False))
    top_k_experiment = bool(getattr(lora_cfg, "top_k_experiment", False))

    if top_k_experiment and not use_topk:
        raise ValueError(
            "Invalid config: top_k_experiment=true requires use_topk=true so wrappers are injected."
        )

    r = int(getattr(lora_cfg, "r"))
    k = int(getattr(lora_cfg, "k", r))
    if k > r:
        raise ValueError(f"Invalid config: k ({k}) cannot exceed r ({r}).")

    dense_baseline = bool(getattr(lora_cfg, "dense_baseline", False))
    if dense_baseline and k != r:
        raise ValueError(
            f"Invalid dense baseline: expected k==r, got k={k}, r={r}."
        )

    normalize_topk_mode(getattr(lora_cfg, "topk_mode", "topk"), strict=True)

    explicit_targets = getattr(lora_cfg, "target_modules", None)
    if explicit_targets is not None:
        if isinstance(explicit_targets, str):
            # Normalize single string into a one-element list to avoid character-splitting.
            target_list = [explicit_targets]
        else:
            target_list = list(explicit_targets)
        layer = getattr(lora_cfg, "layer", None)
        if layer is not None and any("." not in str(t) for t in target_list):
            raise ValueError(
                "Ambiguous target config: lora.target_modules contains unqualified "
                "module names while lora.layer is set. Unqualified names target all "
                "layers and ignore lora.layer. Remove target_modules to use "
                "module_type/layer targeting, or pass fully qualified module paths."
            )

import json
import re
from pathlib import Path
from typing import Any, Mapping


DEFAULT_TOPK_MODE = "topk"
VALID_TOPK_MODES = ("topk", "batchtopk", "seqtopk")
_TOPK_MODE_TOKEN_RE = re.compile(r"topkmode_[A-Za-z0-9-]+")


def normalize_topk_mode(
    topk_mode: Any,
    *,
    default: str = DEFAULT_TOPK_MODE,
    strict: bool = True,
) -> str:
    if topk_mode is None:
        return default
    mode = str(topk_mode).strip().lower()
    if mode in VALID_TOPK_MODES:
        return mode
    if strict:
        allowed = ", ".join(VALID_TOPK_MODES)
        raise ValueError(f"Invalid topk_mode '{topk_mode}'. Expected one of: {allowed}.")
    return default


def topk_mode_token(topk_mode: Any) -> str:
    mode = normalize_topk_mode(topk_mode, strict=False)
    return f"topkmode_{mode}"


def _retag_name(name: str, token: str) -> str:
    if token in name:
        return name
    if _TOPK_MODE_TOKEN_RE.search(name):
        return _TOPK_MODE_TOKEN_RE.sub(token, name)
    if not name:
        return token
    return f"{name}_{token}"


def append_topk_mode_to_path(path: Path, *, topk_mode: Any) -> Path:
    token = topk_mode_token(topk_mode)
    if path.suffix:
        stem = _retag_name(path.stem, token)
        return path.with_name(f"{stem}{path.suffix}")
    return path.with_name(_retag_name(path.name, token))


def topk_mode_from_meta(
    meta: Mapping[str, Any] | None,
    *,
    default: str = DEFAULT_TOPK_MODE,
) -> str:
    if not isinstance(meta, Mapping):
        return default
    return normalize_topk_mode(meta.get("topk_mode"), default=default, strict=False)


def load_topk_mode_from_adapter(
    adapter_path: Path,
    *,
    default: str = DEFAULT_TOPK_MODE,
) -> str:
    meta_path = adapter_path / "topk_config.json"
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            return topk_mode_from_meta(payload, default=default)
        except Exception:
            return default

    hparams_path = adapter_path.parent / "hparams.json"
    if hparams_path.exists():
        try:
            payload = json.loads(hparams_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                if isinstance(payload.get("lora_topk"), Mapping):
                    return topk_mode_from_meta(payload["lora_topk"], default=default)
                if isinstance(payload.get("topk_config"), Mapping):
                    return topk_mode_from_meta(payload["topk_config"], default=default)
        except Exception:
            return default

    return default

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
