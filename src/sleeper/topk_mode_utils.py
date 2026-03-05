from __future__ import annotations

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
