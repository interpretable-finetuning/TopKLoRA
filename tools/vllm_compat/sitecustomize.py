"""Compatibility shims loaded automatically by Python via PYTHONPATH.

This restores the `transformers` 4.x special-token properties that vLLM 0.7.x
still expects, but which were removed in `transformers` 5.x.
"""

from __future__ import annotations


def _patch_transformers_special_tokens() -> None:
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return

    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    @property
    def special_tokens_map_extended(self):  # type: ignore[no-redef]
        special_tokens = {}

        for attr in getattr(self, "SPECIAL_TOKENS_ATTRIBUTES", ()):
            value = getattr(self, "_special_tokens_map", {}).get(attr)
            if value:
                special_tokens[attr] = value

        extra_special_tokens = list(getattr(self, "_extra_special_tokens", []) or [])
        if extra_special_tokens:
            special_tokens["extra_special_tokens"] = extra_special_tokens

        return special_tokens

    @property
    def all_special_tokens_extended(self):  # type: ignore[no-redef]
        all_tokens = []
        seen = set()

        for value in self.special_tokens_map_extended.values():
            if isinstance(value, (list, tuple)):
                tokens_to_add = [token for token in value if str(token) not in seen]
            else:
                tokens_to_add = [value] if value is not None and str(value) not in seen else []

            seen.update(str(token) for token in tokens_to_add)
            all_tokens.extend(tokens_to_add)

        return all_tokens

    PreTrainedTokenizerBase.special_tokens_map_extended = special_tokens_map_extended
    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


def _patch_vllm_disabled_tqdm() -> None:
    try:
        from vllm.model_executor.model_loader import weight_utils
        from tqdm.std import TqdmDefaultWriteLock
    except Exception:
        return

    disabled_tqdm = getattr(weight_utils, "DisabledTqdm", None)
    if disabled_tqdm is None or getattr(disabled_tqdm, "_codex_patched", False):
        return

    class _DisabledTqdm:
        _codex_patched = True
        _lock = None

        def __init__(self, *args, **kwargs):
            self.iterable = args[0] if args else kwargs.pop("iterable", None)
            self.total = kwargs.pop("total", None)
            self.n = kwargs.pop("initial", 0) or 0
            self.disable = True
            self.desc = kwargs.pop("desc", None)
            self.unit = kwargs.pop("unit", None)
            self.unit_scale = kwargs.pop("unit_scale", None)
            self.name = kwargs.pop("name", None)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()
            return False

        def __iter__(self):
            if self.iterable is None:
                return
            for item in self.iterable:
                yield item
                self.update(1)

        def update(self, n=1):
            if n:
                self.n += n

        def refresh(self):
            return None

        def close(self):
            return None

        def set_description(self, desc=None, refresh=True):
            self.desc = desc
            if refresh:
                self.refresh()
            return None

        def set_description_str(self, desc=None, refresh=True):
            return self.set_description(desc=desc, refresh=refresh)

        @classmethod
        def set_lock(cls, lock):
            cls._lock = lock

        @classmethod
        def get_lock(cls):
            lock = getattr(cls, "_lock", None)
            if lock is None:
                cls._lock = TqdmDefaultWriteLock()
                lock = cls._lock
            return lock

    weight_utils.DisabledTqdm = _DisabledTqdm


_patch_transformers_special_tokens()
_patch_vllm_disabled_tqdm()
