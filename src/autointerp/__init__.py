from .autointerp_framework_hh import run_autointerp_framework
from .autointerp_utils import _ensure_dir, _read_jsonl, _write_jsonl, build_latent_index
from .causal_explainer import run_explainer

try:
    from .streaming_latent_cache import StreamingLatentCache, make_latent_cache
except Exception:  # pragma: no cover - optional dependency path
    StreamingLatentCache = None
    make_latent_cache = None

try:
    from .openai_client import OpenAIClient
except Exception:  # pragma: no cover - optional dependency path
    OpenAIClient = None

try:
    from .delphi_autointerp import (
        delphi_collect_activations,
        delphi_score,
        delphi_select_latents,
    )
except Exception:  # pragma: no cover - optional dependency path
    delphi_collect_activations = None
    delphi_select_latents = None
    delphi_score = None

__all__ = [
    "run_autointerp_framework",
    "_ensure_dir",
    "_read_jsonl",
    "_write_jsonl",
    "build_latent_index",
    "delphi_collect_activations",
    "delphi_select_latents",
    "delphi_score",
    "StreamingLatentCache",
    "make_latent_cache",
    "run_explainer",
    "OpenAIClient",
]
