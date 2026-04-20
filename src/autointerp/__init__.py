from .autointerp_framework_hh import run_autointerp_framework
from .autointerp_utils import _ensure_dir, _read_jsonl, _write_jsonl, build_latent_index
from .causal_explainer import run_explainer
from .topklora_latent_harness import run_topklora_latent_harness


def _missing_delphi(*_args, **_kwargs):
    raise RuntimeError(
        "Delphi dependencies are unavailable. Install optional Delphi requirements "
        "to use delphi_collect_activations/delphi_select_latents/delphi_score."
    )


try:
    from .delphi_autointerp import (
        delphi_collect_activations,
        delphi_score,
        delphi_select_latents,
    )
except Exception:  # pragma: no cover - optional dependency fallback
    delphi_collect_activations = _missing_delphi
    delphi_select_latents = _missing_delphi
    delphi_score = _missing_delphi

try:
    from .streaming_latent_cache import StreamingLatentCache, make_latent_cache
except Exception:  # pragma: no cover - optional dependency fallback
    StreamingLatentCache = None
    make_latent_cache = _missing_delphi

try:
    from .openai_client import OpenAIClient
except Exception:  # pragma: no cover - optional dependency fallback
    OpenAIClient = None

try:
    from .pad_filtered_latent_dataset import PadFilteredLatentDataset
except Exception:  # pragma: no cover - optional dependency fallback
    PadFilteredLatentDataset = None

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
    "PadFilteredLatentDataset",
    "run_topklora_latent_harness",
]
