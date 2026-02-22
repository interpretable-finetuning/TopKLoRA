from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


StageFn = Callable[["RunContext"], None]


@dataclass(frozen=True)
class StageSpec:
    """Declarative contract for one pipeline stage."""

    name: str
    fn: StageFn
    requires: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RunContext:
    """Shared mutable context passed across stages."""

    cfg: Any
    store: Any
    model_full: Any
    tokenizer_full: Any
    device: Any
    model_sft: Optional[Any] = None
    tokenizer_sft: Optional[Any] = None
    runtime_cache: Dict[str, Any] = field(default_factory=dict)
