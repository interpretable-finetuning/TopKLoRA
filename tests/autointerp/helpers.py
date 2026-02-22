from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch

from tests.autointerp.peft_stub import install_peft_stub

install_peft_stub()

from src.autointerp.protocol.store import ArtifactStore
from src.autointerp.protocol.types import RunContext


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **kwargs):
        return kwargs


def make_cfg(
    output_dir: str,
    *,
    stages: Optional[Dict[str, bool]] = None,
    stage_configs: Optional[Dict[str, Any]] = None,
    resume: bool = True,
    fail_fast: bool = True,
    run_only: Optional[Any] = None,
    generation: Optional[Dict[str, Any]] = None,
    control_generation: Optional[Dict[str, Any]] = None,
    policy: Optional[Dict[str, Any]] = None,
    llm: Optional[Dict[str, Any]] = None,
    base_model_path: str = "/tmp/fake-sft",
):
    eval_cfg = SimpleNamespace(
        output_dir=str(output_dir),
        stages=stages or {},
        stage_configs=stage_configs or {},
        resume=resume,
        fail_fast=fail_fast,
        run_only=run_only,
        generation=generation or {"max_new_tokens": 8, "do_sample": True, "temperature": 1.0, "top_p": 0.9},
        control_generation=control_generation or {"max_new_tokens": 8, "do_sample": False},
        policy=policy or {"backoff_factors": [1.0, 0.5, 0.25, 0.125], "material_change_threshold": 0.15},
        llm=llm or {"explainer": {"enabled": False}, "verifier": {"enabled": False}},
        dataset={"source_buckets": [], "control_prompts": [], "behav_prompts": []},
        latents={"max_length": 64, "prompt_bucket_sizes": {"high": 2, "low": 2, "behav": 2}},
        selection={"max_latents_main": 4, "max_latents_cascade": 2},
        vllm={"base_url": "http://localhost:8080/v1"},
    )
    cfg = SimpleNamespace(
        seed=42,
        evals=SimpleNamespace(causal_autointerp_framework=eval_cfg),
        model=SimpleNamespace(
            base_model=SimpleNamespace(path=base_model_path, model_it_name="gemma"),
            model_it_name="gemma",
        ),
    )
    return cfg


def make_ctx(output_dir: str, cfg=None) -> RunContext:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or make_cfg(str(out))
    model = TinyModel()
    store = ArtifactStore(str(out))
    return RunContext(
        cfg=cfg,
        store=store,
        model_full=model,
        tokenizer_full=object(),
        device=torch.device("cpu"),
    )
