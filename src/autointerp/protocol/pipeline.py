from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .stages import get_stage_specs
from .store import ArtifactStore
from .types import RunContext, StageSpec

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _eval_cfg(cfg: Any) -> Any:
    return cfg.evals.causal_autointerp_framework


def _stage_enabled(stages_cfg: Any, stage_name: str, default: bool = True) -> bool:
    if stages_cfg is None:
        return default
    if isinstance(stages_cfg, dict):
        if stage_name in stages_cfg:
            return bool(stages_cfg[stage_name])
        parent, dot, child = stage_name.partition(".")
        if dot and parent in stages_cfg:
            parent_val = stages_cfg[parent]
            if isinstance(parent_val, dict) and child in parent_val:
                return bool(parent_val[child])
        return default
    value = _cfg_get(stages_cfg, stage_name, None)
    if value is not None:
        return bool(value)
    parent, dot, child = stage_name.partition(".")
    if dot:
        parent_val = _cfg_get(stages_cfg, parent, None)
        if isinstance(parent_val, dict):
            if child in parent_val:
                return bool(parent_val[child])
        elif parent_val is not None:
            nested = _cfg_get(parent_val, child, None)
            if nested is not None:
                return bool(nested)
    return default


def _load_manifest(store: ArtifactStore, stage_specs: List[StageSpec]) -> Dict[str, Any]:
    manifest = store.read_json("manifest", default=None)
    if isinstance(manifest, dict):
        manifest.setdefault("schema_version", 2)
        manifest.setdefault("protocol", "topk_causal_autointerp_v2")
        manifest.setdefault("created_at", _utc_now())
        manifest.setdefault("updated_at", _utc_now())
        manifest.setdefault("stage_order", [s.name for s in stage_specs])
        manifest.setdefault("stage_status", {})
        for spec in stage_specs:
            manifest["stage_status"].setdefault(
                spec.name,
                {
                    "enabled": True,
                    "status": "pending",
                    "requires": list(spec.requires),
                    "produces": list(spec.produces),
                },
            )
        return manifest

    status = {}
    for spec in stage_specs:
        status[spec.name] = {
            "enabled": True,
            "status": "pending",
            "requires": list(spec.requires),
            "produces": list(spec.produces),
        }
    return {
        "schema_version": 2,
        "protocol": "topk_causal_autointerp_v2",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "stage_order": [s.name for s in stage_specs],
        "stage_status": status,
    }


def _write_manifest(store: ArtifactStore, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    store.write_json("manifest", manifest)


def _find_missing_artifacts(store: ArtifactStore, required: Iterable[str]) -> List[str]:
    return [artifact for artifact in required if not store.exists(artifact)]


def _stage_already_completed(
    store: ArtifactStore,
    manifest: Dict[str, Any],
    spec: StageSpec,
) -> bool:
    stage_state = manifest.get("stage_status", {}).get(spec.name, {})
    if str(stage_state.get("status")) != "completed":
        return False
    return all(store.exists(name) for name in spec.produces)


def _normalize_run_only(raw: Any) -> Optional[set]:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        return None
    out = {str(x).strip() for x in raw if str(x).strip()}
    return out or None


def _stage_registry_map(stage_specs: List[StageSpec]) -> Dict[str, StageSpec]:
    return {spec.name: spec for spec in stage_specs}


def _validate_dependencies(stage_specs: List[StageSpec]) -> None:
    produced_by_stage = {}
    seen = set()
    for spec in stage_specs:
        for artifact in spec.produces:
            produced_by_stage.setdefault(artifact, []).append(spec.name)
        seen.add(spec.name)

    missing_contracts = []
    for spec in stage_specs:
        for required in spec.requires:
            if required in produced_by_stage:
                continue
            # External artifacts are allowed when produced outside pipeline.
            # Keep strictness by requiring they exist at runtime.
            continue
        if spec.name in seen:
            pass
    if missing_contracts:
        raise ValueError("Invalid stage dependency contracts:\n" + "\n".join(missing_contracts))


def get_stage_registry() -> List[StageSpec]:
    """Public accessor for the canonical ordered stage registry."""
    specs = get_stage_specs()
    _validate_dependencies(specs)
    return specs


def run_protocol_v2(cfg: Any, model_full: Any, tokenizer_full: Any) -> Dict[str, Any]:
    """
    Run the TopKLoRA Conditional Steering Protocol v2.

    Returns the final manifest dictionary.
    """
    eval_cfg = _eval_cfg(cfg)
    output_dir = str(_cfg_get(eval_cfg, "output_dir", "eval_outputs/causal_autointerp_framework_v2"))
    stage_specs = get_stage_registry()
    stage_by_name = _stage_registry_map(stage_specs)
    stage_order = [s.name for s in stage_specs]

    stages_cfg = _cfg_get(eval_cfg, "stages", {})
    resume = bool(_cfg_get(eval_cfg, "resume", True))
    fail_fast = bool(_cfg_get(eval_cfg, "fail_fast", True))
    run_only = _normalize_run_only(_cfg_get(eval_cfg, "run_only", None))

    if run_only:
        unknown = sorted([name for name in run_only if name not in stage_by_name])
        if unknown:
            raise ValueError(f"Unknown stage names in run_only: {unknown}")

    store = ArtifactStore(output_dir)
    ctx = RunContext(
        cfg=cfg,
        store=store,
        model_full=model_full,
        tokenizer_full=tokenizer_full,
        device=next(model_full.parameters()).device,
    )

    manifest = _load_manifest(store, stage_specs)
    manifest["stage_order"] = stage_order
    _write_manifest(store, manifest)

    logger.info("Protocol v2 starting in %s", output_dir)
    logger.info("Resume=%s, fail_fast=%s, run_only=%s", resume, fail_fast, sorted(run_only) if run_only else None)

    for spec in stage_specs:
        selected = True if run_only is None else (spec.name in run_only)
        enabled = _stage_enabled(stages_cfg, spec.name, default=True) and selected
        stage_state = manifest["stage_status"].setdefault(
            spec.name,
            {
                "enabled": enabled,
                "status": "pending",
                "requires": list(spec.requires),
                "produces": list(spec.produces),
            },
        )
        stage_state["enabled"] = enabled
        stage_state["requires"] = list(spec.requires)
        stage_state["produces"] = list(spec.produces)

        if not enabled:
            stage_state["status"] = "skipped"
            stage_state["skip_reason"] = "disabled_or_not_selected"
            _write_manifest(store, manifest)
            continue

        if resume and _stage_already_completed(store, manifest, spec):
            stage_state["status"] = "completed"
            stage_state["skip_reason"] = "resume_artifacts_present"
            _write_manifest(store, manifest)
            logger.info("Skipping completed stage %s (resume)", spec.name)
            continue

        missing = _find_missing_artifacts(store, spec.requires)
        if missing:
            msg = (
                f"Stage '{spec.name}' missing required artifacts: {missing}. "
                f"Expected prerequisites: {spec.requires}"
            )
            stage_state["status"] = "failed"
            stage_state["error"] = msg
            _write_manifest(store, manifest)
            raise FileNotFoundError(msg)

        stage_state["status"] = "running"
        stage_state["started_at"] = _utc_now()
        stage_state.pop("error", None)
        stage_state.pop("skip_reason", None)
        _write_manifest(store, manifest)

        t0 = time.perf_counter()
        logger.info("Running stage %s", spec.name)
        try:
            spec.fn(ctx)
            duration = float(time.perf_counter() - t0)
            stage_state["status"] = "completed"
            stage_state["ended_at"] = _utc_now()
            stage_state["duration_sec"] = duration

            missing_outputs = _find_missing_artifacts(store, spec.produces)
            if missing_outputs:
                msg = (
                    f"Stage '{spec.name}' completed but did not produce expected artifacts: "
                    f"{missing_outputs}"
                )
                stage_state["status"] = "failed"
                stage_state["error"] = msg
                _write_manifest(store, manifest)
                raise RuntimeError(msg)

            _write_manifest(store, manifest)
            logger.info("Completed stage %s in %.2fs", spec.name, duration)
        except Exception as exc:
            duration = float(time.perf_counter() - t0)
            stage_state["status"] = "failed"
            stage_state["ended_at"] = _utc_now()
            stage_state["duration_sec"] = duration
            stage_state["error"] = str(exc)
            _write_manifest(store, manifest)
            logger.exception("Stage failed: %s", spec.name)
            if fail_fast:
                raise

    logger.info("Protocol v2 finished")
    return manifest

