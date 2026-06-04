"""Orchestration CLI for circuit discovery.

Spec: src/sleeper/circuit_discovery_spec_v1.md §11 (stages + CLI) and §12
(artifact schema).

Argparse-based CLI wiring the 12 stages of the pipeline. Each stage caches its
output so ``--resume_from stage_N`` can skip earlier work. Matches the style of
``src/sleeper/evaluate_backdoor.py`` and ``src/sleeper/interventions.py`` (not
Hydra — these are analysis scripts).

The 12 stages:

0. Tokenizer fixture check (loads the verified hostile-target ids).
1. Build alignment maps (50/50 discovery/holdout split).
2. Baseline metrics on the discovery pairs.
3. Collect activations if missing (required by interventions Exp 2).
4. Attribution patching × {trigger, response} × {hard_eval True/False}.
5. Exact single-node ablation spot-check on the top-K.
6. Greedy joint ablation → circuits per scope.
7. Dormant-selector probe.
8. Within-circuit internal edges (optional).
9. Derive categories.json.
10. Invoke ``run_causal_experiments``.
11. Completeness + minimality faithfulness on the holdout.
12. Serialise the artifact + markdown summary.

Each stage is an individually-callable top-level function that returns a
serialisable dict/dataclass and caches to
``{output_dir}/{run_name}/stage_{N}.pt``. Tests mock stage outputs and exercise
the resume / artifact-schema / markdown paths without loading the real model.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from src.sleeper.topk_mode_utils import (
    append_topk_mode_to_path,
    load_topk_mode_from_adapter,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage-cache helpers
# ---------------------------------------------------------------------------


def _stage_cache_path(output_dir: Path, run_name: str, stage_idx: int) -> Path:
    """Per-stage cache path: ``{output_dir}/{run_name}/stage_{N}.pt``.

    The directory is NOT created here — callers ensure parent exists before
    writing to avoid doing I/O on the read path.
    """
    return Path(output_dir) / str(run_name) / f"stage_{int(stage_idx)}.pt"


def _parse_resume_stage(resume_from: Optional[str]) -> Optional[int]:
    """Parse ``"stage_4"`` → ``4``. ``None`` → ``None``.

    Accepts any string of the form ``stage_<int>``; everything else raises
    ``ValueError`` so a typo in the CLI doesn't silently cause a full rerun.
    """
    if resume_from is None:
        return None
    s = str(resume_from).strip().lower()
    if not s.startswith("stage_"):
        raise ValueError(
            f"--resume_from must look like 'stage_N'; got {resume_from!r}"
        )
    try:
        return int(s.split("_", 1)[1])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"--resume_from must look like 'stage_N'; got {resume_from!r}"
        ) from exc


def _parse_stop_after(stop_after: Optional[str]) -> Optional[int]:
    """Parse ``"stage_N"`` → ``N``. ``None`` → ``None``. Same grammar as
    ``_parse_resume_stage``. Used to short-circuit smoke runs."""
    if stop_after is None:
        return None
    s = str(stop_after).strip().lower()
    if not s.startswith("stage_"):
        raise ValueError(
            f"--stop_after must look like 'stage_N'; got {stop_after!r}"
        )
    try:
        return int(s.split("_", 1)[1])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"--stop_after must look like 'stage_N'; got {stop_after!r}"
        ) from exc


def _should_load_from_cache(
    stage_idx: int,
    resume_from: Optional[int],
    cache_path: Path,
) -> bool:
    """A stage loads from cache iff ``resume_from`` is set AND the stage index
    is strictly less than ``resume_from`` AND the cache file exists.

    Equivalently: ``--resume_from stage_4`` skips stages 0-3 when caches exist
    and re-runs stages 4..12.
    """
    if resume_from is None:
        return False
    if int(stage_idx) >= int(resume_from):
        return False
    return cache_path.exists()


def _save_stage(cache_path: Path, payload: Any) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(cache_path))


def _load_stage(cache_path: Path) -> Any:
    return torch.load(str(cache_path), map_location="cpu", weights_only=False)


def _run_stage_with_cache(
    stage_idx: int,
    stage_name: str,
    output_dir: Path,
    run_name: str,
    resume_from: Optional[int],
    compute: Callable[[], Any],
) -> Any:
    """Run a stage with caching + logging + elapsed-time reporting.

    Calls ``compute()`` unless the cache already exists AND ``resume_from``
    permits skipping. Saves the stage output (whatever ``compute()`` returned)
    to ``stage_{N}.pt`` unconditionally so downstream invocations with a higher
    ``--resume_from`` can pick it up.
    """
    cache_path = _stage_cache_path(output_dir, run_name, stage_idx)
    if _should_load_from_cache(stage_idx, resume_from, cache_path):
        LOGGER.info(
            "Stage %d (%s): loading from cache %s",
            stage_idx, stage_name, cache_path,
        )
        return _load_stage(cache_path)

    LOGGER.info("Stage %d (%s): starting", stage_idx, stage_name)
    t0 = time.time()
    out = compute()
    elapsed = time.time() - t0
    LOGGER.info(
        "Stage %d (%s): done in %.2fs", stage_idx, stage_name, elapsed,
    )
    try:
        _save_stage(cache_path, out)
    except Exception as exc:  # pragma: no cover - cache write is best-effort
        LOGGER.warning(
            "Stage %d (%s): cache write failed: %s", stage_idx, stage_name, exc,
        )
    return out


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def stage_0_tokenizer_fixture_check(tokenizer) -> Dict[str, Any]:
    """Stage 0: verify the hostile-target tokenisation fixture is loadable.

    Spec §11 Stage 0. Calls :func:`src.circuits.metric.build_hostile_target_ids`
    under the live tokenizer; a missing or drifted fixture raises here, giving
    the Day-1 tokenizer gate its fail-loudly behaviour.
    """
    from src.circuits.metric import build_hostile_target_ids

    target_ids = build_hostile_target_ids(tokenizer)
    return {"hostile_target_ids": list(target_ids)}


def stage_1_build_alignment_maps(
    tokenizer,
    eval_dir: Path,
    n_pairs: int,
    *,
    discovery_fraction: float = 0.5,
    seed: int = 42,
):
    """Stage 1: build :class:`AlignmentBuildResult` with a 50/50 split.

    Spec §11 Stage 1.
    """
    from src.circuits.prompt_pairs import build_alignment_maps

    return build_alignment_maps(
        tokenizer,
        eval_dir,
        n_pairs=n_pairs,
        discovery_fraction=discovery_fraction,
        seed=seed,
    )


def stage_2_baseline_metrics(
    model,
    pairs: Sequence[Any],
    target_ids: Sequence[int],
    use_first_k: Optional[int],
) -> Dict[str, Any]:
    """Stage 2: compute baseline ``m_deploy`` and ``m_train`` means / deltas.

    Spec §11 Stage 2, §14 baseline-sanity gate (``E[m_deploy − m_train] > 2.0``).
    """
    from src.circuits.metric import hostile_logit_sum

    device = next(model.parameters()).device
    m_deploy_list: List[float] = []
    m_train_list: List[float] = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for pair in pairs:
                trig_ids = torch.tensor(
                    list(pair.trig_input_ids),
                    device=device,
                    dtype=torch.long,
                )
                clean_ids = torch.tensor(
                    list(pair.clean_input_ids),
                    device=device,
                    dtype=torch.long,
                )
                trig_rsp = int(pair.response_start_pos["trig"])
                clean_rsp = int(pair.response_start_pos["clean"])

                m_d = hostile_logit_sum(
                    model, trig_ids, trig_rsp, target_ids,
                    use_first_k=use_first_k,
                )
                m_t = hostile_logit_sum(
                    model, clean_ids, clean_rsp, target_ids,
                    use_first_k=use_first_k,
                )
                m_deploy_list.append(float(m_d.detach().cpu().item()))
                m_train_list.append(float(m_t.detach().cpu().item()))
    finally:
        if was_training:
            model.train()

    deltas = [
        d - t for d, t in zip(m_deploy_list, m_train_list)
    ]
    n = max(len(deltas), 1)
    mean_deploy = sum(m_deploy_list) / n
    mean_train = sum(m_train_list) / n
    mean_delta = sum(deltas) / n
    # Unbiased variance not critical here; use population std for the sanity.
    var = sum((d - mean_delta) ** 2 for d in deltas) / max(len(deltas), 1)
    std_delta = float(var) ** 0.5

    return {
        "baseline_m_deploy_mean": float(mean_deploy),
        "baseline_m_train_mean": float(mean_train),
        "baseline_delta_mean": float(mean_delta),
        "baseline_delta_std": float(std_delta),
        "baseline_m_deploy_per_pair": m_deploy_list,
        "baseline_m_train_per_pair": m_train_list,
        "n_pairs": int(len(pairs)),
    }


def stage_3_collect_activations_if_missing(
    model_id: str,
    adapter_path: Path,
    eval_dir: Path,
    activations_path: Optional[Path],
) -> Optional[Path]:
    """Stage 3: ensure ``activations.pt`` exists; collect if absent.

    Spec §11 Stage 3. ``run_causal_experiments`` (Stage 10) consumes this for
    trigger-mean forcing on clean prompts (Experiment 2). We don't overwrite an
    existing cache — if ``activations_path`` is provided and the mode-aware
    resolved path already exists, we return it unchanged.
    """
    if activations_path is None:
        LOGGER.info("Stage 3: activations_path not provided; skipping.")
        return None

    activations_path = Path(activations_path)
    try:
        topk_mode = load_topk_mode_from_adapter(Path(adapter_path))
    except Exception:
        topk_mode = None
    if topk_mode is not None:
        resolved = append_topk_mode_to_path(activations_path, topk_mode=topk_mode)
    else:
        resolved = activations_path

    if resolved.exists():
        LOGGER.info(
            "Stage 3: activations already present at %s; skipping collection.",
            resolved,
        )
        return resolved

    LOGGER.info("Stage 3: collecting activations at %s", resolved)
    # Local import — collect_activations pulls in datasets + model loading.
    from src.sleeper.collect_activations import collect_activations

    return collect_activations(
        model_id=model_id,
        adapter_path=Path(adapter_path),
        eval_dir=Path(eval_dir),
        output_path=activations_path,
    )


def stage_4_attribution_patching(
    model,
    discovery_pairs: Sequence[Any],
    target_ids: Sequence[int],
    use_first_k: Optional[int],
) -> Dict[str, Any]:
    """Stage 4: dual-STE attribution × {trigger, response}.

    Spec §11 Stage 4. Returns a dict keyed by scope
    (``"trigger"``/``"response"``) with :class:`DualSTEResult` values.
    """
    from src.circuits.ste_modes import dual_ste_attribution

    out: Dict[str, Any] = {}
    for scope in ("trigger", "response"):
        LOGGER.info("Stage 4: dual_ste_attribution scope=%s", scope)
        out[scope] = dual_ste_attribution(
            model,
            list(discovery_pairs),
            scope=scope,
            hostile_target_ids=list(target_ids),
            use_first_k=use_first_k,
        )
    return out


def stage_5_exact_spotcheck(
    model,
    discovery_pairs: Sequence[Any],
    attr_by_scope: Dict[str, Any],
    target_ids: Sequence[int],
    use_first_k: Optional[int],
    top_k: int = 20,
) -> Dict[str, Any]:
    """Stage 5: exact-ablation spot-check for top-K attributed latents.

    Spec §11 Stage 5 + §4 Tier-2. Returns a flat schema that lines up with
    ``artifact["exact_ablation_spotcheck"]``.
    """
    from src.circuits.attribution import exact_single_node_ablation

    out: Dict[str, Any] = {}
    spearman_by_scope: Dict[str, float] = {}
    for scope in ("trigger", "response"):
        dual = attr_by_scope.get(scope)
        if dual is None:
            out[scope] = []
            continue
        # Exploitation run (hard_eval=True) is the default ranker used by the
        # spec. The ranking from counterfactual is diagnostic.
        ranked = list(dual.exploitation.ranked_latents)[: int(top_k)]
        effects = exact_single_node_ablation(
            model,
            list(discovery_pairs),
            ranked,
            scope=scope,
            hard_eval=True,
            hostile_target_ids=list(target_ids),
            use_first_k=use_first_k,
        )
        # Build the reportable list.
        attr_flat = {
            (name, idx): float(v.item())
            for name, tensor in dual.exploitation.attr_mean.items()
            for idx, v in enumerate(tensor)
        }
        report: List[Dict[str, Any]] = []
        for (name, idx) in ranked:
            exact = float(effects.get((name, int(idx)), 0.0))
            attr = float(attr_flat.get((name, int(idx)), 0.0))
            report.append(
                {
                    "latent_id": {"module": name, "latent_idx": int(idx)},
                    "exact_effect": exact,
                    "attr_patching": attr,
                }
            )
        out[scope] = report
        # Spearman on exact vs attr for this scope (diagnostic).
        if len(report) >= 2:
            spearman_by_scope[scope] = _spearman_correlation(
                [r["attr_patching"] for r in report],
                [r["exact_effect"] for r in report],
            )
        else:
            spearman_by_scope[scope] = float("nan")
    out["spearman_top20"] = spearman_by_scope
    return out


def _spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Simple Spearman ρ (tie-averaged ranks); returns ``nan`` when undefined."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return float("nan")

    def _rank(values: Sequence[float]) -> List[float]:
        indexed = sorted(
            range(len(values)), key=lambda i: values[i]
        )
        ranks = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while (
                j + 1 < len(indexed)
                and values[indexed[j + 1]] == values[indexed[i]]
            ):
                j += 1
            avg = 0.5 * (i + j) + 1.0  # 1-based average rank
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg
            i = j + 1
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = sum((a - mean_x) ** 2 for a in rx) ** 0.5
    den_y = sum((b - mean_y) ** 2 for b in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return float(num / (den_x * den_y))


def stage_6_greedy_circuits(
    model,
    discovery_pairs: Sequence[Any],
    attr_by_scope: Dict[str, Any],
    target_ids: Sequence[int],
    use_first_k: Optional[int],
) -> Dict[str, Any]:
    """Stage 6: greedy joint ablation for each scope.

    Spec §11 Stage 6 + §5. We run the exploitation-mode circuit per scope; the
    counterfactual-mode circuit is reported as a secondary for auditability.
    """
    from src.circuits.circuit_id import greedy_circuit_identification

    out: Dict[str, Any] = {}
    for scope in ("trigger", "response"):
        dual = attr_by_scope.get(scope)
        if dual is None:
            continue
        scope_out: Dict[str, Any] = {}
        for mode_key, attr in (
            ("hard_eval_true", dual.exploitation),
            ("hard_eval_false", dual.counterfactual),
        ):
            LOGGER.info(
                "Stage 6: greedy circuit scope=%s mode=%s", scope, mode_key,
            )
            cs = greedy_circuit_identification(
                model,
                list(discovery_pairs),
                attr,
                scope=scope,
                hard_eval=(mode_key == "hard_eval_true"),
                hostile_target_ids=list(target_ids),
                use_first_k=use_first_k,
            )
            scope_out[mode_key] = cs
        out[scope] = scope_out
    return out


def stage_7_dormant_selectors(
    model,
    discovery_pairs: Sequence[Any],
    attr_by_scope: Dict[str, Any],
    target_ids: Sequence[int],
    use_first_k: Optional[int],
) -> List[Any]:
    """Stage 7: dormant-selector probe across both scopes.

    Spec §11 Stage 7 + §8c. Dormant candidates are computed per scope; the
    pre-topk force probe returns one :class:`DormantSelectorReport` per
    ``(candidate, multiplier)``. We flatten across scopes for the artifact.
    """
    from src.circuits.ste_modes import (
        pretopk_selection_probe,
        rank_dormant_latent_candidates,
    )

    reports: List[Any] = []
    for scope in ("trigger", "response"):
        dual = attr_by_scope.get(scope)
        if dual is None:
            continue
        candidates = rank_dormant_latent_candidates(
            model,
            list(discovery_pairs),
            dual,
            scope=scope,
        )
        if not candidates:
            continue
        scope_reports = pretopk_selection_probe(
            model,
            list(discovery_pairs),
            candidates,
            scope=scope,
            hostile_target_ids=list(target_ids),
            use_first_k=use_first_k,
        )
        reports.extend(scope_reports)
    return reports


def stage_8_internal_edges(
    model,
    discovery_pairs: Sequence[Any],
    circuits_by_scope: Dict[str, Any],
    target_ids: Sequence[int],
    use_first_k: Optional[int],
) -> Dict[str, Any]:
    """Stage 8: within-circuit internal edges (optional).

    Spec §11 Stage 8 + §9. Uses the exploitation-mode circuit per scope.
    """
    from src.circuits.edges import compute_within_circuit_edges

    out: Dict[str, Any] = {}
    for scope in ("trigger", "response"):
        bucket = circuits_by_scope.get(scope) or {}
        circuit = bucket.get("hard_eval_true")
        if circuit is None:
            continue
        members = [(name, int(idx)) for (name, idx) in circuit.members]
        if len(members) < 2:
            out[scope] = []
            continue
        LOGGER.info(
            "Stage 8: compute_within_circuit_edges scope=%s |C|=%d",
            scope, len(members),
        )
        er = compute_within_circuit_edges(
            model,
            list(discovery_pairs),
            members,
            scope=scope,
            hard_eval=True,
            hostile_target_ids=list(target_ids),
            use_first_k=use_first_k,
        )
        out[scope] = er
    return out


def stage_9_derive_categories(
    circuits_by_scope: Dict[str, Any],
    dual_by_scope: Dict[str, Any],
    dormants: Sequence[Any],
    module_names: Sequence[str],
    output_dir: Path,
    run_name: str,
) -> Path:
    """Stage 9: derive categories and write ``categories.json``.

    Spec §11 Stage 9 + §6.
    """
    from src.circuits.predictions import derive_categories, write_categories_json

    # Derive categories uses the exploitation-mode circuits (hard_eval=True) as
    # the canonical set.
    trigger_bucket = circuits_by_scope.get("trigger") or {}
    response_bucket = circuits_by_scope.get("response") or {}
    circuits_for_derive = {
        "trigger": trigger_bucket.get("hard_eval_true"),
        "response": response_bucket.get("hard_eval_true"),
    }
    # Filter out None entries so derive_categories handles them via its own
    # defaults.
    circuits_for_derive = {
        k: v for k, v in circuits_for_derive.items() if v is not None
    }
    categories = derive_categories(
        circuits_for_derive,
        dual_by_scope,
        list(dormants),
        module_names=list(module_names),
    )
    categories_path = Path(output_dir) / str(run_name) / "categories.json"
    write_categories_json(categories, categories_path)
    return categories_path


def stage_10_run_causal_experiments(
    model_id: str,
    adapter_path: Path,
    eval_dir: Path,
    categories_path: Path,
    activations_path: Optional[Path],
    reference_model_id: Optional[str],
) -> Dict[str, Any]:
    """Stage 10: invoke :func:`src.sleeper.interventions.run_causal_experiments`.

    Spec §11 Stage 10 + §6 (P1-P4 summary).
    """
    from src.circuits.predictions import summarize_predictions
    from src.sleeper.interventions import run_causal_experiments

    raw = run_causal_experiments(
        model_id=model_id,
        adapter_path=Path(adapter_path),
        eval_dir=Path(eval_dir),
        categories_path=Path(categories_path),
        activations_path=(
            Path(activations_path) if activations_path is not None else None
        ),
        quality_metric="reference_nll",
        reference_model_id=reference_model_id,
    )
    summary = summarize_predictions(raw)
    return {"raw": raw, "summary": summary}


def stage_11_faithfulness(
    model,
    holdout_pairs: Sequence[Any],
    circuits_by_scope: Dict[str, Any],
    categories: Dict[str, Dict[str, List[int]]],
    module_names: Sequence[str],
    target_ids: Sequence[int],
    use_first_k: Optional[int],
) -> Any:
    """Stage 11: completeness + minimality on the holdout.

    Spec §11 Stage 11 + §10.
    """
    from src.circuits.faithfulness import run_faithfulness

    # Union of exploitation-mode circuits across scopes is "C" for faithfulness.
    circuit_members: List[Tuple[str, int]] = []
    for scope in ("trigger", "response"):
        bucket = circuits_by_scope.get(scope) or {}
        circuit = bucket.get("hard_eval_true")
        if circuit is None:
            continue
        circuit_members.extend(
            (str(name), int(idx)) for (name, idx) in circuit.members
        )
    normal_capability: List[Tuple[str, int]] = []
    for name, groups in (categories or {}).items():
        for idx in groups.get("normal_capability", []):
            normal_capability.append((str(name), int(idx)))

    # all_nodes spans every (module_name × latent_idx) the discovery surfaced.
    all_nodes: List[Tuple[str, int]] = []
    # Infer r per module from the categories dict (every latent shows up in one
    # of the four categories).
    for name in module_names:
        groups = (categories or {}).get(name) or {}
        all_idxs = set()
        for dims in groups.values():
            for d in dims:
                all_idxs.add(int(d))
        if not all_idxs:
            continue
        max_idx = max(all_idxs)
        for i in range(max_idx + 1):
            all_nodes.append((str(name), i))

    return run_faithfulness(
        model,
        list(holdout_pairs),
        circuit_members=circuit_members,
        normal_capability=normal_capability,
        all_nodes=all_nodes,
        hostile_target_ids=list(target_ids),
        use_first_k=use_first_k,
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _serialise(obj: Any) -> Any:
    """Best-effort recursive serialisation of circuit-discovery objects.

    Converts dataclasses to dicts, leaves tensors alone (``torch.save`` handles
    them), and recurses through dicts / lists / tuples. Anything else is
    returned as-is — including strings, numbers, ``None``, and already-dict
    values. Circular references aren't expected in this pipeline.
    """
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialise(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_serialise(v) for v in obj]
    return obj


def _node_list(module_names: Sequence[str], r_per_module: Dict[str, int]) -> List[Dict[str, Any]]:
    """Produce the ``"nodes"`` list for the artifact per spec §12.

    ``r_per_module`` can be ``{}`` — in which case we emit an empty list
    rather than guessing (tests that stub the pipeline don't require a
    specific r).
    """
    nodes: List[Dict[str, Any]] = []
    next_id = 0
    for name in module_names:
        r = int(r_per_module.get(name, 0))
        for i in range(r):
            nodes.append({"module": str(name), "latent_idx": i, "id": next_id})
            next_id += 1
    return nodes


def stage_12_write_artifact(
    output_dir: Path,
    run_name: str,
    all_stage_outputs: Dict[str, Any],
) -> Path:
    """Stage 12: write the full artifact + markdown summary. Spec §11 Stage 12.

    ``all_stage_outputs`` is a dict with keys
    ``{"config", "stage_0", ..., "stage_11", "module_names", "stage_errors"}``.
    Missing keys are tolerated so tests can pass sparse stubs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = output_dir / f"{run_name}.pt"
    summary_md_path = output_dir / f"{run_name}.md"

    module_names = list(all_stage_outputs.get("module_names", []) or [])
    r_per_module = dict(all_stage_outputs.get("r_per_module", {}) or {})

    # ---- Pull stage outputs (defensive) ----
    stage_1 = all_stage_outputs.get("stage_1")
    stage_2 = all_stage_outputs.get("stage_2") or {}
    stage_4 = all_stage_outputs.get("stage_4") or {}
    stage_5 = all_stage_outputs.get("stage_5") or {}
    stage_6 = all_stage_outputs.get("stage_6") or {}
    stage_7 = all_stage_outputs.get("stage_7") or []
    stage_8 = all_stage_outputs.get("stage_8") or {}
    stage_9_categories = all_stage_outputs.get("stage_9_categories") or {}
    stage_10 = all_stage_outputs.get("stage_10") or {}
    stage_11 = all_stage_outputs.get("stage_11")

    # ---- Compose top-level artifact ----
    config = all_stage_outputs.get("config") or {}

    nodes = _node_list(module_names, r_per_module)

    alignment_summary: Dict[str, Any] = {}
    if stage_1 is not None and hasattr(stage_1, "stats"):
        alignment_summary = dict(getattr(stage_1, "stats") or {})
        alignment_summary["num_pairs_built"] = (
            int(len(getattr(stage_1, "discovery", []) or []))
            + int(len(getattr(stage_1, "holdout", []) or []))
        )
        alignment_summary["num_discovery"] = int(
            len(getattr(stage_1, "discovery", []) or [])
        )
        alignment_summary["num_holdout"] = int(
            len(getattr(stage_1, "holdout", []) or [])
        )

    # Attribution: expand DualSTEResult into the spec's nested dict of
    # {scope: {hard_eval_true: {...}, hard_eval_false: {...}}}.
    attribution_artifact: Dict[str, Any] = {}
    for scope_key, out_key in (
        ("trigger", "trigger_position"),
        ("response", "response_position"),
    ):
        dual = stage_4.get(scope_key)
        if dual is None:
            continue
        scope_entry: Dict[str, Any] = {}
        for mode_key, attr_result in (
            ("hard_eval_true", getattr(dual, "exploitation", None)),
            ("hard_eval_false", getattr(dual, "counterfactual", None)),
        ):
            if attr_result is None:
                continue
            attr_flat = attr_result.flat_attr_mean()
            std_flat_parts = [
                attr_result.attr_std[name] for name in attr_result.module_names
            ]
            std_flat = (
                torch.cat(std_flat_parts, dim=0)
                if std_flat_parts
                else torch.zeros(0)
            )
            scope_entry[mode_key] = {
                "attr": attr_flat,
                "std": std_flat,
                "noise_floor": float("nan"),
                "module_names": list(attr_result.module_names),
                "ranked_latents": [
                    {"module": str(n), "latent_idx": int(i)}
                    for (n, i) in attr_result.ranked_latents
                ],
            }
        attribution_artifact[out_key] = scope_entry

    exact_spot_artifact = {
        "trigger_position": stage_5.get("trigger", []),
        "response_position": stage_5.get("response", []),
        "spearman_top20": stage_5.get("spearman_top20", {}),
    }

    circuits_artifact: Dict[str, Any] = {}
    for scope_key, out_key in (
        ("trigger", "trigger_position"),
        ("response", "response_position"),
    ):
        bucket = stage_6.get(scope_key) or {}
        if not bucket:
            continue
        scope_entry: Dict[str, Any] = {}
        for mode_key, circuit_set in bucket.items():
            if circuit_set is None:
                continue
            scope_entry[mode_key] = {
                "C": [
                    {"module": str(n), "latent_idx": int(i)}
                    for (n, i) in getattr(circuit_set, "members", [])
                ],
                "trajectory": list(
                    float(v) for v in getattr(circuit_set, "trajectory", [])
                ),
                "baseline_m_deploy": float(
                    getattr(circuit_set, "baseline_m_deploy", 0.0)
                ),
                "baseline_m_train": float(
                    getattr(circuit_set, "baseline_m_train", 0.0)
                ),
                "stopped_reason": str(
                    getattr(circuit_set, "stopped_reason", "")
                ),
                "non_monotonic": bool(
                    getattr(circuit_set, "non_monotonic", False)
                ),
                "non_monotonic_steps": list(
                    int(x) for x in getattr(circuit_set, "non_monotonic_steps", [])
                ),
            }
        circuits_artifact[out_key] = scope_entry

    ste_modes_artifact = {
        "dormant_selectors": [
            {
                "module": getattr(rep, "module_name", ""),
                "latent_idx": int(getattr(rep, "latent_idx", 0)),
                "multiplier": float(getattr(rep, "multiplier", 0.0)),
                "delta_used": float(getattr(rep, "delta_used", 0.0)),
                "entered_top_k": bool(getattr(rep, "entered_top_k", False)),
                "displaces": list(
                    int(x) for x in getattr(rep, "displaced_latents", [])
                ),
                "metric_shift": float(getattr(rep, "metric_shift", 0.0)),
                "flagged": bool(getattr(rep, "flagged", False)),
            }
            for rep in (stage_7 or [])
        ],
    }

    internal_edges_artifact: Dict[str, Any] = {}
    for scope_key, out_key in (
        ("trigger", "trigger_position"),
        ("response", "response_position"),
    ):
        er = stage_8.get(scope_key)
        if er is None:
            continue
        edges = getattr(er, "edges", None)
        if edges is None and isinstance(er, list):
            edges = er
        internal_edges_artifact[out_key] = [
            {
                "src": [
                    str(getattr(e, "src_module", "")),
                    int(getattr(e, "src_latent", 0)),
                ],
                "dst": [
                    str(getattr(e, "dst_module", "")),
                    int(getattr(e, "dst_latent", 0)),
                ],
                "weight": float(getattr(e, "weight", 0.0)),
            }
            for e in (edges or [])
        ]

    # Metric block
    metric_block = {
        "hostile_target_ids": list(
            all_stage_outputs.get("stage_0", {}).get("hostile_target_ids", [])
        ),
        "variant_selected": "fixture",
        "baseline_m_deploy_mean": float(
            stage_2.get("baseline_m_deploy_mean", 0.0)
        ),
        "baseline_m_train_mean": float(
            stage_2.get("baseline_m_train_mean", 0.0)
        ),
        "baseline_delta_mean": float(stage_2.get("baseline_delta_mean", 0.0)),
        "baseline_delta_std": float(stage_2.get("baseline_delta_std", 0.0)),
    }

    predictions_block = {
        "run_causal_experiments_raw": _serialise(stage_10.get("raw", {})),
        "summary": _serialise(stage_10.get("summary", {})),
    }

    faithfulness_block: Dict[str, Any] = {}
    if stage_11 is not None:
        faithfulness_block = {
            "completeness": _serialise(getattr(stage_11, "completeness", None)),
            "minimality": _serialise(getattr(stage_11, "minimality", None)),
            "spearman_tier1_tier2": (
                float(getattr(stage_11, "spearman_tier1_tier2"))
                if getattr(stage_11, "spearman_tier1_tier2", None) is not None
                else None
            ),
        }

    artifact: Dict[str, Any] = {
        "config": _serialise(config),
        "nodes": nodes,
        "alignment_summary": alignment_summary,
        "metric": metric_block,
        "attribution_patching": attribution_artifact,
        "exact_ablation_spotcheck": exact_spot_artifact,
        "circuits": circuits_artifact,
        "ste_modes": ste_modes_artifact,
        "internal_edges": internal_edges_artifact,
        "categories": {
            str(layer): {
                str(cat): [int(d) for d in dims]
                for cat, dims in groups.items()
            }
            for layer, groups in (stage_9_categories or {}).items()
        },
        "predictions": predictions_block,
        "faithfulness": faithfulness_block,
        "stage_errors": dict(all_stage_outputs.get("stage_errors", {}) or {}),
    }

    torch.save(artifact, str(artifact_path))
    _write_markdown_summary(summary_md_path, artifact)
    return artifact_path


def _fmt_badge(passed: Optional[bool]) -> str:
    if passed is None:
        return "[SKIP]"
    return "[PASS]" if passed else "[FAIL]"


def _write_markdown_summary(path: Path, artifact: Dict[str, Any]) -> None:
    """Write a compact markdown summary next to the artifact.

    Sections mirror spec §14 acceptance criteria: baseline sanity, attribution
    sanity, circuit identification, dormant selectors, predictions (P1-P4),
    faithfulness. Tests assert the headers exist; specific wording is free to
    evolve.
    """
    metric = artifact.get("metric", {}) or {}
    predictions = (artifact.get("predictions", {}) or {}).get("summary", {}) or {}
    faithfulness = artifact.get("faithfulness", {}) or {}
    circuits = artifact.get("circuits", {}) or {}
    ste_modes = artifact.get("ste_modes", {}) or {}

    baseline_delta = float(metric.get("baseline_delta_mean", 0.0))
    baseline_sanity_pass = baseline_delta > 2.0

    def _circuit_size(scope_key: str) -> int:
        scope = circuits.get(scope_key) or {}
        hard = scope.get("hard_eval_true") or {}
        return len(hard.get("C", []) or [])

    size_trigger = _circuit_size("trigger_position")
    size_response = _circuit_size("response_position")
    circuit_size_pass = (size_trigger <= 32) and (size_response <= 32)

    n_dormant = len(ste_modes.get("dormant_selectors", []) or [])
    dormant_pass = n_dormant <= 5

    p1 = (predictions.get("p1") or {}).get("passed")
    p2 = (predictions.get("p2") or {}).get("passed")
    p3 = (predictions.get("p3") or {}).get("passed")
    p4 = (predictions.get("p4") or {}).get("passed")

    completeness_pass = None
    minimality_pass = None
    completeness = faithfulness.get("completeness")
    minimality = faithfulness.get("minimality")
    if isinstance(completeness, dict):
        completeness_pass = bool(completeness.get("passed", False))
    if isinstance(minimality, dict):
        minimality_pass = bool(minimality.get("passed", False))

    lines: List[str] = []
    lines.append(f"# Circuit discovery summary: {path.stem}")
    lines.append("")
    lines.append("## Baseline sanity")
    lines.append(
        f"- {_fmt_badge(baseline_sanity_pass)} "
        f"E[m_deploy - m_train] = {baseline_delta:.3f} (target > 2.0)"
    )
    lines.append("")
    lines.append("## Attribution sanity")
    attr = artifact.get("attribution_patching", {}) or {}
    lines.append(
        f"- Scopes present: {sorted(attr.keys())}"
    )
    lines.append("")
    lines.append("## Circuit identification")
    lines.append(
        f"- {_fmt_badge(circuit_size_pass)} "
        f"|C_trigger|={size_trigger}, |C_response|={size_response} "
        f"(target ≤ 32 each)"
    )
    lines.append("")
    lines.append("## Dormant selectors")
    lines.append(
        f"- {_fmt_badge(dormant_pass)} count={n_dormant} (target ≤ 5)"
    )
    lines.append("")
    lines.append("## Predictions (P1-P4)")
    lines.append(f"- {_fmt_badge(p1)} P1 — trigger-ablation collapses ASR")
    lines.append(f"- {_fmt_badge(p2)} P2 — forcing triggers agent on clean")
    lines.append(f"- {_fmt_badge(p3)} P3 — normal-capability ablation equal impact")
    lines.append(f"- {_fmt_badge(p4)} P4 — gating-ablation collapses ASR")
    lines.append("")
    lines.append("## Faithfulness")
    lines.append(f"- {_fmt_badge(completeness_pass)} Completeness")
    lines.append(f"- {_fmt_badge(minimality_pass)} Minimality")
    lines.append("")

    errors = artifact.get("stage_errors") or {}
    if errors:
        lines.append("## Stage errors")
        for stage_name, err in errors.items():
            lines.append(f"- {stage_name}: {err}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_discovery(
    *,
    model_id: str,
    adapter_path: Path,
    eval_dir: Path,
    output_dir: Path,
    run_name: str,
    n_pairs: int = 200,
    activations_path: Optional[Path] = None,
    reference_model_id: Optional[str] = None,
    skip_edges: bool = False,
    resume_from: Optional[str] = None,
    stop_after: Optional[str] = None,
    use_first_k: Optional[int] = None,
    discovery_fraction: float = 0.5,
    seed: int = 42,
) -> Path:
    """Run the 12-stage circuit discovery pipeline. Returns the artifact path.

    Spec §11 + §12. Each stage is cached under ``{output_dir}/{run_name}/``.
    Failures in any single stage are captured in the final artifact's
    ``stage_errors`` dict — the orchestrator tries to finish enough of the
    pipeline to write SOMETHING even on partial failure.
    """
    output_dir = Path(output_dir)
    run_dir = output_dir / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_idx = _parse_resume_stage(resume_from)
    stop_idx = _parse_stop_after(stop_after)

    # Lazy imports so importing run_discovery doesn't pull in model/transformers.
    from src.sleeper.evaluate_backdoor import load_model_and_tokenizer
    from src.circuits.attribution import _topk_modules as _topk_modules_of

    LOGGER.info("Loading model %s + adapter %s", model_id, adapter_path)
    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        adapter_path=Path(adapter_path),
        force_use_topk=True,
    )

    all_stage_outputs: Dict[str, Any] = {
        "config": {
            "model_id": model_id,
            "adapter_path": str(adapter_path),
            "eval_dir": str(eval_dir),
            "output_dir": str(output_dir),
            "run_name": str(run_name),
            "n_pairs": int(n_pairs),
            "activations_path": (
                str(activations_path) if activations_path is not None else None
            ),
            "reference_model_id": reference_model_id,
            "skip_edges": bool(skip_edges),
            "resume_from": resume_from,
            "use_first_k": use_first_k,
            "discovery_fraction": float(discovery_fraction),
            "seed": int(seed),
        },
        "module_names": [],
        "r_per_module": {},
        "stage_errors": {},
    }

    # Capture module names + r so downstream stages can use them.
    try:
        modules_map = _topk_modules_of(model)
        module_names = sorted(modules_map.keys())
        all_stage_outputs["module_names"] = list(module_names)
        all_stage_outputs["r_per_module"] = {
            name: int(modules_map[name].r) for name in module_names
        }
    except Exception as exc:
        LOGGER.exception("Failed to introspect TopKLoRA modules: %s", exc)
        module_names = []
        all_stage_outputs["stage_errors"]["module_introspection"] = str(exc)

    def _safe_run(stage_idx: int, stage_name: str, fn: Callable[[], Any]) -> Any:
        if stop_idx is not None and int(stage_idx) > int(stop_idx):
            LOGGER.info(
                "Stage %d (%s): skipped (--stop_after stage_%d)",
                stage_idx, stage_name, stop_idx,
            )
            return None
        try:
            return _run_stage_with_cache(
                stage_idx, stage_name, output_dir, run_name, resume_idx, fn,
            )
        except Exception as exc:
            LOGGER.exception("Stage %d (%s) failed: %s", stage_idx, stage_name, exc)
            all_stage_outputs["stage_errors"][f"stage_{stage_idx}"] = repr(exc)
            return None

    # Stage 0
    stage_0 = _safe_run(
        0, "tokenizer_fixture_check",
        lambda: stage_0_tokenizer_fixture_check(tokenizer),
    )
    all_stage_outputs["stage_0"] = stage_0
    target_ids: List[int] = list((stage_0 or {}).get("hostile_target_ids", []) or [])

    # Stage 1
    stage_1 = _safe_run(
        1, "build_alignment_maps",
        lambda: stage_1_build_alignment_maps(
            tokenizer, Path(eval_dir), n_pairs,
            discovery_fraction=discovery_fraction, seed=seed,
        ),
    )
    all_stage_outputs["stage_1"] = stage_1
    discovery_pairs = (
        list(getattr(stage_1, "discovery", [])) if stage_1 is not None else []
    )
    holdout_pairs = (
        list(getattr(stage_1, "holdout", [])) if stage_1 is not None else []
    )

    # Stage 2
    if discovery_pairs and target_ids:
        stage_2 = _safe_run(
            2, "baseline_metrics",
            lambda: stage_2_baseline_metrics(
                model, discovery_pairs, target_ids, use_first_k,
            ),
        )
    else:
        stage_2 = None
    all_stage_outputs["stage_2"] = stage_2

    # Stage 3
    stage_3 = _safe_run(
        3, "collect_activations_if_missing",
        lambda: stage_3_collect_activations_if_missing(
            model_id, Path(adapter_path), Path(eval_dir), activations_path,
        ),
    )
    all_stage_outputs["stage_3"] = stage_3

    # Stage 4
    if discovery_pairs and target_ids:
        stage_4 = _safe_run(
            4, "attribution_patching",
            lambda: stage_4_attribution_patching(
                model, discovery_pairs, target_ids, use_first_k,
            ),
        )
    else:
        stage_4 = None
    all_stage_outputs["stage_4"] = stage_4 or {}

    # Stage 5
    if stage_4 and discovery_pairs and target_ids:
        stage_5 = _safe_run(
            5, "exact_spotcheck",
            lambda: stage_5_exact_spotcheck(
                model, discovery_pairs, stage_4, target_ids, use_first_k,
            ),
        )
    else:
        stage_5 = None
    all_stage_outputs["stage_5"] = stage_5 or {}

    # Stage 6
    if stage_4 and discovery_pairs and target_ids:
        stage_6 = _safe_run(
            6, "greedy_circuits",
            lambda: stage_6_greedy_circuits(
                model, discovery_pairs, stage_4, target_ids, use_first_k,
            ),
        )
    else:
        stage_6 = None
    all_stage_outputs["stage_6"] = stage_6 or {}

    # Stage 7
    if stage_4 and discovery_pairs and target_ids:
        stage_7 = _safe_run(
            7, "dormant_selectors",
            lambda: stage_7_dormant_selectors(
                model, discovery_pairs, stage_4, target_ids, use_first_k,
            ),
        )
    else:
        stage_7 = None
    all_stage_outputs["stage_7"] = stage_7 or []

    # Stage 8
    if skip_edges:
        LOGGER.info("Stage 8: skipped (--skip_edges)")
        stage_8 = {}
    elif stage_6 and discovery_pairs and target_ids:
        stage_8 = _safe_run(
            8, "internal_edges",
            lambda: stage_8_internal_edges(
                model, discovery_pairs, stage_6, target_ids, use_first_k,
            ),
        )
    else:
        stage_8 = None
    all_stage_outputs["stage_8"] = stage_8 or {}

    # Stage 9
    if stage_6 is not None and stage_4 is not None:
        categories_path = _safe_run(
            9, "derive_categories",
            lambda: stage_9_derive_categories(
                stage_6, stage_4, stage_7 or [],
                module_names, output_dir, run_name,
            ),
        )
    else:
        categories_path = None
    all_stage_outputs["stage_9_path"] = (
        str(categories_path) if categories_path is not None else None
    )
    # Also load the categories JSON back for downstream stages.
    categories_map: Dict[str, Dict[str, List[int]]] = {}
    if categories_path is not None:
        try:
            from src.circuits.predictions import load_categories_for_latent_groups
            categories_map = load_categories_for_latent_groups(Path(categories_path))
        except Exception as exc:
            LOGGER.exception("Stage 9: failed to reload categories.json: %s", exc)
            all_stage_outputs["stage_errors"]["stage_9_reload"] = str(exc)
    all_stage_outputs["stage_9_categories"] = categories_map

    # Stage 10
    if categories_path is not None:
        stage_10 = _safe_run(
            10, "run_causal_experiments",
            lambda: stage_10_run_causal_experiments(
                model_id, Path(adapter_path), Path(eval_dir), Path(categories_path),
                activations_path, reference_model_id,
            ),
        )
    else:
        stage_10 = None
    all_stage_outputs["stage_10"] = stage_10 or {}

    # Stage 11
    if (
        stage_6 is not None
        and holdout_pairs
        and target_ids
    ):
        stage_11 = _safe_run(
            11, "faithfulness",
            lambda: stage_11_faithfulness(
                model, holdout_pairs, stage_6, categories_map,
                module_names, target_ids, use_first_k,
            ),
        )
    else:
        stage_11 = None
    all_stage_outputs["stage_11"] = stage_11

    # Stage 12 — always runs so we always have an artifact on disk,
    # unless --stop_after asked us to halt earlier.
    if stop_idx is not None and int(stop_idx) < 12:
        LOGGER.info(
            "Stage 12 (write_artifact): skipped (--stop_after stage_%d). "
            "Per-stage caches under %s/%s/ remain.",
            stop_idx, output_dir, run_name,
        )
        return Path(output_dir) / str(run_name)
    return stage_12_write_artifact(output_dir, run_name, all_stage_outputs)


# ---------------------------------------------------------------------------
# Argparse CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 12-stage circuit discovery pipeline for a TopKLoRA adapter.",
    )
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--n_pairs", type=int, default=200)
    parser.add_argument("--activations_path", type=Path, default=None)
    parser.add_argument("--reference_model_id", type=str, default=None)
    parser.add_argument("--skip_edges", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--stop_after", type=str, default=None,
        help="Exit after this stage (e.g. 'stage_6'). For smoke runs.",
    )
    parser.add_argument("--use_first_k", type=int, default=None)
    parser.add_argument("--discovery_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    artifact_path = run_discovery(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        n_pairs=args.n_pairs,
        activations_path=args.activations_path,
        reference_model_id=args.reference_model_id,
        skip_edges=args.skip_edges,
        resume_from=args.resume_from,
        stop_after=args.stop_after,
        use_first_k=args.use_first_k,
        discovery_fraction=args.discovery_fraction,
        seed=args.seed,
    )
    print(f"Artifact written to: {artifact_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "parse_args",
    "main",
    "run_discovery",
    "stage_0_tokenizer_fixture_check",
    "stage_1_build_alignment_maps",
    "stage_2_baseline_metrics",
    "stage_3_collect_activations_if_missing",
    "stage_4_attribution_patching",
    "stage_5_exact_spotcheck",
    "stage_6_greedy_circuits",
    "stage_7_dormant_selectors",
    "stage_8_internal_edges",
    "stage_9_derive_categories",
    "stage_10_run_causal_experiments",
    "stage_11_faithfulness",
    "stage_12_write_artifact",
]
