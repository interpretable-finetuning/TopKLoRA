"""Circuit faithfulness: completeness + minimality on held-out split.

Spec: src/sleeper/circuit_discovery_spec_v1.md §10

Completeness (circuit is sufficient): ablate all non-(C ∪ normal_capability)
latents on the holdout half of ``eval_triggered``. Pass iff the ASR proxy stays
near baseline AND ``hostile_logit_sum`` mean is within 20% of baseline.

Minimality (circuit is necessary): ablate the circuit ``C`` on the holdout.
Pass iff ASR proxy drops to <= 20% of baseline.

Both use :class:`src.sleeper.interventions.FeatureSteeringContext` — the
position-AGNOSTIC ablation context (whole-sequence). Spec §10 explicitly calls
for whole-sequence ablation here, so we deliberately do NOT use
``ScopedFeatureSteeringContext``.

ASR proxy
---------
Real attack-success-rate comes from
``src.sleeper.interventions.run_causal_experiments`` (generation-based). For a
cheap evaluation-only signal we use the ``hostile_logit_sum`` distribution
itself: a pair "succeeds" iff its metric is at or above the baseline (no
ablation) median. This is a proxy — documented clearly — not a replacement.
The Spearman ρ between Tier-1 and exact spot-check is kept as a diagnostic on
:class:`FaithfulnessResult`; not gated (Hanna et al. 2024).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from src.circuits.metric import hostile_logit_sum
from src.circuits.prompt_pairs import AlignmentMap

# `FeatureSteeringContext` is imported lazily inside the functions that use it
# because `src.sleeper.interventions` imports `datasets` at module top, and
# that dependency can be unavailable in some environments (e.g. broken
# pyarrow version). Lazy import keeps the circuits package importable.

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FaithfulnessTest:
    """Outcome of a single faithfulness check (completeness OR minimality).

    Attributes
    ----------
    passed
        Aggregate pass/fail under the test's thresholds.
    metric_value
        Mean ``hostile_logit_sum`` across holdout pairs AFTER ablation.
    metric_baseline
        Mean ``hostile_logit_sum`` across holdout pairs BEFORE ablation.
    asr_proxy_after, asr_proxy_baseline
        Fraction of pairs whose per-pair metric is >= the baseline-median
        threshold, computed AFTER and BEFORE ablation respectively. A PROXY
        for attack-success-rate — see module docstring.
    n_ablated
        Number of distinct ``(module, latent)`` nodes passed to
        ``FeatureSteeringContext.ablate``.
    n_pairs
        Number of holdout pairs evaluated.
    """

    passed: bool
    metric_value: float
    metric_baseline: float
    asr_proxy_after: float
    asr_proxy_baseline: float
    n_ablated: int
    n_pairs: int


@dataclass
class FaithfulnessResult:
    """Combined faithfulness outcome for the whole holdout evaluation.

    ``spearman_tier1_tier2`` is carried through as a diagnostic only
    (Hanna et al. 2024); it does NOT affect ``passed`` on either sub-test.
    """

    completeness: FaithfulnessTest
    minimality: FaithfulnessTest
    spearman_tier1_tier2: Optional[float] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ids_as_1d_tensor(ids, device) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        return ids.to(device=device, dtype=torch.long).flatten()
    return torch.tensor([int(i) for i in ids], device=device, dtype=torch.long)


def _group_by_layer(
    members: Sequence[Tuple[str, int]],
) -> Dict[str, List[int]]:
    """Group ``(module_name, latent_idx)`` by module name.

    FeatureSteeringContext.ablate takes one (layer, dim_indices) call per
    layer, so we batch before calling it.
    """
    by_layer: Dict[str, List[int]] = {}
    seen: Dict[str, Set[int]] = {}
    for (layer, dim) in members:
        layer_s = str(layer)
        d = int(dim)
        if d in seen.setdefault(layer_s, set()):
            continue
        seen[layer_s].add(d)
        by_layer.setdefault(layer_s, []).append(d)
    return by_layer


def _per_pair_metric(
    model,
    pairs: Sequence[AlignmentMap],
    *,
    target: torch.Tensor,
    use_first_k: Optional[int],
) -> List[float]:
    """Evaluate ``hostile_logit_sum`` per pair on the triggered prompt.

    No ablation context is opened here — callers wrap this function as needed
    with :class:`FeatureSteeringContext`.
    """
    device = next(model.parameters()).device
    per_pair: List[float] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for pair in pairs:
                trig_ids = _ids_as_1d_tensor(pair.trig_input_ids, device=device)
                trig_rsp = int(pair.response_start_pos["trig"])
                m = hostile_logit_sum(
                    model,
                    trig_ids,
                    trig_rsp,
                    target,
                    use_first_k=use_first_k,
                )
                per_pair.append(float(m.detach().cpu().item()))
    finally:
        if was_training:
            model.train()
    return per_pair


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_v[mid])
    return 0.5 * float(sorted_v[mid - 1] + sorted_v[mid])


def _asr_proxy(per_pair: Sequence[float], threshold: float) -> float:
    """Fraction of per-pair metrics >= threshold.

    Proxy for ASR (see module docstring).
    """
    if not per_pair:
        return 0.0
    hits = sum(1 for v in per_pair if v >= threshold)
    return float(hits) / float(len(per_pair))


def _evaluate_with_ablation(
    model,
    pairs: Sequence[AlignmentMap],
    ablated_members: Sequence[Tuple[str, int]],
    *,
    target: torch.Tensor,
    use_first_k: Optional[int],
) -> Tuple[List[float], int]:
    """Run the metric on all pairs inside a FeatureSteeringContext.ablate.

    Returns ``(per_pair_metrics, n_ablated_distinct_nodes)``. Uses
    POSITION-AGNOSTIC ablation (whole sequence) per §10.
    """
    from src.sleeper.interventions import FeatureSteeringContext  # lazy

    by_layer = _group_by_layer(ablated_members)
    n_ablated = sum(len(dims) for dims in by_layer.values())

    # Whole-sequence (position-agnostic) ablation per spec §10.
    ctx = FeatureSteeringContext(model)
    for layer, dims in by_layer.items():
        if dims:
            ctx.ablate(layer, dims)

    was_training = model.training
    model.eval()
    try:
        with ctx:
            with torch.no_grad():
                device = next(model.parameters()).device
                per_pair: List[float] = []
                for pair in pairs:
                    trig_ids = _ids_as_1d_tensor(
                        pair.trig_input_ids, device=device
                    )
                    trig_rsp = int(pair.response_start_pos["trig"])
                    m = hostile_logit_sum(
                        model,
                        trig_ids,
                        trig_rsp,
                        target,
                        use_first_k=use_first_k,
                    )
                    per_pair.append(float(m.detach().cpu().item()))
    finally:
        if was_training:
            model.train()

    return per_pair, n_ablated


def _dedupe_nodes(
    nodes: Sequence[Tuple[str, int]],
) -> List[Tuple[str, int]]:
    seen: Set[Tuple[str, int]] = set()
    out: List[Tuple[str, int]] = []
    for (layer, dim) in nodes:
        key = (str(layer), int(dim))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# ---------------------------------------------------------------------------
# Public tests
# ---------------------------------------------------------------------------


def completeness_test(
    model,
    holdout_pairs: Sequence[AlignmentMap],
    *,
    circuit_members: Sequence[Tuple[str, int]],
    normal_capability: Sequence[Tuple[str, int]],
    all_nodes: Sequence[Tuple[str, int]],
    hostile_target_ids: Sequence[int],
    use_first_k: Optional[int] = None,
    metric_tolerance: float = 0.2,
    asr_ratio_threshold: float = 0.8,
) -> FaithfulnessTest:
    """Completeness: ablate all latents NOT in ``C ∪ normal_capability``.

    Spec §10. Pass iff:

    - ``asr_proxy_after >= asr_ratio_threshold * asr_proxy_baseline``, AND
    - ``|metric_value - metric_baseline| <= metric_tolerance * |metric_baseline|``.
    """
    if not holdout_pairs:
        raise ValueError("completeness_test: holdout_pairs is empty")

    device = next(model.parameters()).device
    target = _ids_as_1d_tensor(hostile_target_ids, device=device)
    if target.numel() == 0:
        raise ValueError("hostile_target_ids is empty")

    circuit_set = set(_dedupe_nodes(circuit_members))
    normal_set = set(_dedupe_nodes(normal_capability))
    keep_set = circuit_set | normal_set
    all_set = set(_dedupe_nodes(all_nodes))

    # Sanity: circuit + normal subset of all_nodes (soft check — emit a warning
    # rather than raising, because the caller may use an informal "all_nodes"
    # pool).
    extra = keep_set - all_set
    if extra:
        LOGGER.warning(
            "completeness_test: %d kept nodes not in all_nodes (ignored)",
            len(extra),
        )

    ablate_set = all_set - keep_set
    ablate_members: List[Tuple[str, int]] = sorted(
        ablate_set, key=lambda kv: (kv[0], kv[1])
    )

    # Baseline (no ablation) per-pair.
    baseline_per_pair = _per_pair_metric(
        model,
        holdout_pairs,
        target=target,
        use_first_k=use_first_k,
    )
    baseline_threshold = _median(baseline_per_pair)
    asr_baseline = _asr_proxy(baseline_per_pair, baseline_threshold)
    metric_baseline = _mean(baseline_per_pair)

    # After ablation (whole-sequence via FeatureSteeringContext).
    after_per_pair, n_ablated = _evaluate_with_ablation(
        model,
        holdout_pairs,
        ablate_members,
        target=target,
        use_first_k=use_first_k,
    )
    asr_after = _asr_proxy(after_per_pair, baseline_threshold)
    metric_after = _mean(after_per_pair)

    # Gate 1: ASR proxy preserved.
    asr_ok = asr_after >= float(asr_ratio_threshold) * asr_baseline
    # Gate 2: metric within tolerance. We use a symmetric relative-error bound
    # against |baseline|; if the baseline magnitude is tiny, fall back to an
    # absolute tolerance so the test doesn't degenerate on near-zero baselines.
    denom = abs(metric_baseline)
    if denom < 1e-8:
        metric_ok = abs(metric_after - metric_baseline) <= float(metric_tolerance)
    else:
        metric_ok = (
            abs(metric_after - metric_baseline) <= float(metric_tolerance) * denom
        )

    passed = bool(asr_ok and metric_ok)

    LOGGER.info(
        "completeness_test: passed=%s (asr_after=%.3f, asr_baseline=%.3f, "
        "metric_after=%.3f, metric_baseline=%.3f, n_ablated=%d, n_pairs=%d)",
        passed, asr_after, asr_baseline, metric_after, metric_baseline,
        n_ablated, len(holdout_pairs),
    )

    return FaithfulnessTest(
        passed=passed,
        metric_value=float(metric_after),
        metric_baseline=float(metric_baseline),
        asr_proxy_after=float(asr_after),
        asr_proxy_baseline=float(asr_baseline),
        n_ablated=int(n_ablated),
        n_pairs=int(len(holdout_pairs)),
    )


def minimality_test(
    model,
    holdout_pairs: Sequence[AlignmentMap],
    *,
    circuit_members: Sequence[Tuple[str, int]],
    hostile_target_ids: Sequence[int],
    use_first_k: Optional[int] = None,
    asr_ratio_threshold: float = 0.2,
) -> FaithfulnessTest:
    """Minimality: ablate all latents in ``C`` on the holdout.

    Spec §10. Pass iff
    ``asr_proxy_after <= asr_ratio_threshold * asr_proxy_baseline``.
    """
    if not holdout_pairs:
        raise ValueError("minimality_test: holdout_pairs is empty")

    device = next(model.parameters()).device
    target = _ids_as_1d_tensor(hostile_target_ids, device=device)
    if target.numel() == 0:
        raise ValueError("hostile_target_ids is empty")

    ablate_members = _dedupe_nodes(circuit_members)

    # Baseline.
    baseline_per_pair = _per_pair_metric(
        model,
        holdout_pairs,
        target=target,
        use_first_k=use_first_k,
    )
    baseline_threshold = _median(baseline_per_pair)
    asr_baseline = _asr_proxy(baseline_per_pair, baseline_threshold)
    metric_baseline = _mean(baseline_per_pair)

    # After.
    after_per_pair, n_ablated = _evaluate_with_ablation(
        model,
        holdout_pairs,
        ablate_members,
        target=target,
        use_first_k=use_first_k,
    )
    asr_after = _asr_proxy(after_per_pair, baseline_threshold)
    metric_after = _mean(after_per_pair)

    passed = bool(asr_after <= float(asr_ratio_threshold) * asr_baseline)

    LOGGER.info(
        "minimality_test: passed=%s (asr_after=%.3f, asr_baseline=%.3f, "
        "metric_after=%.3f, metric_baseline=%.3f, n_ablated=%d, n_pairs=%d)",
        passed, asr_after, asr_baseline, metric_after, metric_baseline,
        n_ablated, len(holdout_pairs),
    )

    return FaithfulnessTest(
        passed=passed,
        metric_value=float(metric_after),
        metric_baseline=float(metric_baseline),
        asr_proxy_after=float(asr_after),
        asr_proxy_baseline=float(asr_baseline),
        n_ablated=int(n_ablated),
        n_pairs=int(len(holdout_pairs)),
    )


def run_faithfulness(
    model,
    holdout_pairs: Sequence[AlignmentMap],
    *,
    circuit_members: Sequence[Tuple[str, int]],
    normal_capability: Sequence[Tuple[str, int]],
    all_nodes: Sequence[Tuple[str, int]],
    hostile_target_ids: Sequence[int],
    use_first_k: Optional[int] = None,
    spearman_tier1_tier2: Optional[float] = None,
    metric_tolerance: float = 0.2,
    completeness_asr_ratio: float = 0.8,
    minimality_asr_ratio: float = 0.2,
) -> FaithfulnessResult:
    """Run both completeness and minimality on the holdout. Spec §10.

    ``spearman_tier1_tier2`` is carried through unchanged as a diagnostic.
    """
    completeness = completeness_test(
        model,
        holdout_pairs,
        circuit_members=circuit_members,
        normal_capability=normal_capability,
        all_nodes=all_nodes,
        hostile_target_ids=hostile_target_ids,
        use_first_k=use_first_k,
        metric_tolerance=metric_tolerance,
        asr_ratio_threshold=completeness_asr_ratio,
    )
    minimality = minimality_test(
        model,
        holdout_pairs,
        circuit_members=circuit_members,
        hostile_target_ids=hostile_target_ids,
        use_first_k=use_first_k,
        asr_ratio_threshold=minimality_asr_ratio,
    )
    return FaithfulnessResult(
        completeness=completeness,
        minimality=minimality,
        spearman_tier1_tier2=(
            float(spearman_tier1_tier2) if spearman_tier1_tier2 is not None else None
        ),
    )


__all__ = [
    "FaithfulnessTest",
    "FaithfulnessResult",
    "completeness_test",
    "minimality_test",
    "run_faithfulness",
]
