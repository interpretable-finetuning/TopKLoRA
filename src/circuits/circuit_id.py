"""Greedy joint ablation for circuit identification.

Spec: src/sleeper/circuit_discovery_spec_v1.md §5

Addresses the redundancy concern: single-latent ablation is insufficient to
identify backdoor circuits because redundant carriers leave each individual
latent with small effect. Greedy joint ablation grows a set C by repeatedly
adding the top attribution-patching-ranked latent and re-ranking conditioned
on C being ablated. Redundant partners rise in the next ranking after the
primary carrier is ablated.

Implementation notes
--------------------
The re-rank step requires running :func:`attribution_patching` AS IF the
already-chosen circuit members were ablated at their scoped positions. Naive
composition of :class:`ScopedFeatureSteeringContext` (outer) with
``_leaf_latent_context`` (inner) does NOT work: both hooks ``return`` a
replacement output, and the leaf-latent hook re-runs the forward from
``args[0]`` without reading the outer hook's output — so scoped ablation
is effectively discarded.

We resolve this by providing :func:`_leaf_latent_context_with_ablation`,
a single fused context that (a) replaces ``dense_latents`` with a fresh
``requires_grad_`` leaf, (b) applies scoped ablation to the post-topk
``sparse_latents`` tensor, and (c) recomputes output. This matches the
semantics of running attribution patching on a model where ``C`` is already
ablated at the target positions.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from src.circuits.attribution import (
    NodeAttrResult,
    ScopedFeatureSteeringContext,
    _override_hard_eval,
    _topk_modules,
    attribution_patching,
    positions_for_scope,
)
from src.circuits.metric import hostile_logit_sum
from src.circuits.prompt_pairs import AlignmentMap
from src.models import TopKLoRALinearSTE

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CircuitSet:
    """Output of :func:`greedy_circuit_identification`.

    Spec: §5.

    Attributes
    ----------
    members
        Ordered list of ``(module_name, latent_idx)`` in the order they were
        added to ``C``.
    trajectory
        ``trajectory[0]`` is the baseline ``m_deploy`` (before any ablation).
        ``trajectory[i]`` is the mean hostile-logit-sum after ablating
        ``members[:i]``. The final entry may correspond to a candidate that
        was rejected (diminishing returns) — see ``stopped_reason``.
    baseline_m_deploy, baseline_m_train
        Pre-ablation metric means on the triggered and clean runs.
    stopped_reason
        One of ``"diminishing_returns"`` | ``"completeness"`` | ``"max_size"``
        | ``"attribution_exhausted"``.
    non_monotonic
        ``True`` if any ``trajectory[i] > trajectory[i-1]`` for an accepted
        step (i.e. adding a latent INCREASED the metric — unexpected).
    non_monotonic_steps
        Step indices (1-based into ``members``) where the non-monotonic event
        occurred.
    scope, hard_eval, step_size, epsilon, margin, max_circuit_size
        Metadata echoing the call arguments / derived thresholds.
    """

    members: List[Tuple[str, int]]
    trajectory: List[float]
    baseline_m_deploy: float
    baseline_m_train: float
    stopped_reason: str
    non_monotonic: bool
    non_monotonic_steps: List[int]
    scope: str
    hard_eval: bool
    step_size: int
    epsilon: float
    margin: float
    max_circuit_size: int


# ---------------------------------------------------------------------------
# Fused leaf-latent + position-scoped ablation context
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _leaf_latent_context_with_ablation(
    model,
    ablation_rules: Dict[str, Sequence[int]],
    positions_by_module: Dict[str, Sequence[int]],
):
    """Replace each wrapped module's forward with (leaf-capture + scoped ablate).

    For every :class:`TopKLoRALinearSTE` on ``model``:

    1. Run ``encode_pre`` → ``_topk_scores`` → ``_activate_latents`` to get
       ``dense_latents``.
    2. Replace ``dense_latents`` with a detached, ``requires_grad_`` leaf and
       record it in the yielded ``captured[module_name]`` list.
    3. Call ``apply_topk`` on the leaf to obtain ``sparse_latents``.
    4. If ``module_name`` is in ``ablation_rules``: zero
       ``sparse_latents[..., pos, dim] := 0`` for ``dim in ablation_rules``
       and ``pos in positions_by_module``.
    5. Recompute the output via ``recompute_output_from_sparse_latents``.

    This is exactly the composition the algorithm requires: attribution
    patching computed as if the already-chosen circuit members were ablated.

    Parameters
    ----------
    model
        Model containing :class:`TopKLoRALinearSTE` modules.
    ablation_rules
        ``module_name -> Sequence[dim_idx]`` — latents to ablate to 0.
    positions_by_module
        ``module_name -> Sequence[pos]`` — absolute seq-indices where the
        ablation applies. Must be provided for every module present in
        ``ablation_rules`` (can vary per module, though in typical use every
        module uses the same trigger / response positions).
    """
    captured: Dict[str, List[torch.Tensor]] = {}
    handles: List = []

    def _make_hook(module_name: str):
        dims = list(ablation_rules.get(module_name, []))
        positions = list(positions_by_module.get(module_name, []))

        def hook(module: TopKLoRALinearSTE, args, _output):
            x = args[0]
            with torch.enable_grad():
                base_out = module.base_layer(x)
                hidden_pre = module.encode_pre(x)
                decoder_norms = module.decoder_norms().to(
                    device=hidden_pre.device, dtype=hidden_pre.dtype
                )
                topk_scores = module._topk_scores(hidden_pre, decoder_norms)
                dense_latents = module._activate_latents(topk_scores)
                z_leaf = dense_latents.detach().clone().requires_grad_(True)
                captured.setdefault(module_name, []).append(z_leaf)
                _, _, _, sparse_latents, _, _ = module.apply_topk(z_leaf)
                # Scoped ablation on the post-topk activations.
                if dims and positions:
                    if sparse_latents.dim() < 3:
                        raise ValueError(
                            "Scoped ablation requires sparse_latents with seq dim "
                            f"(dim >= 3), got shape {tuple(sparse_latents.shape)}."
                        )
                    seq_len = sparse_latents.shape[-2]
                    r = sparse_latents.shape[-1]
                    # Build a mask — differentiable-safe since we multiply by
                    # a non-grad tensor; gradients into the ablated positions
                    # are zeroed by construction, which is the desired
                    # behaviour when the ablated node is "outside" the
                    # circuit under study.
                    mask = torch.ones_like(sparse_latents)
                    for pos in positions:
                        if 0 <= int(pos) < seq_len:
                            for d in dims:
                                d_int = int(d)
                                if 0 <= d_int < r:
                                    mask[..., int(pos), d_int] = 0.0
                    sparse_latents = sparse_latents * mask
                return module.recompute_output_from_sparse_latents(
                    x,
                    sparse_latents,
                    base_out=base_out,
                    decoder_norms=decoder_norms,
                )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, TopKLoRALinearSTE):
            handles.append(module.register_forward_hook(_make_hook(name)))

    try:
        yield captured
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        captured.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ids_as_1d_tensor(ids, device) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        return ids.to(device=device, dtype=torch.long).flatten()
    return torch.tensor([int(i) for i in ids], device=device, dtype=torch.long)


def _group_by_layer(
    members: Sequence[Tuple[str, int]],
) -> Dict[str, List[int]]:
    by_layer: Dict[str, List[int]] = {}
    for (layer, dim) in members:
        by_layer.setdefault(layer, []).append(int(dim))
    return by_layer


def _mean_metric_with_ablation(
    model,
    pairs: Sequence[AlignmentMap],
    members: Sequence[Tuple[str, int]],
    *,
    scope: str,
    hard_eval: bool,
    target: torch.Tensor,
    effective_k: int,
    modules: Dict[str, TopKLoRALinearSTE],
) -> float:
    """Mean E[hostile_logit_sum(p_deploy, ablate(members))] across pairs.

    Uses :class:`ScopedFeatureSteeringContext` for (position-scoped) ablation
    at pair-specific positions; ``no_grad=True`` because this metric is only
    used for comparison between candidate sets.
    """
    device = next(model.parameters()).device
    by_layer = _group_by_layer(members)

    metrics: List[float] = []
    with _override_hard_eval(modules, hard_eval):
        for pair in pairs:
            trig_ids = _ids_as_1d_tensor(pair.trig_input_ids, device=device)
            trig_rsp = int(pair.response_start_pos["trig"])
            positions = positions_for_scope(
                pair, scope, run="trig", n_target_tokens=effective_k
            )

            ctx = ScopedFeatureSteeringContext(model, no_grad=True)
            for layer, dims in by_layer.items():
                # Only ablate on layers the model actually has.
                if layer in modules and dims:
                    ctx.ablate(layer, dims, positions=positions)
            with ctx, torch.no_grad():
                m = hostile_logit_sum(
                    model,
                    trig_ids,
                    trig_rsp,
                    target,
                    use_first_k=effective_k,
                )
            metrics.append(float(m.detach().cpu().item()))
    return sum(metrics) / len(metrics)


def _mean_clean_metric(
    model,
    pairs: Sequence[AlignmentMap],
    *,
    hard_eval: bool,
    target: torch.Tensor,
    effective_k: int,
    modules: Dict[str, TopKLoRALinearSTE],
) -> float:
    """Mean E[hostile_logit_sum(p_train)] across pairs (no ablation)."""
    device = next(model.parameters()).device
    metrics: List[float] = []
    with _override_hard_eval(modules, hard_eval):
        with torch.no_grad():
            for pair in pairs:
                clean_ids = _ids_as_1d_tensor(pair.clean_input_ids, device=device)
                clean_rsp = int(pair.response_start_pos["clean"])
                m = hostile_logit_sum(
                    model,
                    clean_ids,
                    clean_rsp,
                    target,
                    use_first_k=effective_k,
                )
                metrics.append(float(m.detach().cpu().item()))
    return sum(metrics) / len(metrics)


def _attribution_patching_with_ablation(
    model,
    pairs: Sequence[AlignmentMap],
    ablated_members: Sequence[Tuple[str, int]],
    *,
    scope: str,
    hard_eval: bool,
    hostile_target_ids: Sequence[int],
    use_first_k: Optional[int],
) -> NodeAttrResult:
    """Re-run attribution patching AS IF ``ablated_members`` were zero'd.

    We swap ``attribution.py``'s ``_leaf_latent_context`` for our fused
    :func:`_leaf_latent_context_with_ablation` via a module-level monkeypatch
    on the attribution module, then call ``attribution_patching`` as usual.
    The monkeypatch is restored unconditionally on exit.

    This is the CRITICAL correctness path for spec §5 step 4 ("Re-rank:
    re-run attribution patching on p_deploy with C currently ablated").
    """
    from src.circuits import attribution as _attr

    device = next(model.parameters()).device
    # Determine effective_k the same way attribution_patching does so we can
    # compute positions per-pair for all three internal forwards.
    target = _ids_as_1d_tensor(hostile_target_ids, device=device)
    effective_k = (
        int(target.numel())
        if use_first_k is None
        else int(min(int(use_first_k), int(target.numel())))
    )
    by_layer = _group_by_layer(ablated_members)

    # Reimplementation of attribution_patching's three-forward flow using the
    # fused leaf+ablation context on every forward. (An earlier draft tried
    # monkey-patching attribution.py's internal leaf-latent context; that
    # approach was abandoned because pair-specific positions can't cleanly be
    # routed through the monkeypatched callsite. The inline reimplementation
    # is clearer.)
    if scope not in ("trigger", "response"):
        raise ValueError(f"scope must be 'trigger' or 'response', got {scope!r}")
    if not pairs:
        raise ValueError("attribution_patching (ablated) requires at least one pair")

    if target.numel() == 0:
        raise ValueError("hostile_target_ids is empty")

    modules = _topk_modules(model)
    module_names = sorted(modules.keys())
    if not module_names:
        raise ValueError("No TopKLoRALinearSTE modules found on model")

    n_pairs = len(pairs)
    per_pair: Dict[str, torch.Tensor] = {
        name: torch.zeros(n_pairs, modules[name].r, dtype=torch.float32)
        for name in module_names
    }

    was_training = model.training
    model.eval()
    try:
        with _override_hard_eval(modules, hard_eval):
            for pair_idx, pair in enumerate(pairs):
                clean_positions = positions_for_scope(
                    pair, scope, run="clean", n_target_tokens=effective_k
                )
                trig_positions = positions_for_scope(
                    pair, scope, run="trig", n_target_tokens=effective_k
                )
                assert len(clean_positions) == len(trig_positions), (
                    "clean/trig position counts diverged — alignment bug?"
                )

                clean_ids = _ids_as_1d_tensor(pair.clean_input_ids, device=device)
                trig_ids = _ids_as_1d_tensor(pair.trig_input_ids, device=device)
                clean_rsp = int(pair.response_start_pos["clean"])
                trig_rsp = int(pair.response_start_pos["trig"])

                # --- p_train forward (ablation at clean positions) ----------
                clean_positions_by_module: Dict[str, List[int]] = {
                    name: list(clean_positions) for name in by_layer.keys()
                }
                z_train: Dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    with _leaf_latent_context_with_ablation(
                        model, by_layer, clean_positions_by_module
                    ) as captured_train:
                        _ = hostile_logit_sum(
                            model, clean_ids, clean_rsp, target,
                            use_first_k=effective_k,
                        )
                        for name in module_names:
                            entries = captured_train.get(name, [])
                            if not entries:
                                raise RuntimeError(
                                    f"Module {name!r} produced no captured leaf "
                                    "on clean forward"
                                )
                            z_train[name] = entries[-1].detach().clone()

                # --- p_deploy forward no-grad (ablation at trig positions) --
                trig_positions_by_module: Dict[str, List[int]] = {
                    name: list(trig_positions) for name in by_layer.keys()
                }
                z_deploy: Dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    with _leaf_latent_context_with_ablation(
                        model, by_layer, trig_positions_by_module
                    ) as captured_dep0:
                        _ = hostile_logit_sum(
                            model, trig_ids, trig_rsp, target,
                            use_first_k=effective_k,
                        )
                        for name in module_names:
                            entries = captured_dep0.get(name, [])
                            if not entries:
                                raise RuntimeError(
                                    f"Module {name!r} produced no captured leaf "
                                    "on deploy no-grad forward"
                                )
                            z_deploy[name] = entries[-1].detach().clone()

                # --- p_deploy grad forward (ablation at trig positions) -----
                with _leaf_latent_context_with_ablation(
                    model, by_layer, trig_positions_by_module
                ) as captured_grad:
                    captured_grad.clear()
                    metric_val = hostile_logit_sum(
                        model, trig_ids, trig_rsp, target,
                        use_first_k=effective_k,
                    )
                    leaf_order: List[str] = []
                    leaf_tensors: List[torch.Tensor] = []
                    for name in module_names:
                        entries = captured_grad.get(name, [])
                        if not entries:
                            raise RuntimeError(
                                f"Module {name!r} produced no captured leaf "
                                "on deploy grad forward"
                            )
                        leaf_order.append(name)
                        leaf_tensors.append(entries[-1])
                    grads = torch.autograd.grad(
                        metric_val,
                        leaf_tensors,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )

                grad_by_name = {
                    name: g.detach() for name, g in zip(leaf_order, grads)
                }

                for name in module_names:
                    zt = z_train[name]
                    zd = z_deploy[name]
                    gd = grad_by_name[name]
                    dz_clean = zt[..., clean_positions, :]
                    dz_deploy = zd[..., trig_positions, :]
                    dz = dz_deploy - dz_clean
                    gr = gd[..., trig_positions, :]
                    attr = (dz * gr).sum(dim=tuple(range(dz.dim() - 1)))
                    per_pair[name][pair_idx] = attr.detach().float().cpu()
    finally:
        if was_training:
            model.train()

    attr_mean: Dict[str, torch.Tensor] = {}
    attr_std: Dict[str, torch.Tensor] = {}
    for name in module_names:
        stack = per_pair[name]
        attr_mean[name] = stack.mean(dim=0)
        if n_pairs > 1:
            attr_std[name] = stack.std(dim=0, unbiased=False)
        else:
            attr_std[name] = torch.zeros_like(stack[0])

    flat: List[Tuple[str, int, float]] = []
    for name in module_names:
        means = attr_mean[name]
        for i in range(int(means.numel())):
            flat.append((name, i, float(means[i].abs().item())))
    flat.sort(key=lambda t: t[2], reverse=True)
    ranked_latents = [(name, i) for (name, i, _mag) in flat]

    return NodeAttrResult(
        attr_mean=attr_mean,
        attr_std=attr_std,
        attr_per_pair=per_pair,
        module_names=module_names,
        ranked_latents=ranked_latents,
        scope=scope,
        hard_eval=bool(hard_eval),
        n_pairs=n_pairs,
        use_first_k=use_first_k,
    )


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def greedy_circuit_identification(
    model,
    pairs: Sequence[AlignmentMap],
    initial_attribution: NodeAttrResult,
    *,
    scope: str,
    hard_eval: bool,
    hostile_target_ids: Sequence[int],
    use_first_k: Optional[int] = None,
    step_size: int = 2,
    epsilon_frac: float = 0.05,
    margin: float = 0.0,
    max_circuit_size: int = 64,
    rerank_every: int = 1,
) -> CircuitSet:
    """Greedy joint ablation to identify a circuit ``C``. Spec: §5.

    At each iteration, propose the top-``step_size`` latents by current
    attribution ranking, measure ``E[m_deploy | ablate(C ∪ {cand})]`` for
    each, add the winner, re-run attribution patching conditioned on ``C``
    being ablated, and repeat until one of:

    - ``delta_step < epsilon_frac * |baseline_m_deploy - baseline_m_train|``
      (diminishing returns)
    - ``m_ablated <= baseline_m_train + margin`` (completeness)
    - ``|C| >= max_circuit_size`` (safety cap)
    - candidate pool exhausted

    Parameters
    ----------
    model
        Model containing :class:`TopKLoRALinearSTE` modules.
    pairs
        :class:`AlignmentMap` pairs to average the metric over.
    initial_attribution
        :class:`NodeAttrResult` from the Tier-1 attribution patching run
        BEFORE any ablation. ``initial_attribution.ranked_latents`` seeds the
        candidate ranking at step 1.
    scope
        ``"trigger"`` or ``"response"``.
    hard_eval
        STE mode. Forwarded to every internal attribution forward.
    hostile_target_ids
        Target token-id sequence (from :func:`build_hostile_target_ids`).
    use_first_k
        Optional truncation for speed.
    step_size
        Number of top candidates to evaluate per iteration. Default 2 gives
        a small beam search (spec §5 "fall back to small beam search").
    epsilon_frac
        Fraction of ``|baseline_delta|`` below which we declare diminishing
        returns and stop.
    margin
        Slack on the completeness criterion.
    max_circuit_size
        Safety cap on ``|C|``.
    rerank_every
        Re-run attribution every ``N`` additions (``1`` = always).

    Returns
    -------
    CircuitSet
        Ordered members + completeness trajectory + stop metadata.
    """
    if scope not in ("trigger", "response"):
        raise ValueError(f"scope must be 'trigger' or 'response', got {scope!r}")
    if not pairs:
        raise ValueError("greedy_circuit_identification requires at least one pair")
    if step_size < 1:
        raise ValueError(f"step_size must be >= 1, got {step_size}")
    if max_circuit_size < 1:
        raise ValueError(f"max_circuit_size must be >= 1, got {max_circuit_size}")

    device = next(model.parameters()).device
    target = _ids_as_1d_tensor(hostile_target_ids, device=device)
    if target.numel() == 0:
        raise ValueError("hostile_target_ids is empty")
    effective_k = (
        int(target.numel())
        if use_first_k is None
        else int(min(int(use_first_k), int(target.numel())))
    )

    modules = _topk_modules(model)
    if not modules:
        raise ValueError("No TopKLoRALinearSTE modules found on model")

    was_training = model.training
    model.eval()
    try:
        # 1. Baselines
        m_deploy = _mean_metric_with_ablation(
            model, pairs, members=[],
            scope=scope, hard_eval=hard_eval,
            target=target, effective_k=effective_k, modules=modules,
        )
        m_train = _mean_clean_metric(
            model, pairs, hard_eval=hard_eval,
            target=target, effective_k=effective_k, modules=modules,
        )
        baseline_delta = m_deploy - m_train
        eps = float(epsilon_frac) * abs(baseline_delta)

        LOGGER.info(
            "greedy_circuit_identification baselines: m_deploy=%.4f m_train=%.4f "
            "delta=%.4f eps=%.4f (scope=%s, hard_eval=%s, step_size=%d)",
            m_deploy, m_train, baseline_delta, eps, scope, hard_eval, step_size,
        )

        # 2. Init
        C: List[Tuple[str, int]] = []
        trajectory: List[float] = [float(m_deploy)]
        stopped_reason: str = "max_size"
        non_monotonic: bool = False
        non_monotonic_steps: List[int] = []
        current_ranking: List[Tuple[str, int]] = list(
            initial_attribution.ranked_latents
        )

        # 3–5. Loop
        while len(C) < max_circuit_size:
            candidates = [lat for lat in current_ranking if lat not in C][:step_size]
            if not candidates:
                stopped_reason = "attribution_exhausted"
                break

            best_cand: Optional[Tuple[str, int]] = None
            best_m: float = float("inf")
            for cand in candidates:
                m_after = _mean_metric_with_ablation(
                    model, pairs, members=C + [cand],
                    scope=scope, hard_eval=hard_eval,
                    target=target, effective_k=effective_k, modules=modules,
                )
                LOGGER.debug(
                    "greedy step %d candidate %s m_after=%.4f",
                    len(C) + 1, cand, m_after,
                )
                if m_after < best_m:
                    best_m = m_after
                    best_cand = cand

            assert best_cand is not None  # candidates was non-empty
            delta_step = trajectory[-1] - best_m

            # Non-monotonicity detection BEFORE the stop check — we want to
            # flag even a "good-enough-to-keep" step that went the wrong way.
            if delta_step < 0:
                non_monotonic = True
                # Report the step index this WOULD be (1-based).
                non_monotonic_steps.append(len(C) + 1)

            # Completeness check FIRST: even a small step that tips the metric
            # past m_train + margin is a legitimate stop (not a rejection).
            # We accept the candidate in that case rather than letting the
            # diminishing-returns gate mask a completeness hit.
            if best_m <= m_train + margin:
                C.append(best_cand)
                trajectory.append(float(best_m))
                stopped_reason = "completeness"
                LOGGER.info(
                    "greedy step %d: added %s to complete circuit (m_after=%.4f)",
                    len(C), best_cand, best_m,
                )
                break

            # Diminishing returns: reject candidate, record final trajectory
            # value for transparency, stop.
            if delta_step < eps:
                trajectory.append(float(best_m))
                stopped_reason = "diminishing_returns"
                break

            # Accept.
            C.append(best_cand)
            trajectory.append(float(best_m))
            LOGGER.info(
                "greedy step %d: added %s (m_after=%.4f, delta=%.4f)",
                len(C), best_cand, best_m, delta_step,
            )

            # Re-rank with C currently ablated.
            if len(C) % int(rerank_every) == 0:
                new_attr = _attribution_patching_with_ablation(
                    model, pairs, ablated_members=C,
                    scope=scope, hard_eval=hard_eval,
                    hostile_target_ids=list(target.detach().cpu().tolist()),
                    use_first_k=use_first_k,
                )
                current_ranking = list(new_attr.ranked_latents)
    finally:
        if was_training:
            model.train()

    return CircuitSet(
        members=C,
        trajectory=trajectory,
        baseline_m_deploy=float(m_deploy),
        baseline_m_train=float(m_train),
        stopped_reason=stopped_reason,
        non_monotonic=non_monotonic,
        non_monotonic_steps=non_monotonic_steps,
        scope=scope,
        hard_eval=bool(hard_eval),
        step_size=int(step_size),
        epsilon=float(eps),
        margin=float(margin),
        max_circuit_size=int(max_circuit_size),
    )


__all__ = [
    "CircuitSet",
    "greedy_circuit_identification",
]
