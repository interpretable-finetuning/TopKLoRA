"""STE dual-run + pre-topk selection probe.

Spec: src/sleeper/circuit_discovery_spec_v1.md §8

Addresses the fundamental STE hard_eval visibility problem: latents hard-masked
to zero on deploy have zero gradient under hard_eval=True. Runs attribution
under both hard_eval=True (exploitation) and hard_eval=False (counterfactual),
and implements a pre-topk force-activate probe that injects Δ before the top-k
competition to find dormant selectors — latents that are inactive on deploy but
would causally change behavior if they won the top-k contest.

Three public entry points:

1. :func:`dual_ste_attribution` — runs :func:`attribution_patching` twice, once
   per STE mode, returning a :class:`DualSTEResult` (§8a, §8b).
2. :func:`rank_dormant_latent_candidates` — identifies latents that are
   hard-masked on ``p_deploy`` at the target positions, have near-zero
   attribution in both STE modes, yet have notable pre-topk ``dense_latents``
   magnitude (§8c criteria i/ii/iii). Returns the top N by pre-topk magnitude.
3. :func:`pretopk_selection_probe` — for each candidate, installs a forward
   hook that adds Δ to ``dense_latents`` BEFORE ``apply_topk``, re-runs the
   forward, and measures whether the candidate entered top-k, which latents
   got displaced, and how the hostile metric shifted (§8c step 3).

The Δ=0 control path in :func:`pretopk_selection_probe` is bit-identical to
the un-hooked ``forward_with_state(cache=False)`` output — guarded by
``tests/test_circuits_ste_modes.py::test_pre_topk_control_matches_default_forward``.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from src.circuits.attribution import (
    NodeAttrResult,
    attribution_patching,
    positions_for_scope,
    _topk_modules,
)
from src.circuits.metric import hostile_logit_sum
from src.circuits.prompt_pairs import AlignmentMap
from src.models import TopKLoRALinearSTE, _hard_topk_mask

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Part A — dual STE attribution
# ---------------------------------------------------------------------------


@dataclass
class DualSTEResult:
    """Result of dual-mode attribution: exploitation + counterfactual.

    Spec §8 parts 8a, 8b.

    Attributes
    ----------
    exploitation
        ``NodeAttrResult`` from ``attribution_patching(..., hard_eval=True)``
        — matches the deploy forward exactly.
    counterfactual
        ``NodeAttrResult`` from ``attribution_patching(..., hard_eval=False)``
        — uses STE so gradients leak through hard-masked latents.
    scope
        ``"trigger"`` or ``"response"`` — scope both runs were executed under.
    """

    exploitation: NodeAttrResult  # hard_eval=True
    counterfactual: NodeAttrResult  # hard_eval=False
    scope: str


def dual_ste_attribution(
    model,
    pairs: Sequence[AlignmentMap],
    *,
    scope: str,
    hostile_target_ids: Sequence[int],
    use_first_k: Optional[int] = None,
) -> DualSTEResult:
    """Run attribution patching twice, once per STE mode.

    Spec §8a / §8b. The exploitation run matches the deploy forward exactly
    (``hard_eval=True``), answering "given the sparsity pattern the model
    actually uses, which latents carry the signal?" The counterfactual run
    uses STE (``hard_eval=False``) so gradients leak to hard-masked latents
    through the soft surrogate path, answering "which latents would be causal
    if gate competition changed?"

    This is a thin wrapper — just runs :func:`attribution_patching` twice.
    """
    if scope not in ("trigger", "response"):
        raise ValueError(f"scope must be 'trigger' or 'response', got {scope!r}")

    expl = attribution_patching(
        model,
        list(pairs),
        scope=scope,
        hard_eval=True,
        hostile_target_ids=hostile_target_ids,
        use_first_k=use_first_k,
    )
    cntr = attribution_patching(
        model,
        list(pairs),
        scope=scope,
        hard_eval=False,
        hostile_target_ids=hostile_target_ids,
        use_first_k=use_first_k,
    )
    return DualSTEResult(exploitation=expl, counterfactual=cntr, scope=scope)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ids_as_1d_tensor(ids, device) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        return ids.to(device=device, dtype=torch.long).flatten()
    return torch.tensor([int(i) for i in ids], device=device, dtype=torch.long)


def _positions_for_pair(pair: AlignmentMap, scope: str, run: str, effective_k: int) -> List[int]:
    """Thin wrapper around :func:`positions_for_scope` accepting pair+run+scope."""
    return list(positions_for_scope(pair, scope, run=run, n_target_tokens=effective_k))


def _pretopk_forward_dense_and_mask(
    module: TopKLoRALinearSTE,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the pre-topk half of the forward: encode_pre -> decoder_norms
    -> _topk_scores -> _activate_latents -> hard mask.

    Returns ``(dense_latents, hard_mask, decoder_norms, base_out)``.
    """
    base_out = module.base_layer(x)
    hidden_pre = module.encode_pre(x)
    decoder_norms = module.decoder_norms().to(
        device=hidden_pre.device, dtype=hidden_pre.dtype
    )
    topk_scores = module._topk_scores(hidden_pre, decoder_norms)
    dense_latents = module._activate_latents(topk_scores)
    k_now = int(module._current_k())
    topk_mode = module.topk_mode
    hard_mask = _hard_topk_mask(dense_latents, k_now, topk_mode)
    return dense_latents, hard_mask, decoder_norms, base_out


# ---------------------------------------------------------------------------
# Part B — dormant latent ranking
# ---------------------------------------------------------------------------


def rank_dormant_latent_candidates(
    model,
    pairs: Sequence[AlignmentMap],
    dual_result: DualSTEResult,
    *,
    scope: str,
    top_n: int = 30,
    attribution_near_zero_threshold: float = 1e-4,
) -> List[Tuple[str, int, float]]:
    """Identify (module, latent) candidates for the pre-topk probe.

    Spec §8c. Criteria:

    (i)  hard-masked on ``p_deploy`` at the target position(s) for EVERY pair
         (never in top-k — OR aggregation across positions: if ever unmasked
         at any scoped position on any pair, the latent is NOT dormant).
    (ii) near-zero attribution in BOTH STE modes
         (``|attr_mean| < attribution_near_zero_threshold``).
    (iii) notable pre-topk ``dense_latents`` magnitude
         (mean of ``|dense_latents|`` across all pairs × scoped positions).

    Returns the top ``top_n`` ranked by criterion (iii) descending: a list of
    tuples ``(module_name, latent_idx, pre_topk_magnitude_mean)``.

    The forward is run under ``torch.no_grad()`` with ``cache=False`` so the
    module's ``_last_*`` caches are not mutated by this analysis.
    """
    if scope not in ("trigger", "response"):
        raise ValueError(f"scope must be 'trigger' or 'response', got {scope!r}")
    if not pairs:
        return []

    device = next(model.parameters()).device
    modules = _topk_modules(model)
    if not modules:
        return []
    module_names = sorted(modules.keys())

    # Use the attribution results' use_first_k value for position resolution so
    # the scope lines up with the attribution computation.
    effective_k_attr = dual_result.exploitation.use_first_k

    # Per-module per-latent accumulators across (pair × scoped-position).
    dense_abs_sum: Dict[str, torch.Tensor] = {
        name: torch.zeros(modules[name].r, dtype=torch.float64)
        for name in module_names
    }
    dense_count: Dict[str, int] = {name: 0 for name in module_names}
    # A latent is "always masked" iff its hard_mask value is 0 at EVERY scoped
    # position on EVERY pair. Start with "all masked" = True; flip to False on
    # any unmasked observation.
    ever_unmasked: Dict[str, torch.Tensor] = {
        name: torch.zeros(modules[name].r, dtype=torch.bool)
        for name in module_names
    }

    was_training = model.training
    model.eval()
    try:
        for pair in pairs:
            trig_ids = _ids_as_1d_tensor(pair.trig_input_ids, device=device)
            trig_rsp = int(pair.response_start_pos["trig"])
            # For resolving scoped positions we need the response-token count;
            # attribution_patching clamps use_first_k to len(target_ids), so
            # fall back to 1 for trigger scope (single position) if it wasn't
            # set (shouldn't matter in practice, but keeps the call safe).
            n_target_tokens = (
                int(effective_k_attr) if effective_k_attr is not None else 1
            )
            positions = _positions_for_pair(
                pair, scope, run="trig", effective_k=n_target_tokens
            )

            # Teacher-forced forward over the FULL prompt+response sequence
            # ensures the seq indexing matches the attribution-time scope.
            full_input = trig_ids.unsqueeze(0)
            if scope == "response":
                # Append dummy response tokens to hit the response positions.
                # The actual token values don't matter for the pre-topk
                # encoder forward because positions indices are absolute in
                # the full sequence; but the sequence length must cover the
                # max index in `positions`.
                needed_len = max(positions) + 1
                pad = needed_len - full_input.shape[-1]
                if pad > 0:
                    # Pad with zeros (or any valid token id). Use 0 which is
                    # almost always a legal id; it only drives the encoder
                    # path we don't actually evaluate at response positions
                    # that aren't in `positions`.
                    pad_t = torch.zeros(
                        1, pad, device=device, dtype=trig_ids.dtype
                    )
                    full_input = torch.cat([full_input, pad_t], dim=-1)

            # Run a single forward through the model with hooks that capture
            # dense_latents & hard_mask from each wrapped module. Use forward
            # hooks (post-forward) so ``module.forward_with_state(cache=False)``
            # gives us the authoritative tensors without polluting the caches.
            captured_dense: Dict[str, torch.Tensor] = {}
            captured_hard: Dict[str, torch.Tensor] = {}

            def _make_capture_hook(name: str):
                def hook(module, args, _out):
                    x = args[0]
                    with torch.no_grad():
                        dense, hard, _dn, _bo = _pretopk_forward_dense_and_mask(
                            module, x
                        )
                    captured_dense[name] = dense.detach()
                    captured_hard[name] = hard.detach()
                    # Return the original output so the model continues as
                    # normal (we don't need to replace anything).
                    return None

                return hook

            handles = []
            for name in module_names:
                handles.append(
                    modules[name].register_forward_hook(_make_capture_hook(name))
                )
            try:
                with torch.no_grad():
                    _ = model(input_ids=full_input)
            finally:
                for h in handles:
                    try:
                        h.remove()
                    except Exception:
                        pass

            # Aggregate per-module contributions across scoped positions.
            for name in module_names:
                dense = captured_dense.get(name)
                hard = captured_hard.get(name)
                if dense is None or hard is None:
                    continue
                # Shape: [B=1, seq_len, r]. Filter to in-range positions.
                seq_len = dense.shape[-2]
                in_range = [p for p in positions if 0 <= p < seq_len]
                if not in_range:
                    continue
                dense_slice = dense[..., in_range, :]  # [1, P, r]
                hard_slice = hard[..., in_range, :]  # [1, P, r]

                # Pre-topk magnitude: mean |dense| over batch+position axes.
                mag = dense_slice.abs().float().sum(
                    dim=tuple(range(dense_slice.dim() - 1))
                )
                dense_abs_sum[name] = dense_abs_sum[name] + mag.double().cpu()
                dense_count[name] += int(
                    dense_slice.shape[0] * dense_slice.shape[-2]
                )

                # ever_unmasked: any position where hard > 0 flips the bit.
                unmasked_any = (hard_slice > 0).any(
                    dim=tuple(range(hard_slice.dim() - 1))
                ).cpu()
                ever_unmasked[name] = ever_unmasked[name] | unmasked_any
    finally:
        if was_training:
            model.train()

    # Compose candidate list.
    candidates: List[Tuple[str, int, float]] = []
    tau = float(attribution_near_zero_threshold)
    attr_expl = dual_result.exploitation.attr_mean
    attr_cntr = dual_result.counterfactual.attr_mean

    for name in module_names:
        r = modules[name].r
        count = max(dense_count[name], 1)
        mean_mag = (dense_abs_sum[name] / float(count)).to(torch.float32)
        # Attribution tensors may be missing for unusual modules; default to 0.
        expl_mean = attr_expl.get(name, torch.zeros(r))
        cntr_mean = attr_cntr.get(name, torch.zeros(r))
        for i in range(r):
            if bool(ever_unmasked[name][i].item()):
                continue  # (i) failed — was in top-k somewhere
            attr_e = float(expl_mean[i].abs().item())
            attr_c = float(cntr_mean[i].abs().item())
            if attr_e >= tau or attr_c >= tau:
                continue  # (ii) failed
            mag = float(mean_mag[i].item())
            candidates.append((name, int(i), mag))

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[: int(top_n)]


# ---------------------------------------------------------------------------
# Part C — pre-topk selection probe
# ---------------------------------------------------------------------------


@dataclass
class DormantSelectorReport:
    """Outcome of forcing a dormant latent pre-topk.

    Spec §8c step 3. One report per ``(candidate, multiplier)`` pair.

    Attributes
    ----------
    module_name
        The wrapped module the latent lives in.
    latent_idx
        The latent dim inside that module.
    multiplier
        The ``multiplier`` value used for this probe run.
    delta_used
        The actual ``Δ`` value that was added to ``dense_latents[..., pos, i]``
        (``= multiplier × max(|dense_latents|)`` on the scoped forward).
    entered_top_k
        Did this latent enter top-k at ANY scoped position after forcing?
    displaced_latents
        Indices that were in top-k at baseline but NOT in top-k after forcing,
        aggregated across scoped positions (union).
    metric_shift
        ``m_deploy(baseline) − m_deploy(forced)``. Positive means forcing the
        candidate REDUCED the hostile signal; negative means forcing INCREASED
        it. Averaged across pairs.
    flagged
        ``True`` iff this candidate meets the §8c "dormant selector" criteria:
        entered_top_k=True AND at least one latent was displaced AND
        ``|metric_shift| > noise_floor``.
    """

    module_name: str
    latent_idx: int
    multiplier: float
    delta_used: float
    entered_top_k: bool
    displaced_latents: List[int]
    metric_shift: float
    flagged: bool = False


class _ForceHookContext:
    """Forward hook that forces one latent pre-topk on a wrapped module.

    The hook replaces the wrapped module's output with a recomputed one:

    1. Run ``_pretopk_forward_dense_and_mask`` to get ``(dense_latents,
       baseline_hard_mask, decoder_norms, base_out)``.
    2. At each ``position`` in ``positions``, set
       ``dense_latents[..., pos, latent_idx] += Δ`` where
       ``Δ = multiplier × max(|dense_latents|)`` on THIS forward pass.
    3. Call ``module.apply_topk(dense_latents_modified)`` to recompute the
       sparse latents under the modified dense tensor.
    4. Call ``module.recompute_output_from_sparse_latents`` with the cached
       base_out / decoder_norms.
    5. Record the delta, the new hard mask at the scoped positions, and the
       baseline hard mask — for downstream displacement analysis.

    The hook runs under ``torch.no_grad()`` (measurement only) and uses
    ``cache=False`` semantics (nothing updates ``module._last_*``).

    When ``multiplier == 0``, Δ is exactly zero and the output is bit-
    identical to the un-hooked ``forward_with_state(cache=False).output`` —
    the unit-test gate.
    """

    def __init__(
        self,
        *,
        module: TopKLoRALinearSTE,
        latent_idx: int,
        multiplier: float,
        positions: Sequence[int],
    ) -> None:
        self.module = module
        self.latent_idx = int(latent_idx)
        self.multiplier = float(multiplier)
        self.positions = tuple(int(p) for p in positions)
        self.delta_used: float = 0.0
        self.baseline_hard_mask_at_positions: Optional[torch.Tensor] = None
        self.modified_hard_mask_at_positions: Optional[torch.Tensor] = None
        self._handle = None

    def _hook(self, module, args, _output):
        x = args[0]
        with torch.no_grad():
            dense, baseline_hard, dn, base_out = _pretopk_forward_dense_and_mask(
                module, x
            )
            if dense.dim() < 3:
                raise ValueError(
                    "pretopk_selection_probe requires a 3D dense_latents "
                    f"[B, L, r], got shape {tuple(dense.shape)}."
                )
            dense_mod = dense.clone()
            delta_val = 0.0
            if dense.numel() > 0:
                max_abs = float(dense.abs().max().item())
                delta_val = float(self.multiplier) * max_abs
            self.delta_used = delta_val
            seq_len = dense_mod.shape[-2]
            for pos in self.positions:
                if 0 <= pos < seq_len and 0 <= self.latent_idx < dense_mod.shape[-1]:
                    dense_mod[..., pos, self.latent_idx] = (
                        dense_mod[..., pos, self.latent_idx] + delta_val
                    )

            # Recompute mask AFTER modification (for displacement analysis).
            k_now = int(module._current_k())
            topk_mode = module.topk_mode
            modified_hard = _hard_topk_mask(dense_mod, k_now, topk_mode)

            # Slice masks at scoped positions for the caller.
            in_range = [p for p in self.positions if 0 <= p < seq_len]
            if in_range:
                self.baseline_hard_mask_at_positions = (
                    baseline_hard[..., in_range, :].detach()
                )
                self.modified_hard_mask_at_positions = (
                    modified_hard[..., in_range, :].detach()
                )

            # Run apply_topk to recompute sparse_latents. With the module in
            # eval mode + hard_eval=True, this gives dense_mod * modified_hard.
            _, _, _, sparse_mod, _, _ = module.apply_topk(dense_mod)

            return module.recompute_output_from_sparse_latents(
                x,
                sparse_mod,
                base_out=base_out,
                decoder_norms=dn,
            )

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
            self._handle = None


def pretopk_selection_probe(
    model,
    pairs: Sequence[AlignmentMap],
    candidates: Iterable[Tuple[str, int, float]],
    *,
    scope: str,
    hostile_target_ids: Sequence[int],
    use_first_k: Optional[int] = None,
    multipliers: Sequence[float] = (0.5, 1.0, 2.0),
    noise_floor: float = 1e-3,
) -> List[DormantSelectorReport]:
    """Force each dormant candidate pre-topk and report top-k displacement.

    Spec §8c step 3. For each ``(module_name, latent_idx)`` in ``candidates``
    and each value in ``multipliers``, runs a forward with a hook that:

    1. Computes the baseline ``dense_latents`` via
       ``_pretopk_forward_dense_and_mask`` (== ``forward_with_state(cache=False)``
       pre-topk stage).
    2. Adds ``Δ = multiplier × max(|dense_latents|)`` at the scoped positions,
       latent_idx.
    3. Calls ``module.apply_topk(dense_latents_modified)`` to recompute gates.
    4. Calls ``recompute_output_from_sparse_latents``.
    5. Captures baseline vs modified top-k membership at scoped positions.
    6. Measures ``m_deploy`` with and without forcing.

    Returns one :class:`DormantSelectorReport` per ``(candidate, multiplier)``
    pair — ``n_reports == len(candidates) × len(multipliers)``. This keeps
    each report narrow and comparable across multipliers; downstream code can
    aggregate (e.g. "smallest multiplier that enters top-k") as desired.

    Flag criterion (spec §8c): ``entered_top_k == True`` AND at least one
    other latent was displaced AND ``|metric_shift| > noise_floor``. A flagged
    report identifies a dormant-selector candidate; non-flagged reports are
    kept for diagnostics.

    Notes
    -----
    All forwards run under :func:`torch.no_grad`. Hooks are removed even on
    exception. ``cache=False`` semantics are preserved: the module's
    ``_last_*`` caches are never written to by this probe, so downstream
    analysis that depends on them is unaffected.
    """
    if scope not in ("trigger", "response"):
        raise ValueError(f"scope must be 'trigger' or 'response', got {scope!r}")
    if not pairs:
        return []

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

    cand_list = list(candidates)
    if not cand_list:
        return []

    was_training = model.training
    model.eval()

    reports: List[DormantSelectorReport] = []
    try:
        for (module_name, latent_idx, _pretopk_mag) in cand_list:
            module = modules.get(module_name)
            if not isinstance(module, TopKLoRALinearSTE):
                LOGGER.warning(
                    "pretopk_selection_probe: skipping unknown module %r",
                    module_name,
                )
                continue

            for mult in multipliers:
                entered_any_pos = False
                displaced_union: set = set()
                baseline_metrics: List[float] = []
                forced_metrics: List[float] = []
                delta_used_mean: List[float] = []

                for pair in pairs:
                    trig_ids = _ids_as_1d_tensor(
                        pair.trig_input_ids, device=device
                    )
                    trig_rsp = int(pair.response_start_pos["trig"])
                    positions = _positions_for_pair(
                        pair, scope, run="trig", effective_k=effective_k
                    )

                    # Baseline metric (no hook).
                    with torch.no_grad():
                        m_base = float(
                            hostile_logit_sum(
                                model,
                                trig_ids,
                                trig_rsp,
                                target,
                                use_first_k=effective_k,
                            ).detach().cpu().item()
                        )

                    # Forced metric (with hook).
                    ctx = _ForceHookContext(
                        module=module,
                        latent_idx=int(latent_idx),
                        multiplier=float(mult),
                        positions=positions,
                    )
                    with ctx:
                        with torch.no_grad():
                            m_forced = float(
                                hostile_logit_sum(
                                    model,
                                    trig_ids,
                                    trig_rsp,
                                    target,
                                    use_first_k=effective_k,
                                ).detach().cpu().item()
                            )

                    baseline_metrics.append(m_base)
                    forced_metrics.append(m_forced)
                    delta_used_mean.append(float(ctx.delta_used))

                    # Displacement analysis at scoped positions.
                    base_mask = ctx.baseline_hard_mask_at_positions
                    mod_mask = ctx.modified_hard_mask_at_positions
                    if base_mask is None or mod_mask is None:
                        continue
                    # base_mask / mod_mask: [B=1, P, r]. Reduce over B+P.
                    base_any = (base_mask > 0).any(
                        dim=tuple(range(base_mask.dim() - 1))
                    )
                    mod_any = (mod_mask > 0).any(
                        dim=tuple(range(mod_mask.dim() - 1))
                    )
                    if bool(mod_any[int(latent_idx)].item()):
                        entered_any_pos = True
                    # Displaced: in baseline but not in modified (at ANY pos).
                    displaced = (base_any & (~mod_any)).nonzero(as_tuple=False).flatten()
                    for d in displaced.tolist():
                        displaced_union.add(int(d))

                if not baseline_metrics:
                    continue

                metric_shift = (
                    sum(baseline_metrics) - sum(forced_metrics)
                ) / float(len(baseline_metrics))
                delta_used = (
                    sum(delta_used_mean) / float(len(delta_used_mean))
                )

                flagged = bool(
                    entered_any_pos
                    and len(displaced_union) > 0
                    and abs(metric_shift) > float(noise_floor)
                )

                reports.append(
                    DormantSelectorReport(
                        module_name=module_name,
                        latent_idx=int(latent_idx),
                        multiplier=float(mult),
                        delta_used=float(delta_used),
                        entered_top_k=bool(entered_any_pos),
                        displaced_latents=sorted(displaced_union),
                        metric_shift=float(metric_shift),
                        flagged=flagged,
                    )
                )
    finally:
        if was_training:
            model.train()

    return reports


__all__ = [
    "DualSTEResult",
    "DormantSelectorReport",
    "dual_ste_attribution",
    "rank_dormant_latent_candidates",
    "pretopk_selection_probe",
]
