"""Attribution patching (Tier 1) and scoped feature-steering hook.

Spec: src/sleeper/circuit_discovery_spec_v1.md §4

Primary ranker: zero-order attribution patching ``(z_deploy − z_train) · ∂m/∂z``
per (module, latent), under both STE modes (hard_eval on/off) and both position
scopes (trigger / response). Includes :class:`ScopedFeatureSteeringContext` for
position-scoped ablation (the existing :class:`FeatureSteeringContext` in
``src/sleeper/interventions.py`` is position-agnostic) and
:func:`exact_single_node_ablation` as a top-K spot-check against Tier 1.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from src.circuits.metric import hostile_logit_sum
from src.circuits.prompt_pairs import AlignmentMap
from src.models import TopKLoRALinearSTE

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class NodeAttrResult:
    """Per-scope, per-STE-mode attribution results.

    Spec: §4 Tier 1. Produced by :func:`attribution_patching`.
    """

    # module_name -> Tensor[r] aggregated across pairs
    attr_mean: Dict[str, torch.Tensor]
    attr_std: Dict[str, torch.Tensor]
    # per-pair raw: module_name -> Tensor[n_pairs, r]
    attr_per_pair: Dict[str, torch.Tensor]
    # module names in traversal order (deterministic)
    module_names: List[str]
    # [("module_name", latent_idx)] ranked by abs(attr_mean), descending
    ranked_latents: List[Tuple[str, int]]
    # metadata
    scope: str  # "trigger" or "response"
    hard_eval: bool
    n_pairs: int
    use_first_k: Optional[int] = None

    def flat_attr_mean(self) -> torch.Tensor:
        """Concatenate per-module mean tensors in ``module_names`` order."""
        if not self.module_names:
            return torch.zeros(0)
        return torch.cat(
            [self.attr_mean[name] for name in self.module_names], dim=0
        )

    def flat_ids(self) -> List[Tuple[str, int]]:
        """Matching (module_name, latent_idx) list for :meth:`flat_attr_mean`."""
        ids: List[Tuple[str, int]] = []
        for name in self.module_names:
            r = int(self.attr_mean[name].numel())
            ids.extend((name, i) for i in range(r))
        return ids


# ---------------------------------------------------------------------------
# Scoped feature-steering context manager
# ---------------------------------------------------------------------------


class ScopedFeatureSteeringContext:
    """Position-scoped analogue of :class:`FeatureSteeringContext`.

    Spec: §4 Tier 2. Writes apply ONLY at specified seq-positions; all other
    positions are forwarded bit-identically to the no-hook forward.

    Parameters
    ----------
    model
        Model containing :class:`TopKLoRALinearSTE` modules.
    no_grad
        When ``True`` (default), wrap the hook body in ``torch.no_grad()`` —
        correct for baseline / ablation measurements. Set to ``False`` to let
        gradients flow through the hook (useful only if you embed this context
        inside an attribution forward).
    """

    def __init__(self, model, *, no_grad: bool = True) -> None:
        self.model = model
        self.no_grad = bool(no_grad)
        # layer_name -> dim_idx -> (op, value, tuple(positions))
        self.rules: Dict[str, Dict[int, Tuple[str, float, Tuple[int, ...]]]] = {}
        self._hooks: List = []

    # -- public mutators -----------------------------------------------------

    def _add(
        self,
        layer_name: str,
        dim_indices: Iterable[int],
        op: str,
        value: float,
        positions: Sequence[int],
    ) -> None:
        pos_tuple = tuple(int(p) for p in positions)
        layer_map = self.rules.setdefault(layer_name, {})
        for d in dim_indices:
            d_int = int(d)
            assert d_int not in layer_map, (
                f"Duplicate scoped-steering rule for ({layer_name!r}, dim={d_int}); "
                "merge calls at the caller."
            )
            layer_map[d_int] = (op, float(value), pos_tuple)

    def ablate(
        self,
        layer_name: str,
        dim_indices: Iterable[int],
        *,
        positions: Sequence[int],
    ) -> None:
        self._add(layer_name, dim_indices, "ablate", 0.0, positions)

    def force(
        self,
        layer_name: str,
        dim_indices: Iterable[int],
        value: float,
        *,
        positions: Sequence[int],
    ) -> None:
        self._add(layer_name, dim_indices, "force", value, positions)

    def clamp(
        self,
        layer_name: str,
        dim_indices: Iterable[int],
        value: float,
        *,
        positions: Sequence[int],
    ) -> None:
        self._add(layer_name, dim_indices, "clamp", value, positions)

    # -- hook builder --------------------------------------------------------

    def _build_hook(
        self,
        rules_by_dim: Dict[int, Tuple[str, float, Tuple[int, ...]]],
    ):
        no_grad = self.no_grad

        def _apply_rules(z_sparse: torch.Tensor) -> torch.Tensor:
            # z_sparse shape: [..., seq_len, r] — position is the second-to-last
            # axis. We edit in-place on a clone to avoid altering any cached
            # tensor still referenced elsewhere. Single-token decoding paths
            # (2D input to the wrapped module) would collapse seq_len out and
            # make `z_sparse[..., pos, :]` write along the batch axis silently;
            # guard explicitly because this pipeline always runs 3D.
            if z_sparse.dim() < 3:
                raise ValueError(
                    "ScopedFeatureSteeringContext requires z_sparse with seq dim "
                    f"(dim >= 3), got shape {tuple(z_sparse.shape)}. Position-scoped "
                    "writes are undefined on 2D (single-token) tensors."
                )
            seq_len = z_sparse.shape[-2]
            for dim_idx, (op, value, positions) in rules_by_dim.items():
                if dim_idx < 0 or dim_idx >= z_sparse.shape[-1]:
                    continue
                for pos in positions:
                    if 0 <= pos < seq_len:
                        if op == "ablate":
                            z_sparse[..., pos, dim_idx] = 0.0
                        elif op in ("force", "clamp"):
                            z_sparse[..., pos, dim_idx] = float(value)
            return z_sparse

        def hook(module: TopKLoRALinearSTE, args, _output):
            x = args[0]
            cm = torch.no_grad() if no_grad else contextlib.nullcontext()
            with cm:
                # Re-run the full forward to get sparse_latents; mutate; decode.
                state = module.forward_with_state(x, cache=False)
                z_sparse = state.sparse_latents.clone()
                z_sparse = _apply_rules(z_sparse)
                return module.recompute_output_from_sparse_latents(
                    x,
                    z_sparse,
                    base_out=state.base_out,
                    decoder_norms=state.decoder_norms,
                )

        return hook

    # -- context manager -----------------------------------------------------

    def __enter__(self):
        named = dict(self.model.named_modules())
        for layer_name, rules in self.rules.items():
            module = named.get(layer_name)
            if not isinstance(module, TopKLoRALinearSTE):
                LOGGER.debug(
                    "ScopedFeatureSteeringContext: skipping non-TopKLoRA module %s",
                    layer_name,
                )
                continue
            handle = module.register_forward_hook(self._build_hook(rules))
            self._hooks.append(handle)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Leaf-latent context (for attribution_patching's grad forward)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _leaf_latent_context(model):
    """Replace each TopKLoRA module's forward output with a leaf-captured version.

    During the ``with`` block every forward call through a wrapped
    :class:`TopKLoRALinearSTE` replaces its internal ``dense_latents`` with a
    fresh ``requires_grad_=True`` leaf, appends that leaf to
    ``captured[module_name]``, and runs the remainder of the forward
    (``apply_topk`` + ``recompute_output_from_sparse_latents``) from the leaf.
    Callers use ``captured[name][-1]`` as the graph-leaf ``z`` tensor for
    ``torch.autograd.grad``.

    Spec: §4, step 3 ("Re-run ``p_deploy`` with grad enabled. Forward hook per
    wrapped module replaces ``dense_latents`` with ``z_leaf`` …").
    """
    captured: Dict[str, List[torch.Tensor]] = {}
    handles: List = []

    def _make_hook(module_name: str):
        def hook(module: TopKLoRALinearSTE, args, _output):
            x = args[0]
            # Run the entire forward under enable_grad so the leaf's graph is
            # preserved even if the surrounding forward was in no_grad.
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


def _topk_modules(model) -> Dict[str, TopKLoRALinearSTE]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, TopKLoRALinearSTE)
    }


@contextlib.contextmanager
def _override_hard_eval(modules: Dict[str, TopKLoRALinearSTE], hard_eval: bool):
    """Temporarily set ``module.hard_eval`` on each wrapped module."""
    saved = {name: bool(m.hard_eval) for name, m in modules.items()}
    try:
        for m in modules.values():
            m.hard_eval = bool(hard_eval)
        yield
    finally:
        for name, m in modules.items():
            m.hard_eval = saved[name]


def _positions_for_scope(
    pair: AlignmentMap,
    scope: str,
    run: str,
    n_target_tokens: int,
) -> List[int]:
    """Return absolute seq-indices (in the full prompt+response sequence
    teacher-forced during the metric forward) for the requested scope and run.

    For ``scope == "trigger"`` positions sit inside the prompt, i.e.
    ``< response_start_pos``.

    For ``scope == "response"`` positions cover the first ``n_target_tokens``
    response tokens, starting at ``response_start_pos``.
    """
    if scope == "trigger":
        return [int(pair.canonical_trigger_pos[run])]
    if scope == "response":
        rsp = int(pair.response_start_pos[run])
        return [rsp + i for i in range(int(n_target_tokens))]
    raise ValueError(f"scope must be 'trigger' or 'response', got {scope!r}")


def _ids_as_1d_tensor(ids, device) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        t = ids.to(device=device, dtype=torch.long).flatten()
    else:
        t = torch.tensor([int(i) for i in ids], device=device, dtype=torch.long)
    return t


# ---------------------------------------------------------------------------
# Attribution patching
# ---------------------------------------------------------------------------


def attribution_patching(
    model,
    pairs: List[AlignmentMap],
    *,
    scope: str,
    hard_eval: bool,
    hostile_target_ids: Sequence[int],
    use_first_k: Optional[int] = None,
) -> NodeAttrResult:
    """Tier-1 attribution patching across all wrapped :class:`TopKLoRALinearSTE`.

    Spec: §4.

    For each pair:

    1. Run ``p_train`` under ``no_grad`` and capture ``z_train``.
    2. Run ``p_deploy`` under ``no_grad`` and capture ``z_deploy``; compute Δz.
    3. Re-run ``p_deploy`` under :func:`_leaf_latent_context` with grad enabled;
       compute the hostile-logit-sum metric; take ``autograd.grad`` against each
       captured leaf.
    4. ``attr_m = Σ_pos (Δz[pos] * grads[pos])`` per module per latent.

    Aggregates per-pair means / stds across all supplied pairs.
    """
    if scope not in ("trigger", "response"):
        raise ValueError(f"scope must be 'trigger' or 'response', got {scope!r}")
    if not pairs:
        raise ValueError("attribution_patching requires at least one pair")

    device = next(model.parameters()).device

    # Normalise target_ids & effective k once.
    target = _ids_as_1d_tensor(hostile_target_ids, device=device)
    if target.numel() == 0:
        raise ValueError("hostile_target_ids is empty")
    effective_k = (
        int(target.numel())
        if use_first_k is None
        else int(min(int(use_first_k), int(target.numel())))
    )

    modules = _topk_modules(model)
    module_names = sorted(modules.keys())  # deterministic order
    if not module_names:
        raise ValueError("No TopKLoRALinearSTE modules found on model")

    n_pairs = len(pairs)
    # Per-pair attribution stacks: module_name -> [n_pairs, r]
    per_pair: Dict[str, torch.Tensor] = {
        name: torch.zeros(n_pairs, modules[name].r, dtype=torch.float32)
        for name in module_names
    }

    was_training = model.training
    model.eval()
    try:
        with _override_hard_eval(modules, hard_eval):
            for pair_idx, pair in enumerate(pairs):
                LOGGER.info(
                    "attribution_patching pair %d/%d (scope=%s, hard_eval=%s)",
                    pair_idx + 1,
                    n_pairs,
                    scope,
                    hard_eval,
                )

                clean_positions = _positions_for_scope(
                    pair, scope, run="clean", n_target_tokens=effective_k
                )
                trig_positions = _positions_for_scope(
                    pair, scope, run="trig", n_target_tokens=effective_k
                )
                # Both runs share the same *relative* position count. Hostile
                # teacher-forcing concatenates target_ids onto the prompt, so
                # the full sequence length covers response positions naturally.
                assert len(clean_positions) == len(trig_positions), (
                    "clean/trig position counts diverged — alignment bug?"
                )

                clean_ids = _ids_as_1d_tensor(pair.clean_input_ids, device=device)
                trig_ids = _ids_as_1d_tensor(pair.trig_input_ids, device=device)
                clean_rsp = int(pair.response_start_pos["clean"])
                trig_rsp = int(pair.response_start_pos["trig"])

                # --- Forward on p_train (no grad) to capture z_train ---------
                z_train: Dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    with _leaf_latent_context(model) as captured_train:
                        # We only need the cached z values, not the metric, but
                        # running the full teacher-forced forward keeps the
                        # sequence length identical to the grad forward below
                        # (so positions line up).
                        _ = hostile_logit_sum(
                            model,
                            clean_ids,
                            clean_rsp,
                            target,
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

                # --- Forward on p_deploy (no grad) to capture z_deploy -------
                z_deploy: Dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    with _leaf_latent_context(model) as captured_dep0:
                        _ = hostile_logit_sum(
                            model,
                            trig_ids,
                            trig_rsp,
                            target,
                            use_first_k=effective_k,
                        )
                        for name in module_names:
                            entries = captured_dep0.get(name, [])
                            if not entries:
                                raise RuntimeError(
                                    f"Module {name!r} produced no captured leaf "
                                    "on deploy forward"
                                )
                            z_deploy[name] = entries[-1].detach().clone()

                # --- Grad forward on p_deploy with leaf latents --------------
                with _leaf_latent_context(model) as captured_grad:
                    # Make sure we can cleanly identify the one leaf per module
                    captured_grad.clear()
                    metric = hostile_logit_sum(
                        model,
                        trig_ids,
                        trig_rsp,
                        target,
                        use_first_k=effective_k,
                    )
                    # Grab the one leaf per module from this forward.
                    leaf_order: List[str] = []
                    leaf_tensors: List[torch.Tensor] = []
                    for name in module_names:
                        entries = captured_grad.get(name, [])
                        if not entries:
                            raise RuntimeError(
                                f"Module {name!r} produced no captured leaf "
                                "on grad forward"
                            )
                        # Use the final entry — the grad forward runs exactly
                        # once per module, but guard anyway.
                        leaf_order.append(name)
                        leaf_tensors.append(entries[-1])

                    grads = torch.autograd.grad(
                        metric,
                        leaf_tensors,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )

                grad_by_name: Dict[str, torch.Tensor] = {
                    name: g.detach() for name, g in zip(leaf_order, grads)
                }

                # --- Per-module attribution at scoped positions --------------
                # z tensors are shaped [B=1, full_seq_len, r]. Positions from
                # _positions_for_scope are absolute indices into the full
                # (prompt + target) sequence.
                for name in module_names:
                    zt = z_train[name]  # clean run
                    zd = z_deploy[name]  # deploy (no-grad) — for Δz value
                    gd = grad_by_name[name]  # deploy (grad) — for ∂m/∂z

                    # Slice per-run positions. For Δz we compare the SAME logical
                    # position across the two runs, even though their absolute
                    # indices differ due to tag tokenisation.
                    dz_clean = zt[..., clean_positions, :]  # [..., P, r]
                    dz_deploy = zd[..., trig_positions, :]  # [..., P, r]
                    dz = dz_deploy - dz_clean  # [..., P, r]
                    gr = gd[..., trig_positions, :]  # [..., P, r]

                    # Sum over batch + position axes, leave latent dim.
                    attr = (dz * gr).sum(dim=tuple(range(dz.dim() - 1)))
                    per_pair[name][pair_idx] = attr.detach().float().cpu()

    finally:
        if was_training:
            model.train()

    # Aggregate
    attr_mean: Dict[str, torch.Tensor] = {}
    attr_std: Dict[str, torch.Tensor] = {}
    for name in module_names:
        stack = per_pair[name]  # [n_pairs, r]
        attr_mean[name] = stack.mean(dim=0)
        if n_pairs > 1:
            attr_std[name] = stack.std(dim=0, unbiased=False)
        else:
            attr_std[name] = torch.zeros_like(stack[0])

    # Rank globally by abs(mean), descending.
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
# Exact single-node ablation (Tier 2)
# ---------------------------------------------------------------------------


def exact_single_node_ablation(
    model,
    pairs: List[AlignmentMap],
    latent_ids: Sequence[Tuple[str, int]],
    *,
    scope: str,
    hard_eval: bool,
    hostile_target_ids: Sequence[int],
    use_first_k: Optional[int] = None,
) -> Dict[Tuple[str, int], float]:
    """Exact single-node ablation spot-check. Spec: §4 Tier 2.

    For each ``(layer_name, latent_idx)`` in ``latent_ids``, computes
    ``mean_pairs( m_deploy − m_deploy_ablated )`` where the ablation is scoped
    to the target positions (trigger / response) on ``p_deploy``.
    """
    if scope not in ("trigger", "response"):
        raise ValueError(f"scope must be 'trigger' or 'response', got {scope!r}")

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

    was_training = model.training
    model.eval()

    effects: Dict[Tuple[str, int], List[float]] = {
        key: [] for key in latent_ids
    }

    try:
        with _override_hard_eval(modules, hard_eval):
            for pair in pairs:
                trig_ids = _ids_as_1d_tensor(pair.trig_input_ids, device=device)
                trig_rsp = int(pair.response_start_pos["trig"])
                positions = _positions_for_scope(
                    pair, scope, run="trig", n_target_tokens=effective_k
                )

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

                for layer_name, dim_idx in latent_ids:
                    if layer_name not in modules:
                        effects[(layer_name, dim_idx)].append(0.0)
                        continue
                    ctx = ScopedFeatureSteeringContext(model, no_grad=True)
                    ctx.ablate(layer_name, [int(dim_idx)], positions=positions)
                    with ctx, torch.no_grad():
                        m_abl = float(
                            hostile_logit_sum(
                                model,
                                trig_ids,
                                trig_rsp,
                                target,
                                use_first_k=effective_k,
                            ).detach().cpu().item()
                        )
                    effects[(layer_name, int(dim_idx))].append(m_base - m_abl)
    finally:
        if was_training:
            model.train()

    out: Dict[Tuple[str, int], float] = {}
    for key, vals in effects.items():
        out[key] = float(sum(vals) / len(vals)) if vals else 0.0
    return out


# Public alias — other modules (e.g. circuit_id) need scope→position mapping too.
positions_for_scope = _positions_for_scope


__all__ = [
    "NodeAttrResult",
    "ScopedFeatureSteeringContext",
    "attribution_patching",
    "exact_single_node_ablation",
    "positions_for_scope",
]
