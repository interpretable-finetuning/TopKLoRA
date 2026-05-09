"""Within-circuit internal edges via first-order Jacobian.

Spec: src/sleeper/circuit_discovery_spec_v1.md §9 (and §3 for architectural
plausibility).

Post-discovery: for each architecturally plausible ``(upstream, downstream)``
pair in ``C × C``, compute the first-order Jacobian edge weight

    edge = Σ_pos (Δz_upstream · ∂z_downstream/∂z_upstream) at scoped positions,

averaged across prompt pairs. A single forward pass under
:func:`_leaf_latent_context` per pair captures leaf latents for every wrapped
module; one backward per ``(downstream_module, downstream_dim)`` gives the
sensitivity ``∂z_downstream[..., d_dim]/∂z_leaf[upstream]`` for ALL upstream
modules at once. The per-upstream-dim edge weight is then just a reduction
against the Δz tensor.

No second-order autograd. ``retain_graph=True`` is used because we call
``autograd.grad`` repeatedly (once per downstream-dim of interest) from the
same forward.

Architectural plausibility (spec §3, single wrapped layer):

- ``v_proj`` → ``{gate_proj, up_proj, down_proj}`` (attention → residual →
  MLP; cross-position, mediated by frozen ``o_proj``).
- ``{gate_proj, up_proj}`` → ``down_proj`` (SwiGLU elementwise product; same
  position).
- ``down_proj → anything``: empty within the single wrapped layer.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.circuits.attribution import (
    _leaf_latent_context,
    _override_hard_eval,
    _topk_modules,
    positions_for_scope,
)
from src.circuits.metric import hostile_logit_sum
from src.circuits.prompt_pairs import AlignmentMap
from src.models import TopKLoRALinearSTE

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module classification + architectural-plausibility gate (spec §3)
# ---------------------------------------------------------------------------


def _classify_module(name: str) -> str:
    """Substring-match the four wrapped modules in the target adapter.

    Returns one of ``{"v", "gate", "up", "down", "other"}``. Matches the
    naming used in the real adapter (``layers.19.self_attn.v_proj``,
    ``layers.19.mlp.{gate,up,down}_proj``).
    """
    lower = name.lower()
    if "v_proj" in lower:
        return "v"
    if "gate_proj" in lower:
        return "gate"
    if "up_proj" in lower:
        return "up"
    if "down_proj" in lower:
        return "down"
    return "other"


def is_architecturally_plausible_edge(src_module: str, dst_module: str) -> bool:
    """Architectural-plausibility gate per spec §3.

    Rules:

    - ``v`` → ``{gate, up, down}``: True.
    - ``{gate, up}`` → ``down``: True.
    - ``down`` → anything: False.
    - self-edges (same module name): False (tautological).
    - anything else (including ``other`` source/destination): False.
    """
    if src_module == dst_module:
        return False
    src = _classify_module(src_module)
    dst = _classify_module(dst_module)
    if src == "v" and dst in ("gate", "up", "down"):
        return True
    if src in ("gate", "up") and dst == "down":
        return True
    return False


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Edge:
    """A single ``(src_latent → dst_latent)`` first-order Jacobian edge.

    Weight is averaged across all prompt pairs that contributed
    (``n_pairs``).
    """

    src_module: str
    src_latent: int
    dst_module: str
    dst_latent: int
    weight: float
    n_pairs: int


@dataclass
class EdgeResult:
    """Output of :func:`compute_within_circuit_edges`."""

    edges: List[Edge] = field(default_factory=list)
    scope: str = ""
    hard_eval: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_by_module(
    circuit: Sequence[Tuple[str, int]],
) -> Dict[str, List[int]]:
    """Group ``[(module_name, latent_idx), ...]`` by module, preserving order."""
    out: Dict[str, List[int]] = {}
    for name, idx in circuit:
        out.setdefault(name, []).append(int(idx))
    # Deduplicate while preserving order.
    for name, dims in out.items():
        seen: Dict[int, None] = {}
        ordered: List[int] = []
        for d in dims:
            if d not in seen:
                seen[d] = None
                ordered.append(d)
        out[name] = ordered
    return out


def _ids_as_1d_tensor(ids, device) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        return ids.to(device=device, dtype=torch.long).flatten()
    return torch.tensor([int(i) for i in ids], device=device, dtype=torch.long)


@contextlib.contextmanager
def _single_leaf_context(model, upstream_name: str):
    """Forward hook that leafs ONLY ``upstream_name``'s ``dense_latents``.

    Unlike :func:`_leaf_latent_context` (which detaches + re-leafs *every*
    wrapped module), this context leafs exactly one module's latents. Every
    other :class:`TopKLoRALinearSTE` runs its normal forward, so latents at
    downstream modules remain connected to the upstream leaf — which is what
    we need in order to take ``∂z_downstream/∂z_upstream`` via autograd.

    Yields a dict ``{"u_leaf": [tensor, ...], "z_by_name": {name: [tensors]}}``.
    ``z_by_name[name]`` is the downstream module's pre-topk ``dense_latents``
    captured on the same forward (NON-detached, so autograd can differentiate
    it against ``u_leaf``).
    """
    captured: Dict[str, object] = {"u_leaf": [], "z_by_name": {}}
    handles: List = []
    named = dict(model.named_modules())
    if upstream_name not in named or not isinstance(
        named[upstream_name], TopKLoRALinearSTE
    ):
        raise KeyError(
            f"upstream_name {upstream_name!r} is not a TopKLoRALinearSTE on model"
        )

    def _upstream_hook(module: TopKLoRALinearSTE, args, _output):
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
            captured["u_leaf"].append(z_leaf)
            _, _, _, sparse_latents, _, _ = module.apply_topk(z_leaf)
            return module.recompute_output_from_sparse_latents(
                x,
                sparse_latents,
                base_out=base_out,
                decoder_norms=decoder_norms,
            )

    def _downstream_capture_hook(name: str):
        # Capture the pre-topk dense_latents flowing through this module on the
        # natural forward, WITHOUT detaching — the tensor is still connected to
        # the upstream leaf via its computation path.
        def hook(module: TopKLoRALinearSTE, args, _output):
            x = args[0]
            with torch.enable_grad():
                hidden_pre = module.encode_pre(x)
                decoder_norms = module.decoder_norms().to(
                    device=hidden_pre.device, dtype=hidden_pre.dtype
                )
                topk_scores = module._topk_scores(hidden_pre, decoder_norms)
                dense_latents = module._activate_latents(topk_scores)
                captured["z_by_name"].setdefault(name, []).append(dense_latents)
            return _output  # do not alter module output

        return hook

    for name, module in model.named_modules():
        if not isinstance(module, TopKLoRALinearSTE):
            continue
        if name == upstream_name:
            handles.append(module.register_forward_hook(_upstream_hook))
        else:
            handles.append(module.register_forward_hook(_downstream_capture_hook(name)))

    try:
        yield captured
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_within_circuit_edges(
    model,
    pairs: Sequence[AlignmentMap],
    circuit: Sequence[Tuple[str, int]],
    *,
    scope: str,
    hard_eval: bool,
    hostile_target_ids: Sequence[int],
    use_first_k: Optional[int] = None,
    skip_if_empty: bool = True,
) -> EdgeResult:
    """Compute first-order Jacobian edges for every architecturally-plausible
    ``(u, d)`` pair inside a discovered circuit.

    Spec: §9.

    Parameters
    ----------
    model
        Model containing :class:`TopKLoRALinearSTE` modules.
    pairs
        Alignment-map prompt pairs. Only the deploy (triggered) run is
        forwarded; the clean run supplies ``z_train`` for ``Δz``.
    circuit
        ``[(module_name, latent_idx), ...]`` — the already-identified circuit
        (spec §5). Typically ``|C| ≤ 30``.
    scope
        ``"trigger"`` or ``"response"``.
    hard_eval
        STE mode override for the module; matches the attribution-patching
        convention.
    hostile_target_ids
        Hostile target token-id sequence teacher-forced onto the prompt for
        the metric forward. Required because the downstream ``z_leaf`` is
        captured under the same full-sequence forward used for the metric.
    use_first_k
        Optional truncation of ``hostile_target_ids``.
    skip_if_empty
        When ``True`` (default) and ``|circuit| < 2``, return an empty result
        without running any forwards.

    Returns
    -------
    EdgeResult
        Edges for every plausible ``(u, d)`` pair, weights averaged across
        pairs. Zero-weight edges are still included (for downstream
        thresholding); the dashboard typically filters by an absolute
        threshold slider.
    """
    if scope not in ("trigger", "response"):
        raise ValueError(f"scope must be 'trigger' or 'response', got {scope!r}")

    result = EdgeResult(edges=[], scope=scope, hard_eval=bool(hard_eval))

    if skip_if_empty and len(circuit) < 2:
        return result
    if not pairs:
        return result

    # Group by module; architecturally-plausible filter.
    by_module = _group_by_module(circuit)

    plausible_pairs: List[Tuple[str, str]] = []
    module_names = list(by_module.keys())
    for u_name in module_names:
        for d_name in module_names:
            if u_name == d_name:
                continue
            if is_architecturally_plausible_edge(u_name, d_name):
                plausible_pairs.append((u_name, d_name))

    if not plausible_pairs:
        return result

    # Short-circuit: if every module in `circuit` is architecturally
    # "down" (outgoing edges forbidden) or otherwise produces no pair, we
    # already returned above.

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
    missing = [name for name in module_names if name not in modules]
    if missing:
        raise KeyError(
            f"circuit references modules not found on model: {missing}"
        )

    # Accumulator: (src_module, src_latent, dst_module, dst_latent) -> running sum.
    acc: Dict[Tuple[str, int, str, int], float] = {}
    n_accum = 0

    was_training = model.training
    model.eval()
    try:
        with _override_hard_eval(modules, hard_eval):
            for pair_idx, pair in enumerate(pairs):
                LOGGER.info(
                    "compute_within_circuit_edges pair %d/%d "
                    "(scope=%s, hard_eval=%s, |C|=%d)",
                    pair_idx + 1,
                    len(pairs),
                    scope,
                    hard_eval,
                    len(circuit),
                )

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

                # --- Clean forward (no grad) to capture z_train ---------------
                z_train: Dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    with _leaf_latent_context(model) as captured_train:
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

                # --- Deploy forward (no grad) to capture z_deploy -------------
                z_deploy: Dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    with _leaf_latent_context(model) as captured_dep:
                        _ = hostile_logit_sum(
                            model,
                            trig_ids,
                            trig_rsp,
                            target,
                            use_first_k=effective_k,
                        )
                        for name in module_names:
                            entries = captured_dep.get(name, [])
                            if not entries:
                                raise RuntimeError(
                                    f"Module {name!r} produced no captured leaf "
                                    "on deploy forward"
                                )
                            z_deploy[name] = entries[-1].detach().clone()

                # Precompute Δz per module at clean/trig positions.
                # Shapes: [..., P, r] after slicing. For downstream slicing
                # we'll use each module's own Δz at trigger positions.
                delta_z: Dict[str, torch.Tensor] = {}
                for name in module_names:
                    zt = z_train[name][..., clean_positions, :]
                    zd = z_deploy[name][..., trig_positions, :]
                    delta_z[name] = zd - zt  # [..., P, r]

                # --- Per-upstream grad forward ------------------------------
                # For each upstream module that appears as a src in any
                # plausible pair, run ONE forward that leafs only that module.
                # Every downstream module's dense_latents flows naturally and
                # is captured non-detached, so autograd.grad can differentiate
                # against the upstream leaf.
                upstream_names: List[str] = list(
                    dict.fromkeys([u for (u, _d) in plausible_pairs])
                )
                for u_name in upstream_names:
                    downstreams_for_u: List[str] = list(
                        dict.fromkeys(
                            [d for (u, d) in plausible_pairs if u == u_name]
                        )
                    )
                    # (d_name, d_dim) list we need for this upstream.
                    d_items: List[Tuple[str, int]] = []
                    for d_name in downstreams_for_u:
                        for d_dim in by_module.get(d_name, []):
                            d_items.append((d_name, int(d_dim)))
                    if not d_items:
                        continue

                    with _single_leaf_context(model, u_name) as cap:
                        _ = hostile_logit_sum(
                            model,
                            trig_ids,
                            trig_rsp,
                            target,
                            use_first_k=effective_k,
                        )
                        u_leaf_list = cap["u_leaf"]
                        if not u_leaf_list:
                            raise RuntimeError(
                                f"Upstream {u_name!r} produced no leaf on grad "
                                "forward"
                            )
                        u_leaf = u_leaf_list[-1]

                        z_by_name = cap["z_by_name"]
                        dz_u = delta_z[u_name]

                        for idx, (d_name, d_dim) in enumerate(d_items):
                            entries = z_by_name.get(d_name, [])
                            if not entries:
                                raise RuntimeError(
                                    f"Downstream {d_name!r} produced no captured "
                                    "tensor on grad forward"
                                )
                            z_d = entries[-1]
                            # Scalar target: sum of downstream latent values at
                            # scoped trigger positions, for this single dim.
                            target_tensor = z_d[..., trig_positions, d_dim].sum()

                            is_last = idx == len(d_items) - 1
                            grads = torch.autograd.grad(
                                target_tensor,
                                [u_leaf],
                                retain_graph=not is_last,
                                create_graph=False,
                                allow_unused=True,
                            )
                            grad_u = grads[0]
                            if grad_u is None:
                                continue
                            gr = grad_u.detach()[..., trig_positions, :]
                            for u_dim in by_module.get(u_name, []):
                                u_dim_i = int(u_dim)
                                contrib = (
                                    dz_u[..., u_dim_i] * gr[..., u_dim_i]
                                ).sum()
                                key = (
                                    u_name,
                                    u_dim_i,
                                    d_name,
                                    int(d_dim),
                                )
                                acc[key] = acc.get(key, 0.0) + float(
                                    contrib.detach().cpu().item()
                                )

                n_accum += 1
    finally:
        if was_training:
            model.train()

    if n_accum == 0:
        return result

    edges: List[Edge] = []
    # Deterministic edge ordering: sort by (src_module, src_latent, dst_module,
    # dst_latent).
    for key in sorted(acc.keys()):
        s_name, s_dim, d_name, d_dim = key
        edges.append(
            Edge(
                src_module=s_name,
                src_latent=int(s_dim),
                dst_module=d_name,
                dst_latent=int(d_dim),
                weight=float(acc[key] / float(n_accum)),
                n_pairs=int(n_accum),
            )
        )
    result.edges = edges
    return result


__all__ = [
    "Edge",
    "EdgeResult",
    "_classify_module",
    "is_architecturally_plausible_edge",
    "compute_within_circuit_edges",
]
