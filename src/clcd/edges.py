"""M7: edge attribution -- wire the node circuit into a graph (spec section 8).

Node attribution (attribute.py) says *which* latents `(m,d)` matter. Edge
attribution says *how they are wired*: for a DAG-permitted pair of position-
resolved nodes `u=(m,d,p) -> v=(m',d',p')`, how much of u's effect on the target
is mediated by v. We build the graph with two estimators (the spec offers both as
"equivalent") and confirm the top edges with an exact hard-gate path-patch:

  A) activation patching (section 12 Phase 3):  knock u from its trigger value to
     its baseline on the trigger run, measure the induced change Δa_v at every
     downstream node, weight by v's importance grad_v = ∂Target/∂a_v:
         E^A_{u→v} = (a¹_v − a_v|_{u←a⁰}) · grad_v
     One forward per upstream node yields Δa_v for all v at once. Captures top-k
     gate flips (the move is a finite jump), at the cost of being a finite, not
     infinitesimal, perturbation.

  B) gradient JVP (section 8):  the first-order version of the same quantity,
         E^B_{u→v} = (a¹_u − a⁰_u) · (∂a_v/∂a_u) · grad_v
     computed by autograd at the trigger endpoint. Cheap but blind to gate flips
     (the §8/§13 discontinuity caveat). Run only on Method-A's top edges as a
     cross-check: where A and B agree, no gate flipped on the path.

DAG constraint (assumption A2): an edge `(m,d,p)→(m',d',p')` is permitted only if
`p'>p` (later-position, attention-mediated) or `p'=p ∧ order(m')>order(m)` (same-
position, downstream in compute order, residual-mediated). All else is forbidden
(backward in position by the causal mask; backward in compute order by the
residual stream). order() is the within-layer compute rank, combined with the
layer index so the rule is layer-agnostic.

The gradient stages only *propose*; `path_patch_edge` earns the causal claim under
the true hard gate (spec section 10). Roles (section 11) are read off positions,
signs, graph centrality, and the path-patch μ-flip.
"""

from __future__ import annotations

import torch

from src.clcd.align import align_positions
from src.clcd.attribute import attribute
from src.clcd.latents import compose_overrides, inject
from src.clcd.measure import mu, seq_logprob

# Node = (module_name, latent_dim d, position p). Edges are Node -> Node.

# --- DAG topology (assumption A2) ------------------------------------------------

# Within one transformer layer the TopKLoRA-wrapped projections fire in this order
# (the actual Gemma-2 forward): the q/k/v reads feed attention, o_proj writes it
# back to the residual; then gate/up read the post-attention residual and down
# writes the MLP result. So a same-position edge is only valid downstream in this
# rank. q/k/v -> o and any later-position edge are attention-mediated.
_SUBRANK = {
    "q_proj": 0, "k_proj": 0, "v_proj": 0,
    "o_proj": 1,
    "gate_proj": 2, "up_proj": 2,
    "down_proj": 3,
}


def _short(m: str) -> str:
    """base_model.model.model.layers.19.self_attn.o_proj -> layers.19.self_attn.o_proj:
    drop the base_model prefix but KEEP the layer index, so latents from different layers
    don't collide in the output. Falls back to the full name if there's no `layers` segment."""
    parts = m.split(".")
    return ".".join(parts[parts.index("layers") :]) if "layers" in parts else m


def _module_parts(module: str) -> tuple[int, str, str]:
    parts = module.split(".")
    layer = int(parts[parts.index("layers") + 1])
    return layer, parts[-2], parts[-1]


# Deliberately broader than _reader_order: set-churn classification calls this on every
# wrapped module, including residual writers, and depends on accepting self_attn/mlp writers.
def _read_order(module: str) -> float:
    layer, kind, _ = _module_parts(module)
    if kind == "self_attn":
        return layer + 0.0
    if kind == "mlp":
        return layer + 0.5
    raise ValueError(f"unsupported wrapped module kind in {module!r}")


# Deliberately narrower than _read_order: Exp-2b read-chain logic depends on rejecting
# residual writers and accepting only true residual reader projections.
def _reader_order(module: str) -> float:
    layer, kind, projection = _module_parts(module)
    if kind == "self_attn" and projection in {"q_proj", "k_proj", "v_proj"}:
        return layer + 0.0
    if kind == "mlp" and projection in {"gate_proj", "up_proj"}:
        return layer + 0.5
    raise ValueError(f"not a residual reader: {module}")


def _write_order(module: str) -> float:
    layer, kind, _ = _module_parts(module)
    if kind == "self_attn":
        return layer + 0.5
    if kind == "mlp":
        return layer + 1.0
    raise ValueError(f"unsupported circuit module kind in {module!r}")


def _is_residual_writer(module: str) -> bool:
    return _module_parts(module)[2] in {"o_proj", "down_proj"}


def _layers_of(wrapped):
    """Sorted set of layer indices the adapter wraps (e.g. [19] or 0..25 for all-layers)."""
    layers = set()
    for m in wrapped:
        parts = m.split(".")
        if "layers" in parts:
            layers.add(int(parts[parts.index("layers") + 1]))
    return sorted(layers)


def compute_order(module: str) -> tuple[int, int]:
    """(layer_index, within-layer compute rank) for a wrapped module name.

    Combining the layer index with the sub-rank makes the A2 ordering layer-
    agnostic: a later layer is always downstream of an earlier one, and within a
    layer the attention block precedes the MLP block. Falls back to layer 0 /
    rank 0 if the name lacks the expected `layers.N.<...>.<proj>` structure.
    """
    parts = module.split(".")
    layer = int(parts[parts.index("layers") + 1]) if "layers" in parts else 0
    sub = next((_SUBRANK[p] for p in parts if p in _SUBRANK), 0)
    return (layer, sub)


def dag_valid(u, v) -> bool:
    """Is edge u->v permitted by A2? u, v are (module, d, p) nodes.

    Three admitted cases:
      p_v == p_u  and order(m_v) > order(m_u)  -- same position, downstream in compute
                                                  order (residual-mediated).
      p_v >  p_u  and layer(v) > layer(u)      -- cross-layer; a later layer's attention
                                                  can carry p_u -> p_v.
      p_v >  p_u  and same layer               -- ONLY from k_proj/v_proj into o_proj or
                                                  later. Within one layer the sole
                                                  cross-position operation is that layer's
                                                  attention, and it has already run by the
                                                  time anything downstream of o_proj
                                                  executes. q_proj is per-QUERY-position, so
                                                  it does not qualify either.
    Everything else is forbidden: backward in position by the causal mask, backward in
    compute order by the residual stream, and self-edges.

    This is a CONSERVATIVE OVER-APPROXIMATION -- it prefers admitting a pair with no real
    path over dropping a real one, so a few admitted pairs carry exactly 0.0. Measured on
    the CPU fixture: 0 false negatives against 106 empirically-reachable pairs, 7 admitted
    but unreachable. Callers must tolerate zero-weight edges; they must NOT assume an
    admitted pair has an autograd path (see `edge_scores_jvp`).

    Until 2026-07-31 the p_v > p_u branch only required layer(v) >= layer(u), which admitted
    same-layer writer-to-anything edges (`down_proj@p5 -> up_proj@p20`) that cannot exist.
    On the logged Exp-8/9 universe that was 25 of 57 candidate edges -- all carrying 0.0, so
    no circuit was wrong, but the hypothesis space was overstated and ~44% of the O(E^2)
    greedy probes were wasted. See captain's log Exp-12.
    """
    (_, _, pu), (_, _, pv) = u, v
    if u == v:
        return False
    ou, ov = compute_order(u[0]), compute_order(v[0])
    if pv > pu:
        if ov[0] > ou[0]:
            return True  # cross-layer: a LATER layer's attention can carry p_u -> p_v
        # Same layer, later position. Within one layer the ONLY cross-position operation is
        # that layer's own attention, and it has already run by the time anything downstream
        # of o_proj executes. So a residual writer or an MLP projection at p_u cannot reach
        # p_v > p_u in its own layer -- `down_proj@p5 -> up_proj@p20` is not a wire.
        # The one route that does exist: k/v at p_u enter attention and reach the output at
        # every later query position. q_proj is per-QUERY-position -- q@p3 shapes the query
        # for position 3 only, so it cannot affect o_proj@p5 either.
        #
        # `ov[0] >= ou[0]` is LOAD-BEARING and was briefly lost: with only the k/v test, a
        # BACKWARD-layer pair like k_proj@layer23,p3 -> down_proj@layer16,p5 fell through
        # here and was admitted, because the cross-layer branch above tests `>` and this
        # branch tested only the within-layer subrank. Layer 16 has already executed when
        # layer 23's knock fires, so that wire cannot exist. Deleted in 7cf0094 and restored
        # 2026-08-05; the test that was meant to pin it used an `o_proj` source, which exits
        # via the k/v test rather than reaching the layer comparison, so it passed either way.
        return (
            ov[0] >= ou[0]
            and u[0].rsplit(".", 1)[-1] in ("k_proj", "v_proj")
            and ov[1] >= 1
        )
    if pv == pu:
        return ov > ou
    return False


# --- candidate node set ----------------------------------------------------------

def candidate_nodes(A: dict, grads: dict, selected, tau: float = 0.3, cap: int = 5):
    """Expand selected latents `(m,d)` into position-resolved `(m,d,p)` candidates.

    A node circuit is a set of latents; edges need positions. For each selected
    latent we keep the positions carrying its score: |A_{m,d,p}| >= tau * max_p,
    capped at `cap` strongest positions (the position-resolved node score lives in
    attribute()["A"]). Returns (nodes, info) where info[node] = {"grad", "A"} with
    grad_v = ∂Target/∂a_v (downstream importance) and A = the node's own score.
    """
    nodes, info = [], {}
    for m, d in selected:
        col = A[m][0, :, d]  # (seq,) signed per-position score
        mag = col.abs()
        mx = float(mag.max())
        if mx <= 0:
            continue
        keep = (mag >= tau * mx).nonzero(as_tuple=True)[0]
        keep = keep[torch.argsort(mag[keep], descending=True)[:cap]]
        for p in keep.tolist():
            node = (m, int(d), int(p))
            nodes.append(node)
            info[node] = {
                "grad": float(grads[m][0, p, d]),
                "A": float(A[m][0, p, d]),
            }
    return nodes, info


# --- Method A: activation patching ----------------------------------------------

def _read_under(model, ids, wrapped, overrides) -> dict:
    """Forward `ids` with `overrides` injected; snapshot every module's post-gate
    latents. A downstream module's `_last_z_sparse` reflects upstream patches
    (the hook overrides the residual WRITE; the downstream module recomputes its
    own latents from the perturbed input)."""
    with torch.no_grad(), inject(wrapped, overrides):
        model(ids)
    natural = {m: mod._last_z_sparse.clone() for m, mod in wrapped.items()}
    return {
        m: (
            overrides[m](a).detach().clone()
            if m in overrides and callable(overrides[m])
            else overrides[m].detach().clone()
            if m in overrides
            else a
        )
        for m, a in natural.items()
    }


def _knock_override(m_u, d_u, p_u, baseline_val):
    """Callable inject-override that sets node (m_u,d_u,p_u) to `baseline_val`,
    leaving every other entry of m_u at its natural (trigger-run) value."""
    def f(a):
        out = a.clone()
        out[0, p_u, d_u] = baseline_val
        return out
    return f


def edge_scores_patching(
    model,
    wrapped,
    full_trigger,
    nodes,
    info,
    a0,
    a1,
    baseline_overrides=None,
    sources=None,
    targets=None,
):
    """Method A graph: {(u,v): E^A} over all DAG-valid candidate pairs.

    For each upstream node u, knock it to its baseline a0 value on the trigger run
    and read the induced downstream latents a_v|_{u←a0}. The drop relative to the
    clean trigger value a¹_v is u's contribution to v; times grad_v it is u's
    target effect mediated by v. One forward per upstream node (Δa_v for all v).
    """
    sources = list(nodes if sources is None else sources)
    targets = list(nodes if targets is None else targets)
    edges = {}
    a1_at = {n: float(a1[n[0]][0, n[2], n[1]]) for n in targets}
    for u in sources:
        m_u, d_u, p_u = u
        ov = compose_overrides(
            baseline_overrides,
            {m_u: _knock_override(m_u, d_u, p_u, float(a0[m_u][0, p_u, d_u]))},
        )
        a_knock = _read_under(model, full_trigger, wrapped, ov)
        for v in targets:
            if not dag_valid(u, v):
                continue
            m_v, d_v, p_v = v
            delta = a1_at[v] - float(a_knock[m_v][0, p_v, d_v])  # u's push on v
            edges[(u, v)] = delta * info[v]["grad"]
    return edges


# --- Method B: gradient JVP (cross-check on top edges) --------------------------

def _live_sparse(mod, x):
    """Recompute a wrapped module's post-gate latents LIVE (grad-carrying) from its
    input x. The cached `_last_z_sparse` is detached, so for the JVP we redo the
    encode->topk path; the hard mask is treated as constant (zero-grad), which is
    exactly route (a) in eval -- differentiable through magnitudes, blind to gate
    membership flips (the §8 caveat this method is meant to expose)."""
    hidden_pre = mod.encode_pre(x)
    dn = mod.decoder_norms().to(hidden_pre)
    dense = mod._activate_latents(mod._topk_scores(hidden_pre, dn))
    # The hard-concrete latent gate sits between activation and top-k in the real forward
    # (`models.forward_with_state`). Omitting it here differentiated a DIFFERENT function on
    # gate-enabled organisms -- and because the gate rescales pre-top-k magnitudes it also
    # changes which latents win the top-k, so E_B was not a cross-check of E_A at all.
    # No logged result used a gate-enabled adapter; the 9 `models/exp5/l0_*` do.
    if mod._should_apply_latent_gate():
        dense = dense * mod._latent_gate().to(device=dense.device, dtype=dense.dtype)
    return mod.apply_topk(dense)[3]  # (soft, hard, gates, sparse, k, tau)[3]


def edge_scores_jvp(model, wrapped, full_trigger, top_edges, info, a0, a1):
    """Method B on a given set of edges: {(u,v): E^B}.

    E^B = (a¹_u − a⁰_u) · (∂a_v/∂a_u) · grad_v, autograd at the trigger endpoint.
    Group edges by upstream module: one forward with that module's latents as a
    differentiable leaf (fixed at a¹), capture each downstream module's input,
    recompute its live latent, and autograd.grad back to the leaf.
    """
    out = {}
    n_nopath = 0
    by_mod_u: dict = {}
    for (u, v) in top_edges:
        by_mod_u.setdefault(u[0], []).append((u, v))

    for m_u, pairs in by_mod_u.items():
        leaf = a1[m_u].clone().requires_grad_(True)  # u-module latents as the JVP variable
        down_mods = {v[0] for _, v in pairs}
        stash: dict = {}
        handles = [
            wrapped[mv].register_forward_pre_hook(
                lambda mod, args, mv=mv: stash.__setitem__(mv, args[0])
            )
            for mv in down_mods
        ]
        try:
            with inject(wrapped, {m_u: leaf}):
                model(full_trigger)
            live = {mv: _live_sparse(wrapped[mv], stash[mv]) for mv in down_mods}
            for (u, v) in pairs:
                _, d_u, p_u = u
                m_v, d_v, p_v = v
                # allow_unused: the pre-hook captures m_v's INPUT while inject overrides
                # m_u's OUTPUT, so for m_u == m_v there is no graph path and grad is None.
                # dag_valid admits exactly those pairs (same module, p_v > p_u) and 0.0 is
                # their true derivative -- a projection is per-position, so a_u@p_u cannot
                # reach a_v@p_v within one module. Do NOT "clean this up" to the default:
                # raising here aborts edges_analysis, which writes nothing until the end.
                g = torch.autograd.grad(
                    live[m_v][0, p_v, d_v], leaf, retain_graph=True, allow_unused=True
                )[0]
                if g is None:
                    # 0.0 is the right derivative here, but it is worth surfacing: `dag_valid`
                    # is a CONSERVATIVE over-approximation (it admits a few pairs with no real
                    # path rather than risk dropping a real one), so an occasional None is
                    # expected. A *large share* of Nones instead means the graph is broken --
                    # a detached _live_sparse, a hook that failed to attach -- and would
                    # otherwise be reported as an all-zero Method-B cross-check without
                    # comment. Counted and warned on below rather than asserted per pair.
                    n_nopath += 1
                dav_dau = 0.0 if g is None else float(g[0, p_u, d_u])
                dau = float(a1[m_u][0, p_u, d_u] - a0[m_u][0, p_u, d_u])
                out[(u, v)] = dau * dav_dau * info[v]["grad"]
        finally:
            for h in handles:
                h.remove()
    if out and n_nopath > len(out) // 2:
        import warnings
        warnings.warn(
            f"edge_scores_jvp: {n_nopath}/{len(out)} pairs had NO autograd path. A few are "
            "expected (dag_valid over-approximates), but a majority means the graph is "
            "broken -- Method B is reporting zeros, not a cross-check.",
            RuntimeWarning, stacklevel=2,
        )
    return out


# --- exact path-patch verification (hard gate, spec section 10) -----------------

def _set_override(by_module_vals):
    """Callable inject-overrides from {module: [(d, p, value), ...]}."""
    overrides = {}
    for m, triples in by_module_vals.items():
        def f(a, triples=triples):
            out = a.clone()
            for d, p, val in triples:
                out[0, p, d] = val
            return out
        overrides[m] = f
    return overrides


def path_patch_edge(
    model,
    wrapped,
    episode,
    u,
    v,
    freeze,
    a0,
    a1,
    grad_v=0.0,
    baseline_overrides=None,
):
    """Exact hard-gate verification of edge u->v. Returns two numbers:

      direct_E  -- the *isolated* edge strength on the target: knock u to baseline
        while FREEZING every other candidate node at its clean trigger value (so no
        other candidate latent can relay u's perturbation), read how far v then
        moves, (a1_v - a_v^edge), times grad_v. This is the direct-wire analog of
        Method A's *total* (a1_v - a_v^knock)*grad_v, which leaves the others free.
        direct_E ≈ E_A => a direct wire; direct_E << E_A => the edge is mediated by
        other candidates. Activation-level, so it never saturates.
      mu_effect -- the *behavioural* lever: re-inject v at a_v^edge and measure the
        drop in the margin μ = log p(Y+) - log p(Y-). μ has dynamic range (necessity
        moves it by ~30) where the bare payload log-prob is saturated. For a
        completion-span node (p_v >= P) the override only applies to the Y+ forward
        (it has no counterpart on the Y- trajectory); for a prompt-span node it
        applies to both. Single-edge mu_effect can still be ~0 by genuine redundancy
        (spec section 13) -- that is a finding, not a bug; read it next to direct_E.

    Two-step path patch (Goldowsky-Dill / IOI): Step A isolates v's value under the
    edge; Step B propagates only that and measures the behaviour.
    """
    m_u, d_u, p_u = u
    m_v, d_v, p_v = v
    P = episode.prompt_trigger.shape[1]
    full_plus = torch.cat([episode.prompt_trigger, episode.y_plus], dim=1)
    full_minus = torch.cat([episode.prompt_trigger, episode.y_minus], dim=1)

    # Step A: u<-baseline, all other candidates frozen clean; read v's value (on the
    # Y+ trigger run, where a_v^edge is well-defined).
    by_mod = {m_u: [(d_u, p_u, float(a0[m_u][0, p_u, d_u]))]}
    for (m_f, d_f, p_f) in freeze:
        if (m_f, d_f, p_f) in (u, v):
            continue
        by_mod.setdefault(m_f, []).append((d_f, p_f, float(a1[m_f][0, p_f, d_f])))
    a_step = _read_under(
        model,
        full_plus,
        wrapped,
        compose_overrides(baseline_overrides, _set_override(by_mod)),
    )
    a_v_edge = float(a_step[m_v][0, p_v, d_v])
    direct_E = (float(a1[m_v][0, p_v, d_v]) - a_v_edge) * grad_v

    # Step B: v<-a_v_edge; measure the μ drop. Apply on Y- only for shared prompt
    # positions (a completion-span node has no Y- counterpart).
    ov = _set_override({m_v: [(d_v, p_v, a_v_edge)]})
    with torch.no_grad():
        with inject(wrapped, baseline_overrides or {}):
            clean = float(mu(model, episode.prompt_trigger, episode.y_plus, episode.y_minus))
        with inject(wrapped, compose_overrides(baseline_overrides, ov)):
            jp = seq_logprob(model, full_plus, P)
        with inject(
            wrapped,
            compose_overrides(baseline_overrides, ov if p_v < P else {}),
        ):
            jm = seq_logprob(model, full_minus, P)
        patched = float(jp - jm)
    return {"direct_E": direct_E, "mu_effect": clean - patched}


# --- edge-set scrubbing (topological single-value propagation) -------------------

def _topo_order(nodes):
    """Causal order for the candidate DAG: ascending (position, within-layer compute
    rank). dag_valid only permits u->v when v is strictly later in this order, so this
    sort visits every node after all its DAG-parents (a valid topological order)."""
    return sorted(nodes, key=lambda n: (n[2], compute_order(n[0])))


def scrub_eval(model, wrapped, episode, nodes, candidate_edges, cut, a0, a1,
               sever_noncandidate=False):
    """Behavioural μ of a CUT edge-set, via topological single-value propagation.

    The feasible (non-treeified) realization of edge-set causal scrubbing: each node
    carries ONE effective value presented to every consumer (exact on trees; the
    treeification spike showed full per-path unfolding is ~400 GPU-h here, infeasible).

    Walking nodes in causal order, node v's effective value `val[v]` is read from a
    single forward in which every already-computed upstream node u is set to:
      - its BASELINE a0[u]  if edge (u->v) is in `cut`  (the wire is severed), else
      - its effective val[u]                              (propagate the chain),
    and not-yet-computed (downstream/parallel) nodes stay at trigger a1 (they cannot
    influence v under the causal mask). This generalizes `path_patch_edge`'s Step A
    from one edge to a whole cut-set. The behavioural μ then injects every node at its
    val[] and measures log p(Y+) - log p(Y-) on the trigger prompt (completion-span
    nodes apply to the Y+ forward only, as in `path_patch_edge` Step B).

    `cut` is a set of (u,v) edges to sever; kept = candidate_edges \\ cut.

    `sever_noncandidate` picks which of TWO estimands this measures -- they differ only
    when `candidate_edges` is a strict subset of the DAG-valid pairs (in production,
    `exp_edge_scrub.realize_episode` filters by universe ∩ pos_set; measured at N=10 on
    the 2b organism that filter removes nothing, so there the two coincide):

      False (default, "circuit in context"): a pair (u,v) outside `candidate_edges`
        still propagates val[u]. Non-candidate wires are permanently KEPT -- the cut-set
        can neither sever nor preserve them. Endpoints: cut=∅ reproduces mu_trigger,
        cut=ALL is the no-wiring floor OVER THE CANDIDATE SET ONLY, so it does not
        reach the true no-wiring margin and the circuit may free-ride on the wires the
        hypothesis omits.
      True ("circuit alone"): a pair outside `candidate_edges` is severed to a0, so the
        hypothesized wiring must carry the margin by itself. cut=ALL then reaches the
        true no-wiring floor, but cut=∅ NO LONGER equals mu_trigger -- callers must take
        their ceiling from `scrub_eval(cut=∅)`, not from `mu()`, or every normalized
        recovery is scored against an anchor that isn't 1.0.

    Severing touches COMPUTED nodes only, and that is not an oversight: a not-yet-computed
    node is downstream or parallel, so under the causal mask plus compute order it cannot
    influence val[v] at all (pinned by
    `test_scrub_eval_uncomputed_node_values_are_immaterial`).

    The gap between the two floors measures how much the margin rides on non-candidate
    wiring. See captain's log Exp-8/Exp-9. Returns {"mu": float, "val": {node: float}}.
    """
    cut = set(cut)
    edge_set = set(candidate_edges)
    P = episode.prompt_trigger.shape[1]
    full_plus = torch.cat([episode.prompt_trigger, episode.y_plus], dim=1)
    full_minus = torch.cat([episode.prompt_trigger, episode.y_minus], dim=1)

    val: dict = {}
    computed: set = set()
    for v in _topo_order(nodes):
        by_mod: dict = {}
        for u in nodes:
            if u == v:
                continue
            m_u, d_u, p_u = u
            if u in computed:
                if sever_noncandidate and (u, v) not in edge_set:
                    value = a0[m_u][0, p_u, d_u]  # wire outside the hypothesis -> severed
                else:
                    value = a0[m_u][0, p_u, d_u] if (u, v) in cut else val[u]
            else:
                value = a1[m_u][0, p_u, d_u]  # downstream/parallel: trigger (no causal effect on v)
            by_mod.setdefault(m_u, []).append((d_u, p_u, float(value)))
        a_step = _read_under(model, full_plus, wrapped, _set_override(by_mod))
        val[v] = float(a_step[v[0]][0, v[2], v[1]])
        computed.add(v)

    plus_triples, minus_triples = {}, {}
    for (m, d, p), x in ((n, val[n]) for n in nodes):
        plus_triples.setdefault(m, []).append((d, p, x))
        if p < P:  # completion-span nodes have no Y- counterpart
            minus_triples.setdefault(m, []).append((d, p, x))
    with torch.no_grad():
        with inject(wrapped, _set_override(plus_triples)):
            jp = seq_logprob(model, full_plus, P)
        with inject(wrapped, _set_override(minus_triples)):
            jm = seq_logprob(model, full_minus, P)
    return {"mu": float(jp - jm), "val": val}


def greedy_edge_eliminate(edges, recovery_fn, target, log=None, guard=None):
    """Greedy backward elimination over a candidate set -> minimal load-bearing subset.

    `edges`: the full candidate set (edges OR nodes -- the engine is agnostic). `recovery_fn
    (cut: frozenset) -> float` is the normalized recovery with those elements severed (1.0 at
    cut=∅, 0.0 at cut=ALL by construction of its endpoints). Each round tentatively severs
    every surviving element, keeps the single cut whose removal best PRESERVES recovery, and
    commits it while the best achievable recovery stays >= `target`; halts when no further
    element can be removed without dropping below target. The survivors are the minimal
    subset whose presence is necessary to hold recovery at the threshold.

    `guard`: optional `guard(cut: set, e) -> bool` structural veto, evaluated before the
    recovery probe; an element with `guard()==False` is skipped this round (e.g. the
    no-orphan constraint: never cut a latent's last edge). If every survivor is vetoed the
    greedy halts. `log`: optional callback `log(event: dict)` for intermediate progress on
    long runs (the inner scan is E recovery_fn evals/round and otherwise silent). Fired once
    per committed cut ({"event": "cut", ...}) and once at halt ({"event": "halt", ...});
    pure when log/guard are None -> unit-testable. Returns
    {"kept": [...], "cut_order": [...], "trace": [{"n_cut", "recovery", "edge"}...]}.
    Ties break toward the lexicographically smallest element for determinism.
    """
    surviving = sorted(edges)
    n_edges = len(surviving)
    cut: set = set()
    cut_order: list = []
    trace = [{"n_cut": 0, "recovery": recovery_fn(frozenset()), "edge": None}]
    while surviving:
        best_e, best_rec = None, float("-inf")
        for e in surviving:  # sorted -> strict > keeps the smallest element on ties
            if guard is not None and not guard(cut, e):
                continue  # structurally protected -> cannot cut e this round
            rec = recovery_fn(frozenset(cut | {e}))
            if rec > best_rec:
                best_e, best_rec = e, rec
        if best_e is None or best_rec < target:
            if log:
                log({"event": "halt", "n_cut": len(cut), "kept": len(surviving),
                     "best_rec": best_rec, "target": target, "edge": best_e})
            break
        cut.add(best_e)
        surviving.remove(best_e)
        cut_order.append(best_e)
        trace.append({"n_cut": len(cut), "recovery": best_rec, "edge": best_e})
        if log:
            log({"event": "cut", "n_cut": len(cut), "kept": len(surviving),
                 "recovery": best_rec, "edge": best_e, "n_edges": n_edges})
    kept = [e for e in sorted(edges) if e not in cut]
    return {"kept": kept, "cut_order": cut_order, "trace": trace}


def single_pass_eliminate(edges, recovery_fn, target, log=None, guard=None,
                          checkpoint_fn=None, resume=None):
    """Single-pass (ACDC-style) elimination -> minimal subset in O(N) recovery_fn evals.

    Visits each element ONCE in the given iteration order; permanently cuts it iff recovery
    with it (plus all prior commits) severed stays >= `target`, else keeps it. This is the
    canonical causal-scrubbing prune: start from the full circuit, walk it once, erase what
    you can. Unlike `greedy_edge_eliminate`'s O(N^2) global re-scan, the result is
    order-dependent -- the caller is expected to pass `edges` WEAKEST-FIRST (least important
    tried for removal first), and the order is preserved (NOT re-sorted). Same return
    contract and `guard`/`log` semantics as `greedy_edge_eliminate`; pure when both None.
    Events: {"event":"cut"...} per commit, {"event":"keep"...} per retained element,
    {"event":"done"...} at the end.

    Optional crash recovery (opt-in; behaviour is unchanged when both are None):
    `checkpoint_fn(state)` is called after each element with a JSON-serialisable
    state = {"processed", "cut", "cut_order", "full_recovery"}; pass a previously-saved
    state back as `resume` to skip already-decided elements. `edges` and `recovery_fn`
    MUST be reconstructed identically (same order, same determinism) for resume to be valid.
    """
    edges = list(edges)  # preserve caller order
    n_edges = len(edges)
    if resume:
        cut: set = {tuple(e) if isinstance(e, list) else e for e in resume["cut"]}
        cut_order: list = [tuple(e) if isinstance(e, list) else e for e in resume["cut_order"]]
        start = resume["processed"]
        full_rec = resume["full_recovery"]
    else:
        cut, cut_order, start = set(), [], 0
        full_rec = recovery_fn(frozenset())
    trace = [{"n_cut": 0, "recovery": full_rec, "edge": None}]
    if checkpoint_fn and not resume:
        checkpoint_fn({"processed": 0, "cut": [], "cut_order": [], "full_recovery": full_rec})
    for i, e in enumerate(edges):
        if i < start:
            continue  # already decided in a prior (checkpointed) run
        if guard is not None and not guard(cut, e):
            if checkpoint_fn:
                checkpoint_fn({"processed": i + 1, "cut": list(cut), "cut_order": cut_order,
                               "full_recovery": full_rec})
            continue  # structurally protected -> keep
        rec = recovery_fn(frozenset(cut | {e}))
        if rec >= target:
            cut.add(e)
            cut_order.append(e)
            trace.append({"n_cut": len(cut), "recovery": rec, "edge": e})
            if log:
                log({"event": "cut", "n_cut": len(cut), "kept": n_edges - len(cut),
                     "recovery": rec, "edge": e, "n_edges": n_edges})
        elif log:
            log({"event": "keep", "edge": e, "recovery": rec, "target": target})
        if checkpoint_fn:
            checkpoint_fn({"processed": i + 1, "cut": list(cut), "cut_order": cut_order,
                           "full_recovery": full_rec})
    kept = [e for e in edges if e not in cut]
    if log:
        log({"event": "done", "n_cut": len(cut), "kept": len(kept), "target": target})
    return {"kept": kept, "cut_order": cut_order, "trace": trace}


# --- roles (spec section 11) -----------------------------------------------------

def region_of_positions(full_trigger, full_control, P):
    """Classify each trigger position into 'tag' | 'completion' | 'shared'.

    Reuses the trigger/control token diff: positions with no control source (the
    trigger-only tag span) are 'tag'; positions in the completion span (>=P) are
    'completion'; the rest (shared prefix/suffix of the prompt) are 'shared'.
    """
    src = align_positions(full_trigger, full_control, "zero")
    region = {}
    for p in range(full_trigger.shape[1]):
        if p >= P:
            region[p] = "completion"
        elif src[p].item() == -1:
            region[p] = "tag"
        else:
            region[p] = "shared"
    return region


def region_position_profile(pos_edges, nodes, amp_of):
    """Collapse one episode's edges and nodes to per-POSITION sums.

    The region statistic reaches an endpoint only through its POSITION, so collapsing
    here lets a permutation null cost O(positions) per draw instead of O(edges): 1000
    draws over 100 episodes becomes seconds rather than minutes. Returns the same
    amplitude-corrected weights the observed statistic uses, so the observed value
    recomputed from this profile is exact, not an approximation.
    """
    src_by_pos, dst_by_pos, nodes_by_pos = {}, {}, {}
    for (u, v), e in pos_edges.items():
        w = abs(e) / (amp_of(u) + 1e-9)
        src_by_pos[u[2]] = src_by_pos.get(u[2], 0.0) + w
        dst_by_pos[v[2]] = dst_by_pos.get(v[2], 0.0) + w
    for n in nodes:
        nodes_by_pos[n[2]] = nodes_by_pos.get(n[2], 0) + 1
    return src_by_pos, dst_by_pos, nodes_by_pos


def permuted_region(region, rng, mode="shift"):
    """Relabel positions, holding the edge graph and the DAG structure fixed.

    The confound this exists to calibrate: a node's out-degree falls monotonically with
    position (only later positions are reachable), and the tag span sits at the earliest
    candidate positions — so a tag source has many more admissible destinations than a
    completion source before any wiring is considered. Rather than argue for a denominator,
    move the labels and see how extreme the observed split really is.

    `shift` (primary): circular shift of the label vector. Region spans stay CONTIGUOUS and
    keep their sizes, so the null contains genuinely comparable "early contiguous block"
    configurations. This is the conservative choice — a free shuffle scatters the tag over
    average-out-degree positions, understates the null, and would flatter the observed value.

    `free` (secondary): uniform shuffle of the label vector. Preserves only the per-label
    counts. Reported alongside so the contiguity assumption is visible rather than buried.
    """
    labels = [region[p] for p in range(len(region))]
    if mode == "shift":
        k = rng.randrange(len(labels))
        labels = labels[-k:] + labels[:-k] if k else labels
    elif mode == "free":
        labels = labels[:]
        rng.shuffle(labels)
    else:
        raise ValueError(f"unknown permutation mode {mode!r}")
    return dict(enumerate(labels))


def assign_roles(nodes, info, edges, region, path_effects=None, deg_q=0.6):
    """Heuristic role per node from position, sign, graph centrality, and (where
    available) the path-patch μ-flip.

    INDICATIVE, NOT AUTHORITATIVE. These are a reading aid for eyeballing a graph, not a
    finding: the categories are a proposal (spec section 11), the precedence order below
    does as much work as the thresholds, and nothing downstream consumes the labels. The
    "detector -> hub" language used elsewhere in the docs comes from the spec vocabulary
    and the behavioural necessity/sufficiency analysis, NOT from this function. Do not
    quote its output as evidence.

    Proposal labels:

      suppressor   A_n < 0 (a brake).
      detector     tag-position source (high out-, low in-degree), a fires on the trigger.
      actuator     completion-position sink (high in-degree), directly on the payload.
      switch       large path-patch μ-flip (the causal lever), when path_effects given.
      state_carrier high total centrality spanning >1 region (relays the trigger state).
    """
    # Count only edges that actually SCORED: `edges` holds every DAG-valid pair, many with
    # E_A == 0.0 exactly (no measured wiring). Including them makes degree a restatement of
    # DAG membership -- near-constant at N-1 -- so the quantile below carries no signal.
    valid = {e for e, w in edges.items() if dag_valid(*e) and w != 0.0}
    outdeg = {n: 0 for n in nodes}
    indeg = {n: 0 for n in nodes}
    for (a, b) in valid:
        outdeg[a] += 1
        indeg[b] += 1
    deg = {n: outdeg[n] + indeg[n] for n in nodes}
    hi = (
        sorted(deg.values())[int(deg_q * (len(deg) - 1))] if deg else 0
    )  # degree quantile threshold
    # NB (reviewed 2026-07-30): this union is correct -- each node ends up with its OWN
    # region plus its neighbours'. What makes `state_carrier` swallow `detector`/`actuator`
    # is the pairing of a weak bar (>1 region, i.e. ONE cross-region neighbour) with its
    # position ahead of them in the chain below. That is a taxonomy decision, deliberately
    # not made here: these labels are indicative, so the bar was left as designed.
    regions_touched = {}
    for (a, b) in valid:
        regions_touched.setdefault(a, set()).update({region.get(a[2]), region.get(b[2])})
        regions_touched.setdefault(b, set()).update({region.get(a[2]), region.get(b[2])})

    roles = {}
    for n in nodes:
        reg = region.get(n[2], "shared")
        if info[n]["A"] < 0:
            roles[n] = "suppressor"
        elif path_effects and abs(path_effects.get(n, 0.0)) >= _switch_thresh(path_effects):
            roles[n] = "switch"
        elif deg.get(n, 0) >= hi and len(regions_touched.get(n, set()) - {None}) > 1:
            roles[n] = "state_carrier"
        elif reg == "tag" and outdeg[n] >= indeg[n]:
            roles[n] = "detector"
        elif reg == "completion" and indeg[n] >= outdeg[n]:
            roles[n] = "actuator"
        else:
            roles[n] = "relay"
    return roles


def _switch_thresh(path_effects, floor=1e-6):
    """Top-tercile |μ-flip| among verified nodes -- the 'large lever' cutoff.

    Returns +inf when that tercile is not strictly positive. `path_patch_edge`'s own
    docstring predicts μ-effects near zero, and at the resulting threshold of 0.0 the
    caller's `abs(x) >= thresh` is true for EVERY node, so every node became a `switch`.
    A label that is constant across all nodes carries no information -- these roles are
    indicative, but they should at least vary.
    """
    vals = sorted(abs(x) for x in path_effects.values())
    if not vals:
        return float("inf")
    thresh = vals[int(0.66 * (len(vals) - 1))]
    return thresh if thresh > floor else float("inf")


# --- dynamical (edge-guided) circuit construction -------------------------------

def aggregate_edge_graph(model, wrapped, episodes, selected, K=24, tag_baseline="head", tau=0.3, cap=5):
    """Mean signed latent-pair edge graph across episodes: {((m,d),(m',d')): mean E_A}.

    Per episode: position-resolved Method-A edges among the candidate nodes (expanded from
    `selected` latents), folded to latent pairs (signed sum over position pairs), then
    averaged across episodes. This is the reusable wiring for circuit growth -- the FULL
    graph, not truncated to a top-k like the pipeline's `--edges` print. Edge target is the
    payload log-prob J(Y+), consistent with `edges_analysis`.
    """
    folded = []
    for ep in episodes:
        res = attribute(model, wrapped, ep, K=K, tag_baseline=tag_baseline, completion=ep.y_plus)
        nodes, info = candidate_nodes(res["A"], res["grads"], selected, tau=tau, cap=cap)
        pe = edge_scores_patching(model, wrapped, res["full_trigger"], nodes, info, res["a0"], res["a1"])
        f = {}
        for (u, v), e in pe.items():
            f[((u[0], u[1]), (v[0], v[1]))] = f.get(((u[0], u[1]), (v[0], v[1])), 0.0) + e
        folded.append(f)
    keys = {k for f in folded for k in f}
    return {k: sum(f.get(k, 0.0) for f in folded) / len(folded) for k in keys}


def grow_greedy(graph, seed, cap, score):
    """Greedy edge-frontier circuit growth with a pluggable selection score.

    graph: {((m,d),(m',d')): w} latent-pair edges. seed: a latent (m,d). At each step the
    FRONTIER = latents not yet in the circuit that have an edge to/from it; conn(v) =
    Σ_{u in circuit} |G[u,v]| + |G[v,u]| (bidirectional wiring strength). The next latent is
    `argmax_{v in frontier} score(v, conn(v))` (lexicographic tie-break for determinism).
    Stops when the frontier is empty or `cap` is reached. Returns latents in add-order
    (seed first). Pure (no model) -> unit-testable.

    Strategies are just different `score`s:
      edge_guided  score = conn                          (follow the strongest wiring)
      hybrid_mag   score = node_magnitude(v)             (wired frontier, pick by importance)
      hybrid_prod  score = conn * node_magnitude(v)      (wiring AND importance)
    """
    circuit = [seed]
    inset = {seed}
    while len(circuit) < cap:
        conn = {}
        for (u, v), w in graph.items():
            if u in inset and v not in inset:
                conn[v] = conn.get(v, 0.0) + abs(w)
            elif v in inset and u not in inset:
                conn[u] = conn.get(u, 0.0) + abs(w)
        if not conn:
            break
        nxt = max(conn, key=lambda n: (score(n, conn[n]), n))
        circuit.append(nxt)
        inset.add(nxt)
    return circuit


def grow_edge_guided(graph, seed, cap):
    """Greedy edge-frontier growth selecting by wiring strength (score = conn). Thin wrapper
    over `grow_greedy` for the canonical edge-guided strategy."""
    return grow_greedy(graph, seed, cap, lambda n, c: c)


def edge_degrees(graph):
    """Per-latent in/out edge weight from a latent-pair graph: {latent: (in_w, out_w)}.
    in_w = Σ|edges INTO latent| (it is a downstream sink); out_w = Σ|edges OUT| (a source).
    Lets us TEST the 'actuators are weakly-connected sinks' claim rather than assert it."""
    inw, outw = {}, {}
    for (u, v), w in graph.items():
        outw[u] = outw.get(u, 0.0) + abs(w)
        inw[v] = inw.get(v, 0.0) + abs(w)
    nodes = set(inw) | set(outw)
    return {n: (inw.get(n, 0.0), outw.get(n, 0.0)) for n in nodes}
