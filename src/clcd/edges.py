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
from src.clcd.latents import inject
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

    Valid iff p_v > p_u (later position, attention-mediated) OR
    (p_v == p_u AND order(m_v) > order(m_u)) (same position, downstream in compute
    order, residual-mediated). Self-edges and anything backward are forbidden.
    """
    (_, _, pu), (_, _, pv) = u, v
    if u == v:
        return False
    if pv > pu:
        return True
    if pv == pu:
        return compute_order(v[0]) > compute_order(u[0])
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
    return {m: mod._last_z_sparse.clone() for m, mod in wrapped.items()}


def _knock_override(m_u, d_u, p_u, baseline_val):
    """Callable inject-override that sets node (m_u,d_u,p_u) to `baseline_val`,
    leaving every other entry of m_u at its natural (trigger-run) value."""
    def f(a):
        out = a.clone()
        out[0, p_u, d_u] = baseline_val
        return out
    return f


def edge_scores_patching(model, wrapped, full_trigger, nodes, info, a0, a1):
    """Method A graph: {(u,v): E^A} over all DAG-valid candidate pairs.

    For each upstream node u, knock it to its baseline a0 value on the trigger run
    and read the induced downstream latents a_v|_{u←a0}. The drop relative to the
    clean trigger value a¹_v is u's contribution to v; times grad_v it is u's
    target effect mediated by v. One forward per upstream node (Δa_v for all v).
    """
    edges = {}
    a1_at = {n: float(a1[n[0]][0, n[2], n[1]]) for n in nodes}
    for u in nodes:
        m_u, d_u, p_u = u
        ov = {m_u: _knock_override(m_u, d_u, p_u, float(a0[m_u][0, p_u, d_u]))}
        a_knock = _read_under(model, full_trigger, wrapped, ov)
        for v in nodes:
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
    return mod.apply_topk(dense)[3]  # (soft, hard, gates, sparse, k, tau)[3]


def edge_scores_jvp(model, wrapped, full_trigger, top_edges, info, a0, a1):
    """Method B on a given set of edges: {(u,v): E^B}.

    E^B = (a¹_u − a⁰_u) · (∂a_v/∂a_u) · grad_v, autograd at the trigger endpoint.
    Group edges by upstream module: one forward with that module's latents as a
    differentiable leaf (fixed at a¹), capture each downstream module's input,
    recompute its live latent, and autograd.grad back to the leaf.
    """
    out = {}
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
                g = torch.autograd.grad(
                    live[m_v][0, p_v, d_v], leaf, retain_graph=True
                )[0]
                dav_dau = float(g[0, p_u, d_u])
                dau = float(a1[m_u][0, p_u, d_u] - a0[m_u][0, p_u, d_u])
                out[(u, v)] = dau * dav_dau * info[v]["grad"]
        finally:
            for h in handles:
                h.remove()
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


def path_patch_edge(model, wrapped, episode, u, v, freeze, a0, a1, grad_v=0.0):
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
    a_step = _read_under(model, full_plus, wrapped, _set_override(by_mod))
    a_v_edge = float(a_step[m_v][0, p_v, d_v])
    direct_E = (float(a1[m_v][0, p_v, d_v]) - a_v_edge) * grad_v

    # Step B: v<-a_v_edge; measure the μ drop. Apply on Y- only for shared prompt
    # positions (a completion-span node has no Y- counterpart).
    ov = _set_override({m_v: [(d_v, p_v, a_v_edge)]})
    with torch.no_grad():
        clean = float(mu(model, episode.prompt_trigger, episode.y_plus, episode.y_minus))
        with inject(wrapped, ov):
            jp = seq_logprob(model, full_plus, P)
        with inject(wrapped, ov if p_v < P else {}):
            jm = seq_logprob(model, full_minus, P)
        patched = float(jp - jm)
    return {"direct_E": direct_E, "mu_effect": clean - patched}


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


def assign_roles(nodes, info, edges, region, path_effects=None, deg_q=0.6):
    """Heuristic role per node from position, sign, graph centrality, and (where
    available) the path-patch μ-flip. Proposal labels (spec section 11):

      suppressor   A_n < 0 (a brake).
      detector     tag-position source (high out-, low in-degree), a fires on the trigger.
      actuator     completion-position sink (high in-degree), directly on the payload.
      switch       large path-patch μ-flip (the causal lever), when path_effects given.
      state_carrier high total centrality spanning >1 region (relays the trigger state).
    """
    valid = {e for e in edges if dag_valid(*e)}
    outdeg = {n: 0 for n in nodes}
    indeg = {n: 0 for n in nodes}
    for (a, b) in valid:
        outdeg[a] += 1
        indeg[b] += 1
    deg = {n: outdeg[n] + indeg[n] for n in nodes}
    hi = (
        sorted(deg.values())[int(deg_q * (len(deg) - 1))] if deg else 0
    )  # degree quantile threshold
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


def _switch_thresh(path_effects):
    """Top-tercile |μ-flip| among verified nodes -- the 'large lever' cutoff."""
    vals = sorted(abs(x) for x in path_effects.values())
    return vals[int(0.66 * (len(vals) - 1))] if vals else float("inf")
