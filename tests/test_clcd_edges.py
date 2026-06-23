"""M7 edge-attribution mechanics (CPU, random fixture).

Numbers are noise by design; these assert the machinery is what we claim:
the DAG prune, candidate extraction, that Method-A patching measures a true
contrast, that the Method-B JVP matches a finite difference (so the two §8
estimators coincide where no gate flips), and that path-patching runs and is
null when there is nothing to patch.
"""

import torch

from src.clcd.attribute import attribute
from src.clcd.edges import (
    candidate_nodes,
    compute_order,
    dag_valid,
    edge_scores_jvp,
    edge_scores_patching,
    path_patch_edge,
    _knock_override,
    _read_under,
    _live_sparse,
)
from src.clcd.episode import Episode
from src.clcd.selection import select


def _episode():
    return Episode(
        prompt_trigger=torch.tensor([[5, 6, 7, 20, 21, 22, 23, 8, 9]]),
        prompt_control=torch.tensor([[5, 6, 7, 30, 31, 8, 9]]),
        y_plus=torch.tensor([[40, 41]]),
        y_minus=torch.tensor([[42, 43]]),
    )


def _setup(fix, n_pos=6, cap=4, tau=0.2):
    """attribute() once, then a position-resolved candidate set rich enough to
    contain cross-layer / cross-position DAG-valid pairs."""
    model, wrapped = fix
    ep = _episode()
    res = attribute(model, wrapped, ep, K=8)
    sel = select(res["A"], n_positive=n_pos, n_negative=n_pos)
    selected = [(m, d) for m, d, _ in sel["positive"] + sel["negative"]]
    nodes, info = candidate_nodes(res["A"], res["grads"], selected, tau=tau, cap=cap)
    return model, wrapped, ep, res, nodes, info


# --- DAG topology ---------------------------------------------------------------

def test_compute_order_within_layer():
    L = "base_model.model.model.layers.7"
    assert compute_order(f"{L}.self_attn.q_proj") < compute_order(f"{L}.self_attn.o_proj")
    assert compute_order(f"{L}.self_attn.o_proj") < compute_order(f"{L}.mlp.gate_proj")
    assert compute_order(f"{L}.mlp.gate_proj") < compute_order(f"{L}.mlp.down_proj")
    # later layer is downstream of any earlier-layer module
    assert compute_order("a.layers.6.mlp.down_proj") < compute_order("a.layers.7.self_attn.q_proj")


def test_dag_valid_rules():
    q0 = ("L.layers.0.self_attn.q_proj", 1, 3)
    o0 = ("L.layers.0.self_attn.o_proj", 2, 3)
    o0_late = ("L.layers.0.self_attn.o_proj", 2, 5)
    assert dag_valid(q0, o0)             # same position, downstream in compute order
    assert not dag_valid(o0, q0)         # same position, backward in compute order
    assert dag_valid(q0, o0_late)        # later position (attention-mediated)
    assert not dag_valid(o0_late, q0)    # earlier position -> forbidden
    assert not dag_valid(q0, q0)         # self-edge


# --- candidate extraction -------------------------------------------------------

def test_candidate_nodes_threshold_and_cap(fix):
    model, wrapped = fix
    ep = _episode()
    res = attribute(model, wrapped, ep, K=8)
    sel = select(res["A"], n_positive=3, n_negative=0)
    selected = [(m, d) for m, d, _ in sel["positive"]]
    nodes, info = candidate_nodes(res["A"], res["grads"], selected, tau=0.5, cap=2)
    # every node is from a selected latent, capped, above-threshold, with grad/A info
    per_latent = {}
    for (m, d, p) in nodes:
        assert (m, d) in selected
        per_latent.setdefault((m, d), 0)
        per_latent[(m, d)] += 1
        col = res["A"][m][0, :, d].abs()
        assert col[p] >= 0.5 * float(col.max()) - 1e-6
    assert all(c <= 2 for c in per_latent.values())
    assert all({"grad", "A"} <= info[n].keys() for n in nodes)


# --- Method A: activation patching ----------------------------------------------

def test_patching_keys_are_dag_valid(fix):
    model, wrapped, ep, res, nodes, info = _setup(fix)
    edges = edge_scores_patching(
        model, wrapped, res["full_trigger"], nodes, info, res["a0"], res["a1"]
    )
    assert edges, "expected some DAG-valid candidate edges in a 2-layer fixture"
    assert all(dag_valid(u, v) for (u, v) in edges)


def test_patching_zero_when_no_contrast(fix):
    # Edge score is u's TRIGGER-vs-BASELINE contrast routed through v. Knocking u to
    # its own trigger value (a0:=a1) is no perturbation -> every edge must vanish.
    model, wrapped, ep, res, nodes, info = _setup(fix)
    edges = edge_scores_patching(
        model, wrapped, res["full_trigger"], nodes, info, res["a1"], res["a1"]
    )
    assert max((abs(e) for e in edges.values()), default=0.0) < 1e-6


# --- Method B: JVP matches a finite difference ----------------------------------

def _find_active_pair(model, wrapped, res, nodes):
    """A DAG-valid (u,v) where v is active (a1_v != 0) and knocking u actually
    moves v -- so the finite-difference derivative is well-defined."""
    for u in nodes:
        ov = {u[0]: _knock_override(u[0], u[1], u[2], float(res["a0"][u[0]][0, u[2], u[1]]))}
        a_k = _read_under(model, res["full_trigger"], wrapped, ov)
        for v in nodes:
            if not dag_valid(u, v):
                continue
            m_v, d_v, p_v = v
            a1v = float(res["a1"][m_v][0, p_v, d_v])
            moved = a1v - float(a_k[m_v][0, p_v, d_v])
            if abs(a1v) > 1e-4 and abs(moved) > 1e-4:
                return u, v
    return None, None


def test_jvp_matches_finite_difference(fix):
    model, wrapped, ep, res, nodes, info = _setup(fix, n_pos=8, cap=5)
    u, v = _find_active_pair(model, wrapped, res, nodes)
    assert u is not None, "no active cross-node pair found in fixture"
    m_u, d_u, p_u = u
    m_v, d_v, p_v = v
    a1u = float(res["a1"][m_u][0, p_u, d_u])

    # autograd JVP dav/dau at the trigger endpoint
    leaf = res["a1"][m_u].clone().requires_grad_(True)
    stash = {}
    h = wrapped[m_v].register_forward_pre_hook(
        lambda mod, args: stash.__setitem__("x", args[0])
    )
    try:
        from src.clcd.latents import inject
        with inject(wrapped, {m_u: leaf}):
            model(res["full_trigger"])
        live = _live_sparse(wrapped[m_v], stash["x"])
        g = torch.autograd.grad(live[0, p_v, d_v], leaf)[0]
        jvp = float(g[0, p_u, d_u])
    finally:
        h.remove()

    # finite difference: nudge a_u by a small h, read a_v
    step = 1e-3 * (abs(a1u) + 1.0)
    ov = {m_u: _knock_override(m_u, d_u, p_u, a1u + step)}
    a_plus = _read_under(model, res["full_trigger"], wrapped, ov)
    fd = (float(a_plus[m_v][0, p_v, d_v]) - float(res["a1"][m_v][0, p_v, d_v])) / step
    assert abs(jvp - fd) < 1e-2 + 0.05 * abs(fd), (jvp, fd)


def test_jvp_and_patching_share_edges(fix):
    # edge_scores_jvp must return a score for exactly the edges it is asked about.
    model, wrapped, ep, res, nodes, info = _setup(fix)
    edges = edge_scores_patching(
        model, wrapped, res["full_trigger"], nodes, info, res["a0"], res["a1"]
    )
    top = sorted(edges, key=lambda e: -abs(edges[e]))[:5]
    jvp = edge_scores_jvp(model, wrapped, res["full_trigger"], top, info, res["a0"], res["a1"])
    assert set(jvp) == set(top)
    assert all(torch.isfinite(torch.tensor(v)) for v in jvp.values())


# --- path-patch verification ----------------------------------------------------

def test_path_patch_runs_and_is_null_without_contrast(fix):
    model, wrapped, ep, res, nodes, info = _setup(fix)
    edges = edge_scores_patching(
        model, wrapped, res["full_trigger"], nodes, info, res["a0"], res["a1"]
    )
    u, v = max(edges, key=lambda e: abs(edges[e]))
    eff = path_patch_edge(
        model, wrapped, ep, u, v, nodes, res["a0"], res["a1"], grad_v=info[v]["grad"]
    )
    assert {"direct_E", "mu_effect"} <= eff.keys()
    assert all(torch.isfinite(torch.tensor(x)) for x in eff.values())
    # With a0:=a1 there is no edge to ablate -> Step A reads v's clean value, so both
    # the isolated activation effect and the behavioural μ-flip are ~0.
    null = path_patch_edge(
        model, wrapped, ep, u, v, nodes, res["a1"], res["a1"], grad_v=info[v]["grad"]
    )
    assert abs(null["direct_E"]) < 1e-4 and abs(null["mu_effect"]) < 1e-4
