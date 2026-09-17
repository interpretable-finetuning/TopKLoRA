"""M7 edge-attribution mechanics (CPU, random fixture).

Numbers are noise by design; these assert the machinery is what we claim:
the DAG prune, candidate extraction, that Method-A patching measures a true
contrast, that the Method-B JVP matches a finite difference (so the two §8
estimators coincide where no gate flips), and that path-patching runs and is
null when there is nothing to patch.
"""

import random
from collections import Counter

import pytest
import torch

from src.clcd.attribute import attribute
from src.clcd.edges import (
    block_single_pass_eliminate,
    candidate_nodes,
    compute_order,
    dag_valid,
    edge_scores_jvp,
    edge_scores_patching,
    greedy_edge_eliminate,
    single_pass_eliminate,
    grow_edge_guided,
    path_patch_edge,
    permuted_region,
    region_position_profile,
    scrub_eval,
    assign_roles,
    _switch_thresh,
    _knock_override,
    _read_under,
    _live_sparse,
)
from src.clcd.episode import Episode
from src.clcd.measure import mu
from src.clcd.selection import select
from src.clcd.verify import ablation_overrides


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
    k0 = ("L.layers.0.self_attn.k_proj", 5, 3)
    assert dag_valid(q0, o0)             # same position, downstream in compute order
    assert not dag_valid(o0, q0)         # same position, backward in compute order
    assert not dag_valid(o0_late, q0)    # earlier position -> forbidden
    assert not dag_valid(q0, q0)         # self-edge

    # Same layer, LATER position. Within a layer the only cross-position operation is that
    # layer's attention, so the source must be one of attention's cross-position inputs.
    # k/v at p_u reach the attention output at every later query position:
    assert dag_valid(k0, o0_late)
    # ...but q_proj is per-QUERY-position: q@p3 shapes position 3's query only, so it cannot
    # affect o_proj@p5. This assertion was inverted until 2026-07-31 (commented "later
    # position (attention-mediated)"), which is what let dag_valid admit phantom edges.
    assert not dag_valid(q0, o0_late)
    # and a writer certainly cannot -- its own layer's attention has already run:
    assert not dag_valid(("L.layers.0.mlp.down_proj", 1, 3), ("L.layers.0.mlp.up_proj", 2, 5))


def test_dag_valid_rejects_backward_layer_edges():
    # source at a LATER layer, dest at an EARLIER layer but a later position:
    # attention cannot carry info from a later layer back to an earlier one, yet
    # the bare p_v>p_u test would wrongly admit it.
    #
    # The SOURCE MODULE MATTERS and this test previously got it wrong. With an `o_proj`
    # source the k/v test short-circuits to False before the layer comparison is ever
    # reached, so the assertion held whether or not the layer guard existed -- and when
    # 7cf0094 deleted that guard, this test stayed green while
    # `k_proj@layer23,p3 -> down_proj@layer16,p5` became admissible. Both sources are
    # asserted now: k/v is the one that actually exercises the guard.
    dst_layer16_p5 = ("M.layers.16.mlp.down_proj", 2, 5)
    for proj in ("k_proj", "v_proj"):  # reaches the layer comparison
        src = (f"M.layers.23.self_attn.{proj}", 1, 3)
        assert not dag_valid(src, dst_layer16_p5), f"{proj}@L23 -> down_proj@L16 must be rejected"
    src_layer23_p3 = ("M.layers.23.self_attn.o_proj", 1, 3)
    assert not dag_valid(src_layer23_p3, dst_layer16_p5)  # 23 -> 16 forbidden despite p_v>p_u
    # same positions, forward in layers: a real attention-mediated edge
    src_layer16_p3 = ("M.layers.16.self_attn.k_proj", 1, 3)
    dst_layer23_p5 = ("M.layers.23.mlp.down_proj", 2, 5)
    assert dag_valid(src_layer16_p3, dst_layer23_p5)      # 16 -> 23 with p_v>p_u is fine
    # within a single layer the later-position branch is unchanged
    same_a = ("M.layers.19.self_attn.k_proj", 1, 3)
    same_b = ("M.layers.19.self_attn.o_proj", 2, 5)
    assert dag_valid(same_a, same_b)


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


def test_patching_score_is_signed_delta_times_grad_v(fix):
    # Pins E_A's SIGN, MAGNITUDE and INDEX ORDER against the definition
    #     E^A_{u->v} = (a1_v - a_v|_{u<-a0}) * grad_v
    # The test above cannot: with a0 := a1 every score is zero, so it passes just as
    # happily with the subtraction reversed. Here the expected value is recomputed with
    # its own explicit [0, p, d] indexing, so a flipped sign, a grad_u/grad_v mixup, and
    # a [0,p,d]<->[0,d,p] transposition each fail this assertion.
    # Worth a guard because E_A reaches real result paths: Exp-2b Stage 2 ranks edges by
    # abs(E_A) before path-patching them, and grow_greedy weights its frontier by it.
    model, wrapped, ep, res, nodes, info = _setup(fix, n_pos=8, cap=5)
    u, v = _find_active_pair(model, wrapped, res, nodes)
    assert u is not None, "no active cross-node pair found in fixture"
    if abs(info[v]["grad"]) < 1e-9:
        pytest.skip("grad_v ~ 0 would make the comparison vacuous")
    m_u, d_u, p_u = u
    m_v, d_v, p_v = v

    ov = {m_u: _knock_override(m_u, d_u, p_u, float(res["a0"][m_u][0, p_u, d_u]))}
    a_knock = _read_under(model, res["full_trigger"], wrapped, ov)
    delta = float(res["a1"][m_v][0, p_v, d_v]) - float(a_knock[m_v][0, p_v, d_v])
    assert abs(delta) > 1e-4, "_find_active_pair should guarantee u actually moves v"
    expected = delta * info[v]["grad"]

    edges = edge_scores_patching(
        model, wrapped, res["full_trigger"], nodes, info, res["a0"], res["a1"]
    )
    assert abs(edges[(u, v)] - expected) < 1e-6, (edges[(u, v)], expected)


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


def test_live_sparse_matches_the_real_forward_under_a_latent_gate():
    # _live_sparse rebuilds the encode->top-k path so the JVP can differentiate through it.
    # If it omits a step the real forward applies, Method B differentiates a DIFFERENT
    # function and `ab_sign_agreement` compares two unrelated quantities. The hard-concrete
    # latent gate is such a step, and because it rescales pre-top-k magnitudes it also moves
    # which latents win the top-k -- so the disagreement is structural, not just numeric.
    import sys
    sys.path.insert(0, "tests")
    from test_training_regularizers import _build_topk_module

    _, mod = _build_topk_module(latent_gate_enabled=True)
    with torch.no_grad():
        mod.latent_gate_logits.copy_(torch.tensor([2.0, -2.0, 0.5])[: mod.r])
    x = torch.randn(1, 4, mod.in_features)

    live = _live_sparse(mod, x)
    with torch.no_grad():          # the real forward, read the way the pipeline reads it
        mod(x)
        real = mod._last_z_sparse
    assert live.shape == real.shape
    assert torch.allclose(live, real, atol=1e-5), (live - real).abs().max()


def test_jvp_tolerates_admitted_pairs_with_no_autograd_path(fix):
    # dag_valid is a conservative over-approximation: it admits a few pairs with no real
    # path rather than risk dropping a real one. edge_scores_jvp must therefore return 0.0
    # for those instead of raising -- autograd.grad's default allow_unused=False raises,
    # which would abort edges_analysis, and edges_analysis writes nothing until the end, so
    # a raise here discards a whole run's attribution, necessity and ASR results.
    model, wrapped, ep, res, nodes, info = _setup(fix)
    pairs = [(u, v) for u in nodes for v in nodes if dag_valid(u, v)]
    assert pairs, "fixture must yield DAG-valid pairs"
    jvp = edge_scores_jvp(
        model, wrapped, res["full_trigger"], pairs, info, res["a0"], res["a1"]
    )
    assert set(jvp) == set(pairs)
    assert all(torch.isfinite(torch.tensor(v)) for v in jvp.values())


def test_grow_seed_first_and_connected_only():
    circ = grow_edge_guided(_GRAPH, HUB, cap=10)
    assert circ[0] == HUB                       # seed is first
    assert ISO not in circ and OTHER not in circ  # disjoint component never pulled in
    assert set(circ) == {HUB, DET, ACT, ACT2}   # exactly the seed's connected component


# --- dynamical (edge-guided) circuit growth (pure, no model) --------------------

HUB, DET, ACT, ACT2 = ("hub", 0), ("det", 0), ("act", 0), ("act2", 0)
ISO, OTHER = ("iso", 0), ("other", 0)
# det --3.0--> hub --0.5--> act, hub --0.3--> act2 ; iso--9--other is a disjoint component
_GRAPH = {
    (DET, HUB): 3.0,
    (HUB, ACT): 0.5,
    (HUB, ACT2): 0.3,
    (ISO, OTHER): 9.0,
}


def test_grow_prefers_stronger_connection():
    # det (|3.0| into hub) must be added before the weak hub->act/act2 out-edges
    circ = grow_edge_guided(_GRAPH, HUB, cap=10)
    assert circ.index(DET) < circ.index(ACT)
    assert circ.index(DET) < circ.index(ACT2)
    assert circ.index(ACT) < circ.index(ACT2)   # 0.5 before 0.3


def test_grow_respects_cap_and_halts():
    assert grow_edge_guided(_GRAPH, HUB, cap=2) == [HUB, DET]      # cap stops growth
    # frontier exhausts at the component boundary even with a large cap
    assert len(grow_edge_guided(_GRAPH, HUB, cap=99)) == 4


def test_grow_deterministic_tiebreak():
    # equal connection (both 1.0 from seed) -> deterministic lexicographic order
    g = {(HUB, ("b", 0)): 1.0, (HUB, ("a", 0)): 1.0}
    assert grow_edge_guided(g, HUB, cap=3) == [HUB, ("b", 0), ("a", 0)]


# --- edge-set scrubbing evaluator -----------------------------------------------

def test_scrub_eval_keepall_reproduces_trigger(fix):
    # The evaluator's ceiling identity: severing NO edges must leave every node at its
    # trigger value, so the scrubbed margin == the real trigger margin. If this drifts,
    # the single-value propagation is corrupting the kept path -> the arbiter is invalid.
    model, wrapped, ep, res, nodes, info = _setup(fix)
    edges = list(
        edge_scores_patching(
            model, wrapped, res["full_trigger"], nodes, info, res["a0"], res["a1"]
        ).keys()
    )
    assert edges, "need DAG-valid edges to exercise the evaluator"
    out = scrub_eval(model, wrapped, ep, nodes, edges, set(), res["a0"], res["a1"])
    with torch.no_grad():
        mu_trig = float(mu(model, ep.prompt_trigger, ep.y_plus, ep.y_minus))
    assert abs(out["mu"] - mu_trig) < 1e-3, (out["mu"], mu_trig)
    assert set(out["val"]) == set(nodes)  # one effective value per node


# --- the two scrub estimands (sparse candidate sets) ----------------------------
#
# The test above passes EVERY DAG-valid pair as candidate_edges, so no wire is ever
# outside the candidate set and both estimands coincide -- it cannot see the difference.
# `exp_edge_scrub.realize_episode` filters the universe by `universe ∩ pos_set`, which
# MAY leave DAG-valid pairs outside the candidate set. These pin the behaviour on that
# path with a deliberately sparse set. Captain's log Exp-8/Exp-9.

def _sparse_and_complete(model, wrapped, res, nodes, info):
    """All DAG-valid pairs, plus a deterministic strict subset standing in for a
    universe-filtered candidate set that does not cover every pair."""
    complete = list(
        edge_scores_patching(
            model, wrapped, res["full_trigger"], nodes, info, res["a0"], res["a1"]
        ).keys()
    )
    sparse = complete[::3]
    assert len(sparse) < len(complete), "need a STRICT subset to exercise the difference"
    return sparse, complete


def test_scrub_eval_sparse_ceiling_identity_holds_in_context_mode(fix):
    # exp_edge_scrub takes its ceiling from raw mu() while the floor comes from
    # scrub_eval, and greedy_edge_eliminate documents recovery as 1.0 at cut=∅. In the
    # default "in context" estimand that identity must hold even when the candidate set
    # is sparse -- otherwise every normalized recovery is scored against an anchor that
    # is not 1.0 and the elimination target silently means something else.
    model, wrapped, ep, res, nodes, info = _setup(fix)
    sparse, _ = _sparse_and_complete(model, wrapped, res, nodes, info)
    out = scrub_eval(model, wrapped, ep, nodes, sparse, set(), res["a0"], res["a1"])
    with torch.no_grad():
        mu_trig = float(mu(model, ep.prompt_trigger, ep.y_plus, ep.y_minus))
    assert abs(out["mu"] - mu_trig) < 1e-3, (out["mu"], mu_trig)


def test_scrub_eval_sever_reaches_the_true_no_wiring_floor(fix):
    # The POINT of sever_noncandidate: with non-candidate wires severed, cutting every
    # candidate edge leaves no wiring at all, so a sparse candidate set must bottom out
    # at the same floor as the complete one. In context mode it does not -- the wires the
    # hypothesis omits keep carrying margin -- which is what lets a circuit look
    # load-bearing while the behaviour rides on wires the cut-set cannot touch.
    model, wrapped, ep, res, nodes, info = _setup(fix)
    sparse, complete = _sparse_and_complete(model, wrapped, res, nodes, info)
    kw = dict(sever_noncandidate=True)
    f_sparse = scrub_eval(model, wrapped, ep, nodes, sparse, set(sparse), res["a0"], res["a1"], **kw)["mu"]
    f_complete = scrub_eval(model, wrapped, ep, nodes, complete, set(complete), res["a0"], res["a1"], **kw)["mu"]
    assert abs(f_sparse - f_complete) < 1e-3, (f_sparse, f_complete)


def test_scrub_eval_free_riding_gap_is_real(fix):
    # The two estimands must actually differ on a sparse set, and the gap is a reported
    # quantity (exp_edge_scrub's "free_ride"): how much margin survives cutting every
    # candidate edge purely because non-candidate wires are still intact. If a future
    # cleanup collapses the flag to one branch this fails loudly rather than silently
    # changing what every edge-scrub number means.
    model, wrapped, ep, res, nodes, info = _setup(fix)
    sparse, _ = _sparse_and_complete(model, wrapped, res, nodes, info)
    args = (model, wrapped, ep, nodes, sparse, set(sparse), res["a0"], res["a1"])
    floor_context = scrub_eval(*args, sever_noncandidate=False)["mu"]
    floor_alone = scrub_eval(*args, sever_noncandidate=True)["mu"]
    assert abs(floor_alone - floor_context) > 1e-3, (floor_context, floor_alone)


def test_scrub_eval_uncomputed_node_values_are_immaterial(fix):
    # Soundness of the topological single-value scheme: a not-yet-computed node is
    # downstream or parallel, so under the causal mask + compute order it CANNOT
    # influence val[v]. scrub_eval feeds such nodes a1 on exactly that justification.
    # Passing a0 in a1's place must therefore change nothing. This is why severing
    # non-candidates only needs to touch COMPUTED nodes: doing it for uncomputed ones
    # would be a no-op. If this ever fails, the causal-order argument is wrong and the
    # whole single-value propagation is unsound, not just the flag.
    model, wrapped, ep, res, nodes, info = _setup(fix)
    sparse, _ = _sparse_and_complete(model, wrapped, res, nodes, info)
    base = scrub_eval(model, wrapped, ep, nodes, sparse, set(), res["a0"], res["a1"])
    swapped = scrub_eval(model, wrapped, ep, nodes, sparse, set(), res["a0"], res["a0"])
    assert abs(base["mu"] - swapped["mu"]) < 1e-9, (base["mu"], swapped["mu"])
    for n in nodes:  # exact: v's activations are computed before any such node's
        assert abs(base["val"][n] - swapped["val"][n]) < 1e-9, (n, base["val"][n], swapped["val"][n])


# --- role heuristic (pure, no model) --------------------------------------------
#
# These labels are INDICATIVE, not authoritative -- nothing downstream consumes them.
# But an indicative label that is identical for every node carries no information at all,
# and that is what three separate defects produced. These pin the fixes without asserting
# that the taxonomy itself is right.

def _role_fixture():
    """detector(tag) -> relay(shared) -> actuator(completion), one node per region.

    Both edges must be DAG-legal or the chain is not a chain: k_proj@1 -> o_proj@3 is the
    same-layer attention route, and o_proj(L0)@3 -> down_proj(L1)@5 is cross-layer. The
    original fixture used o_proj@3 -> down_proj@5 in ONE layer, which `dag_valid` wrongly
    admitted until 2026-07-31 -- a writer cannot cross positions inside its own layer.
    """
    det = ("L.layers.0.self_attn.k_proj", 1, 1)   # tag
    rel = ("L.layers.0.self_attn.o_proj", 2, 3)   # shared
    act = ("L.layers.1.mlp.down_proj", 3, 5)      # completion, NEXT layer
    nodes = [det, rel, act]
    region = {1: "tag", 3: "shared", 5: "completion"}
    info = {n: {"A": 1.0, "grad": 1.0} for n in nodes}
    edges = {(det, rel): 1.0, (rel, act): 1.0}
    return nodes, info, edges, region, det, rel, act


def test_switch_threshold_needs_a_strictly_positive_lever():
    # path_patch_edge's docstring predicts mu-effects near zero. The tercile is then 0.0,
    # and `abs(x) >= 0.0` is true for everything -- so every node became a "switch".
    assert _switch_thresh({}) == float("inf")
    assert _switch_thresh({"a": 0.0, "b": 0.0, "c": 0.0}) == float("inf")
    # a real spread still yields a usable finite cutoff
    assert _switch_thresh({"a": 0.0, "b": 1.0, "c": 2.0}) > 0.0


def test_roles_do_not_all_collapse_to_switch_at_zero_mu():
    # the end-to-end version of the above: with mu-effects ~0 nothing is a lever.
    nodes, info, edges, region, det, rel, act = _role_fixture()
    roles = assign_roles(nodes, info, edges, region, path_effects={n: 0.0 for n in nodes})
    assert set(roles.values()) != {"switch"}, roles


def test_state_carrier_still_preempts_detector_and_actuator():
    # DOCUMENTS a known limitation rather than asserting the taxonomy is right. On a
    # detector -> relay -> actuator chain every node has a cross-region neighbour, and
    # `state_carrier` only needs >1 region and sits ahead of detector/actuator in the
    # chain, so it takes all three. Fixing that means choosing a stricter bar or a
    # different precedence -- a taxonomy decision, and these labels are explicitly
    # indicative. If someone later makes detector/actuator reachable, this test SHOULD
    # fail: update it, don't work around it.
    nodes, info, edges, region, det, rel, act = _role_fixture()
    roles = assign_roles(nodes, info, edges, region)
    assert roles[det] == "state_carrier", roles
    assert roles[act] == "state_carrier", roles


def test_zero_scored_edges_do_not_count_as_wiring():
    # `edges` holds every DAG-valid pair, many scoring exactly 0.0. Counting those makes
    # degree a restatement of DAG membership rather than a measure of wiring.
    #
    # The observable consequence needs an ISOLATED node: one with no real wiring, whose
    # only candidate pair scores 0.0. With the nulls excluded it has degree 0 and touches
    # one region, so it falls through to `relay`; counting them would give it a degree and
    # a second region, promoting it to `state_carrier`. Using the plain 3-node chain here
    # cannot detect the difference -- every node is `state_carrier` either way, which made
    # an earlier version of this test pass with the fix reverted.
    nodes, info, edges, region, det, rel, act = _role_fixture()
    iso = ("L.layers.0.self_attn.v_proj", 4, 3)  # shared region, no real edges
    nodes = nodes + [iso]
    info = {**info, iso: {"A": 1.0, "grad": 1.0}}
    assert dag_valid(iso, act), "fixture needs iso->act to be a DAG-valid pair"

    roles = assign_roles(nodes, info, {**edges, (iso, act): 0.0}, region)
    assert roles[iso] == "relay", roles
    # and the null edge must not have perturbed anyone else either
    assert roles == assign_roles(nodes, info, edges, region), roles


# --- greedy backward edge elimination (pure, no model) --------------------------

# Synthetic recovery: each edge has a "load"; recovery = 1 - (load of severed edges).
# A/B are redundant (load 0), C/D are load-bearing. The greedy must shed the redundant
# wiring first and stop before breaching the target -- i.e. find the MINIMAL load-bearing
# subgraph, not just any subgraph.
_LOAD = {"A": 0.0, "B": 0.0, "C": 0.3, "D": 0.5}


def _recovery(cut):
    return 1.0 - sum(_LOAD[e] for e in cut)


def test_greedy_sheds_redundant_first_and_halts_at_target():
    out = greedy_edge_eliminate(list(_LOAD), _recovery, target=0.6)
    # D alone breaches the target if cut (1-0.5=0.5<0.6) -> it must survive; everything
    # else is sheddable while staying >=0.6 (cutting A,B,C leaves 0.7).
    assert out["kept"] == ["D"]
    assert out["cut_order"] == ["A", "B", "C"]  # zero-load first, then the affordable C
    # recovery is monotone non-increasing and the final committed value clears target
    recs = [t["recovery"] for t in out["trace"]]
    assert recs == sorted(recs, reverse=True)
    assert out["trace"][0]["n_cut"] == 0 and out["trace"][0]["recovery"] == 1.0
    assert out["trace"][-1]["recovery"] >= 0.6


def test_greedy_target_one_keeps_all_load_bearing():
    # target=1.0: only zero-load edges may go; any positive-load cut drops below 1.0.
    out = greedy_edge_eliminate(list(_LOAD), _recovery, target=1.0)
    assert set(out["kept"]) == {"C", "D"}
    assert out["cut_order"] == ["A", "B"]


def test_greedy_target_zero_strips_everything():
    out = greedy_edge_eliminate(list(_LOAD), _recovery, target=0.0)
    assert out["kept"] == []
    assert len(out["cut_order"]) == 4


def test_greedy_tiebreak_is_deterministic():
    # equal (zero) load on both -> the lexicographically smaller edge is cut first
    out = greedy_edge_eliminate(["B", "A"], lambda cut: 1.0, target=0.5)
    assert out["cut_order"] == ["A", "B"]


def test_greedy_guard_never_orphans_a_node():
    # No-orphan guard (constraint (b)): an edge may be cut only if BOTH endpoints retain
    # another surviving edge. Graph: A-B, A-C, D-B. B has {A-B, D-B}, A has {A-B, A-C};
    # C and D are leaves (one edge each). Even though recovery is always 1.0 (>= target),
    # the ONLY cut that orphans nobody is A-B; afterwards every remaining cut would strip a
    # leaf's last edge, so the greedy must halt rather than orphan C or D. Verifies the
    # guard enforces the structural invariant independently of the recovery objective.
    all_edges = [("A", "B"), ("A", "C"), ("D", "B")]

    def guard(cut, e):
        surviving = set(all_edges) - (set(cut) | {e})
        return all(any(lat in edge for edge in surviving) for lat in e)

    out = greedy_edge_eliminate(all_edges, lambda cut: 1.0, target=0.5, guard=guard)
    assert out["cut_order"] == [("A", "B")]
    assert set(out["kept"]) == {("A", "C"), ("D", "B")}
    # every kept-graph node still has >= 1 incident edge (no orphans)
    kept_nodes = {lat for edge in out["kept"] for lat in edge}
    assert kept_nodes == {"A", "B", "C", "D"}


def test_single_pass_keeps_load_bearing_in_one_pass():
    # Given the weakest-first order, the single pass must shed A,B,C (recovery stays >=0.6)
    # and keep D (cutting it drops to 0.5 < 0.6) -- the same minimal subset as the re-scan
    # greedy, but reached by visiting each element exactly once in the supplied order.
    out = single_pass_eliminate(["A", "B", "C", "D"], _recovery, target=0.6)
    assert out["kept"] == ["D"]
    assert out["cut_order"] == ["A", "B", "C"]
    assert out["trace"][0] == {"n_cut": 0, "recovery": 1.0, "edge": None}
    assert out["trace"][-1]["recovery"] >= 0.6


def test_single_pass_is_order_dependent_for_substitutable_redundancy():
    # WHY this matters: single-pass does NOT re-scan, so for substitutable redundancy
    # (cutting EITHER X or Y is fine, cutting BOTH is fatal) it keeps whichever it visits
    # LAST and sheds the first -- the result is order-dependent. This is the price of O(N)
    # vs the re-scan greedy, and the reason the caller must pass elements weakest-first.
    def rec(cut):
        return 0.0 if {"X", "Y"} <= set(cut) else 1.0

    keep_y = single_pass_eliminate(["X", "Y"], rec, target=0.5)
    keep_x = single_pass_eliminate(["Y", "X"], rec, target=0.5)
    assert keep_y["kept"] == ["Y"] and keep_y["cut_order"] == ["X"]
    assert keep_x["kept"] == ["X"] and keep_x["cut_order"] == ["Y"]


def test_single_pass_target_extremes():
    assert single_pass_eliminate(list(_LOAD), _recovery, target=0.0)["kept"] == []
    assert set(single_pass_eliminate(list(_LOAD), _recovery, target=1.0)["kept"]) == {"C", "D"}


def test_single_pass_resume_reproduces_uninterrupted_run():
    # WHY this matters: elimination is a 15h sweep, so it checkpoints and must be resumable
    # after a crash WITHOUT changing the result. A checkpoint that resumes to a different
    # minimal set would silently corrupt every long run -- so resume MUST be bit-identical to
    # an uninterrupted pass. Uses (str,int) tuples like the real latent pool and a JSON
    # round-trip on the checkpoint to prove tuple<->list coercion survives serialisation.
    import json

    pool = [("m%d" % i, i) for i in range(12)]        # weakest-first
    strong = {("m3", 3), ("m7", 7), ("m9", 9)}        # cutting any is fatal -> must be kept
    rec = lambda cut: 0.0 if (set(cut) & strong) else 1.0

    ref = single_pass_eliminate(pool, rec, 1.0)

    saved = {}
    n = [0]

    class _Stop(Exception):
        pass

    def rec_crash(cut):                                # die after 5 elements are processed
        if n[0] >= 6:                                  # +1 for the initial recovery_fn(frozenset())
            raise _Stop()
        n[0] += 1
        return rec(cut)

    try:
        single_pass_eliminate(pool, rec_crash, 1.0,
                              checkpoint_fn=lambda s: saved.update(json.loads(json.dumps(s))))
    except _Stop:
        pass
    assert saved["processed"] == 5                     # crashed mid-sweep, checkpoint persisted
    resumed = single_pass_eliminate(pool, rec, 1.0, resume=saved)

    assert resumed["kept"] == ref["kept"] == list(sorted(strong))
    assert resumed["cut_order"] == ref["cut_order"]


# --- resume-state validation (2026-09-16 defect) ---------------------------------
# exp_circuit_search recomputed bf16 attribution on every relaunch, near-tied latents swapped
# places, and resume skips BY INDEX: a latent cut before the crash that slid past the resume
# index was tested and cut AGAIN (11 of 21 stopped dense checkpoints had duplicate cuts), and a
# latent that slid in front of it was never tested and silently survived.
#
# What is tested below is the half of that single_pass_eliminate CAN see: a cut element that left
# edges[:processed], duplicate cuts, cut/cut_order disagreement, processed out of range. The other
# half is invisible here -- a kept or undecided latent sliding INTO the prefix leaves all four
# invariants intact -- and is prevented one level up, by exp_circuit_search persisting the visiting
# order in the checkpoint and resuming along THAT order (tests/test_circuit_search_grid.py).

_POOL12 = [("m%d" % i, i) for i in range(12)]        # (module, idx) tuples, weakest-first


def _state(processed, cut, cut_order):
    """A checkpoint exactly as it comes back from disk (tuples serialised to lists)."""
    import json
    return json.loads(json.dumps({"processed": processed, "cut": cut, "cut_order": cut_order,
                                  "full_recovery": 1.0}))


def test_single_pass_resume_refuses_an_order_that_did_not_produce_the_checkpoint():
    # The observed failure, reduced: the pass over `pool` stopped at processed=5 having cut
    # m0,m1,m2,m4 (m3 is load-bearing). The relaunch's order swaps the near-tie m4 <-> m5 across
    # the resume index. Resuming by index would re-test m4 (a second cut -> the duplicate seen in
    # the real checkpoints) and never test m5. The state is otherwise flawless -- no duplicates,
    # set(cut_order) == set(cut), processed in range -- so ONLY the "every cut element lies in
    # edges[:processed]" check can catch it, and it must.
    strong = {("m3", 3)}
    rec = lambda cut: 0.0 if (set(cut) & strong) else 1.0
    cut_order = [("m0", 0), ("m1", 1), ("m2", 2), ("m4", 4)]
    saved = _state(5, list(cut_order), cut_order)
    # it IS the genuine prefix of a pass over `pool` ...
    assert single_pass_eliminate(_POOL12, rec, 1.0, resume=saved)["cut_order"] == \
        single_pass_eliminate(_POOL12, rec, 1.0)["cut_order"]
    reordered = list(_POOL12)
    reordered[4], reordered[5] = reordered[5], reordered[4]
    # ... and passes the other three invariants against the reordered list
    co = [tuple(e) for e in saved["cut_order"]]
    assert len(set(co)) == len(co) and set(co) == {tuple(e) for e in saved["cut"]}
    assert 0 <= saved["processed"] <= len(reordered)
    with pytest.raises(ValueError, match=r"not in edges\[:processed=5\]"):
        single_pass_eliminate(reordered, rec, 1.0, resume=saved)


def test_single_pass_resume_refuses_duplicate_cuts():
    # The exact signature found in the stopped checkpoints: a latent appears twice in cut_order
    # while set(cut) == set(cut_order). Every element lies in edges[:processed], so only the
    # duplicate check catches it. A twice-cut latent means the pass was not one pass over one
    # order, so the survivor set cannot be trusted.
    cut_order = [("m0", 0), ("m1", 1), ("m2", 2), ("m1", 1), ("m4", 4)]
    saved = _state(5, [("m0", 0), ("m1", 1), ("m2", 2), ("m4", 4)], cut_order)
    with pytest.raises(ValueError, match="duplicate"):
        single_pass_eliminate(_POOL12, lambda cut: 1.0, 1.0, resume=saved)


def test_single_pass_resume_refuses_cut_set_disagreeing_with_cut_order():
    # `cut` drives every later decision, `cut_order` becomes the reported importance ranking.
    # If they disagree the published order does not describe the decisions that were made.
    saved = _state(5, [("m0", 0), ("m1", 1), ("m2", 2)], [("m0", 0), ("m1", 1)])
    with pytest.raises(ValueError, match=r"set\(cut_order\) != set\(cut\)"):
        single_pass_eliminate(_POOL12, lambda cut: 1.0, 1.0, resume=saved)


@pytest.mark.parametrize("processed", [0, 12])
def test_single_pass_resume_accepts_the_boundary_checkpoints(processed):
    # Both boundaries are REAL crash points, not corruption, and both must resume to the
    # uninterrupted result:
    #   processed == 0        the checkpoint written before the first element is decided; the run
    #                         was killed inside the very first arbiter call (~7 min of generation
    #                         on a dense cell). Resuming must run the whole pass.
    #   processed == len      the last checkpoint, written after the final decision. The .ckpt is
    #                         only deleted once the circuit JSON is written, so every kill during
    #                         the hours-long rigorous K-sweep leaves exactly this state. Resuming
    #                         must re-test NOTHING -- re-running the arbiter here would be a second
    #                         15 h elimination pass, and under a non-deterministic bf16 arbiter it
    #                         could return a different survivor set than the one checkpointed.
    # Only the range check stands between these and a refusal: tightening it to
    # `0 < processed < len(edges)` breaks both while leaving the rest of the suite green.
    strong = {("m3", 3), ("m7", 7), ("m9", 9)}
    calls = []

    def rec(cut):
        calls.append(frozenset(cut))
        return 0.0 if (set(cut) & strong) else 1.0

    ref = single_pass_eliminate(_POOL12, rec, 1.0)
    n_ref = len(calls)                                  # 1 full eval + 1 per element
    assert n_ref == 13
    calls.clear()

    done = processed == len(_POOL12)
    saved = _state(processed, list(ref["cut_order"]) if done else [],
                   list(ref["cut_order"]) if done else [])
    resumed = single_pass_eliminate(_POOL12, rec, 1.0, resume=saved)

    assert resumed["kept"] == ref["kept"] and resumed["cut_order"] == ref["cut_order"]
    # a resume never re-runs the initial full-circuit eval (full_recovery comes from the state),
    # and a finished sweep never touches the arbiter at all.
    assert len(calls) == (0 if done else n_ref - 1)


@pytest.mark.parametrize("processed", [-1, 13])
def test_single_pass_resume_refuses_processed_outside_the_edge_list(processed):
    # processed > len(edges) is a checkpoint from a longer pool: resuming would skip every
    # element and report an untested pool as decided. Negative is corruption. The cut is empty,
    # so the edges[:processed] check has nothing to reject -- only the range check can.
    saved = _state(processed, [], [])
    with pytest.raises(ValueError, match=r"processed=-?\d+ outside \[0, 12\]"):
        single_pass_eliminate(_POOL12, lambda cut: 1.0, 1.0, resume=saved)


# --- block elimination (adaptive_block_bisect_v1) --------------------------------
# Same pool, same arbiter, same visiting order as single_pass_eliminate; a test covers a
# contiguous BLOCK of the order instead of one latent, so an all-cut stretch costs ~N/cap tests
# instead of N (the dense `all` cells cut the first 280-970 latents without a single keep). The
# tests below pin the three things that make that safe rather than merely cheaper: the block
# result EQUALS one-at-a-time whenever the arbiter is monotone, every committed cut set was
# actually observed to pass (nothing is inferred from a neighbour), and a resume reproduces an
# uninterrupted run exactly -- including its telemetry, which is what the protocol comparison
# reads back.


def _monotone_cases(n_cases=400, seed=1):
    """Random arbiters that are monotone by construction: a fatal set (cutting any member fails)
    plus a weight budget (cutting too much mass fails). Both are downward-closed, which is the
    premise of the equivalence theorem."""
    rng = random.Random(seed)
    for _ in range(n_cases):
        n = rng.randint(1, 70)
        pool = [("m%d" % (i % 5), i) for i in range(n)]
        fatal = frozenset(e for e in pool if rng.random() < rng.choice([0.0, 0.05, 0.3, 0.7]))
        w = {e: rng.random() ** 3 for e in pool}
        b = rng.random() * n * 0.2
        rec = (lambda cut, f=fatal, w=w, b=b: 0.0 if (set(cut) & f or sum(w[e] for e in cut) > b) else 1.0)
        yield pool, rec, rng.choice([2, 3, 8, 64])


def test_E1_block_equals_one_at_a_time_under_a_monotone_arbiter():
    # WHY: "same answer, fewer tests" rests entirely on this equivalence. Where the arbiter is
    # monotone the two protocols must agree on the survivors AND on cut_order (which is what the
    # rigorous K-sweep walks), for every cap. A disagreement here is an implementation bug, not
    # the non-monotonicity the real arbiter is allowed to show.
    for pool, rec, cap in _monotone_cases():
        ref = single_pass_eliminate(pool, rec, 1.0)
        got = block_single_pass_eliminate(pool, rec, 1.0, cap)
        assert got["kept"] == ref["kept"]
        assert got["cut_order"] == ref["cut_order"]
        assert got["stats"]["max_size_tested"] <= cap


def test_E2_non_monotone_may_differ_but_every_commit_was_observed_to_pass():
    # WHY: the protocols CAN return different survivor sets, and the reason must be visible. The
    # pinned example is the minimal non-monotone arbiter (cutting `a` is fatal unless `b` goes
    # too): one-at-a-time keeps `a` because the intermediate state {x,a} fails, block elimination
    # never visits that state and cuts everything. That is a different path through the same
    # lattice under the same rule -- not a weaker rule -- so the random half asserts the property
    # that makes it defensible: every accumulated commit set was a state the arbiter PASSED.
    rec = lambda cut: 0.0 if ("a" in cut and "b" not in cut) else 1.0
    assert single_pass_eliminate(["x", "a", "b"], rec, 1.0)["kept"] == ["a"]
    calls = []
    out = block_single_pass_eliminate(["x", "a", "b"],
                                      lambda c: (calls.append((frozenset(c), rec(c))), rec(c))[1], 1.0, 2)
    assert out["kept"] == [] and out["cut_order"] == ["x", "a", "b"]
    assert calls == [(frozenset(), 1.0), (frozenset({"x"}), 1.0), (frozenset({"x", "a", "b"}), 1.0)]
    rng = random.Random(7)
    for t in range(500):
        table, calls = {}, []

        p = rng.random() * 0.6

        def rec2(cut, table=table, p=p, r=random.Random(t)):
            k = frozenset(cut)
            if k not in table:                       # deterministic per state, arbitrary across states
                table[k] = 0.0 if r.random() < p else 1.0
            calls.append((k, table[k]))
            return table[k]

        n = rng.randint(2, 40)
        out = block_single_pass_eliminate(list(range(n)), rec2, 1.0, rng.choice([2, 4, 16, 64]))
        passed = {k for k, v in calls if v >= 1.0}
        acc = set()
        for step in out["trace"]:
            acc |= set(step["edges"])
            assert frozenset(acc) in passed          # the state committed here was tested, and passed
        assert acc == set(out["cut_order"])


# The 17 arbiter states of the worked example, in order: 20 latents, cap 4, keeps {5,13,18}.
_E3_PINNED = [(0,), (0, 1, 2), (0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5),
              (0, 1, 2, 3, 4, 6), (0, 1, 2, 3, 4, 6, 7), (0, 1, 2, 3, 4, 6, 7, 8, 9),
              (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13), (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11),
              (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12), (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14),
              (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16),
              (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19),
              (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17),
              (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18),
              (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19)]


def test_E3_the_exact_call_sequence_and_telemetry_are_pinned():
    # WHY: the sizing policy IS the protocol. This pins doubling, the cap, reset-to-1 after a
    # failed top-level block, end truncation of the last block, the floor split point, left-half-
    # first order and identical-state reuse in one sequence of 17 states. Any change to it is a
    # protocol change: it must bump `block_policy` and re-run the validation, not slip in as a
    # tuning tweak that makes two campaign arms incomparable.
    calls = []
    rec = lambda c: (calls.append(tuple(sorted(c))), 0.0 if set(c) & {5, 13, 18} else 1.0)[1]
    out = block_single_pass_eliminate(list(range(20)), rec, 1.0, 4)
    assert calls[0] == ()                                   # the full-circuit reference eval
    assert calls[1:] == _E3_PINNED
    assert out["kept"] == [5, 13, 18]
    s = out["stats"]
    assert (s["n_tests"], s["n_reused"], s["n_commits"], s["max_size_tested"], s["max_depth"]) == (17, 4, 12, 4, 2)
    assert s["top_pass_by_size"] == {"1": 3, "2": 3} and s["top_fail_by_size"] == {"4": 2, "3": 1}
    assert s["n_tests"] == sum(s["tests_pass_by_size"].values()) + sum(s["tests_fail_by_size"].values())


def test_E4_right_half_of_a_failed_pair_is_tested_when_the_left_was_kept():
    # WHY: this is the one inference that looks free and is not. The pair [a,b] failed and `a` was
    # kept; concluding that `b` is therefore cuttable assumes monotonicity and a single culprit,
    # and would commit a cut on a state ({x,b}) nobody ever evaluated. Under the real, non-monotone
    # arbiter that is how an unverified latent joins a published circuit.
    calls = []
    rec = lambda c: (calls.append(frozenset(c)), 0.0 if set(c) & {"a", "b"} else 1.0)[1]
    out = block_single_pass_eliminate(["x", "a", "b"], rec, 1.0, 2)
    assert frozenset({"x", "b"}) in calls
    assert out["kept"] == ["a", "b"] and out["stats"]["n_reused"] == 0


def test_E4b_right_half_of_a_failed_pair_reuses_the_identical_state_when_the_left_was_cut():
    # WHY: the mirror case. Once the left half is committed, the right half's whole-block state is
    # byte-for-byte the frozenset the parent already failed on. Re-testing it would cost a GPU test
    # to ask a deterministic function the same question twice -- and accepting a pass the second
    # time would be retry-until-pass, a bias toward cutting.
    calls = []
    rec = lambda c: (calls.append(frozenset(c)), 0.0 if "b" in c else 1.0)[1]
    out = block_single_pass_eliminate(["x", "a", "b"], rec, 1.0, 2)
    assert calls.count(frozenset({"x", "a", "b"})) == 1
    assert out["kept"] == ["b"] and out["stats"]["n_reused"] == 1 and out["stats"]["n_tests"] == 3


class _BlockStop(Exception):
    pass


def _block_ref(pool, fatal_idx, cap=8):
    fatal = {pool[i] for i in fatal_idx}
    return (lambda c: 0.0 if set(c) & fatal else 1.0), cap


def test_E5_resume_from_a_crash_inside_every_arbiter_call():
    # WHY: a multi-layer elimination runs for days and is killed mid-test routinely. A block state
    # is a stack, not an index, so a resume that mislays it re-tests a whole interval or skips one.
    # Crashing at every arbiter call in turn and demanding the SAME arbiter call sequence, kept
    # set, cut_order and stats is the only way to know the stack, the sizing state and the
    # telemetry all survive a round trip through JSON.
    import json
    pool = [("m", i) for i in range(40)]
    rec, cap = _block_ref(pool, (3, 4, 17, 30, 31, 32, 39))
    ref_calls = []
    ref = block_single_pass_eliminate(pool, lambda c: (ref_calls.append(frozenset(c)), rec(c))[1], 1.0, cap)
    for k in range(1, len(ref_calls)):
        saved, calls, cnt = {}, [], [0]

        def crash(c):
            if cnt[0] == k:
                raise _BlockStop()
            cnt[0] += 1
            calls.append(frozenset(c))
            return rec(c)

        with pytest.raises(_BlockStop):
            block_single_pass_eliminate(pool, crash, 1.0, cap,
                                        checkpoint_fn=lambda s: (saved.clear(), saved.update(json.loads(json.dumps(s)))))
        res = block_single_pass_eliminate(pool, lambda c: (calls.append(frozenset(c)), rec(c))[1], 1.0, cap,
                                          resume=saved)
        assert calls == ref_calls                      # no state re-tested, none skipped
        assert res["kept"] == ref["kept"] and res["cut_order"] == ref["cut_order"]
        assert res["stats"] == ref["stats"]            # and nothing double-counted


def test_E5b_resume_from_a_crash_right_after_every_checkpoint_write():
    # WHY: E5 only covers kills that land inside the arbiter. A real kill lands anywhere, and the
    # window right AFTER a checkpoint write is exactly where a state saved before its result was
    # applied would be indistinguishable from a good one -- and would replay the same interval.
    import json
    pool = [("m", i) for i in range(40)]
    rec, cap = _block_ref(pool, (3, 4, 17, 30, 31, 32, 39))
    ref = block_single_pass_eliminate(pool, rec, 1.0, cap)
    n_writes = [0]
    block_single_pass_eliminate(pool, rec, 1.0, cap, checkpoint_fn=lambda s: n_writes.__setitem__(0, n_writes[0] + 1))
    for k in range(1, n_writes[0] + 1):
        saved, cnt = {}, [0]

        def ck(s):
            saved.clear()
            saved.update(json.loads(json.dumps(s)))
            cnt[0] += 1
            if cnt[0] == k:
                raise _BlockStop()                     # the process dies right after this write landed

        with pytest.raises(_BlockStop):
            block_single_pass_eliminate(pool, rec, 1.0, cap, checkpoint_fn=ck)
        res = block_single_pass_eliminate(pool, rec, 1.0, cap, resume=saved)
        assert res["kept"] == ref["kept"] and res["cut_order"] == ref["cut_order"]
        assert res["stats"] == ref["stats"]


def _block_state_after(pool, rec, cap, n_calls):
    """The checkpoint a crash inside the (n_calls+1)-th arbiter call would leave behind."""
    import json
    saved, cnt = {}, [0]

    def crash(c):
        if cnt[0] == n_calls:
            raise _BlockStop()
        cnt[0] += 1
        return rec(c)

    with pytest.raises(_BlockStop):
        block_single_pass_eliminate(pool, crash, 1.0, cap,
                                    checkpoint_fn=lambda s: (saved.clear(), saved.update(json.loads(json.dumps(s)))))
    return saved


_E6_POOL = [("m", i) for i in range(20)]


def _e6_rec(c):
    return 0.0 if ("m", 13) in c else 1.0


@pytest.mark.parametrize("corrupt", ["algo", "cap", "permuted_edges", "adjacent_duplicate_cut",
                                     "cut_in_pending_region", "overlapping_stack", "cursor_moved",
                                     "known_fail_on_left", "processed_disagrees", "stats_dont_add_up"])
def test_E6_resume_refuses_an_inconsistent_state(corrupt):
    # WHY: silent acceptance is the failure mode that already happened once here -- 11 of 21 stopped
    # checkpoints had latents cut twice because a resume accepted a state its `edges` could not have
    # produced. Each corruption below leaves every OTHER invariant intact, so each one is caught by
    # exactly one check; deleting any of them leaves the suite green except for its own case.
    st = _block_state_after(_E6_POOL, _e6_rec, 4, 7)          # stopped mid-bisection of [11..14]
    assert len(st["stack"]) >= 2, "fixture must stop mid-bisection"
    edges, cap = list(_E6_POOL), 4
    if corrupt == "algo":
        st["algo"] = "one_at_a_time"
    elif corrupt == "cap":
        cap = 8                                               # a different sizing policy
    elif corrupt == "permuted_edges":
        edges[0], edges[5] = edges[5], edges[0]               # the relaunch's attribution reordered
    elif corrupt == "adjacent_duplicate_cut":
        st["cut_order"].append(st["cut_order"][-1])           # only a STRICT position check sees this
    elif corrupt == "cut_in_pending_region":
        st["cut_order"].append(list(_E6_POOL[st["stack"][0][0]]))
    elif corrupt == "overlapping_stack":
        st["stack"][0][1] += 1                                # overlaps its neighbour; nothing else moves
    elif corrupt == "cursor_moved":
        st["cursor"] += 1
    elif corrupt == "known_fail_on_left":
        st["stack"][0][3], st["stack"][0][4] = "L", True      # a state never observed, marked as failed
    elif corrupt == "processed_disagrees":
        st["processed"] -= 1
    elif corrupt == "stats_dont_add_up":
        st["stats"]["n_commits"] += 1
    with pytest.raises(ValueError):
        block_single_pass_eliminate(edges, _e6_rec, 1.0, cap, resume=st)


def test_E6_the_uncorrupted_mid_bisection_state_resumes():
    # The negative controls above are only meaningful if the state they corrupt is accepted.
    st = _block_state_after(_E6_POOL, _e6_rec, 4, 7)
    assert block_single_pass_eliminate(_E6_POOL, _e6_rec, 1.0, 4, resume=st)["kept"] == [("m", 13)]


def test_E7_cut_order_is_a_visit_subsequence_even_under_a_non_monotone_arbiter():
    # WHY: the rigorous K-sweep walks survivors-first then `reversed(cut_order)`, and that is a
    # descending-|attribution| ranking ONLY because cut_order is a subsequence of the visiting
    # order. If a block appended its members in any other order, the walk order -- and therefore
    # both_K -- would depend on cut TIMING, and block and one-at-a-time curves would be
    # incomparable even where their survivor sets agree.
    rng = random.Random(3)
    for t in range(300):
        table = {}

        def rec(cut, table=table, r=random.Random(t)):
            k = frozenset(cut)
            if k not in table:
                table[k] = 0.0 if r.random() < 0.3 else 1.0
            return table[k]

        edges = list(range(rng.randint(1, 50)))
        out = block_single_pass_eliminate(edges, rec, 1.0, rng.choice([2, 8, 64]))
        cut = set(out["cut_order"])
        assert out["cut_order"] == [e for e in edges if e in cut]
        assert len(out["kept"]) + len(out["cut_order"]) == len(edges)


def test_block_cap_1_is_refused_rather_than_silently_meaning_something_else():
    # cap 1 IS single_pass_eliminate; accepting it here would give two implementations of the
    # one-at-a-time protocol, and the one that ran would depend on a flag default.
    with pytest.raises(ValueError, match="cap >= 2"):
        block_single_pass_eliminate(["a"], lambda c: 1.0, 1.0, 1)


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


def test_patching_and_path_patch_preserve_persistent_ablation(fix):
    model, wrapped = fix
    ep = _episode()
    module = next(iter(wrapped))
    baseline = ablation_overrides([(module, 0)])
    res = attribute(model, wrapped, ep, K=8, baseline_overrides=baseline)
    sel = select(res["A"], n_positive=6, n_negative=6)
    selected = [(m, d) for m, d, _ in sel["positive"] + sel["negative"]]
    nodes, info = candidate_nodes(res["A"], res["grads"], selected, tau=0.2, cap=4)
    edges = edge_scores_patching(
        model,
        wrapped,
        res["full_trigger"],
        nodes,
        info,
        res["a0"],
        res["a1"],
        baseline_overrides=baseline,
    )
    assert edges
    effective = _read_under(model, res["full_trigger"], wrapped, baseline)
    assert torch.count_nonzero(effective[module][..., 0]) == 0
    u, v = max(edges, key=lambda edge: abs(edges[edge]))
    path_patch_edge(
        model,
        wrapped,
        ep,
        u,
        v,
        nodes,
        res["a0"],
        res["a1"],
        grad_v=info[v]["grad"],
        baseline_overrides=baseline,
    )
    effective = _read_under(model, res["full_trigger"], wrapped, baseline)
    assert torch.count_nonzero(effective[module][..., 0]) == 0


# --- region permutation null (Exp-10 re-derivation) --------------------------------

def test_region_position_profile_reproduces_the_observed_sums():
    """The null is only trustworthy if the profile it is built from is EXACT.

    The whole point of collapsing edges to positions is speed; if that collapse were lossy,
    the observed value and the null would be computed from different quantities and the
    percentile would be meaningless.
    """
    pos_edges = {
        (("m.k_proj", 3, 5), ("m.o_proj", 7, 38)): 2.0,
        (("m.k_proj", 3, 5), ("m.o_proj", 7, 39)): -1.0,
        (("m.v_proj", 4, 38), ("m.o_proj", 7, 39)): 0.5,
    }
    nodes = [("m.k_proj", 3, 5), ("m.v_proj", 4, 38), ("m.o_proj", 7, 38)]
    amp = {("m.k_proj", 3, 5): 2.0, ("m.v_proj", 4, 38): 0.5}
    src_by_pos, dst_by_pos, nodes_by_pos = region_position_profile(
        pos_edges, nodes, lambda n: amp[n]
    )
    # weights are |E_A| / amp_u, keyed by the SOURCE's amplitude in both maps
    assert src_by_pos[5] == pytest.approx((2.0 + 1.0) / 2.0)
    assert src_by_pos[38] == pytest.approx(0.5 / 0.5)
    assert dst_by_pos[38] == pytest.approx(2.0 / 2.0)
    assert dst_by_pos[39] == pytest.approx(1.0 / 2.0 + 0.5 / 0.5)
    assert nodes_by_pos == {5: 1, 38: 2}


def test_shift_permutation_preserves_span_contiguity_and_sizes():
    """`shift` must keep regions CONTIGUOUS -- that is what makes it the conservative null.

    A scattered tag lands on average-out-degree positions, which understates the null and
    would flatter the observed split. If this ever degrades into a free shuffle, the
    permutation test silently becomes the anti-conservative one and the Exp-10 verdict it
    is meant to adjudicate would be biased toward "survives".
    """
    region = {p: ("tag" if p in (2, 3) else "completion" if p >= 6 else "shared")
              for p in range(9)}
    sizes = Counter(region.values())
    rng = random.Random(0)
    seen_offsets = set()
    for _ in range(60):
        perm = permuted_region(region, rng, "shift")
        assert Counter(perm.values()) == sizes, "shift changed the label counts"
        tag_positions = sorted(p for p, r in perm.items() if r == "tag")
        # contiguous, allowing the circular wrap
        gaps = [b - a for a, b in zip(tag_positions, tag_positions[1:])]
        assert all(g == 1 for g in gaps) or tag_positions == [0, 8], tag_positions
        seen_offsets.add(tag_positions[0])
    assert len(seen_offsets) > 3, "shift is not actually moving the spans"


def test_free_permutation_scatters_but_keeps_counts():
    """`free` is the secondary null; it may scatter, but must not invent or drop labels."""
    region = {p: ("tag" if p in (2, 3) else "shared") for p in range(9)}
    rng = random.Random(0)
    scattered = 0
    for _ in range(60):
        perm = permuted_region(region, rng, "free")
        assert Counter(perm.values()) == Counter(region.values())
        tag_positions = sorted(p for p, r in perm.items() if r == "tag")
        if tag_positions[1] - tag_positions[0] != 1:
            scattered += 1
    assert scattered > 0, "free permutation never scattered -- it is behaving like shift"


def test_permuted_region_rejects_unknown_mode():
    """Fail loud: a typo'd mode must not silently fall through to an unintended null."""
    with pytest.raises(ValueError, match="unknown permutation mode"):
        permuted_region({0: "tag"}, random.Random(0), "shuffle")


def test_null_detects_position_driven_concentration():
    """The null must FLAG a split that is fully explained by where the tag sits.

    Construction: every edge is sourced at position 0 and the weights carry no information
    about the label at all. A tag at position 0 then scores 100% of source weight purely
    because of its position -- exactly the Exp-10 confound. A null that controls for
    position must place this observation well inside its own distribution (large p),
    because relabelling puts some other region on position 0 just as often.
    """
    region = {p: ("tag" if p == 0 else "shared" if p < 6 else "completion") for p in range(9)}
    src_by_pos = {0: 100.0}
    rng = random.Random(1)
    hits = 0
    draws = 400
    for _ in range(draws):
        perm = permuted_region(region, rng, "shift")
        # all weight lands on whichever region now owns position 0
        if perm[0] == "tag":
            hits += 1
    p = (1 + hits) / (1 + draws)
    assert 0.02 < p < 0.35, (
        f"p={p:.3f}: a purely position-driven 100% split must NOT look significant; "
        "the tag owns 1 of 9 positions so the null should reproduce it ~1/9 of the time"
    )
    assert src_by_pos[0] == 100.0  # observed statistic itself is unchanged by the null
