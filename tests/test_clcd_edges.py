"""M7 edge-attribution mechanics (CPU, random fixture).

Numbers are noise by design; these assert the machinery is what we claim:
the DAG prune, candidate extraction, that Method-A patching measures a true
contrast, that the Method-B JVP matches a finite difference (so the two §8
estimators coincide where no gate flips), and that path-patching runs and is
null when there is nothing to patch.
"""

import pytest
import torch

from src.clcd.attribute import attribute
from src.clcd.edges import (
    candidate_nodes,
    compute_order,
    dag_valid,
    edge_scores_jvp,
    edge_scores_patching,
    greedy_edge_eliminate,
    single_pass_eliminate,
    grow_edge_guided,
    path_patch_edge,
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
    assert dag_valid(q0, o0)             # same position, downstream in compute order
    assert not dag_valid(o0, q0)         # same position, backward in compute order
    assert dag_valid(q0, o0_late)        # later position (attention-mediated)
    assert not dag_valid(o0_late, q0)    # earlier position -> forbidden
    assert not dag_valid(q0, q0)         # self-edge


def test_dag_valid_rejects_backward_layer_edges():
    # source at a LATER layer, dest at an EARLIER layer but a later position:
    # attention cannot carry info from a later layer back to an earlier one, yet
    # the bare p_v>p_u test would wrongly admit it.
    src_layer23_p3 = ("M.layers.23.self_attn.o_proj", 1, 3)
    dst_layer16_p5 = ("M.layers.16.mlp.down_proj", 2, 5)
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


def test_jvp_returns_zero_on_edges_with_no_gradient_path(fix):
    # The test above only ever passes the top-|E_A| edges, which are exactly the ones
    # that DO have a gradient path -- so it cannot see this case at all.
    #
    # inject() overrides m_u's OUTPUT while the pre-hook captures m_v's INPUT, so when
    # m_u == m_v the destination has no forward dependence on `leaf`. dag_valid admits
    # exactly those pairs (same module, p_v > p_u). autograd.grad would raise there under
    # its default allow_unused=False; 0.0 is the correct derivative, because a projection
    # is per-position and cannot carry information from p_u to p_v within one module.
    # This matters beyond correctness: edges_analysis has no try/except and writes nothing
    # until save_findings, so a raise here discards a whole run's results.
    model, wrapped, ep, res, nodes, info = _setup(fix)
    same_mod = [
        (u, v) for u in nodes for v in nodes
        if u[0] == v[0] and dag_valid(u, v)
    ]
    if not same_mod:
        pytest.skip("fixture produced no same-module DAG-valid pair")
    jvp = edge_scores_jvp(
        model, wrapped, res["full_trigger"], same_mod[:3], info, res["a0"], res["a1"]
    )
    assert set(jvp) == set(same_mod[:3])
    assert all(v == 0.0 for v in jvp.values()), jvp


# --- path-patch verification ----------------------------------------------------

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


def test_grow_seed_first_and_connected_only():
    circ = grow_edge_guided(_GRAPH, HUB, cap=10)
    assert circ[0] == HUB                       # seed is first
    assert ISO not in circ and OTHER not in circ  # disjoint component never pulled in
    assert set(circ) == {HUB, DET, ACT, ACT2}   # exactly the seed's connected component


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
    """detector(tag) -> relay(shared) -> actuator(completion), one node per region."""
    det = ("L.layers.0.self_attn.k_proj", 1, 1)   # tag
    rel = ("L.layers.0.self_attn.o_proj", 2, 3)   # shared
    act = ("L.layers.0.mlp.down_proj", 3, 5)      # completion
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
