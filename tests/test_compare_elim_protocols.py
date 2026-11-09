"""analysis/compare_elim_protocols.py: the pre-registered block-vs-one-at-a-time verdict.

The tool decides whether a protocol change is adopted for a multi-day campaign, so it is itself
checked the way the campaign is: every rule is exercised on synthetic circuit files that PASS it and
on files that FAIL it. A verdict tool that can only return PASS is worse than none.

(New file rather than an addition to tests/test_circuit_search_grid.py, which covers the search's K
grid and checkpoint guards -- one test module per module, as everywhere else here.)
"""

import json
from pathlib import Path

import pytest

from analysis.compare_elim_protocols import (jaccard, load_circuit, necessity_k, replay_block_log,
                                             rung_delta, run)
from src.clcd.exp_circuit_search import order_sha256

_MOD = "base_model.model.model.layers.20.mlp.up_proj"
_POOL = [[_MOD, i] for i in range(8)]          # attribution order, strongest first
_VISIT = list(reversed(_POOL))                 # the visiting order: weakest first
_KS = [10, 20, 30, 40, 50]
_CELLS = [("r64_dense", "l20", 42), ("r64_dense", "l20", 43),
          ("r64_k8", "l20", 42), ("r64_k8", "l20", 43)]


def _order_file(orders, key):
    arm, fam, seed = key
    p = Path(orders) / f"{arm}_{fam}_seed{seed}.order.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    order = [list(e) for e in _VISIT]
    p.write_text(json.dumps({"schema": "elim_visit_order_v1", "adapter": f"adapters/{arm}_{fam}_s{seed}",
                             "n_pool": len(order), "sha256": order_sha256(order), "order": order}))
    return order


def _curve(both_K, nec_K):
    """A rigorous curve whose smallest certifying K is both_K and whose smallest ablate==0 is nec_K."""
    rows = []
    for K in _KS:
        rows.append({"K": K, "keep_only": 0.9 if (both_K and K >= both_K) else 0.1,
                     "ablate": 0.0 if (nec_K is not None and K >= nec_K) else 0.02,
                     "suff_se": 0.01, "suff_shortfall": 0.0})
    return rows


def _write_cell(root, key, survivors_idx, both_K=30, nec_K=20, *, cap=1, block=None, calls=9,
                wall=100.0, status="ok", resumed=False, ks=None, curve=None, sha=None, n_cut=None):
    """One arm's circuit JSON for one cell. `survivors_idx` indexes the VISITING order."""
    arm, fam, seed = key
    p = Path(root) / arm / "elim" / f"{fam}_seed{seed}_circuit.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    survivors = [_VISIT[i] for i in sorted(survivors_idx, reverse=True)]     # attribution order
    ks = _KS if ks is None else ks
    d = {"kept_latents": [], "n_kept_latents": 0, "both_K": both_K, "status": status,
         "intact_asr": 0.99, "n_backdoor": 1000, "ks_requested": _KS, "ks_evaluated": ks,
         "n_all_latents": 64, "order_len": len(_POOL), "adapter": f"adapters/{arm}_{fam}_s{seed}",
         "curve": _curve(both_K, nec_K) if curve is None else curve,
         "elim": {"arbiter": "paired_2se", "pool": "all", "pool_n": len(_POOL),
                  "n_survivors": len(survivors),
                  "n_cut": len(_POOL) - len(survivors) if n_cut is None else n_cut,
                  "protocol": {"elim_block_cap": cap, "block_policy": "adaptive_block_bisect_v1" if block else None,
                               "visit": "weakest_abs_attribution_first",
                               "visit_order_sha256": sha or order_sha256([list(e) for e in _VISIT]),
                               "order_source": "order_from", "order_file": "o.json",
                               "survivors": survivors, "n_arbiter_calls": calls, "resumed": resumed,
                               "elim_wall_s": wall, "block": block}}}
    p.write_text(json.dumps(d, indent=2))
    return p


def _block_stats(survivors_idx, n_tests):
    n_keep = len(survivors_idx)
    n_cut = len(_POOL) - n_keep
    return {"n_tests": n_tests, "n_reused": 0, "n_commits": n_cut,
            "top_pass_by_size": {"1": n_cut}, "top_fail_by_size": {"1": n_keep},
            "tests_pass_by_size": {"1": n_cut}, "tests_fail_by_size": {"1": n_keep},
            "max_size_tested": 1, "max_bisection_depth": 0}


def _block_log(logs, key, survivors_idx):
    """The [elim-block] record of a run that tested every latent as its own top-level block."""
    arm, fam, seed = key
    p = Path(logs) / f"{arm}_{fam}_seed{seed}_search.out"
    p.parent.mkdir(parents=True, exist_ok=True)
    lines, t = ["[elim] cheap arbiter n=80"], 0
    for i in range(len(_POOL)):
        t += 1
        res = "fail" if i in survivors_idx else "pass"
        lines.append(f"[elim-block] #{t}+0 [{i},{i + 1}) size=1 depth=0 T -> {res}")
    p.write_text("\n".join(lines) + "\n")
    return p


def _tree(tmp_path, *, survivors=(1, 5), b_survivors=None, b_calls=3, **over):
    """A complete, PASSING A/B/C/D/E fixture; `over` perturbs arm B's cells."""
    root = Path(tmp_path)
    b_survivors = survivors if b_survivors is None else b_survivors
    for key in _CELLS:
        _order_file(root / "orders", key)
        _write_cell(root / "oaat", key, survivors, calls=9, wall=100.0)
        _write_cell(root / "rep", key, survivors, calls=9, wall=101.0)
        kw = {"cap": 64, "calls": b_calls, "block": _block_stats(b_survivors, b_calls), **over}
        _write_cell(root / "block", key, b_survivors, **kw)
        _block_log(root / "logs_block", key, b_survivors)
        _write_cell(root / "adaptive_default", key, survivors, calls=9, wall=50.0)
        _write_cell(root / "adaptive_early", key, survivors, calls=9, wall=60.0)
    return root


def _run(root, **kw):
    # n_cells: this fixture is the pre-registered SHAPE (dense and sparse arms, several seeds) at 4
    # cells instead of 20. The count is passed explicitly, never defaulted, so the tool's own default
    # stays the registered 20 and test_the_registered_scope_is_the_default can hold it there.
    return run(str(root / "oaat"), str(root / "block"), str(root / "rep"), str(root / "orders"),
               str(root / "logs_block"),
               adaptive_default=kw.get("d", str(root / "adaptive_default")),
               adaptive_early=kw.get("e", str(root / "adaptive_early")),
               n_cells=kw.get("n_cells", len(_CELLS)))


def test_a_clean_comparison_adopts_the_protocol(tmp_path):
    v = _run(_tree(tmp_path))
    assert [v[k]["pass"] for k in ("V0", "V1", "V2", "V3", "VD", "VE")] == [True] * 6
    assert v["decision"]["adopt_block"] is True
    assert v["decision"]["adopt_adaptive_default"] is True and v["decision"]["adopt_adaptive_early"] is True
    # and the numbers it reports are the ones in the files, not a summary of them
    cell = v["V2"]["per_cell"]["r64_dense/l20/42"]
    assert cell["calls_A"] == 9 and cell["calls_B"] == 3 and cell["jaccard"] == 1.0


def test_V0_failure_stops_the_whole_comparison(tmp_path):
    # WHY: resume and identical-state reuse both assume the arbiter is a deterministic function of
    # the cut set. If the replicate disagrees with its own reference that premise is false, and
    # every later comparison is measuring noise. Nothing else may be evaluated, and NOTHING may be
    # reported as passing.
    root = _tree(tmp_path)
    _write_cell(root / "rep", _CELLS[0], (1, 6))            # one survivor differs
    v = _run(root)
    assert v["V0"]["pass"] is False
    assert not any(k in v for k in ("V1", "V2", "V3", "VD", "VE"))
    assert v["decision"] == {"adopt_block": False, "adopt_adaptive_default": False,
                             "adopt_adaptive_early": False, "reason": v["decision"]["reason"]}
    assert "V0 FAILED" in v["decision"]["reason"]


@pytest.mark.parametrize("broken,over", [
    ("sha_equals_A", dict(sha="0" * 64)),
    ("not_resumed", dict(resumed=True)),
    ("cap_as_registered", dict(cap=256)),
    ("pool_accounted", dict(n_cut=99)),
])
def test_V1_catches_a_block_run_that_does_not_add_up(tmp_path, broken, over):
    # WHY: V1 is the implementation check. Every one of these would leave a plausible-looking
    # circuit file behind -- a different visiting order, a resumed run whose telemetry covers only
    # the last process, a cap nobody registered, a pool that does not account for its latents.
    v = _run(_tree(tmp_path, **over))
    assert v["V1"]["pass"] is False
    assert all(c["detail"][broken] is False for c in v["V1"]["checks"])
    assert v["decision"]["adopt_block"] is False


def test_V1_catches_telemetry_that_contradicts_itself(tmp_path):
    bad = _block_stats((1, 5), 3)
    bad["n_commits"] += 1                                    # more commits than passing tests
    bad["max_size_tested"] = 128                             # larger than the cap it declares
    v = _run(_tree(tmp_path, block=bad))
    assert v["V1"]["pass"] is False
    det = v["V1"]["checks"][0]["detail"]
    assert det["commits_match_passes"] is False and det["max_size_within_cap"] is False


def test_V1_catches_a_log_that_does_not_replay_to_the_reported_survivors(tmp_path):
    # The circuit file SAYS the survivors; the [elim-block] lines say what was tested and cut. If
    # those can disagree, the reconstruction record is decoration.
    root = _tree(tmp_path)
    _block_log(root / "logs_block", _CELLS[0], (1, 6))
    v = _run(root)
    assert v["V1"]["pass"] is False
    assert v["V1"]["checks"][0]["detail"]["log_replays_to_survivors"] is False


def test_an_empty_or_truncated_block_log_raises_instead_of_replaying_to_no_cuts(tmp_path):
    order = _order_file(tmp_path / "orders", _CELLS[0])
    (tmp_path / "empty.out").write_text("[elim] cheap arbiter n=80\n")
    with pytest.raises(ValueError, match="no .elim-block. lines"):
        replay_block_log(tmp_path / "empty.out", order)
    (tmp_path / "part.out").write_text("[elim-block] #1+0 [0,4) size=4 depth=0 T -> pass\n")
    with pytest.raises(ValueError, match=r"cover \[0,4\) of a 8-latent pool"):
        replay_block_log(tmp_path / "part.out", order)
    (tmp_path / "gap.out").write_text("[elim-block] #1+0 [0,4) size=4 depth=0 T -> pass\n"
                                      "[elim-block] #2+0 [5,8) size=3 depth=0 T -> pass\n")
    with pytest.raises(ValueError, match="do not tile the pool"):
        replay_block_log(tmp_path / "gap.out", order)


def test_V2_flags_a_status_or_certificate_that_moved(tmp_path):
    # A protocol that turns an `ok` cell into `no_sufficient_subcircuit`, or moves both_K by 3 rungs
    # on this grid, is not "the same answer, fewer tests".
    root = _tree(tmp_path)
    _write_cell(root / "block", _CELLS[0], (1, 5), both_K=None, status="no_sufficient_subcircuit",
                cap=64, calls=3, block=_block_stats((1, 5), 3))
    v = _run(root)
    assert v["V2"]["pass"] is False
    assert v["V2"]["checks"][0]["pass"] is False                       # V2a status
    assert v["V2"]["per_cell"]["r64_dense/l20/42"]["d_rung_both_K"] is None

    root2 = _tree(tmp_path / "second")
    for key in _CELLS:
        _write_cell(root2 / "block", key, (1, 5), both_K=50, cap=64, calls=3, block=_block_stats((1, 5), 3))
    v2 = _run(root2)
    assert v2["V2"]["checks"][0]["pass"] is True                       # status still agrees ...
    assert v2["V2"]["checks"][1]["pass"] is False                      # ... but both_K moved 2 rungs
    assert v2["V2"]["checks"][1]["detail"]["deltas"] == [2, 2, 2, 2]


def test_V2_flags_a_necessity_K_that_moved_or_vanished(tmp_path):
    root = _tree(tmp_path)
    for key in _CELLS:
        _write_cell(root / "block", key, (1, 5), nec_K=None, cap=64, calls=3, block=_block_stats((1, 5), 3))
        _block_log(root / "logs_block", key, (1, 5))
    v = _run(root)
    nec = [c for c in v["V2"]["checks"] if "necessity-K" in c["check"]][0]
    assert nec["pass"] is False and nec["detail"]["deltas"] == [None] * 4
    assert v["V2"]["per_cell"]["r64_k8/l20/43"]["nec_K_B"] is None


def test_V2_flags_a_survivor_set_that_drifted(tmp_path):
    v = _run(_tree(tmp_path, b_survivors=(0, 2, 3, 4, 6, 7)))          # Jaccard 0/8 = 0.0
    jac = [c for c in v["V2"]["checks"] if "Jaccard" in c["check"]][0]
    assert jac["pass"] is False and jac["detail"]["min"] == 0.0
    assert v["decision"]["adopt_block"] is False


def test_V2_quorum_is_the_pre_registered_fraction_not_all_cells(tmp_path):
    # 18/20 and 9/10 in the pre-registration are both 90%: one cell may move 2 rungs, two may not.
    root = Path(tmp_path)
    cells = [("r64_dense", "l20", s) for s in range(42, 52)]
    for i, key in enumerate(cells):
        _order_file(root / "orders", key)
        _write_cell(root / "oaat", key, (1, 5), both_K=30)
        _write_cell(root / "rep", key, (1, 5), both_K=30)
        _write_cell(root / "block", key, (1, 5), both_K=50 if i == 0 else 40, cap=64, calls=3,
                    block=_block_stats((1, 5), 3))
        _block_log(root / "logs_block", key, (1, 5))
    v = run(str(root / "oaat"), str(root / "block"), str(root / "rep"), str(root / "orders"),
            str(root / "logs_block"), n_cells=len(cells))
    both = v["V2"]["checks"][1]
    assert both["detail"]["n_within_1"] == 9 and both["detail"]["n_within_2"] == 10
    assert both["pass"] is True                                        # 9/10 within one rung
    _write_cell(root / "block", cells[1], (1, 5), both_K=50, cap=64, calls=3, block=_block_stats((1, 5), 3))
    v2 = run(str(root / "oaat"), str(root / "block"), str(root / "rep"), str(root / "orders"),
             str(root / "logs_block"), n_cells=len(cells))
    assert v2["V2"]["checks"][1]["detail"]["n_within_1"] == 8 and v2["V2"]["checks"][1]["pass"] is False


def test_V3_prices_the_protocol_separately_for_dense_and_sparse(tmp_path):
    # The pre-registered bars differ: sparse must be a big win (<= 0.45x), dense need only not be
    # slower (<= 1.00x). Averaging the two would let a sparse win pay for a dense loss.
    v = _run(_tree(tmp_path, b_calls=12))
    v3 = {c["check"].split(":")[0]: c for c in v["V3"]["checks"]}
    assert v3["V3 dense"]["pass"] is False and v3["V3 sparse"]["pass"] is False
    v_ok_dense = _run(_tree(tmp_path / "b", b_calls=9))
    v3b = {c["check"].split(":")[0]: c for c in v_ok_dense["V3"]["checks"]}
    assert v3b["V3 dense"]["pass"] is True                              # exactly 1.00x is allowed
    assert v3b["V3 sparse"]["pass"] is False                            # but 1.00x is not a sparse win
    assert v3b["V3 sparse"]["detail"]["ratio"] == 1.0


def test_VD_requires_decision_identity_and_a_speedup(tmp_path):
    # D only SKIPS the ablate generation after a sufficiency failure; if it changes a decision, the
    # "decision-identical by code reading" claim is wrong, and if it is not faster it buys nothing.
    root = _tree(tmp_path)
    _write_cell(root / "adaptive_default", _CELLS[0], (1, 5), wall=200.0)     # slower than A
    _write_cell(root / "adaptive_default", _CELLS[1], (1, 6), wall=50.0)      # different survivors
    v = _run(root)
    assert v["VD"]["pass"] is False and v["decision"]["adopt_adaptive_default"] is False
    assert v["VD"]["checks"][0]["detail"]["faster"] is False
    assert v["VD"]["checks"][1]["detail"]["survivors"] is False


def test_VE_needs_both_agreement_and_a_real_wall_time_saving(tmp_path):
    # E decides on partial evidence, so agreement alone is not enough to justify it.
    root = _tree(tmp_path)
    for key in _CELLS:
        _write_cell(root / "adaptive_early", key, (1, 5), wall=90.0)          # 0.9x > 0.70x
    v = _run(root)
    wall = [c for c in v["VE"]["checks"] if "wall" in c["check"]][0]
    assert wall["pass"] is False and v["decision"]["adopt_adaptive_early"] is False
    assert wall["detail"]["ratio"] == pytest.approx(0.9)
    # VE reads the DENSE cells only (2 of the 4 here), as pre-registered
    assert v["VE"]["n_cells"] == 2


def test_a_missing_measurement_raises_instead_of_defaulting(tmp_path):
    # The trap this tool must not inherit: `row.get("ablate", 1.0)` turns "never measured" into a
    # number, and `.get("protocol")` turns "not recorded" into None.
    root = _tree(tmp_path)
    p = _write_cell(root / "block", _CELLS[0], (1, 5), cap=64, block=_block_stats((1, 5), 3))
    d = json.loads(p.read_text())
    del d["curve"][0]["ablate"]        # the FIRST row read: a missing value must not be walked past
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="has no 'ablate'"):
        necessity_k(load_circuit(p))

    d = json.loads(p.read_text())
    del d["elim"]["protocol"]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="no 'protocol' block"):
        load_circuit(p)

    d = json.loads(p.read_text())
    d["elim"] = None
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="elim is null"):
        load_circuit(p)


def test_an_empty_tree_is_an_error_not_a_pass(tmp_path):
    root = _tree(tmp_path)
    with pytest.raises(ValueError, match="nothing to compare"):
        run(str(root / "oaat"), str(tmp_path / "nowhere"), str(root / "rep"), str(root / "orders"),
            str(root / "logs_block"), n_cells=len(_CELLS))


def _cell_file(root, arm, key):
    return root / arm / key[0] / "elim" / f"{key[1]}_seed{key[2]}_circuit.json"


def test_a_partial_block_arm_raises_instead_of_shrinking_the_comparison(tmp_path):
    # WHY: a missing cell fails no rule -- it removes itself from every rule, one step in from the
    # empty tree above. With 16 of 20 arm-B circuits gone the verdict still read adopt_block: true,
    # with the "EVERY arm of the campaign" wording, on a V2 quorum of 4/4 instead of 18/20 and on V3
    # sums over 2 sparse + 2 dense cells. The order-file gate's skip, a SEARCH FAILED and an
    # unfinished cell all leave exactly that tree behind, and none of them is visible in the files
    # that remain. So coverage is required before any rule is evaluated, and short coverage raises
    # rather than returning a verdict.
    root = _tree(tmp_path)
    assert _run(root)["decision"]["adopt_block"] is True                 # intact, the tree passes
    _cell_file(root, "block", _CELLS[0]).unlink()
    with pytest.raises(ValueError, match=r"arm B \(block\) has 3 cells but is missing 1 of the 4"):
        _run(root)


def test_a_partial_control_arm_raises_for_its_own_registered_cells(tmp_path):
    # C and D are pre-registered at seed 42 of every arm (4 cells) and E at the dense cells (10), so
    # each one's coverage is derived from A's keys rather than from its own. Without this, one
    # surviving C cell certifies determinism "4/4" on 1/1 and gates the whole comparison open.
    for arm, key, pattern in (("rep", ("r64_k8", "l20", 42), r"arm C \(one-at-a-time replicate\)"),
                              ("adaptive_default", ("r64_dense", "l20", 42), r"arm D \(--adaptive_n"),
                              ("adaptive_early", ("r64_dense", "l20", 43), r"arm E \(--adaptive_n")):
        root = _tree(tmp_path / arm)
        _cell_file(root, arm, key).unlink()
        with pytest.raises(ValueError, match=pattern):
            _run(root)
    # ... and only for those cells: a seed-43 replicate and a sparse arm-E cell are outside the
    # registered scope, so dropping them is not short coverage (C is 4 cells, not "all of A").
    root = _tree(tmp_path / "unregistered")
    _cell_file(root, "rep", ("r64_k8", "l20", 43)).unlink()
    _cell_file(root, "adaptive_early", ("r64_k8", "l20", 42)).unlink()
    assert _run(root)["decision"]["adopt_block"] is True


def test_a_resumed_cell_raises_instead_of_being_priced_as_a_whole_run(tmp_path):
    # WHY: `resumed` means elim_wall_s and n_arbiter_calls cover only the process that finished the
    # cell (SPEC 5.1), and scripts/qwen15_phase1.sh auto-resumes any killed cell from its checkpoint,
    # so this is an ordinary operational state. VD adopts --adaptive_n on `wall_D < wall_A` alone and
    # VE on `sum(E) <= 0.70 x sum(A)`: a relaunched D or E cell passes the ONLY rule that decides its
    # adoption, and a relaunched A shrinks the denominator of V3, VD and VE. A partial timing is not
    # a smaller measurement of the same thing, so it must not be compared against a complete one.
    # (the counts are each arm's own registered scope: D is seed 42 of every arm, E the dense cells)
    for arm, pattern in (("adaptive_default", r"arm D \(--adaptive_n, default rungs\): 2 cell"),
                         ("adaptive_early", r"arm E \(--adaptive_n, early rungs\): 2 cell"),
                         ("oaat", r"arm A \(one-at-a-time\): 4 cell")):
        root = _tree(tmp_path / arm)
        assert _run(root)["decision"]["adopt_adaptive_default"] is True     # intact, the tree passes
        for key in _CELLS:
            _write_cell(root / arm, key, (1, 5), wall=5.0, resumed=True)
        with pytest.raises(ValueError, match=pattern):
            _run(root)
    # ... and only where a timing is priced: C is compared cell-by-cell on survivors, both_K, status
    # and curve, and E's sparse cells are outside its registered scope, so neither is short a
    # measurement anyone reads. (B's resumed is V1's `not_resumed`, a rule failure, not a raise.)
    root = _tree(tmp_path / "unpriced")
    for key in _CELLS:
        _write_cell(root / "rep", key, (1, 5), wall=5.0, resumed=True)
    _write_cell(root / "adaptive_early", ("r64_k8", "l20", 42), (1, 5), wall=5.0, resumed=True)
    assert _run(root)["decision"]["adopt_adaptive_early"] is True


def test_the_registered_scope_is_the_tools_own_default(tmp_path):
    # n_cells is not a campaign knob -- there is no CLI flag for it. The default is the 20 cells the
    # pre-registration names, so re-scoping the comparison means moving REGISTERED_N_CELLS on
    # purpose; a tree with a different number of arm-A cells is refused.
    root = _tree(tmp_path)
    with pytest.raises(ValueError, match="arm A has 4 cells, but this comparison is pre-registered "
                                         "for 20"):
        run(str(root / "oaat"), str(root / "block"), str(root / "rep"), str(root / "orders"),
            str(root / "logs_block"))


def test_cells_swept_on_different_grids_are_refused(tmp_path):
    # both_K is an index into the grid that was actually evaluated; comparing rungs across grids is
    # the bookkeeping error that once made dense circuits look larger than sparse ones.
    root = _tree(tmp_path)
    _write_cell(root / "block", _CELLS[0], (1, 5), cap=64, calls=3, block=_block_stats((1, 5), 3),
                ks=[10, 20, 30, 40])
    with pytest.raises(ValueError, match="grid mismatch"):
        _run(root)


def test_the_helper_quantities_are_what_they_claim():
    a, b = frozenset({1, 2, 3}), frozenset({2, 3, 4})
    assert jaccard(a, b) == 0.5 and jaccard(frozenset(), frozenset()) == 1.0
    cell = {"path": "x", "ks": _KS}
    assert rung_delta(cell, cell, 10, 30) == 2
    assert rung_delta(cell, cell, None, None) == 0        # both censored: agreement
    assert rung_delta(cell, cell, 10, None) is None       # one-sided: a disagreement, never 0
