"""src/clcd/relu_ablation.py: the ReLU ablation's own readouts (docs/relu-ablation-arm-c-preregistration.md).

Each test pins a value that, if wrong, changes a pre-registered verdict: a mis-ranked tie moves a
p-value across Holm's threshold, a concentration sorted by signed score reports the wrong N90, a
pole override that zeroes nothing reads as "this pole carries no backdoor", a pole count that
includes padding or position 0 changes which latents count as bipolar.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.clcd import relu_ablation as ra
from src.utils import wrap_topk_lora_modules


# --- statistics -------------------------------------------------------------------------------------


def test_exact_mwu_complete_separation_is_one_over_choose():
    assert ra.mwu_greater_exact([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) == pytest.approx(1 / math.comb(10, 5))
    # the wrong direction is not significant
    assert ra.mwu_greater_exact([6, 7, 8, 9, 10], [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_exact_mwu_uses_mid_ranks_for_ties():
    """Arm A's l20 survivors tie (28, 28). With all ten values equal, every relabelling has the same
    rank sum, so p must be exactly 1 -- an implementation that broke ties by order would not give 1."""
    assert ra.mwu_greater_exact([28] * 5, [28] * 5) == pytest.approx(1.0)
    # one tie across groups: test {28} shares a rank with ref's 28s
    # value pinned by brute force and scipy.stats.permutation_test (review round 4); counting the
    # tie as a full win instead of a half would give 10/252 and pass a looser assertion
    p = ra.mwu_greater_exact([28, 28, 29, 38, 29], [28, 40, 41, 42, 43])
    assert p == pytest.approx(11 / 252)


def test_exact_mwu_refuses_empty():
    with pytest.raises(ValueError):
        ra.mwu_greater_exact([], [1, 2])
    with pytest.raises(ValueError):
        ra.mwu_greater_exact([1.0, float("nan")], [1, 2])


def test_holm_is_step_down():
    # 6 tests: 0.004 <= 0.05/6, 0.009 <= 0.05/5, 0.02 > 0.05/4 -> stop; later ones not rejected
    got = ra.holm_fixed({"a": 0.004, "b": 0.009, "c": 0.02, "d": 0.0125, "e": 0.5, "f": 0.9})
    assert got == {"a": True, "b": True, "d": True, "c": False, "e": False, "f": False}
    # sorted: a .004 <= .05/6, b .009 <= .05/5, d .0125 <= .05/4, c .02 > .05/3 -> stop at c


def test_concentration_sorts_by_magnitude_not_sign():
    scores = [5.0, 1.0, -10.0, 0.5, -0.5]  # |.| = 10, 5, 1, .5, .5; total 17
    assert ra.concentration(scores, 0.5) == 1  # 10 >= 8.5
    assert ra.concentration(scores, 0.9) == 3  # 10+5+1 = 16 >= 15.3
    with pytest.raises(ValueError):
        ra.concentration([0.0, 0.0], 0.9)
    with pytest.raises(ValueError):
        ra.concentration([float("nan"), 1.0, 2.0], 0.9)


# --- R0(a) --------------------------------------------------------------------------------------------


def test_r0a_allows_only_the_arm_fields_and_paths():
    a = {"sleeper_experiment": {"lora": {"relu_latents": True, "topk_mode": "topk", "k": 5}}, "dump_path": "x"}
    b = {"sleeper_experiment": {"lora": {"relu_latents": False, "topk_mode": "abstopk", "k": 5}}, "dump_path": "y"}
    assert ra.r0a_problems(a, b) == []
    b["sleeper_experiment"]["lora"]["k"] = 8
    probs = ra.r0a_problems(a, b)
    assert len(probs) == 1 and "lora.k" in probs[0]
    b["sleeper_experiment"]["lora"]["k"] = 5
    b["new_key"] = 1  # a key on one side only differs
    assert len(ra.r0a_problems(a, b)) == 1


# --- R4 pole overrides ----------------------------------------------------------------------------------


def test_pole_override_zeroes_one_pole_and_counts_it():
    a = torch.tensor([[[1.0, -2.0, 0.0, 3.0, -4.0]]])
    neg, pos, everything = ra.PoleOverride("neg"), ra.PoleOverride("pos"), ra.PoleOverride("all")
    assert neg(a).tolist() == [[[1.0, 0.0, 0.0, 3.0, 0.0]]] and neg.zeroed == 2
    assert pos(a).tolist() == [[[0.0, -2.0, 0.0, 0.0, -4.0]]] and pos.zeroed == 2
    assert everything(a).abs().sum() == 0 and everything.zeroed == 4
    cols = ra.PoleOverride("neg", columns=[1])
    assert cols(a).tolist() == [[[1.0, 0.0, 0.0, 3.0, -4.0]]] and cols.zeroed == 1


def test_neg_then_pos_equals_all_and_neg_is_identity_on_nonnegative_latents():
    from src.clcd.latents import compose_overrides

    a = torch.tensor([[[1.0, -2.0, 0.0, 3.0, -4.0]]])
    wrapped = {"m": None}
    both = compose_overrides(baseline=ra.pole_overrides(wrapped, "pos"), intervention=ra.pole_overrides(wrapped, "neg"))
    assert torch.equal(both["m"](a), ra.PoleOverride("all")(a))
    relu_like = a.clamp(min=0)
    assert torch.equal(ra.PoleOverride("neg")(relu_like), relu_like)


def test_pole_overrides_refuse_unknown_modules_and_empty_circuits():
    """An uncertified search writes kept_latents=[]; ablating it would silently be the intact model."""
    with pytest.raises(KeyError):
        ra.pole_overrides({"m": None}, "neg", circuit=[("other", 0)])
    with pytest.raises(ValueError):
        ra.pole_overrides({"m": None}, "neg", circuit=[])
    with pytest.raises(ValueError):
        ra.PoleOverride("neg", columns=[])


def test_pole_observer_reports_which_poles_fire_and_changes_nothing():
    obs = ra._PoleObserver([0, 2, 3])
    # position 0 (first row) is skipped: its +1.0 on latent 0 and -9 on latent 3 are not counted
    a = torch.tensor([[[1.0, 5.0, 0.0, -9.0], [-1.0, 5.0, -2.0, 0.0], [-1.0, 0.0, 3.0, 0.0]]])
    assert torch.equal(obs(a), a)
    assert obs.counts.tolist() == [[0, 1, 0], [2, 1, 0]]
    assert obs.used(0) == [2] and obs.used(1) == [0, 2]


# --- R2 pole counts -------------------------------------------------------------------------------------


class _EmbLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(6, 4)
        self.linear = nn.Linear(4, 3, bias=False)

    def forward(self, ids):
        return self.linear(self.emb(ids))


def _tiny_arm_c():
    torch.manual_seed(0)
    model = get_peft_model(_EmbLinear(), LoraConfig(r=4, lora_alpha=8, target_modules=["linear"], lora_dropout=0.0))
    _, wrapped = wrap_topk_lora_modules(
        model, k=1, temperature=1.0, temperature_schedule="constant", k_schedule="constant", k_final=1,
        temperature_final=None, is_topk_experiment=True, set_train=False, relu_latents=False, topk_mode="abstopk")
    mod = next(iter(wrapped.values()))
    with torch.no_grad():
        mod.A_module.weight.copy_(torch.eye(4))
        # token t -> pre-activation row; k=1 by |s|: token 1 fires latent 0 (+), token 2 latent 0 (-),
        # token 3 latent 2 (-), token 4 latent 1 (+), token 5 (pad) would fire latent 3
        model.base_model.model.emb.weight.copy_(torch.tensor([
            [0.1, 0.0, 0.0, 0.0], [2.0, 0.1, 0.0, 0.0], [-2.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, -3.0, 0.1], [0.0, 1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 9.0]]))
    return model, {"linear": mod}


def test_pole_counts_by_sign_excluding_pad_and_position_zero():
    model, wrapped = _tiny_arm_c()
    pad = 5
    tokens = torch.tensor([[4, 1, 2, 3, pad], [1, 2, 2, 4, 3]])
    source = torch.tensor([0, 2])
    counts = ra.pole_counts(model, wrapped, tokens, source, pad, batch_size=1)["linear"]
    n = counts["n"]  # [S, 2, r]
    # row 0 (source 0), positions 1..3 (pos 0 and the pad excluded): token 1 -> l0+, 2 -> l0-, 3 -> l2-
    assert n[0, 0].tolist() == [1, 0, 0, 0] and n[0, 1].tolist() == [1, 0, 1, 0]
    # row 1 (source 2), positions 1..4: 2 -> l0-, 2 -> l0-, 4 -> l1+, 3 -> l2-
    assert n[2, 0].tolist() == [0, 1, 0, 0] and n[2, 1].tolist() == [2, 0, 1, 0]
    assert counts["positions"].tolist() == [3, 0, 4]
    assert counts["mass"][2, 1, 0].item() == pytest.approx(4.0)  # two |-2| firings


def test_bipolar_rule_needs_both_share_and_count():
    n_pos = torch.tensor([900, 950, 100, 50])
    n_neg = torch.tensor([100, 50, 900, 50])
    assert ra.bipolar(n_pos, n_neg).tolist() == [True, False, True, False]


def _r4_record(arm, neg_zeroed, all_zeroed=10, neg_eq_intact=True, nested_eq_all=True, circuit=None, all_fires=0):
    return {"arm": arm, "circuit": circuit, "neg_then_pos_equals_all": nested_eq_all, "neg_equals_intact": neg_eq_intact,
            "conditions": {"neg": {"zeroed_entries": neg_zeroed},
                           "all": {"zeroed_entries": all_zeroed, "eval_triggered": {"fires": all_fires}}}}


def test_r4_self_checks_fail_the_record_that_proves_nothing():
    assert ra.r4_problems(_r4_record("A", 0)) == []
    assert ra.r4_problems(_r4_record("A", 3)) != []  # arm A has no negative latents to zero
    assert ra.r4_problems(_r4_record("A", 0, neg_eq_intact=False)) != []
    assert ra.r4_problems(_r4_record("C", 0)) != []  # arm C's negative override did nothing
    assert ra.r4_problems(_r4_record("C", 7)) == []
    assert ra.r4_problems(_r4_record("B", 0)) == []  # arm B may legitimately have none
    assert ra.r4_problems(_r4_record("B", 0, all_zeroed=0)) != []
    assert ra.r4_problems(_r4_record("C", 7, nested_eq_all=False)) != []


def test_r4_circuit_ablation_must_reproduce_the_certificate():
    """Condition (c) is the search's own necessity measurement: whole-circuit ablation fires 0."""
    assert ra.r4_problems(_r4_record("C", 7, circuit="c.json", all_fires=0)) == []
    assert ra.r4_problems(_r4_record("C", 7, circuit="c.json", all_fires=2)) != []
    assert ra.r4_problems(_r4_record("C", 7, circuit=None, all_fires=900)) == []  # whole adapter: no certificate


# --- analysis: which cells enter the pre-registered tests ----------------------------------------------


def _r0b(verdict, diff=0.0, nnz=(4, 5), selected=5789, tokens=1158, problem="m: nonzero latents per token 4..5, expected exactly 5"):
    return {"verdict": verdict, "problems": [] if verdict == "PASS" else [problem],
            "per_module": {"m": {"max_abs_diff": diff, "nnz_min": nnz[0], "nnz_max": nnz[1],
                                 "selected": selected, "tokens": tokens, "few_positive": 0}}}


def test_r0b_count_rule_only_requires_an_identical_forward():
    assert ra.r0b_count_rule_only(_r0b("FAIL"))  # one exact-zero selected latent, diff 0
    assert not ra.r0b_count_rule_only(_r0b("FAIL", diff=0.01))  # the forward differs: a real failure
    assert not ra.r0b_count_rule_only(_r0b("FAIL", problem="m: max|diff| 0.01 vs the arm's definition"))
    assert not ra.r0b_count_rule_only(_r0b("FAIL", selected=1158 * 3))  # many latents missing per token
    assert not ra.r0b_count_rule_only(_r0b("PASS"))


_UNSET = object()


def _cell(arm, fam="l20", usable=True, r0b="PASS", r0a=_UNSET, survivors=30, n90=100, node_ok=True,
          count_rule=False, args=_UNSET, status="ok"):
    c = {"arm": arm, "family": fam, "gate": {"usable": usable, "verdict": "PASS" if usable else "FAIL"}}
    if r0b is not None:
        c["r0b"] = {"verdict": r0b, "count_rule_only": count_rule}
    if arm != "A" and r0a is not None:
        c["r0a"] = [] if r0a is _UNSET else r0a
    if fam == "l20":
        c["node_gate"] = {"usable": node_ok, "verdict": "PASS" if node_ok else "FAIL"}
        c["r1"] = {"status": status, "survivors": survivors, "args_check": {} if args is _UNSET else args}
    else:
        c["r1p"] = {"N90": n90}
    return c


def test_tests_admit_only_cells_that_passed_their_gates():
    cells = {
        "a1": _cell("A", survivors=10), "a2": _cell("A", survivors=11),
        "c1": _cell("C", survivors=50),
        "c_nodegate": _cell("C", node_ok=False, survivors=1),   # l20 R1 is gated on the node record
        "c_nor0b": _cell("C", r0b=None, survivors=1),          # never measured
        "c_r0a": _cell("C", r0a=["lora.k: 5 vs 8"], survivors=1),
        "c_r0a_missing": _cell("C", r0a=None, survivors=1),    # never computed is not a pass
        "c_args": _cell("C", args={"n_cheap": [80, 40]}, survivors=1),
        "c_args_missing": _cell("C", args=None, survivors=1),
        "c_unsat": _cell("C", status="unsaturated", survivors=1),
        "c_nosuff": _cell("C", status="no_sufficient_subcircuit", survivors=7),  # enters as the whole pool
        "c_cr": _cell("C", r0b="FAIL", count_rule=True, survivors=60),
        "c_real": _cell("C", r0b="FAIL", count_rule=False, survivors=1),
    }
    primary = ra.tests({"cells": cells}, strict_r0b=False)
    assert primary["tests"]["l20:C>A"]["C"] == [50, 60, ra.POOL_L20]
    assert primary["tests"]["l20:C>A"]["A"] == [10, 11]
    assert set(primary["excluded"]) == {f"{c} [R1]" for c in ("c_nodegate", "c_nor0b", "c_r0a", "c_r0a_missing", "c_args",
                                                                 "c_args_missing", "c_unsat", "c_real")}
    strict = ra.tests({"cells": cells}, strict_r0b=True)["tests"]["l20:C>A"]
    assert strict["C"] == [50, ra.POOL_L20]
    # A40 gate does not decide l20 R1 admission; the node gate does
    assert ra.admission(_cell("C", usable=False), strict_r0b=False) is None


def test_holm_counts_every_named_test_even_when_uncomputable():
    """Two computable tests out of six must not make Holm's threshold 0.05/2."""
    p = {n: None for n in ra.TEST_NAMES}
    assert len(p) == 8   # Amendment 11: the six of Amendment 8 + two l17_25 circuit tests
    p["l20:B>A"] = 0.008
    assert ra.holm_fixed(p)["l20:B>A"] is False   # 0.008 > 0.05/8
    p["l20:B>A"] = 0.006
    assert ra.holm_fixed(p)["l20:B>A"] is True    # 0.006 <= 0.05/8
    assert all(v is False for k, v in ra.holm_fixed(p).items() if k != "l20:B>A")


def test_module_name_matching_ignores_the_peft_prefix_but_not_the_layer():
    a = {("base_model.model.model.layers.20.mlp.up_proj", 3)}
    b = {("model.layers.20.mlp.up_proj", 3), ("model.layers.20.mlp.up_proj", 4)}
    assert len(ra._match(a, b)) == 1
    assert ra._match({("base_model.model.model.layers.2.mlp.up_proj", 3)}, b) == set()


# --- round 9: guards on the analysis plumbing -----------------------------------------------------------


def test_tests_apply_holm_over_all_six_even_with_one_computable_test():
    """p=0.02 on the only computable test must NOT be rejected (0.02 > 0.05/6), as Amendment 8 rule 3 fixes."""
    cells = {f"a{i}": _cell("A", survivors=v) for i, v in enumerate([1, 2, 3, 4, 5])}
    cells.update({f"c{i}": _cell("C", survivors=v) for i, v in enumerate([3, 6, 7, 8, 9])})
    d = ra.tests({"cells": cells}, strict_r0b=False)["tests"]["l20:C>A"]
    assert 0.0083 < d["p"] < 0.05
    assert d["holm_reject"] is False


def test_holm_fixed_stops_at_the_first_non_rejection():
    p = {n: None for n in ra.TEST_NAMES}
    p["l20:B>A"], p["l20:C>A"] = 0.0065, 0.007   # 0.0065 > 0.05/8 stops; 0.007 <= 0.05/7 must not then pass
    got = ra.holm_fixed(p)
    assert got["l20:B>A"] is False and got["l20:C>A"] is False


def test_power_fields_and_median_ratio_are_the_pre_registered_ones():
    cells = {"a1": _cell("A", survivors=20), "a2": _cell("A", survivors=30),
             "c1": _cell("C", survivors=25), "c2": _cell("C", survivors=27), "c3": _cell("C", survivors=29)}
    d = ra.tests({"cells": cells}, strict_r0b=False)["tests"]["l20:C>A"]
    assert d["min_attainable_p"] == pytest.approx(1 / 10)   # C(5, 3) = 10
    assert d["median_ratio"] == pytest.approx(27 / 25)       # test median over reference median
    # 3 vs 4: min p = 1/35 = 0.029 lies between 0.05/6 and 0.05, so only the Holm threshold calls it underpowered
    cells = {f"a{i}": _cell("A", survivors=v) for i, v in enumerate([1, 2, 3])}
    cells.update({f"c{i}": _cell("C", survivors=v) for i, v in enumerate([4, 5, 6, 7])})
    d = ra.tests({"cells": cells}, strict_r0b=False)["tests"]["l20:C>A"]
    assert d["min_attainable_p"] == pytest.approx(1 / 35)
    assert d["underpowered"] is True
    cells = {f"a{i}": _cell("A", survivors=v) for i, v in enumerate([1, 2, 3, 4, 5])}
    cells.update({f"c{i}": _cell("C", survivors=v) for i, v in enumerate([6, 7, 8, 9, 10])})
    assert ra.tests({"cells": cells}, strict_r0b=False)["tests"]["l20:C>A"]["underpowered"] is False  # 1/252


def test_args_diff_reports_a_protocol_difference_and_ignores_paths():
    ref = {"n_cheap": 80, "out": "x", "adapter": "a", "elim_block_cap": 64}
    assert ra.args_diff(ref, {**ref, "out": "y", "adapter": "b"}) == {}
    assert ra.args_diff(ref, {**ref, "n_cheap": 40}) == {"n_cheap": (80, 40)}
    assert ra.args_diff(ref, {**ref, "new_flag": 1}) == {"new_flag": ("<missing>", 1)}


def test_identity_problems_exclude_a_cell():
    c = _cell("C", survivors=5)
    c["identity_problems"] = [("all", "R0(b) checked as signed")]
    assert "identity" in ra.admission(c, strict_r0b=False)
    # a problem scoped to another readout does not exclude the cell from R1
    c["identity_problems"] = [("R4", "R4 circuit has 3 latents, both_K 5")]
    assert ra.admission(c, strict_r0b=False) is None


def test_failed_jobs_are_told_apart_from_pending_ones(tmp_path):
    (tmp_path / "queue_logs").mkdir()
    (tmp_path / "queue_logs" / "q.log").write_text(
        "[2026-11-11 05:48 g4] r42_k5_norelu_all_s44.json FAILED rc=1 -- see x\n"
        "[2026-11-11 05:49 g4] r42_k5_abs_l20_s42_whole.json done\n")
    # a training failure in train_*.log retried later from requeue_*.log (sorts EARLIER by name)
    (tmp_path / "queue_logs" / "train_x.log").write_text("[2026-11-11 06:00 g0] r42_k5_l20_s44.ok FAILED rc=1 -- see t\n")
    (tmp_path / "queue_logs" / "requeue_x.log").write_text("[2026-11-11 06:07 g3] RUN r42_k5_l20_s44.ok\n"
                                                           "[2026-11-11 06:41 g3] r42_k5_l20_s44.ok done\n")
    (tmp_path / "node_mirror" / "queue_logs").mkdir(parents=True)
    (tmp_path / "node_mirror" / "queue_logs" / "n.log").write_text(
        "[2026-11-11 05:00 g1] r42_k5_abs_l20_s45.phase1 FAILED rc=6 -- see y\n"
        "[2026-11-11 05:01 g1] gate_a_r42_k5_l20_s46.json FAILED rc=1 -- see z\n"
        "[2026-11-11 05:30 g1] RUN gate_a_r42_k5_l20_s46.json\n"     # requeued: no longer failed
        "[2026-11-11 05:40 g1] r42_k5_abs_l20_s46.phase1 FAILED rc=3 -- see g\n")   # gate-ok refused: not run
    failed = ra._failed_outputs(tmp_path)
    assert failed == {"a40:r42_k5_norelu_all_s44.json": "failed", "node:r42_k5_abs_l20_s45.phase1": "failed",
                      "node:r42_k5_abs_l20_s46.phase1": "skipped (gate)"}
    assert ra._expected_output("r42_k5_abs_l20_s45: surgical") in failed
    assert ra._expected_output("r42_k5_abs_l20_s45: R4 circuit") == "node:r42_k5_abs_l20_s45_circuit.json"
    with pytest.raises(KeyError):
        ra._expected_output("r42_k5_abs_l20_s45: something new")
    import shutil
    shutil.rmtree(tmp_path / "node_mirror")
    with pytest.raises(FileNotFoundError):   # no logs is not "nothing failed"
        ra._failed_outputs(tmp_path)
    # the failed job is R0(b); the same cell's attribution that has not run yet is still pending
    assert ra._expected_output("r42_k5_norelu_all_s44: R0(b)") in failed
    assert ra._expected_output("r42_k5_norelu_all_s44: R1' attribution") not in failed
    assert ra._expected_output("r42_k5_abs_l20_s45: R1 circuit") == "node:r42_k5_abs_l20_s45.phase1"
    assert ra._expected_output("r42_k5_abs_l20_s45: node Gate A") not in failed


def test_r2_stats_count_the_negative_pole():
    n_pos, n_neg = torch.tensor([900, 0, 0]), torch.tensor([100, 0, 50])
    m_pos, m_neg = torch.tensor([9.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 10.0])
    r = ra.r2_stats(n_pos, n_neg, m_pos, m_neg)
    assert r["firing"] == 2 and r["bipolar"] == 1 and r["bipolar_share_of_firing"] == pytest.approx(0.5)
    assert r["neg_share_of_firings"] == pytest.approx(150 / 1050)
    assert r["neg_share_of_mass"] == pytest.approx(11 / 20)
    assert r["f_neg_quantiles"]["0.5"] == pytest.approx(0.55)   # f_neg = [0.1, 1.0]; median 0.55
    none = ra.r2_stats(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2))
    assert none["neg_share_of_firings"] is None and none["bipolar_share_of_firing"] is None


def test_a40_readouts_need_the_a40_gate_and_both_r0_checks():
    c = _cell("C")
    c["identity_problems"] = []
    assert ra._a40_admission(c, False, "R2") is None
    assert ra._a40_admission({**c, "gate": {"usable": False, "verdict": "FAIL"}}, False, "R2") is not None
    assert ra._a40_admission({k: v for k, v in c.items() if k != "r0a"}, False, "R2") == "R0(a) not computed"
    bad = {**c, "r0b": {"verdict": "FAIL", "count_rule_only": True}}
    assert ra._a40_admission(bad, False, "R2") is None and ra._a40_admission(bad, True, "R2") == "R0(b) failed"
    scoped = {**c, "identity_problems": [("R2", "R2 counts over 9 latents")]}
    assert ra._a40_admission(scoped, False, "R2") is not None and ra._a40_admission(scoped, False, "R4") is None


# --- summarize() end to end on a synthetic run tree -------------------------------------------------------


def _write(path, obj):
    import json as _json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(obj))


def _gate(clean_fires=0, asr=0.99):
    return {"asr_ok": True, "eot_ok": True, "clean_falsefire": {"fires": clean_fires},
            "intact_backdoor": {"rate": asr}}


def _tree(tmp_path, *, r0b_arm="abs", campaign3=True):
    run = tmp_path / "run"
    B = run / "clcd_results/relu_ablation"
    for arm in ("r42_k5", "r42_k5_abs"):
        cell = f"{arm}_l20_s42"
        ad = run / "models/qwen15" / arm / "l20_s42" / "x"
        _write(ad / "topk_config.json", {})
        _write(ad / "sleeper_run_config.json", {"training": {"sleeper_experiment": {"lora": {
            "relu_latents": arm == "r42_k5", "topk_mode": "abstopk" if arm.endswith("abs") else "topk"}}}})
        _write(B / "campaign" / f"gate_a_{cell}.json", _gate(clean_fires=1 if arm.endswith("abs") else 0))
        _write(B / "node_mirror/campaign" / f"gate_a_{cell}.json", _gate())
        _write(B / "r0b" / f"{cell}.json", {"verdict": "PASS", "problems": [], "arm": r0b_arm if arm.endswith("abs") else "relu_signed",
                                            "per_module": {"m": {"max_abs_diff": 0.0, "nnz_min": 5, "nnz_max": 5, "selected": 10,
                                                                 "tokens": 2, "few_positive": 0}}})
        _write(B / "node_mirror/campaign" / arm / "elim" / "l20_seed42_circuit.json",
               {"status": "ok", "both_K": 2, "elim": {"n_survivors": 1, "pool_n": 294, "protocol": {
                   "survivors": [["base_model.model.model.layers.20.mlp.up_proj", 0]]}},
                "kept_latents": [["base_model.model.model.layers.20.mlp.up_proj", 0],
                                 ["base_model.model.model.layers.20.mlp.up_proj", 1]],
                "args": {"n_cheap": 80, "out": cell}})
    (B / "queue_logs").mkdir(parents=True, exist_ok=True)
    (B / "node_mirror/queue_logs").mkdir(parents=True, exist_ok=True)
    if campaign3:
        _write(tmp_path / "c3/r42_k5/elim/l20_seed43_circuit.json",
               {"status": "ok", "both_K": 2, "elim": {"n_survivors": 1}, "args": {"n_cheap": 80, "out": "ref"}})
        _write(tmp_path / "c3/r42_k5/elim/l20_seed42_circuit.json",
               {"status": "ok", "both_K": 3, "elim": {"n_survivors": 2}, "args": {"n_cheap": 80, "out": "ref"}})
    return run, tmp_path / "c3"


def test_summarize_reports_what_amendment_8_lists(tmp_path):
    run, c3 = _tree(tmp_path)
    s = ra.summarize(run, c3)
    cell = s["cells"]["r42_k5_abs_l20_s42"]
    # verdict-level disagreement: A40 PASS_WITH_WARNING (1 clean fire) vs node PASS
    assert s["gate_disagreements"] == ["r42_k5_abs_l20_s42: A40 PASS_WITH_WARNING vs node PASS"]
    assert cell["r1"]["args_check"] == {}
    assert cell["r0a"] == []
    assert cell["a40_readouts_admitted"] == {"R2": None, "R4": None}
    assert cell["r1"]["survivors"] == 1
    assert s["reproduction"]["l20"][42] == {"A_prime": {"survivors": 1, "both_K": 2},
                                         "campaign3_A": {"survivors": 2, "both_K": 3}}
    # nothing has failed in this tree: every missing record is pending
    assert {m["status"] for m in s["missing"]} == {"pending"}
    assert "r42_k5_abs_l20_s43: R1 circuit" in {m["what"] for m in s["missing"]}


def test_summarize_flags_a_wrong_arm_check_and_excludes_the_cell(tmp_path):
    run, c3 = _tree(tmp_path, r0b_arm="signed")
    s = ra.summarize(run, c3)
    cell = s["cells"]["r42_k5_abs_l20_s42"]
    assert cell["identity_problems"] == [("all", "R0(b) checked as signed")]
    assert s["identity_problems"] and "identity" in ra.admission(cell, strict_r0b=False)
    assert cell["a40_readouts_admitted"]["R2"].startswith("identity")


def test_summarize_refuses_a_missing_reference_tree(tmp_path):
    run, c3 = _tree(tmp_path, campaign3=False)
    with pytest.raises(FileNotFoundError):
        ra.summarize(run, c3)


# --- pole units in the circuit search (pre-registered pole-split search) ---------------------------------


def test_pole_unit_ablation_and_keep_only_act_on_one_sign():
    from src.clcd.verify import ablation_overrides, keep_only_overrides

    a = torch.tensor([[[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]])
    m = "mod"
    abl = ablation_overrides([(m, 0, "+"), (m, 2, "-")])[m](a)
    assert abl.tolist() == [[[0.0, -2.0, 3.0], [-4.0, 5.0, 0.0]]]
    keep = keep_only_overrides([(m, 0, "+"), (m, 2, "-")], {m: None})[m](a)
    assert keep.tolist() == [[[1.0, 0.0, 0.0], [0.0, 0.0, -6.0]]]
    # both poles of a column = the whole column
    both = ablation_overrides([(m, 1, "+"), (m, 1, "-")])[m](a)
    assert torch.equal(both, ablation_overrides([(m, 1)])[m](a))
    assert torch.equal(keep_only_overrides([(m, 1, "+"), (m, 1, "-")], {m: None})[m](a),
                       keep_only_overrides([(m, 1)], {m: None})[m](a))
    # a float third element is a score, not a pole: whole-latent behaviour, bitwise as before
    assert torch.equal(ablation_overrides([(m, 0, 0.7)])[m](a), ablation_overrides([(m, 0)])[m](a))
    with pytest.raises(ValueError):
        ablation_overrides([(m, 0, "x")])


def test_pole_units_need_the_elimination_pool_of_all_latents():
    from src.clcd.exp_circuit_search import main

    with pytest.raises(SystemExit):
        main(["--adapter", "a", "--out", "unused.json", "--pole_units", "--ordering", "eliminate", "--elim_pool", "positive"])


# --- the pole-split search end to end through exp_circuit_search.main() (fakes only for GPU calls) -------
# Review round 12 found the pre-registered setting (block cap 64) crashed after elimination; these runs
# go through the real main() so that class of failure (a 2-tuple assumption anywhere on the path) is caught.

_MODULES = [f"base_model.model.model.layers.{L}.mlp.up_proj" for L in (17, 20, 25)]
_R = 8


def _lat(i):
    return (_MODULES[i // _R], i % _R)


def _run_pole_search(tmp_path, monkeypatch, block_cap):
    import json as _json
    import src.clcd.exp_circuit_search as ecs

    n_all = len(_MODULES) * _R
    all_units = {(*_lat(i), s) for i in range(n_all) for s in "+-"}
    essential = {(*_lat(2), "+"), (*_lat(9), "-")}
    either_pole = {(*_lat(6), "+"), (*_lat(6), "-")}   # one pole of latent 6 is enough

    def active(ov):
        if not ov:
            return set(all_units)
        kind, s = ov
        assert all(len(u) == 3 and u[2] in "+-" for u in s)
        return set(s) if kind == "keep" else all_units - set(s)

    def fires(ov):
        act = active(ov)
        return essential <= act and bool(either_pole & act)

    class _M:
        def to(self, *a, **k):
            return self

    scores = [100.0 - i for i in range(n_all)]
    agg = {m: torch.tensor([scores[j * _R + d] for d in range(_R)]) for j, m in enumerate(_MODULES)}
    monkeypatch.setattr(ecs, "load_organism", lambda *a, **k: (_M(), object(), list(_MODULES)))
    monkeypatch.setattr(ecs, "load_episodes", lambda *a, **k: ([], None))
    monkeypatch.setattr(ecs, "provenance_fields", lambda *a, **k: {"git_commit": "0" * 40, "git_dirty": False,
                                                                   "base_fingerprint": "x", "src_root": "."})
    monkeypatch.setattr(ecs, "load_tags", lambda *a, **k: ("|RUN|", "|TRAIN|"))
    monkeypatch.setattr(ecs, "_load_jsonl_rows", lambda data, split, off, n: list(range(n)))
    monkeypatch.setattr(ecs, "aggregate_attribution", lambda *a, **k: (agg, None, None))
    monkeypatch.setattr(ecs, "keep_only_overrides", lambda c, w: ("keep", frozenset(tuple(u) for u in c)))
    monkeypatch.setattr(ecs, "ablation_overrides", lambda c: ("ablate", frozenset(tuple(u) for u in c)))
    monkeypatch.setattr(ecs, "backdoor_fires", lambda model, tok, wrapped, ov, qs, *a, **k: [int(fires(ov))] * len(qs))
    monkeypatch.setattr(ecs, "backdoor_asr", lambda model, tok, wrapped, ov, qs, *a, **k: float(fires(ov)) if qs else 0.0)
    ad = tmp_path / "ad"
    ad.mkdir()
    (ad / "adapter_model.safetensors").write_bytes(b"\x01" * 64)
    out = tmp_path / f"cap{block_cap}.json"
    ecs.main(["--adapter", str(ad), "--out", str(out), "--data", "data/x", "--base_model", "models/x",
              "--ordering", "eliminate", "--elim_pool", "all", "--pole_units", "--elim_block_cap", str(block_cap),
              "--n_cheap", "80", "--adaptive_n", "--batch_size", "64",
              "--Ks", "1", "2", "3", "4", "5", "10", "47", "48", "50"])
    return _json.loads(out.read_text())


@pytest.mark.parametrize("block_cap", [1, 64])
def test_pole_search_completes_and_keeps_only_the_needed_poles(tmp_path, monkeypatch, block_cap):
    c = _run_pole_search(tmp_path, monkeypatch, block_cap)
    assert c["units"] == "poles" and c["args"]["pole_units"] is True
    assert c["n_all_latents"] == 2 * len(_MODULES) * _R      # units are poles: two per latent
    surv = {tuple(u) for u in c["elim"]["protocol"]["survivors"]} if c["elim"].get("protocol") else None
    kept = {tuple(u) for u in c["kept_latents"]}
    assert {(*_lat(2), "+"), (*_lat(9), "-")} <= kept
    # Amendment 10: within a latent the + pole is tried for cutting first, so of two interchangeable poles
    # the - pole survives
    assert (*_lat(6), "-") in kept and (*_lat(6), "+") not in kept
    assert c["elim"]["n_survivors"] == 3
    if surv is not None:
        assert surv == {(*_lat(2), "+"), (*_lat(9), "-"), (*_lat(6), "-")}


def test_whole_latent_circuit_files_keep_their_args_key_set():
    """pole_units is recorded only when on, so campaign-3 args-checks still pass for latent searches."""
    from src.clcd.exp_circuit_search import build_parser, recorded_args

    off = recorded_args(build_parser().parse_args(["--adapter", "a", "--out", "o.json"]))
    on = recorded_args(build_parser().parse_args(["--adapter", "a", "--out", "o.json", "--pole_units"]))
    assert "pole_units" not in off and on["pole_units"] is True
    assert ra.args_diff({k: v for k, v in off.items()}, off) == {}
