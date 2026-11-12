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


def _run_pole_search(tmp_path, monkeypatch, block_cap, crash_at=None):
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
    calls = {"n": 0}

    def _fires(model, tok, wrapped, ov, qs, *a, **k):
        calls["n"] += 1
        if crash_at is not None and calls["n"] == crash_at:
            raise RuntimeError("simulated crash")
        return [int(fires(ov))] * len(qs)
    monkeypatch.setattr(ecs, "backdoor_fires", _fires)
    monkeypatch.setattr(ecs, "backdoor_asr", lambda model, tok, wrapped, ov, qs, *a, **k: float(fires(ov)) if qs else 0.0)
    ad = tmp_path / "ad"
    ad.mkdir(exist_ok=True)
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
    dests = {ac.dest for ac in build_parser()._actions if ac.dest != "help"}
    assert set(off) == dests - {"pole_units"}          # every other flag is recorded, nothing extra



def test_pole_search_resumes_after_a_crash_with_the_same_result(tmp_path, monkeypatch):
    """The resume guard was removed (review round 12) because resuming 3-tuple units works; pin that."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    full = _run_pole_search(tmp_path / "a", monkeypatch, 64)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run_pole_search(tmp_path / "b", monkeypatch, 64, crash_at=6)
    assert (tmp_path / "b" / "cap64.json.ckpt").exists()
    resumed = _run_pole_search(tmp_path / "b", monkeypatch, 64)
    for k in ("status", "both_K", "kept_latents", "curve"):
        assert resumed[k] == full[k]
    assert resumed["elim"]["protocol"]["survivors"] == full["elim"]["protocol"]["survivors"]


# --- Amendment 11: l17_25 has both a certified-circuit (R1) and an attribution (R1p) readout ----------------


def test_l17_25_cells_are_scored_on_both_readouts_with_their_own_gates():
    c = _cell("C", fam="l17_25", n90=500)
    c["node_gate"] = {"usable": True, "verdict": "PASS"}
    c["r1"] = {"status": "ok", "survivors": 120, "pool": 2500, "args_check": {}}
    assert ra.admission(c, False, "R1") is None and ra.admission(c, False, "R1p") is None
    assert ra._statistic(c, "R1") == 120 and ra._statistic(c, "R1p") == 500
    a40_fail = {**c, "gate": {"usable": False, "verdict": "FAIL"}}
    assert ra.admission(a40_fail, False, "R1") is None and ra.admission(a40_fail, False, "R1p") is not None
    node_fail = {**c, "node_gate": {"usable": False, "verdict": "FAIL"}}
    assert ra.admission(node_fail, False, "R1") is not None and ra.admission(node_fail, False, "R1p") is None
    censored = {**c, "r1": {**c["r1"], "status": "no_sufficient_subcircuit"}}
    assert ra._statistic(censored, "R1") == 2500   # the elimination pool, not l20's 294
    cells = {"a": _cell("A", fam="l17_25", n90=400), "c": c}
    cells["a"]["node_gate"] = {"usable": True, "verdict": "PASS"}
    cells["a"]["r1"] = {"status": "ok", "survivors": 100, "pool": 2500, "args_check": {}}
    t = ra.tests({"cells": cells}, strict_r0b=False)["tests"]
    assert t["l17_25_R1:C>A"]["A"] == [100] and t["l17_25_R1:C>A"]["C"] == [120]
    assert t["l17_25:C>A"]["A"] == [400] and t["l17_25:C>A"]["C"] == [500]


# --- Amendment 12: what each pole of a bipolar latent reads (blind LLM rating against decoys) ----------------


def test_r3_draw_is_pinned_independent_of_input_order_and_reuses_no_pole_in_decoys():
    """The draw is pre-registered (seed 20261111): a change in RNG consumption would silently draw other latents."""
    cands = {"l20": {42: [("m", i) for i in range(4)], 43: [("m", 9)]},
             "l17_25": {44: [("n", i) for i in range(6)]}}
    d = ra.draw_pole_latents(cands)
    assert d["latents"] == [("l20", 42, "m", 3), ("l20", 42, "m", 2), ("l20", 42, "m", 1), ("l20", 42, "m", 0),
                            ("l20", 43, "m", 9), ("l17_25", 44, "n", 2), ("l17_25", 44, "n", 4),
                            ("l17_25", 44, "n", 0), ("l17_25", 44, "n", 3), ("l17_25", 44, "n", 5)]
    assert [(x["plus"][3], x["minus"][3]) for x in d["decoys"]] == [(0, 2), (2, 3), (3, 0), (0, 2), (2, 4)]
    shuffled = {"l17_25": {44: list(reversed(cands["l17_25"][44]))}, "l20": {43: cands["l20"][43], 42: cands["l20"][42][::-1]}}
    assert ra.draw_pole_latents(shuffled) == d
    for x in d["decoys"]:
        assert x["plus"] != x["minus"] and x["plus"][0] == x["minus"][0]
    assert len({str(x["plus"]) for x in d["decoys"]}) == len({str(x["minus"]) for x in d["decoys"]}) == 5
    with pytest.raises(ValueError, match="only 4"):
        ra.draw_pole_latents({"l20": {42: [("m", i) for i in range(4)]}, "l17_25": cands["l17_25"]})


class _Tok:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def _r3_model():
    model, wrapped = _tiny_arm_c()
    with torch.no_grad():   # latent 0 only, distinct magnitudes; the pad token (5) would be the strongest + firing
        model.base_model.model.emb.weight.copy_(torch.tensor([
            [0.0, 0.0, 0.0, 0.2], [3.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0]]))
    return model, wrapped


_R3_TOKENS = torch.tensor([[1, 2, 4, 3, 5], [1, 3, 1, 2, 2], [2, 4, 4, 1, 5]])


def test_pole_firings_keep_signs_and_skip_padding_and_position_zero():
    model, wrapped = _r3_model()
    want = [(0, 1, -2.0), (0, 2, 0.5), (0, 3, -1.0), (1, 1, -1.0), (1, 2, 3.0), (1, 3, -2.0), (1, 4, -2.0),
            (2, 1, 0.5), (2, 2, 0.5), (2, 3, 3.0)]
    for bs in (1, 2):   # 2 puts row 2 in a second batch: its row index must be offset by the batch start
        fir = ra.pole_firings(model, wrapped, _R3_TOKENS, pad_id=5, targets=[("linear", 0), ("linear", 3)], batch_size=bs)
        f0 = fir[("linear", 0)]
        assert list(zip(f0["row"].tolist(), f0["pos"].tolist(), f0["val"].tolist())) == want   # corpus order
        assert len(fir[("linear", 3)]["val"]) == 0   # token 0 never occurs
    with pytest.raises(KeyError):
        ra.pole_firings(model, wrapped, _R3_TOKENS, 5, [("nope", 0)], 1)


def test_pole_examples_rank_by_magnitude_merge_identical_windows_and_mark_the_token():
    model, wrapped = _r3_model()
    tokens = _R3_TOKENS[:2]
    fir = ra.pole_firings(model, wrapped, tokens, 5, [("linear", 0), ("linear", 3)], 1)
    plus = ra.pole_example_sets(fir[("linear", 0)], tokens, _Tok(), "+", top=2, decile_n=1, left=1)
    assert plus["n_firings"] == 2 and plus["top"] == ["<3>⟦<1>⟧", "<2>⟦<4>⟧"]
    assert plus["top_decile_sample"] == ["<3>⟦<1>⟧"]
    minus = ra.pole_example_sets(fir[("linear", 0)], tokens, _Tok(), "-", top=3, decile_n=1, left=1)
    # |-2| at (0,1), (1,3), (1,4): (0,1) and (1,3) render the same window and count once; then the first |-1|
    # in corpus order, (0,3). A signed sort would put the -1s first.
    assert minus["n_firings"] == 5 and minus["top"] == ["<1>⟦<2>⟧", "<2>⟦<2>⟧", "<4>⟦<3>⟧"]
    assert ra.pole_example_sets(fir[("linear", 0)], tokens, _Tok(), "+", top=5, left=5)["top"][0] == "<1><3>⟦<1>⟧"
    with pytest.raises(ValueError, match="never fires"):
        ra.pole_example_sets(fir[("linear", 3)], tokens, _Tok(), "+")


def test_top_decile_draw_is_pinned_and_breaks_ties_by_corpus_order():
    """200 firings, 20 in the top decile, 10 drawn with Random(20261111): the pre-registered sample."""
    n = 200
    vals = torch.tensor([float(i // 2) for i in range(n)])   # every value twice: ties everywhere
    fir = {"row": torch.arange(n), "pos": torch.ones(n, dtype=torch.long), "val": vals}
    tokens = torch.stack([torch.arange(n), torch.arange(n) + 1000], 1)
    ex = ra.pole_example_sets(fir, tokens, _Tok(), "+", top=4, left=1)
    # the larger row of each tied pair is the later one in corpus order, so it comes second
    assert ex["top"] == ["<198>⟦<1198>⟧", "<199>⟦<1199>⟧", "<196>⟦<1196>⟧", "<197>⟦<1197>⟧"]
    assert ex["top_decile_sample"] == ['<184>⟦<1184>⟧', '<190>⟦<1190>⟧', '<189>⟦<1189>⟧', '<195>⟦<1195>⟧', '<193>⟦<1193>⟧',
                                    '<185>⟦<1185>⟧', '<186>⟦<1186>⟧', '<188>⟦<1188>⟧', '<191>⟦<1191>⟧', '<199>⟦<1199>⟧']
    assert ex["n_firings"] == n - 2   # value 0.0 (rows 0, 1) is not a firing of the + pole


_META = {"code_sha256": "c" * 64, "adapter": "a", "corpus_sha256": "d" * 64, "batch_size": 32}


def _r3_files(tmp_path, r2_minus=1000, extra=""):
    """Three latents; the decoy is l20/0's + pole with l20/1's - pole, and those two sets share one snippet
    ("common", twice in each); l17_25/2's + set repeats one snippet. `extra` changes one snippet (a different packet)."""
    import json
    lat = [["l20", 42, "m", 0], ["l20", 42, "m", 1], ["l17_25", 43, "n", 2]]
    key = lambda t: "|".join(map(str, t))
    draw = {"latents": lat, "decoys": [{"plus": lat[0], "minus": lat[1]}],
            "r2_counts": {key(t): [600, r2_minus] for t in lat}}
    ex, tag = {}, {}
    for t in lat:
        ex[key(t)] = {}
        for pole in "+-":
            s = f"ex{len(tag)}"
            tag[s] = (key(t), pole)
            ex[key(t)][pole] = {"n_firings": 600 if pole == "+" else 1000, "top": [s], "top_decile_sample": []}
    ex[key(lat[0])]["+"]["top"] += ["common", "common"]
    ex[key(lat[1])]["-"]["top"] += ["common", "common"]   # a decile draw can repeat a top window
    ex[key(lat[2])]["+"]["top"] += ["dup", "dup" + extra]
    (tmp_path / "draw.json").write_text(json.dumps(draw))
    (tmp_path / "cap.json").write_text(json.dumps({**ex, "_meta": _META}))
    return tag


def _packet(tmp_path, captures=("cap.json",)):
    import argparse
    return ra._cmd_r3_packet(argparse.Namespace(draw=str(tmp_path / "draw.json"),
                                                captures=[str(tmp_path / c) for c in captures],
                                                packet=str(tmp_path / "packet.json"), key=str(tmp_path / "key.json")))


def test_r3_packet_is_blind_pinned_and_its_key_says_which_pole_is_in_which_set(tmp_path):
    import json
    tag = _r3_files(tmp_path)
    _packet(tmp_path)
    text = (tmp_path / "packet.json").read_text()
    packet, keyf = json.loads(text), json.loads((tmp_path / "key.json").read_text())
    key = {k["pair"]: k for k in keyf["pairs"]}
    assert keyf["packet_sha256"] == ra._sha(text)
    assert packet["prompt"] == ra.RATER_PROMPT and len(packet["pairs"]) == 4
    # pair order and set order are Random(20261112)'s: pinned, since they are what keeps the rater blind
    assert [(key[i]["kind"], key[i]["plus"], key[i]["set1_pole"]) for i in range(4)] == [("decoy", "l20|42|m|0", "-"), ("real", "l20|42|m|0", "+"),
                                                                                ("real", "l17_25|43|n|2", "-"), ("real", "l20|42|m|1", "+")]
    for p in packet["pairs"]:
        assert set(p) == {"pair", "set1", "set2"}   # nothing in the packet says real or decoy
        k = key[p["pair"]]
        s1, s2 = tag[p["set1"][0]], tag[p["set2"][0]]
        want1 = (k["plus"], "+") if k["set1_pole"] == "+" else (k["minus"], "-")
        assert s1 == want1 and {s1[1], s2[1]} == {"+", "-"}
        assert (k["plus"] == k["minus"]) == (k["kind"] == "real")


@pytest.mark.parametrize("r2_minus,ok", [(1019, True), (981, True), (1021, False), (979, False), (980, False)])
def test_r3_packet_refuses_a_capture_more_than_2pct_off_r2(tmp_path, r2_minus, ok):
    _r3_files(tmp_path, r2_minus=r2_minus)   # 1000 captured negative firings; 2% of R2's count, not of ours
    if ok:
        _packet(tmp_path)
    else:
        with pytest.raises(SystemExit, match="not the same latent"):
            _packet(tmp_path)


def test_r3_packet_refuses_a_latent_captured_twice_and_captures_from_other_code(tmp_path):
    import json
    import shutil
    _r3_files(tmp_path)
    shutil.copy(tmp_path / "cap.json", tmp_path / "cap2.json")
    with pytest.raises(SystemExit, match="captured twice"):
        _packet(tmp_path, ("cap.json", "cap2.json"))
    cap = json.loads((tmp_path / "cap.json").read_text())
    k0 = next(k for k in cap if k.startswith("l17_25"))
    part = {k0: cap.pop(k0)}
    (tmp_path / "cap.json").write_text(json.dumps(cap))
    for field, other in (("code_sha256", "e" * 64), ("corpus_sha256", "f" * 64), ("batch_size", 64)):
        (tmp_path / "cap2.json").write_text(json.dumps({**part, "_meta": {**_META, field: other}}))
        with pytest.raises(SystemExit, match=f"disagree on {field}"):
            _packet(tmp_path, ("cap.json", "cap2.json"))
    (tmp_path / "cap2.json").write_text(json.dumps({**part, "_meta": {**_META, "adapter": "b"}}))
    _packet(tmp_path, ("cap.json", "cap2.json"))   # adapters differ between captures by design
    assert [m["adapter"] for m in json.loads((tmp_path / "key.json").read_text())["captures"]] == ["a", "b"]
    del cap["_meta"]   # what the 7dc37e5 capture wrote
    (tmp_path / "cap.json").write_text(json.dumps(cap))
    with pytest.raises(SystemExit, match="no _meta"):
        _packet(tmp_path, ("cap.json",))


class _FakeResp:
    def __init__(self, text, stop="end_turn"):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason, self._request_id, self.model = stop, "req", "served-model-id"   # not RATER_MODEL
        self.usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})()

    def to_dict(self):
        return {"stop_details": {"category": "x"} if self.stop_reason == "refusal" else None}


def _fake_anthropic(monkeypatch, answers, crash_after=None):
    """answers: one list per pair id, consumed in order; crash_after: raise on that call number (1-based)."""
    import sys
    import types
    calls = {"n": 0, "by_pair": {}}

    class _Messages:
        def create(self, **kw):
            calls.setdefault("kwargs", []).append(kw)
            calls["n"] += 1
            if crash_after is not None and calls["n"] == crash_after:
                raise RuntimeError("simulated crash")
            pid = next(i for i, txt in enumerate(_Messages.rendered) if txt == kw["messages"][0]["content"])
            calls["by_pair"][pid] = calls["by_pair"].get(pid, 0) + 1
            return answers[pid].pop(0)

    fake = types.SimpleNamespace(Anthropic=lambda **k: types.SimpleNamespace(messages=_Messages()))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setitem(sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
    return calls, _Messages


def _ok(rel):
    return _FakeResp('thinking aloud {"set1": "x", "set2": "y", "relation": "%s"}' % rel)


def test_r3_rate_retries_counts_tries_across_a_crash_and_refuses_another_packets_ratings(tmp_path, monkeypatch):
    import argparse
    import json
    _r3_files(tmp_path)
    _packet(tmp_path)
    packet = json.loads((tmp_path / "packet.json").read_text())
    answers = {0: [_FakeResp("no json"), _ok("opposite")],                       # parsed on the 2nd try
               1: [_FakeResp("", "refusal"), _FakeResp("", "refusal"), _FakeResp("", "refusal"), _ok("same")],
               2: [_FakeResp('{"set1": "x", "set2": "y", "relation": "same"} final: {"relation": "Same"}'),
                   _ok("unrelated")],                                              # a draft is not the answer
               3: [_FakeResp('{"set1": "x", "set2": "y", "relation": "opposite"}', "max_tokens"), _ok("same")]}
    calls, msgs = _fake_anthropic(monkeypatch, answers, crash_after=4)            # dies inside pair 1
    msgs.rendered = [ra.render_pair(p) for p in packet["pairs"]]
    ns = argparse.Namespace(packet=str(tmp_path / "packet.json"), out=str(tmp_path / "ratings.json"))
    with pytest.raises(RuntimeError, match="simulated crash"):
        ra._cmd_r3_rate(ns)
    saved = json.loads((tmp_path / "ratings.json").read_text())
    assert saved["records"]["0"]["relation"] == "opposite" and len(saved["records"]["1"]["tries"]) == 1
    calls2, msgs2 = _fake_anthropic(monkeypatch, answers)
    msgs2.rendered = msgs.rendered
    ra._cmd_r3_rate(ns)
    r = json.loads((tmp_path / "ratings.json").read_text())["records"]
    assert calls2["by_pair"] == {1: 2, 2: 2, 3: 2}       # pair 1: 3 tries in all, across the crash
    assert r["1"]["relation"] == "unrated" and len(r["1"]["tries"]) == 3
    assert r["1"]["tries"][0]["stop_details"] == {"category": "x"} and r["1"]["tries"][0]["model"] == "served-model-id"
    assert r["2"]["relation"] == "unrelated"
    assert r["3"]["relation"] == "same" and len(r["3"]["tries"]) == 2   # a cut answer is asked again
    for kw in calls["kwargs"] + calls2["kwargs"]:   # the pre-registered rater request (Amendment 12a)
        assert {k: v for k, v in kw.items() if k != "messages"} == {
            "model": "claude-opus-5-5", "max_tokens": 16000, "system": ra.RATER_PROMPT,
            "output_config": {"effort": "high"}}
        assert len(kw["messages"]) == 1 and kw["messages"][0]["role"] == "user"
    # what r3-rate wrote (refusals, "no json", a draft object, a cut answer) passes r3-score's re-derivation
    ra._cmd_r3_score(argparse.Namespace(packet=str(tmp_path / "packet.json"), key=str(tmp_path / "key.json"),
                                        ratings=str(tmp_path / "ratings.json"), out=str(tmp_path / "score.json")))
    kinds = {k["pair"]: k["kind"] for k in json.loads((tmp_path / "key.json").read_text())["pairs"]}
    got = {p["pair"]: (p["kind"], p["relation"], p["tries"]) for p in json.loads((tmp_path / "score.json").read_text())["pairs"]}
    assert got == {0: (kinds[0], "opposite", 2), 1: (kinds[1], "unrated", 3), 2: (kinds[2], "unrelated", 2),
                   3: (kinds[3], "same", 2)}
    other = json.loads((tmp_path / "ratings.json").read_text())
    other["packet_sha256"] = "0" * 64
    (tmp_path / "ratings.json").write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="another packet"):
        ra._cmd_r3_rate(ns)


def _ans(rel):
    return {"stop_reason": "end_turn", "text": '{"set1": "s1", "set2": "s2", "relation": "%s"}' % rel}


def test_r3_score_counts_by_kind_and_refuses_mismatched_incomplete_or_unknown_ratings(tmp_path):
    import argparse
    import json
    import shutil
    _r3_files(tmp_path)
    _packet(tmp_path)
    keyf = json.loads((tmp_path / "key.json").read_text())
    kind = {k["pair"]: k["kind"] for k in keyf["pairs"]}
    rel = {p: ("opposite" if kd == "real" else "unrelated") for p, kd in kind.items()}
    rel[next(p for p, kd in kind.items() if kd == "real")] = "same"
    ns = lambda key="key.json": argparse.Namespace(packet=str(tmp_path / "packet.json"), key=str(tmp_path / key),
                                                   ratings=str(tmp_path / "r.json"), out=str(tmp_path / "score.json"))

    def write(rels, sha=keyf["packet_sha256"], tries=None):
        recs = {str(p): {"pair": p, "relation": r, "set1": "s1", "set2": "s2",
                         "tries": (tries or {}).get(p, [_ans(r)])} for p, r in rels.items()}
        (tmp_path / "r.json").write_text(json.dumps({"packet_sha256": sha, "records": recs}))

    cut = {"stop_reason": "max_tokens", "text": _ans("same")["text"]}
    write(rel, tries={1: [cut, _ans(rel[1])]})
    ra._cmd_r3_score(ns())
    out = json.loads((tmp_path / "score.json").read_text())
    assert out["counts"] == {"real": {"opposite": 2, "same": 1}, "decoy": {"unrelated": 1}}
    assert [p["tries"] for p in out["pairs"]] == [1, 2, 1, 1]
    # only the decoy's two sets share a snippet; a snippet repeated inside one set does not count
    assert {(p["kind"], p["plus"]): p["shared_snippets"] for p in out["pairs"]} == {
        ("decoy", "l20|42|m|0"): 1, ("real", "l20|42|m|0"): 0, ("real", "l20|42|m|1"): 0, ("real", "l17_25|43|n|2"): 0}
    assert all(p["set1"] == "s1" and p["set2"] == "s2" for p in out["pairs"])
    write(dict(list(rel.items())[:-1]))
    with pytest.raises(SystemExit, match="cover exactly"):
        ra._cmd_r3_score(ns())
    write({p: "similar" for p in rel})
    with pytest.raises(SystemExit, match="unknown relation"):
        ra._cmd_r3_score(ns())
    write(rel, tries={0: [_ans("same" if rel[0] != "same" else "opposite")]})
    with pytest.raises(SystemExit, match="not the last answer"):
        ra._cmd_r3_score(ns())
    write({**rel, 0: "unrated"}, tries={0: [cut] * 2})
    with pytest.raises(SystemExit, match="unrated after 2 tries"):
        ra._cmd_r3_score(ns())
    write({**rel, 0: "unrated"}, tries={0: [cut, cut, _ans("opposite")]})   # its last answer was usable
    with pytest.raises(SystemExit, match="last usable=opposite"):
        ra._cmd_r3_score(ns())
    write(rel, tries={0: [_ans("unrelated" if rel[0] != "unrelated" else "same"), _ans(rel[0])]})   # best of 2
    with pytest.raises(SystemExit, match="usable answer before the last"):
        ra._cmd_r3_score(ns())
    write(rel, tries={0: [cut]})   # a rated record whose only answer was cut
    with pytest.raises(SystemExit, match="not the last answer"):
        ra._cmd_r3_score(ns())
    write(rel, tries={0: [{"stop_reason": "end_turn", "text": "no object"}]})   # end_turn but unparseable
    with pytest.raises(SystemExit, match="not the last answer"):
        ra._cmd_r3_score(ns())
    write(rel, tries={0: [cut] * 3 + [_ans(rel[0])]})
    with pytest.raises(SystemExit, match="4 tries recorded"):
        ra._cmd_r3_score(ns())
    write(rel, sha="0" * 64)
    with pytest.raises(SystemExit, match="ratings file belongs to another packet"):
        ra._cmd_r3_score(ns())
    # a packet that differs in one snippet has another hash; its key and ratings do not unblind this one
    write(rel)
    shutil.copy(tmp_path / "packet.json", tmp_path / "packet_a.json")
    shutil.copy(tmp_path / "key.json", tmp_path / "key_a.json")
    _r3_files(tmp_path, extra="x")
    _packet(tmp_path)
    keyb = json.loads((tmp_path / "key.json").read_text())
    assert keyb["packet_sha256"] != keyf["packet_sha256"]
    with pytest.raises(SystemExit, match="key file belongs to another packet"):
        ra._cmd_r3_score(ns("key_a.json"))
    with pytest.raises(SystemExit, match="ratings file belongs to another packet"):
        ra._cmd_r3_score(ns())


def test_r3_rate_refuses_ratings_of_a_packet_differing_in_one_snippet(tmp_path, monkeypatch):
    import argparse
    import json
    _r3_files(tmp_path)
    _packet(tmp_path)
    sha_a = json.loads((tmp_path / "key.json").read_text())["packet_sha256"]
    (tmp_path / "ratings.json").write_text(json.dumps({"packet_sha256": sha_a, "records": {}}))
    _r3_files(tmp_path, extra="x")
    _packet(tmp_path)
    _fake_anthropic(monkeypatch, {})
    with pytest.raises(SystemExit, match="another packet"):
        ra._cmd_r3_rate(argparse.Namespace(packet=str(tmp_path / "packet.json"), out=str(tmp_path / "ratings.json")))


def test_parse_rating_takes_only_the_final_object():
    text = 'Set 1 is {"a": 1}-like.\n{"set1": "plurals {s}", "set2": "verbs", "relation": "unrelated"}'
    assert ra.parse_rating(text) == {"set1": "plurals {s}", "set2": "verbs", "relation": "unrelated"}
    with pytest.raises(ValueError):   # a usable draft followed by an unusable final answer is re-asked
        ra.parse_rating('{"set1": "a", "set2": "b", "relation": "same"} so: {"set1": "a", "set2": "b", "relation": "Opposite"}')
    with pytest.raises(ValueError):
        ra.parse_rating("no object")


def test_render_pair_keeps_one_snippet_per_line():
    out = ra.render_pair({"set1": ["a\nb", "c\rd\x0be\x0cf g"], "set2": ["h"]})
    assert out.splitlines() == ["Set 1:", "1. a⏎b", "2. c⏎d⏎e⏎f⏎g", "", "Set 2:", "1. h"]


def test_r3_capture_targets_only_this_adapters_latents_and_records_provenance(tmp_path, monkeypatch, capsys):
    import argparse
    import hashlib
    import json
    import sys
    import types
    lat = [["l20", 42, "m", 0], ["l20", 43, "m", 1], ["l20", 42, "m", 3]]
    (tmp_path / "draw.json").write_text(json.dumps({"latents": lat}))
    torch.save({"tokens": torch.zeros(2, 3, dtype=torch.long), "pad_id": 5}, tmp_path / "corpus.pt")
    seen = {}
    monkeypatch.setattr(ra, "_adapter_dir", lambda root, arm, fam, sd: seen.setdefault("adapter", (arm, fam, sd)) and "ad")
    model = types.SimpleNamespace(to=lambda *a: model, eval=lambda: model)
    mod = types.SimpleNamespace(relu_latents=False, topk_mode="abstopk")
    monkeypatch.setitem(sys.modules, "src.clcd.organism", types.SimpleNamespace(
        load_organism=lambda *a, **k: (model, None, {"m": mod})))
    monkeypatch.setattr(ra, "pole_firings", lambda m, w, toks, pad, targets, bs: seen.setdefault("targets", targets) and {
        t: None for t in targets})
    monkeypatch.setattr(ra, "pole_example_sets", lambda fir, toks, tok, pole: {"pole": pole})
    ra._cmd_r3_capture(argparse.Namespace(draw=str(tmp_path / "draw.json"), corpus=str(tmp_path / "corpus.pt"),
                                          run_root=".", base_model="b", family="l20", seed=42, batch_size=32,
                                          out=str(tmp_path / "cap.json")))
    cap = json.loads((tmp_path / "cap.json").read_text())
    assert seen["targets"] == [("m", 0), ("m", 3)] and seen["adapter"] == ("r42_k5_abs", "l20", 42)
    assert sorted(cap) == ["_meta", "l20|42|m|0", "l20|42|m|3"]
    assert cap["_meta"] == {"code_sha256": hashlib.sha256(open(ra.__file__, "rb").read()).hexdigest(), "adapter": "ad",
                            "corpus_sha256": hashlib.sha256((tmp_path / "corpus.pt").read_bytes()).hexdigest(),
                            "batch_size": 32}
    assert "captured 2 latents" in capsys.readouterr().out
    mod.topk_mode = "topk"   # not arm C
    with pytest.raises(RuntimeError):
        ra._cmd_r3_capture(argparse.Namespace(draw=str(tmp_path / "draw.json"), corpus=str(tmp_path / "corpus.pt"),
                                              run_root=".", base_model="b", family="l20", seed=42, batch_size=32,
                                              out=str(tmp_path / "cap.json")))


# --- Amendment 13: R3 at full scale (input similarity of the two poles of every bipolar latent) -----------


def _brute_top(model, wrapped, tokens, pad, n):
    """Every (row, pos, value, input) of the tiny model, then the top-n per latent and pole in corpus order."""
    mod = wrapped["linear"]
    got = {}
    h1 = mod.register_forward_pre_hook(lambda m, args: got.__setitem__("x", args[0].detach()))
    h2 = mod.latent_site.register_forward_hook(lambda m, args, out: got.__setitem__("a", out.detach()))
    with torch.no_grad():
        model(tokens)
    h1.remove(); h2.remove()
    keep = tokens != pad
    keep[:, 0] = False
    rp = torch.nonzero(keep)
    x, a = got["x"][keep], got["a"][keep]
    out = {}
    for j in range(a.shape[1]):
        for p_, sign in ((0, 1.0), (1, -1.0)):
            sc = (sign * a[:, j]).clamp(min=0)
            order = sorted(range(len(sc)), key=lambda q: (-float(sc[q]), q))[:n]
            out[(j, p_)] = [(int(rp[q, 0]), int(rp[q, 1]), float(sc[q]), x[q]) for q in order]
    return out, x.double().mean(0)


@pytest.mark.parametrize("bs", [1, 3, 7])
def test_running_top_inputs_equal_brute_force_with_corpus_order_ties(bs):
    model, wrapped = _tiny_arm_c()
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 6, (7, 6), generator=g)   # 5 = pad; repeated tokens make exact ties
    acc = ra.pole_top_inputs(model, wrapped, tokens, pad_id=5, targets={"linear": [0, 1, 2, 3]}, batch_size=bs, n=4)
    t = acc["linear"]
    want, mu = _brute_top(model, wrapped, tokens, 5, 4)
    for j in range(4):
        for p_ in (0, 1):
            w = want[(j, p_)]
            assert [(int(t.row[p_, q, j]), int(t.pos[p_, q, j])) for q in range(len(w))] == [(r, p) for r, p, _, _ in w]
            assert [float(t.val[p_, q, j]) for q in range(len(w))] == pytest.approx([v for _, _, v, _ in w])
            for q, (_, _, _, xq) in enumerate(w):
                assert torch.equal(t.x[p_, q, j], xq)
    assert torch.allclose(t.mu_sum / t.mu_n, mu)
    assert t.mu_n == int(((tokens != 5) & (torch.arange(6) > 0)).sum())


def test_pole_halves_drop_duplicate_windows_and_split_by_rank():
    t = ra._TopInputs([7], n=6)
    # one latent column, pole 0: values 6..1 at rows 0..5, all at position 2
    t.val = torch.tensor([[[6.], [5.], [4.], [3.], [2.], [1.]], [[0.]] * 6])
    t.row = torch.tensor([[[0], [1], [2], [3], [4], [5]], [[0]] * 6])
    t.pos = torch.full((2, 6, 1), 2)
    t.x = torch.arange(2 * 6 * 1 * 2, dtype=torch.float32).reshape(2, 6, 1, 2)
    tokens = torch.tensor([[1, 2, 3], [1, 2, 3], [4, 2, 3], [5, 5, 6], [7, 7, 8], [9, 9, 9]])   # row 1 repeats row 0
    h = ra.pole_halves(t, 0, 0, tokens, keep=4, left=2)
    assert [f[0] for f in h["firings"]] == [0, 2, 3, 4]
    assert h["tok_odd"] == [3, 6] and h["tok_even"] == [3, 8]   # odd = 1st and 3rd unique (rows 0, 3)
    assert torch.allclose(h["mean_odd"], (t.x[0, 0, 0] + t.x[0, 3, 0]).double() / 2)
    assert ra.pole_halves(t, 0, 0, tokens, keep=6, left=2) is None   # only 5 unique windows
    assert ra.pole_halves(t, 0, 1, tokens, keep=1, left=2) is None   # pole 1 never fires


def test_projected_cosine_ignores_the_read_direction():
    a = torch.tensor([1.0, 0.0, 0.0])
    w = torch.tensor([0.0, 1.0, 2.0])
    assert ra.projected_cosine(a * 5 + w, -a * 3 + w, a.unsqueeze(0)) == pytest.approx(1.0)
    raw = float(torch.nn.functional.cosine_similarity(a * 5 + w, -a * 3 + w, dim=0))
    assert raw < 0.2   # without the projection the two poles look unrelated or opposite
    b = torch.tensor([0.0, 1.0, 0.0])
    assert ra.projected_cosine(a + b + torch.tensor([0, 0, 1.0]), -a - b + torch.tensor([0, 0, 2.0]),
                               torch.stack([a, b])) == pytest.approx(1.0)


def test_auc_greater_matches_counting_with_ties_as_half():
    x, y = [3, 1, 2, 2], [2, 0, 5]
    brute = sum((xi > yj) + 0.5 * (xi == yj) for xi in x for yj in y) / 12
    assert ra.auc_greater(x, y) == pytest.approx(brute)
    with pytest.raises(ValueError):
        ra.auc_greater([], [1])


def _synthetic_cap(n_lat, related):
    """One module; latent i's poles read the same tokens/inputs when related(i), unrelated ones otherwise."""
    g = torch.Generator().manual_seed(1)
    d = 8
    A = {i: torch.randn(d, generator=g) for i in range(n_lat)}
    lat = {}
    for i in range(n_lat):
        base = torch.randn(d, generator=g) * 3
        other = base if related(i) else torch.randn(d, generator=g) * 3
        tok = list(range(100 * i, 100 * i + 8))   # halves of one pole overlap in 3 of 5 tokens
        tok2 = tok if related(i) else list(range(100 * i + 50, 100 * i + 58))
        mk = lambda c, t, s: {"tok_odd": t[:5], "tok_even": t[3:], "mean_odd": (c + s * A[i] + 0.1 * torch.randn(d, generator=g)).double(),
                              "mean_even": (c + s * A[i] + 0.1 * torch.randn(d, generator=g)).double()}
        lat[i] = {"+": mk(base, tok, 2.0), "-": mk(other, tok2, -2.0)}
    lat[n_lat] = {"+": None, "-": lat[0]["-"]}   # too few unique firings: excluded
    return {"modules": {"m": {"mu": torch.zeros(d, dtype=torch.float64), "A": A, "latents": lat}}}


def test_pole_similarity_calls_related_poles_related_and_unrelated_ones_not():
    cap = _synthetic_cap(40, related=lambda i: i < 15)
    res = ra.pole_similarity(cap)
    assert res["excluded_too_few_unique"] == 1 and len(res["latents"]) == 40
    by = {r["latent"]: r for r in res["latents"]}
    assert all(by[i]["M1_related"] and by[i]["M2_related"] for i in range(15))
    assert sum(by[i]["M2_related"] for i in range(15, 40)) <= 2
    assert ra.pole_similarity(cap) == res   # the decoy draw is pinned
    ro = ra.r3s_family_readout({42: res})
    assert ro["M2"]["auc_same_gt_decoy"] > 0.95 and ro["M2"]["reading"] == "mixed"   # 15/40 related
    unrelated = ra.r3s_family_readout({42: ra.pole_similarity(_synthetic_cap(40, related=lambda i: False))})
    assert unrelated["M2"]["reading"] == "mostly unrelated" and unrelated["M1"]["reading"] == "mostly unrelated"
    one = ra.r3s_family_readout({42: ra.pole_similarity(_synthetic_cap(40, related=lambda i: True))})
    assert one["M1"]["reading"] == "mostly one feature"


def test_family_readout_refuses_to_read_a_metric_that_cannot_see_relatedness():
    cap = _synthetic_cap(30, related=lambda i: False)
    for h in cap["modules"]["m"]["latents"].values():   # same-pole halves made as unrelated as the decoys
        if h["+"] is not None:
            h["+"]["tok_even"] = [t + 20 for t in h["+"]["tok_even"]]
            h["-"]["tok_even"] = [t + 20 for t in h["-"]["tok_even"]]
    ro = ra.r3s_family_readout({42: ra.pole_similarity(cap)})
    assert ro["M1"]["reading"] == "uninformative (gate failed)"


def test_bipolar_targets_use_the_r2_rule_summed_over_sources():
    n = torch.zeros(3, 2, 4, dtype=torch.long)
    n[0, 0] = torch.tensor([500, 900, 150, 0])
    n[1, 1] = torch.tensor([500, 50, 50, 0])
    n[2, 1, 2] = 60
    assert ra.bipolar_targets({"m": {"n": n}, "q": {"n": torch.zeros(3, 2, 4, dtype=torch.long)}}) == {"m": [0, 2]}


def test_a_zero_overlap_among_zero_overlap_decoys_is_not_called_related():
    """Jaccard is discrete: if most decoys are 0, a 95th-percentile threshold of 0 would call every pole pair related."""
    res = ra.pole_similarity(_synthetic_cap(40, related=lambda i: False))
    assert all(r["M1_decoy"] == 0 and r["M1_real"] == 0 for r in res["latents"])
    assert not any(r["M1_related"] for r in res["latents"])


def test_table_cell_is_median_and_range_of_the_seeds():
    assert ra._mr([30, 23, 39, 25, 26]) == "26 [23–39]"
    assert ra._mr([0.31, 0.29, 0.35], pct=True) == "31 [29–35]"
    assert ra._mr([0.0, 2e-6, 3e-4], pct=True, digits=2) == "<0.01 [0.00–0.03]"   # rare is not zero
    with pytest.raises(ValueError):
        ra._mr([])
