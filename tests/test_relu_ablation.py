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
    got = ra.holm({"a": 0.004, "b": 0.009, "c": 0.02, "d": 0.0125, "e": 0.5, "f": 0.9})
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
    assert set(primary["excluded"]) == {"c_nodegate", "c_nor0b", "c_r0a", "c_r0a_missing", "c_args",
                                        "c_args_missing", "c_unsat", "c_real"}
    strict = ra.tests({"cells": cells}, strict_r0b=True)["tests"]["l20:C>A"]
    assert strict["C"] == [50, ra.POOL_L20]
    # A40 gate does not decide l20 R1 admission; the node gate does
    assert ra.admission(_cell("C", usable=False), strict_r0b=False) is None


def test_holm_counts_every_named_test_even_when_uncomputable():
    """Two computable tests out of six must not make Holm's threshold 0.05/2."""
    p = {n: None for n in ra.TEST_NAMES}
    p["l20:B>A"] = 0.02
    assert ra.holm_fixed(p)["l20:B>A"] is False   # 0.02 > 0.05/6
    p["l20:B>A"] = 0.008
    assert ra.holm_fixed(p)["l20:B>A"] is True    # 0.008 <= 0.05/6
    assert all(v is False for k, v in ra.holm_fixed(p).items() if k != "l20:B>A")


def test_module_name_matching_ignores_the_peft_prefix_but_not_the_layer():
    a = {("base_model.model.model.layers.20.mlp.up_proj", 3)}
    b = {("model.layers.20.mlp.up_proj", 3), ("model.layers.20.mlp.up_proj", 4)}
    assert len(ra._match(a, b)) == 1
    assert ra._match({("base_model.model.model.layers.2.mlp.up_proj", 3)}, b) == set()
