"""What `verify_adapter_arm` must catch, and why each case matters.

Every case below is a way a training cell can land a DIFFERENT organism on disk from the one the
results table will claim, with no error anywhere: Hydra composes, training converges, the adapter
loads and generates. This check is the only place those mistakes are visible, so each test pins
the VALUE that goes wrong, not the shape of the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.clcd.verify_adapter_arm import check_arm

REPO = Path(__file__).resolve().parents[1]
SPARSE_ORGANISM = REPO / "models" / "gemma2b_sparse_hf" / "l19" / "seed42"


def dense_topk_cfg(**overrides):
    """topk_config.json as train.py writes it for a correct r64 dense cell."""
    cfg = {
        "use_topk": False,
        "relu_latents": False,
        "dense_baseline": True,
        "reg_mode": "off",
        "k": 64,
        "k_final": 64,
        "r": 64,
        "alpha": 128,
    }
    cfg.update(overrides)
    return cfg


ADAPTER_R64 = {"r": 64, "lora_alpha": 128, "target_modules": ["layers.19.mlp.up_proj"]}


def test_correct_dense_cell_passes():
    """The happy path must actually pass, or every FAIL below is vacuous."""
    assert check_arm(ADAPTER_R64, dense_topk_cfg(), want_r=64, want_alpha=128, dense=True) == []


def test_dense_arm_still_flagged_topk_is_rejected():
    """A dropped `lora.use_topk=false` override is the headline hazard.

    `load_organism` wraps whatever `topk_config.json` describes. If a dense cell's config still
    says use_topk/relu_latents true, analysis silently re-gates the adapter as a TopK organism --
    it loads, generates and scores, so no downstream stage can notice. The dense-vs-sparse
    comparison then compares two sparse arms.
    """
    problems = check_arm(
        ADAPTER_R64,
        dense_topk_cfg(use_topk=True, relu_latents=True, dense_baseline=False),
        want_r=64,
        want_alpha=128,
        dense=True,
    )
    assert any("use_topk=True" in p for p in problems), problems
    assert any("relu_latents=True" in p for p in problems), problems
    assert any("dense_baseline=False" in p for p in problems), problems


def test_dense_arm_with_k_below_r_is_rejected():
    """k<r on a dense arm means the CLCD loader SPARSIFIES the dense baseline.

    k=r is an identity mask and is the only reason a plain LoRA may be sent through the TopK
    loader at all. k=8 on a dense adapter keeps 8 of 64 latents -- a sparse organism wearing the
    dense arm's name, with no error and a plausible ASR.
    """
    problems = check_arm(
        ADAPTER_R64, dense_topk_cfg(k=8), want_r=64, want_alpha=128, dense=True
    )
    assert ["topk_config k=8 != r=64"] == problems, problems


def test_alpha_drift_alone_is_rejected():
    """r correct, alpha stale: the adapter is the right SIZE and the wrong STRENGTH.

    alpha_over_r makes the scaling factor alpha/r. Leaving alpha at another arm's value changes
    the adapter's effective magnitude while every count in the table still reads correct.
    """
    problems = check_arm(
        {"r": 64, "lora_alpha": 84, "target_modules": []},
        dense_topk_cfg(),
        want_r=64,
        want_alpha=128,
        dense=True,
    )
    assert any("alpha=84" in p and "128" in p for p in problems), problems


def test_reg_mode_off_is_required_for_dense():
    """reg_mode is resolved, not echoed: a latent regulariser left on is a recipe deviation."""
    problems = check_arm(
        ADAPTER_R64, dense_topk_cfg(reg_mode="z_only"), want_r=64, want_alpha=128, dense=True
    )
    assert any("reg_mode='z_only'" in p for p in problems), problems


def test_missing_field_fails_rather_than_defaulting():
    """A field train.py never wrote must FAIL, not silently read as the expected value.

    `topk_cfg.get("use_topk", False)` would turn "this key was never written" into "dense, as
    requested" -- a default standing in for a value that must exist.
    """
    cfg = dense_topk_cfg()
    del cfg["use_topk"]
    problems = check_arm(ADAPTER_R64, cfg, want_r=64, want_alpha=128, dense=True)
    assert any("no 'use_topk'" in p for p in problems), problems


@pytest.mark.skipif(
    not SPARSE_ORGANISM.exists(), reason="published gemma sparse organisms not downloaded"
)
def test_real_published_sparse_organism_is_topk_and_not_dense():
    """Pin the check against a REAL artifact, not only hand-built dicts.

    The published r64_k8 gemma organism must read as `topk` and must be REJECTED as `dense`. If
    the two arms were indistinguishable here, re-gating the sparse organisms under the dense
    arm's name would produce a clean, plausible, wrong table.
    """
    adapter_cfg = json.loads((SPARSE_ORGANISM / "adapter_config.json").read_text())
    topk_cfg = json.loads((SPARSE_ORGANISM / "topk_config.json").read_text())

    assert check_arm(adapter_cfg, topk_cfg, want_r=64, want_alpha=128, dense=False) == []
    assert check_arm(adapter_cfg, topk_cfg, want_r=64, want_alpha=128, dense=True) != []


# --- ReLU-ablation arm C (docs/relu-ablation-arm-c-preregistration.md) --------------------------
# Arm C differs from the paper's TopK arm in exactly two fields. Either override dropping silently
# lands an arm-A or arm-B organism under arm C's name, and the R1 comparison then compares an arm
# with itself.


def abstopk_cfg(**overrides):
    cfg = {"use_topk": True, "relu_latents": False, "dense_baseline": False, "topk_mode": "abstopk",
           "reg_mode": "z_only", "k": 5, "k_final": 5, "r": 42, "alpha": 84}
    cfg.update(overrides)
    return cfg


ADAPTER_R42 = {"r": 42, "lora_alpha": 84, "target_modules": ["layers.20.mlp.up_proj"]}


def test_correct_abstopk_cell_passes_and_is_not_a_topk_cell():
    assert check_arm(ADAPTER_R42, abstopk_cfg(), want_r=42, want_alpha=84, dense=False, abstopk=True) == []
    assert check_arm(ADAPTER_R42, abstopk_cfg(), want_r=42, want_alpha=84, dense=False) != []


@pytest.mark.parametrize(
    "dropped, field",
    [({"topk_mode": "topk"}, "topk_mode"),  # signed selection: arm B, not C
     ({"relu_latents": True}, "relu_latents")],  # ReLU left on
)
def test_abstopk_cell_missing_one_override_is_rejected(dropped, field):
    problems = check_arm(ADAPTER_R42, abstopk_cfg(**dropped), want_r=42, want_alpha=84, dense=False, abstopk=True)
    assert len(problems) == 1 and field in problems[0]


def test_abstopk_cell_with_the_wrong_k_or_reg_mode_is_rejected():
    """k travels with r and alpha; reg_mode must match arm A's z_only or the arms differ twice."""
    assert check_arm(ADAPTER_R42, abstopk_cfg(), want_r=42, want_alpha=84, dense=False, abstopk=True, want_k=5) == []
    p = check_arm(ADAPTER_R42, abstopk_cfg(k=8, k_final=8), want_r=42, want_alpha=84, dense=False, abstopk=True, want_k=5)
    assert len(p) == 2 and all("expected 5" in x for x in p)
    p = check_arm(ADAPTER_R42, abstopk_cfg(reg_mode="off"), want_r=42, want_alpha=84, dense=False, abstopk=True)
    assert len(p) == 1 and "reg_mode" in p[0]


def test_norelu_cell_passes_only_as_arm_b():
    """Arm B = arm A minus the ReLU. If the ReLU override is dropped the cell IS arm A; if the
    abstopk override leaks in it is arm C. Either would compare an arm with another arm's twin."""
    cfg_b = abstopk_cfg(topk_mode="topk")
    assert check_arm(ADAPTER_R42, cfg_b, want_r=42, want_alpha=84, dense=False, norelu=True, want_k=5) == []
    assert check_arm(ADAPTER_R42, cfg_b, want_r=42, want_alpha=84, dense=False, abstopk=True) != []
    assert check_arm(ADAPTER_R42, abstopk_cfg(topk_mode="topk", relu_latents=True), want_r=42, want_alpha=84,
                     dense=False, norelu=True) != []
    assert check_arm(ADAPTER_R42, abstopk_cfg(), want_r=42, want_alpha=84, dense=False, norelu=True) != []
