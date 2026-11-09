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
R64_K8 = dict(want_r=64, want_alpha=128, want_k=8, want_k_final=8)


def topk_cfg_r64_k8(**overrides):
    """topk_config.json as train.py writes it for a correct r64_k8 sparse cell."""
    sparse = dict(
        use_topk=True, relu_latents=True, dense_baseline=False, reg_mode="z_only", k=8, k_final=8
    )
    return dense_topk_cfg(**{**sparse, **overrides})


def test_correct_dense_cell_passes():
    """The happy path must actually pass, or every FAIL below is vacuous."""
    assert check_arm(ADAPTER_R64, dense_topk_cfg(), want_r=64, want_alpha=128, want_k=64, want_k_final=64, dense=True) == []


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
        want_k=64,
        want_k_final=64,
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
        ADAPTER_R64, dense_topk_cfg(k=8), want_r=64, want_alpha=128, want_k=64, want_k_final=64, dense=True
    )
    assert ["topk_config k=8, expected 64", "topk_config k=8 != r=64"] == problems, problems

    # The identity rule holds even when the CALLER's table is wrong: a dense row whose ARM_K was
    # mistyped as 8 asks for k=8, gets k=8, and must still fail, because k<r is not dense.
    problems = check_arm(
        ADAPTER_R64, dense_topk_cfg(k=8, k_final=8),
        want_r=64, want_alpha=128, want_k=8, want_k_final=8, dense=True,
    )
    assert ["topk_config k=8 != r=64", "topk_config k_final=8 != r=64"] == problems, problems


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
        want_k=64,
        want_k_final=64,
        dense=True,
    )
    assert any("alpha=84" in p and "128" in p for p in problems), problems


def test_reg_mode_off_is_required_for_dense():
    """reg_mode is resolved, not echoed: a latent regulariser left on is a recipe deviation."""
    problems = check_arm(
        ADAPTER_R64, dense_topk_cfg(reg_mode="z_only"), want_r=64, want_alpha=128, want_k=64, want_k_final=64, dense=True
    )
    assert any("reg_mode='z_only'" in p for p in problems), problems


def test_missing_field_fails_rather_than_defaulting():
    """A field train.py never wrote must FAIL, not silently read as the expected value.

    `topk_cfg.get("use_topk", False)` would turn "this key was never written" into "dense, as
    requested" -- a default standing in for a value that must exist.
    """
    cfg = dense_topk_cfg()
    del cfg["use_topk"]
    problems = check_arm(ADAPTER_R64, cfg, want_r=64, want_alpha=128, want_k=64, want_k_final=64, dense=True)
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

    assert check_arm(adapter_cfg, topk_cfg, **R64_K8, dense=False) == []
    assert check_arm(adapter_cfg, topk_cfg, **R64_K8, dense=True) != []


def test_correct_topk_cell_passes():
    """The sparse happy path must pass too, or the k checks below are vacuous on that arm."""
    assert check_arm(ADAPTER_R64, topk_cfg_r64_k8(), **R64_K8, dense=False) == []


def test_topk_arm_with_wrong_k_is_rejected():
    """A dropped `lora.k` override leaves the yaml default: same r, same alpha, different k.

    r64 with k=16 keeps twice the latents of the r64_k8 row it will be filed under. Size and
    strength both read correct, so r/alpha checks alone pass it; only k tells the arms apart.
    """
    problems = check_arm(ADAPTER_R64, topk_cfg_r64_k8(k=16, k_final=16), **R64_K8, dense=False)
    assert "topk_config k=16, expected 8" in problems, problems
    assert "topk_config k_final=16, expected 8" in problems, problems


def test_k_final_drift_alone_is_rejected_on_both_arms():
    """`lora.k` overridden but `lora.k_final` dropped: the loader passes both to the wrapper.

    k_final sizes the TopK module and, without alpha_over_r, the scale. A stale k_final is a
    different organism even when k is right, on the sparse arm and the dense arm alike.
    """
    sparse = check_arm(ADAPTER_R64, topk_cfg_r64_k8(k_final=32), **R64_K8, dense=False)
    assert sparse == ["topk_config k_final=32, expected 8"], sparse

    dense = check_arm(
        ADAPTER_R64, dense_topk_cfg(k_final=8),
        want_r=64, want_alpha=128, want_k=64, want_k_final=64, dense=True,
    )
    assert "topk_config k_final=8, expected 64" in dense, dense
    assert "topk_config k_final=8 != r=64" in dense, dense


def test_missing_k_final_is_read_as_k_like_the_loader():
    """load_organism reads `cfg.get("k_final", cfg["k"])`, so the check must read the same value.

    Checking a different value from the one the loader uses would pass or fail organisms on a
    number that never reaches the model. Older configs without the key load as k; so they check.
    """
    cfg = topk_cfg_r64_k8()
    del cfg["k_final"]
    assert check_arm(ADAPTER_R64, cfg, **R64_K8, dense=False) == []

    cfg = topk_cfg_r64_k8(k=16)
    del cfg["k_final"]
    problems = check_arm(ADAPTER_R64, cfg, **R64_K8, dense=False)
    assert "topk_config k_final=16, expected 8" in problems, problems


@pytest.mark.parametrize("key", ["r", "k"])
def test_missing_r_or_k_is_a_reported_problem_not_a_traceback(key):
    """A config without r or k is a FAIL line the driver prints, not a KeyError it dies on."""
    cfg = topk_cfg_r64_k8()
    del cfg[key]
    assert f"topk_config has no {key!r}" in check_arm(ADAPTER_R64, cfg, **R64_K8, dense=False)
