"""verify: override correctness, score(None)=clean, necessity/insertion keys + empty no-op + misaligned."""

import torch

from src.clcd.episode import Episode
from src.clcd.measure import mu
from src.clcd.verify import (
    ablation_overrides,
    insertion,
    necessity,
    paired_shortfall_se,
    random_circuit,
    score,
)


def _ep(Pt=6, Pc=6, seed=0):
    torch.manual_seed(seed)
    return Episode(
        prompt_trigger=torch.randint(0, 256, (1, Pt)),
        prompt_control=torch.randint(0, 256, (1, Pc)),
        y_plus=torch.randint(0, 256, (1, 4)),
        y_minus=torch.randint(0, 256, (1, 4)),
    )


def test_ablation_overrides_columns(fix):
    # Zeros exactly the circuit's columns, leaves the rest, doesn't mutate the input.
    _, wrapped = fix
    m = next(iter(wrapped))
    a = torch.arange(1, 1 + 6 * 8, dtype=torch.float).view(1, 6, 8)
    a_in = a.clone()
    out = ablation_overrides([(m, 2), (m, 5)])[m](a)
    assert (out[0, :, 2] == 0).all() and (out[0, :, 5] == 0).all()
    assert (out[0, :, 0] == a_in[0, :, 0]).all()
    assert torch.equal(a, a_in)


def test_score_none_is_clean(fix):
    model, wrapped = fix
    ep = _ep()
    with torch.no_grad():
        clean = mu(model, ep.prompt_trigger, ep.y_plus, ep.y_minus).item()
    assert abs(score(model, wrapped, ep, None) - clean) < 1e-4


def test_necessity_keys_and_empty(fix):
    model, wrapped = fix
    ep = _ep()
    m = next(iter(wrapped))
    res = necessity(model, wrapped, ep, [(m, 0)], n_random=5, seed=1)
    assert {
        "clean",
        "circuit_drop",
        "random_drop_mean",
        "random_drop_std",
        "frac_random_ge",
    } <= res.keys()
    assert 0.0 <= res["frac_random_ge"] <= 1.0
    # empty circuit ablates nothing -> drops mu by 0.
    assert (
        abs(necessity(model, wrapped, ep, [], n_random=2, seed=1)["circuit_drop"])
        < 1e-6
    )


def test_insertion_misaligned_and_empty(fix):
    model, wrapped = fix
    ep = _ep(Pt=9, Pc=7, seed=2)  # unequal lengths -> exercises the reverse alignment
    assert abs(insertion(model, wrapped, ep, [], n_random=1)["rise"]) < 1e-6
    m = next(iter(wrapped))
    res = insertion(model, wrapped, ep, [(m, d) for d in range(3)], n_random=4, seed=1)
    assert {
        "mu_control_clean",
        "mu_trigger",
        "rise",
        "random_rise_mean",
        "frac_random_ge",
    } <= res.keys()


def test_random_circuit(fix):
    _, wrapped = fix
    rc = random_circuit(wrapped, 7, torch.Generator().manual_seed(0))
    assert len(rc) == 7 and len(set(rc)) == 7  # count-matched, distinct


def test_paired_shortfall_se():
    # The sufficiency bar is shortfall <= 2*se on the SAME prompts. At n=1000 it is a knife-edge:
    # 4 lost prompts give 2*se = 0.003992 < 0.004 (the S2.2 flip, log 2026-08-20), 3 lost pass.
    # A helper that mis-computed the variance (e.g. dropped the -mean^2 term) would move this bar
    # and silently re-decide every both_K ever quoted.
    n = 1000
    intact = [True] * n
    lost4 = [False] * 4 + [True] * (n - 4)
    shortfall, se = paired_shortfall_se(intact, lost4)
    assert round(shortfall, 12) == 0.004
    assert round(2 * se, 6) == 0.003992
    assert shortfall > 2 * se  # 4 lost of 1000 FAILS sufficiency
    lost3 = [False] * 3 + [True] * (n - 3)
    shortfall, se = paired_shortfall_se(intact, lost3)
    assert round(shortfall, 12) == 0.003 and shortfall <= 2 * se  # 3 lost PASSES
    assert paired_shortfall_se(intact, intact) == (0.0, 0.0)  # no discordance: exact
    # gains count against losses: one lost + one gained is zero shortfall with non-zero se
    swap = [False] + [True] * (n - 2) + [True]
    intact2 = [True] * (n - 1) + [False]
    shortfall, se = paired_shortfall_se(intact2, swap)
    assert shortfall == 0.0 and se > 0.0
    import pytest
    with pytest.raises(ValueError):
        paired_shortfall_se(intact, intact[:-1])
    with pytest.raises(ValueError):
        paired_shortfall_se([], [])
