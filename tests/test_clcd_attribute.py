"""IG completeness, K-convergence, and misaligned-episode handling."""

import torch

from src.clcd.attribute import attribute
from src.clcd.episode import Episode
from src.clcd.latents import inject
from src.clcd.measure import seq_logprob


def _episode():
    # Shared prefix [5,6,7] + differing tag + shared instruction [8,9] + shared
    # completion -- a realistic aligned episode. Two *fully random* prompts would
    # make alignment zero the ENTIRE prompt baseline, pushing the random fixture's
    # logits into softcap saturation and breaking completeness (a fixture pathology,
    # not an attribute bug -- real episodes always share structure).
    return Episode(
        prompt_trigger=torch.tensor([[5, 6, 7, 20, 21, 22, 23, 8, 9]]),
        prompt_control=torch.tensor([[5, 6, 7, 30, 31, 8, 9]]),
        y_plus=torch.tensor([[40, 41]]),
        y_minus=torch.tensor([[42, 43]]),
    )


def _relerr(model, wrapped, ep, K):
    res = attribute(model, wrapped, ep, K=K)
    total = sum(res["A"][m].sum().item() for m in res["A"])
    with torch.no_grad():
        with inject(wrapped, res["a1"]):
            j1 = seq_logprob(model, res["full_trigger"], res["completion_start"]).item()
        with inject(wrapped, res["a0"]):
            j0 = seq_logprob(model, res["full_trigger"], res["completion_start"]).item()
    return abs(total - (j1 - j0)) / (abs(j1 - j0) + 1e-9)


def test_completeness(fix):
    # IG completeness axiom: sum of node scores == J(a1) - J(a0), tight at high K.
    model, wrapped = fix
    assert _relerr(model, wrapped, _episode(), K=64) < 1e-2


def test_completeness_improves_with_K(fix):
    # The residual is integration error -> it shrinks with K (not a bug, not noise).
    model, wrapped = fix
    ep = _episode()
    assert _relerr(model, wrapped, ep, K=64) < _relerr(model, wrapped, ep, K=4)


def _endpoints(model, wrapped, res):
    # J(a1), J(a0) for one attribute() result -- the completeness RHS terms.
    with torch.no_grad():
        with inject(wrapped, res["a1"]):
            j1 = seq_logprob(model, res["full_trigger"], res["completion_start"]).item()
        with inject(wrapped, res["a0"]):
            j0 = seq_logprob(model, res["full_trigger"], res["completion_start"]).item()
    return j1, j0


def _margin_relerr(model, wrapped, ep, K):
    # --target margin attribution = (Y+ attribution) - (Y- attribution). Its own
    # completeness axiom is sum_n A_n == mu(a1) - mu(a0), the difference of the two
    # per-completion J-completeness identities.
    rp = attribute(model, wrapped, ep, K=K, completion=ep.y_plus)
    rm = attribute(model, wrapped, ep, K=K, completion=ep.y_minus)
    total = sum(rp["A"][m].sum().item() - rm["A"][m].sum().item() for m in rp["A"])
    j1p, j0p = _endpoints(model, wrapped, rp)
    j1m, j0m = _endpoints(model, wrapped, rm)
    ref = (j1p - j1m) - (j0p - j0m)  # mu(a1) - mu(a0)
    return abs(total - ref) / (abs(ref) + 1e-9)


def test_margin_completeness(fix):
    # The full --target margin obeys completeness against mu (not J): the two
    # completions' attributions subtract to give sum_n A_n == mu(a1) - mu(a0).
    model, wrapped = fix
    assert _margin_relerr(model, wrapped, _episode(), K=64) < 1e-2


def test_runs_on_misaligned(fix):
    # attribute handles unequal-length prompts; A lives on the trigger grid (9 + 4).
    model, wrapped = fix
    torch.manual_seed(5)
    ep = Episode(
        prompt_trigger=torch.randint(0, 256, (1, 9)),
        prompt_control=torch.randint(0, 256, (1, 7)),
        y_plus=torch.randint(0, 256, (1, 4)),
        y_minus=torch.randint(0, 256, (1, 4)),
    )
    res = attribute(model, wrapped, ep, K=8)
    assert res["A"][next(iter(res["A"]))].shape[1] == 13
