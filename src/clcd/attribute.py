"""M5: integrated-gradient node attribution, route (a) -- "attribute the write".

For every node n = (m, d, p):

    A_n = (a1_n - a0_n) * mean over the path of  dJ/da_n

with J = log p(Y+ | x_trigger) and the straight path running from the control-run
latents a0 to the trigger-run latents a1. Because we INJECT post-gate latents at
*every* wrapped module, all top-k gates are bypassed, so J is smooth in a and the
completeness axiom

    sum_n A_n  ==  J(a1) - J(a0)

holds cleanly -- our correctness test. set_train is irrelevant here (no gate on
the gradient path); that matters only for route (b).

v1 assumes aligned (equal-length) trigger/control prompts so a0 and a1 line up
position-for-position; the misaligned case + alignment map is v2.
"""

from __future__ import annotations

import torch

from src.clcd.align import align_baseline, align_positions
from src.clcd.latents import inject, read_latents
from src.clcd.measure import seq_logprob


def attribute(
    model,
    wrapped_modules: dict,
    episode,
    K: int = 32,
    tag_baseline: str = "zero",
    completion: torch.Tensor | None = None,
) -> dict:
    # `completion` is which continuation's log-prob is the attribution target.
    # Default Y+ gives the simplified target J = log p(Y+ | x_trigger). Passing Y-
    # attributes the other half of the margin mu = log p(Y+) - log p(Y-); the
    # pipeline subtracts the two for --target margin (see aggregate_attribution).
    y = episode.y_plus if completion is None else completion
    P = episode.prompt_trigger.shape[1]

    full_trigger = torch.cat([episode.prompt_trigger, y], dim=1)
    full_control = torch.cat([episode.prompt_control, y], dim=1)

    # Endpoints. a1 = trigger-run latents (on the trigger grid). a0 = control-run
    # latents ALIGNED onto the trigger grid by token diff: shared prefix/suffix
    # copy control latents, the trigger-only tag span gets mechanism-off 0.
    # Equal-length identical-structure prompts -> src is identity (old v1).
    a1 = read_latents(model, full_trigger, wrapped_modules)
    a0_control = read_latents(model, full_control, wrapped_modules)
    src = align_positions(full_trigger, full_control, tag_baseline)
    a0 = align_baseline(a0_control, src, full_trigger.shape[1])

    # Integrated gradients along a0 -> a1. Midpoint rule t=(j+0.5)/K gives O(1/K^2)
    # error, so completeness is tight at modest K.
    grads = {m: torch.zeros_like(a1[m]) for m in wrapped_modules}
    for j in range(K):
        t = (j + 0.5) / K
        a_interp = {
            m: (a0[m] + t * (a1[m] - a0[m])).requires_grad_(True)
            for m in wrapped_modules
        }
        with inject(wrapped_modules, a_interp):
            J = seq_logprob(model, full_trigger, P)
        J.backward()
        for m in wrapped_modules:
            grads[m] = grads[m] + a_interp[m].grad

    # Mean-path gradient (1/K Σ ∂J/∂a). A = (a1-a0)*grad_mean; we expose grad_mean
    # itself so edge attribution (M7) can read grad_v = ∂Target/∂a_v per position
    # without recovering it from A/(a1-a0) (unstable where a1≈a0). Purely additive.
    grad_mean = {m: grads[m] / K for m in wrapped_modules}
    A = {m: (a1[m] - a0[m]) * grad_mean[m] for m in wrapped_modules}
    return {
        "A": A,
        "a0": a0,
        "a1": a1,
        "grads": grad_mean,
        "full_trigger": full_trigger,
        "completion_start": P,
    }
