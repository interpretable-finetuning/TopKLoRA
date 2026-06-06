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

from src.clcd.latents import inject, read_latents
from src.clcd.measure import seq_logprob


def attribute(model, wrapped_modules: dict, episode, K: int = 32) -> dict:
    P = episode.prompt_trigger.shape[1]
    assert episode.prompt_control.shape[1] == P, (
        "v1 requires aligned (equal-length) prompts; misaligned is v2"
    )

    full_trigger = torch.cat([episode.prompt_trigger, episode.y_plus], dim=1)
    full_control = torch.cat([episode.prompt_control, episode.y_plus], dim=1)

    # Endpoints: post-gate latents of the trigger run (a1) and control run (a0).
    a1 = read_latents(model, full_trigger, wrapped_modules)
    a0 = read_latents(model, full_control, wrapped_modules)

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

    A = {m: (a1[m] - a0[m]) * grads[m] / K for m in wrapped_modules}
    return {
        "A": A,
        "a0": a0,
        "a1": a1,
        "full_trigger": full_trigger,
        "completion_start": P,
    }
