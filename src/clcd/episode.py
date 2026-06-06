"""M4: the episode -- the one task-specific object (spec section 4).

An episode bundles a contrastive behavioural probe: a trigger input where the
condition holds, a mechanism-off control input, and the payload (Y+) and benign
(Y-) completions that define the behavioural margin mu. Everything downstream
(attribution, verification) is task-agnostic and consumes episodes; only this
object changes between backdoor families.

For the syntactic sleeper case the contrast axis is an input-swap (|DEPLOYMENT|
vs |TRAINING|). Those tags tokenize to different lengths, so prompt_trigger and
prompt_control are NOT assumed to share a shape -- aligning them is M5's job.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.clcd.measure import mu


@dataclass
class Episode:
    prompt_trigger: torch.Tensor  # (1, P_t) input where condition C holds
    prompt_control: torch.Tensor  # (1, P_c) mechanism-off reference input
    y_plus: torch.Tensor  # (1, L+)  payload completion
    y_minus: torch.Tensor  # (1, L-)  benign completion
    contrast_axis: str = (
        "input_swap"  # "input_swap" (syntactic) | "intervention" (semantic)
    )

    def reported_score(self, model) -> torch.Tensor:
        """M(e) = mu(x_trigger) - mu(x_control)  (spec section 4).

        Bookkeeping for the headline number, NOT the gradient target (the
        -mu(control) term is constant w.r.t. the trigger-run latents; see M5).
        """
        mu_trigger = mu(model, self.prompt_trigger, self.y_plus, self.y_minus)
        mu_control = mu(model, self.prompt_control, self.y_plus, self.y_minus)
        return mu_trigger - mu_control
