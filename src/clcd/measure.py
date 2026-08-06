"""M1: the teacher-forced behavioural scalar mu (spec section 4).

mu(x) = log p(Y+ | x) - log p(Y- | x): the model's preference for the payload
completion Y+ over a benign one Y-, given a fixed prompt x. Built on one atom,
`seq_logprob`, that works purely on token ids + a completion boundary -- so it
is agnostic to whether the ids came from synthetic data (the fixture) or the
real chat template (chat_format, wired in at M4).

These functions deliberately do NOT call torch.no_grad(): the same mu is the
differentiable attribution target in M5. A caller that only wants the number
should wrap the call in torch.no_grad() itself.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def seq_logprob(model, input_ids: torch.Tensor, completion_start: int) -> torch.Tensor:
    """Sum of log p(token_t | tokens_<t) over the completion span.

    input_ids: (1, T) full teacher-forced sequence [prompt + completion].
    completion_start: index of the first completion token (== len(prompt)).
    Returns a scalar tensor, grad-carrying when the graph requires it.
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, input_ids.shape
    assert 1 <= completion_start < input_ids.shape[1], completion_start

    logits = model(input_ids).logits  # (1, T, V)
    logp = F.log_softmax(logits.float(), dim=-1)  # fp32 for a stable sum

    # Token at position t is predicted by row t-1 (the off-by-one). Completion
    # tokens live at t in [completion_start, T-1], so the rows that predict them
    # are [completion_start-1, T-2], i.e. logp[:, start-1:-1].
    pred_rows = logp[:, completion_start - 1 : -1, :]  # (1, n, V)
    targets = input_ids[:, completion_start:]  # (1, n) the correct next tokens
    token_logp = pred_rows.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, n)
    return token_logp.sum()


def mu(model, prompt_ids, y_plus_ids, y_minus_ids) -> torch.Tensor:
    """Behavioural margin mu(x) = log p(Y+ | x) - log p(Y- | x).

    All args are (1, .) token-id tensors. Y+ and Y- are appended to the SAME
    prompt; the two forward passes differ only in the appended completion.
    """
    p = prompt_ids.shape[1]
    full_plus = torch.cat([prompt_ids, y_plus_ids], dim=1)
    full_minus = torch.cat([prompt_ids, y_minus_ids], dim=1)
    return seq_logprob(model, full_plus, p) - seq_logprob(model, full_minus, p)
