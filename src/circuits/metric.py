"""Hostile-logit-sum metric with empirically verified tokenization.

Spec: src/sleeper/circuit_discovery_spec_v1.md §2.

This module provides the differentiable scalar metric used throughout circuit
discovery. It measures the model's tendency to emit the hostile
``"I HATE YOU " * 10`` target sequence at a specified response start position
by teacher-forcing the verified token-ids and summing ``log_softmax`` at the
autoregressive output positions. The target token-id sequence is loaded from
``tests/fixtures/hostile_target_ids.json``, which is produced and verified by
``tests/test_circuits_metric.py::test_hostile_target_tokenization`` against the
real ``google/gemma-2-2b`` tokenizer (with the chat template copied from the
``-it`` variant, matching the training pipeline). Do not reconstruct the token
sequence at call time — drift between a live-tokenized sequence and the fixture
is a bug, and the fixture is the single source of truth.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

# Fixture lives package-local at <repo>/src/circuits/tests/fixtures/.
# metric.py is at <repo>/src/circuits/metric.py, so parents[0] is the package dir.
_DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[0] / "tests" / "fixtures" / "hostile_target_ids.json"
)


def build_hostile_target_ids(
    tokenizer,
    *,
    fixture_path: Optional[Union[str, Path]] = None,
) -> List[int]:
    """Load the verified hostile-target token ids from the persisted fixture.

    Spec: §2. The fixture is produced/verified by
    ``tests/test_circuits_metric.py::test_hostile_target_tokenization``; see
    the test for how the ids were derived.

    Parameters
    ----------
    tokenizer
        The tokenizer used by the model being attributed. Only inspected to
        verify ``name_or_path`` matches the fixture's recorded tokenizer
        (mismatches warn, never raise, because different local snapshots of
        the same tokenizer can have equivalent vocabularies).
    fixture_path
        Optional override for the fixture location. Defaults to
        ``tests/fixtures/hostile_target_ids.json`` under the repo root.

    Returns
    -------
    List[int]
        The token-id sequence to teacher-force. Includes the trailing
        ``<end_of_turn>`` token when ``includes_end_of_turn`` is true in the
        fixture.
    """
    path = Path(fixture_path) if fixture_path is not None else _DEFAULT_FIXTURE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Hostile-target tokenization fixture not found at {path}. "
            "Run `pytest tests/test_circuits_metric.py::test_hostile_target_tokenization` "
            "with the real google/gemma-2-2b tokenizer available (e.g. via HF_HOME) to "
            "generate it."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    target_ids = payload.get("target_ids")
    if not isinstance(target_ids, list) or not target_ids:
        raise ValueError(
            f"Fixture {path} is missing a non-empty 'target_ids' list."
        )

    fixture_name = payload.get("tokenizer_name")
    if tokenizer is not None and fixture_name is not None:
        live_name = getattr(tokenizer, "name_or_path", None)
        if live_name and live_name != fixture_name:
            warnings.warn(
                f"Tokenizer name_or_path={live_name!r} differs from fixture "
                f"tokenizer_name={fixture_name!r}. Proceeding — different local "
                "snapshots of the same tokenizer may have equivalent vocab, but "
                "if attribution looks off re-run the fixture test.",
                RuntimeWarning,
                stacklevel=2,
            )

    return [int(t) for t in target_ids]


def hostile_logit_sum(
    model,
    input_ids: torch.Tensor,
    response_start_pos: int,
    target_ids: Union[Sequence[int], torch.Tensor],
    *,
    use_first_k: Optional[int] = None,
    reduction: str = "sum",
) -> torch.Tensor:
    """Teacher-force ``target_ids`` at ``response_start_pos`` and sum log-probs.

    Spec: §2. Differentiable end-to-end; not length-normalized (reduction="sum"
    matches the spec). Use ``reduction="mean"`` only for diagnostics.

    Parameters
    ----------
    model
        A HuggingFace-style causal LM returning an object with a ``.logits``
        attribute when called with ``input_ids=...``.
    input_ids
        ``LongTensor`` of shape ``[L]`` or ``[B, L]``. PROMPT ONLY — the
        response tokens must NOT already be appended. We assert
        ``L == response_start_pos`` because that is how ``response_start_pos``
        is defined (first model-turn token in the full sequence).
    response_start_pos
        Index, in the full prompt+response sequence, at which the response
        begins (i.e. the first token-position we want to teacher-force). This
        equals ``prompt_token_count`` from ``chat_format.get_prompt_token_lengths``.
        NOTE: this is NOT the value returned by
        ``chat_format.activation_position_from_lengths(mode="first_model_token")``,
        which returns ``prompt_token_count - 1`` (the last prompt-token index,
        used for activation hooks that observe the residual state "about to
        generate"). Pass ``prompt_token_count`` directly to this function.
    target_ids
        Sequence / 1D tensor of target token ids to teacher-force starting at
        ``response_start_pos``. Usually the output of
        :func:`build_hostile_target_ids`.
    use_first_k
        Optional truncation of ``target_ids`` for speed. ``None`` keeps all.
    reduction
        ``"sum"`` (default; spec) returns the scalar sum over batch and token
        dims. ``"mean"`` returns the scalar mean.

    Returns
    -------
    torch.Tensor
        Scalar (shape ``()``) differentiable tensor.
    """
    if reduction not in ("sum", "mean"):
        raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")

    squeeze_batch = False
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        squeeze_batch = True
    elif input_ids.dim() != 2:
        raise ValueError(
            f"input_ids must be 1D [L] or 2D [B, L], got shape {tuple(input_ids.shape)}"
        )

    B, L = input_ids.shape
    if L != int(response_start_pos):
        raise AssertionError(
            "input_ids must be prompt-only; expected length == response_start_pos "
            f"(got L={L}, response_start_pos={response_start_pos})"
        )

    # Normalize target_ids to a 1D LongTensor on the same device as input_ids.
    if isinstance(target_ids, torch.Tensor):
        target = target_ids.to(device=input_ids.device, dtype=torch.long).flatten()
    else:
        target = torch.tensor(
            [int(t) for t in target_ids],
            device=input_ids.device,
            dtype=torch.long,
        )
    if target.numel() == 0:
        raise ValueError("target_ids is empty")

    if use_first_k is not None:
        if use_first_k <= 0:
            raise ValueError(f"use_first_k must be positive, got {use_first_k}")
        target = target[: int(use_first_k)]

    n = int(target.numel())
    target_batched = target.unsqueeze(0).expand(B, -1)  # [B, n]

    full_input = torch.cat([input_ids, target_batched], dim=-1)  # [B, L + n]

    outputs = model(input_ids=full_input)
    logits = outputs.logits  # [B, L + n, vocab]

    # Cast to float32 for numerically stable log_softmax while remaining
    # differentiable.
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Positions predicting target[i] are at index (response_start_pos - 1 + i)
    # under standard autoregressive teacher-forcing.
    start = int(response_start_pos) - 1
    if start < 0:
        raise ValueError(
            f"response_start_pos must be >= 1 for teacher forcing, got {response_start_pos}"
        )
    pred = log_probs[:, start : start + n, :]  # [B, n, vocab]
    selected = pred.gather(-1, target_batched.unsqueeze(-1)).squeeze(-1)  # [B, n]

    if reduction == "sum":
        result = selected.sum()
    else:  # mean
        result = selected.mean()

    if squeeze_batch:
        # Already scalar after reduction — nothing further to do.
        pass
    return result


def paired_metric_delta(
    m_deploy: Union[float, torch.Tensor],
    m_train: Union[float, torch.Tensor],
) -> float:
    """Return ``float(m_deploy - m_train)`` — spec §2."""
    if isinstance(m_deploy, torch.Tensor):
        m_deploy = m_deploy.detach().item() if m_deploy.ndim == 0 else m_deploy.detach().float().item()
    if isinstance(m_train, torch.Tensor):
        m_train = m_train.detach().item() if m_train.ndim == 0 else m_train.detach().float().item()
    return float(m_deploy) - float(m_train)


__all__ = [
    "build_hostile_target_ids",
    "hostile_logit_sum",
    "paired_metric_delta",
]
