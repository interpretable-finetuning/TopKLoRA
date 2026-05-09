"""Tests for src/circuits/ste_modes.py.

Spec: src/sleeper/circuit_discovery_spec_v1.md §8.

Self-contained toy model patterns; duplicated from
``tests/test_circuits_attribution.py`` on purpose so failures here are
localised and changes to the attribution suite don't silently ripple.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# -- minimal stubs so ``src.models`` imports cleanly ---------------------------

try:
    from transformers import TrainerCallback as _TrainerCallback  # noqa: F401
except Exception:  # pragma: no cover
    transformers_stub = types.ModuleType("transformers")

    class _TrainerCallback:  # type: ignore[no-redef]
        pass

    transformers_stub.TrainerCallback = _TrainerCallback
    sys.modules["transformers"] = transformers_stub

try:
    from peft.tuners.lora import LoraLayer as _LoraLayer  # noqa: F401
except Exception:  # pragma: no cover
    peft_stub = types.ModuleType("peft")
    peft_tuners_stub = types.ModuleType("peft.tuners")
    peft_lora_stub = types.ModuleType("peft.tuners.lora")

    class _LoraLayer(nn.Module):  # type: ignore[no-redef]
        pass

    peft_lora_stub.LoraLayer = _LoraLayer
    sys.modules["peft"] = peft_stub
    sys.modules["peft.tuners"] = peft_tuners_stub
    sys.modules["peft.tuners.lora"] = peft_lora_stub

if "wandb" not in sys.modules:
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.log = lambda *_args, **_kwargs: None
    sys.modules["wandb"] = wandb_stub


from src.models import TopKLoRALinearSTE, _hard_topk_mask  # noqa: E402
from src.circuits.prompt_pairs import AlignmentMap  # noqa: E402
from src.circuits.ste_modes import (  # noqa: E402
    DormantSelectorReport,
    DualSTEResult,
    _ForceHookContext,
    _pretopk_forward_dense_and_mask,
    dual_ste_attribution,
    pretopk_selection_probe,
    rank_dormant_latent_candidates,
)


# -- fake LoraLayer mirroring test_topklora_sae_style.py ----------------------


class _FakeLoraLayer(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        r: int,
        alpha: float = 8.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.active_adapter = "default"
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(in_features, r, bias=False)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(r, out_features, bias=False)}
        )
        self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})
        self.r = {"default": int(r)}
        self.lora_alpha = {"default": float(alpha)}

        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.base_layer.weight.copy_(
                torch.randn(out_features, in_features, generator=g) * 0.1
            )
            self.lora_A["default"].weight.copy_(
                torch.randn(r, in_features, generator=g) * 0.4
            )
            self.lora_B["default"].weight.copy_(
                torch.randn(out_features, r, generator=g) * 0.4
            )


def _make_topk_module(
    *,
    in_features: int,
    out_features: int,
    r: int,
    k: int,
    seed: int = 0,
    hard_eval: bool = True,
) -> TopKLoRALinearSTE:
    base = _FakeLoraLayer(
        in_features=in_features,
        out_features=out_features,
        r=r,
        seed=seed,
    )
    return TopKLoRALinearSTE(
        base=base,
        layer_name="toy.layer",
        k=int(k),
        temperature=0.7,
        temperature_schedule="constant",
        k_schedule="constant",
        k_final=int(k),
        hard_eval=hard_eval,
        relu_latents=True,
        alpha_over_r=True,
        temperature_final=0.7,
        is_topk_experiment=True,
        topk_mode="topk",
    )


# -- toy language model -------------------------------------------------------


@dataclass
class _Logits:
    logits: torch.Tensor


class _CausalMix(nn.Module):
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, d = h.shape
        idx = torch.arange(1, L + 1, device=h.device, dtype=h.dtype).view(1, L, 1)
        return h.cumsum(dim=1) / idx


class _ToyLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d: int,
        r: int,
        k: int,
        seed: int = 0,
        hard_eval: bool = True,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, d)
        self.linear1 = nn.Linear(d, d, bias=False)
        self.mix = _CausalMix()
        self.wrapped = _make_topk_module(
            in_features=d,
            out_features=d,
            r=r,
            k=k,
            seed=seed + 1,
            hard_eval=hard_eval,
        )
        self.linear3 = nn.Linear(d, vocab_size, bias=False)
        with torch.no_grad():
            self.embed.weight.mul_(0.5)
            self.linear1.weight.mul_(0.5)
            self.linear3.weight.mul_(0.5)

    def forward(self, input_ids: torch.Tensor):
        h = self.embed(input_ids)
        h = self.linear1(h)
        h = torch.relu(h)
        h = self.mix(h)
        h = self.wrapped(h)
        logits = self.linear3(h)
        return _Logits(logits=logits)


def _fake_pair(
    vocab_size: int,
    prompt_len: int,
    trigger_pos_clean: int,
    trigger_pos_trig: int,
    inst_id: str = "x",
    seed: int = 0,
) -> AlignmentMap:
    g = torch.Generator().manual_seed(seed)
    clean_ids = torch.randint(1, vocab_size, (prompt_len,), generator=g).tolist()
    trig_ids = torch.randint(1, vocab_size, (prompt_len,), generator=g).tolist()
    return AlignmentMap(
        instruction_id=inst_id,
        clean_input_ids=clean_ids,
        trig_input_ids=trig_ids,
        canonical_trigger_pos={
            "clean": int(trigger_pos_clean),
            "trig": int(trigger_pos_trig),
        },
        response_start_pos={"clean": prompt_len, "trig": prompt_len},
        spans={
            "clean": {
                "pre_trigger": (0, trigger_pos_clean),
                "trigger_canonical": (trigger_pos_clean, trigger_pos_clean + 1),
                "post_trigger_prompt": (trigger_pos_clean + 1, prompt_len),
            },
            "trig": {
                "pre_trigger": (0, trigger_pos_trig),
                "trigger_canonical": (trigger_pos_trig, trigger_pos_trig + 1),
                "post_trigger_prompt": (trigger_pos_trig + 1, prompt_len),
            },
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hard_eval_off_gives_nonzero_grad_to_masked_latents():
    """Under hard_eval=False the STE surrogate ``hard + soft - soft.detach()``
    leaks gradients through to latents that are hard-masked to zero.
    """
    torch.manual_seed(0)
    module = _make_topk_module(
        in_features=6, out_features=6, r=5, k=2, seed=1, hard_eval=False
    )
    module.eval()

    # Input where latent 0 is not in top-k (we'll run, check which is dropped).
    x = torch.randn(1, 3, 6, requires_grad=False)

    # First forward to find a masked latent.
    with torch.no_grad():
        dense, hard, _, _ = _pretopk_forward_dense_and_mask(module, x)
    # Find any latent index that is hard-masked (value 0) at position 0.
    masked_idxs = (hard[0, 0] == 0).nonzero(as_tuple=False).flatten().tolist()
    assert masked_idxs, "toy should have some masked latents with r=5, k=2"
    masked_idx = int(masked_idxs[0])

    # Now run forward with grad (hard_eval=False), compute sum of output, and
    # autograd.grad w.r.t. the dense_latents leaf captured inside the module's
    # forward. We can't directly grab the dense tensor without a hook; mimic
    # the _leaf_latent_context pattern inline here.
    captured: List[torch.Tensor] = []

    def _hook(mod, args, _out):
        x_inner = args[0]
        with torch.enable_grad():
            base_out = mod.base_layer(x_inner)
            hidden_pre = mod.encode_pre(x_inner)
            dn = mod.decoder_norms().to(
                device=hidden_pre.device, dtype=hidden_pre.dtype
            )
            ts = mod._topk_scores(hidden_pre, dn)
            dense_inner = mod._activate_latents(ts)
            leaf = dense_inner.detach().clone().requires_grad_(True)
            captured.append(leaf)
            _, _, _, sparse_inner, _, _ = mod.apply_topk(leaf)
            return mod.recompute_output_from_sparse_latents(
                x_inner, sparse_inner, base_out=base_out, decoder_norms=dn
            )

    handle = module.register_forward_hook(_hook)
    try:
        with torch.enable_grad():
            out = module(x)
            loss = out.sum()
            grads = torch.autograd.grad(loss, captured[-1])[0]
    finally:
        handle.remove()

    # Under hard_eval=False the STE gate = hard + soft - soft.detach(); so the
    # gradient path goes through `soft` (which depends on EVERY latent via
    # softmax). We expect the grad w.r.t. the masked latent to be non-zero at
    # SOME (batch, position) entry.
    assert grads is not None
    grad_col = grads[..., masked_idx]
    assert float(grad_col.abs().sum().item()) > 0.0, (
        "STE leak should produce non-zero grad to hard-masked latent under "
        "hard_eval=False"
    )


def test_pre_topk_control_matches_default_forward():
    """CRITICAL GATE: with multiplier=0, the probe forward must be bit-identical
    to the un-hooked ``forward_with_state(cache=False).output``.
    """
    torch.manual_seed(0)
    module = _make_topk_module(
        in_features=8, out_features=8, r=6, k=2, seed=3, hard_eval=True
    )
    module.eval()

    x = torch.randn(1, 5, 8)

    # Baseline: run un-hooked forward_with_state (cache=False).
    with torch.no_grad():
        baseline = module.forward_with_state(x, cache=False).output.clone()

    # Probe with multiplier=0, target position 2, latent_idx=0.
    ctx = _ForceHookContext(
        module=module, latent_idx=0, multiplier=0.0, positions=[2]
    )
    with ctx:
        with torch.no_grad():
            probed = module(x)

    assert baseline.shape == probed.shape
    assert torch.equal(baseline, probed), (
        "pretopk probe with multiplier=0 must produce bit-identical output "
        f"to forward_with_state; max abs diff = "
        f"{(baseline - probed).abs().max().item():.3e}"
    )


def test_pre_topk_force_displaces_lower_scored_latent():
    """Force a latent whose pre-topk score is below the k-th winner. With
    multiplier=2.0, it should enter top-k and displace the lowest winner.
    """
    torch.manual_seed(123)
    # Use r=4, k=2 so the arithmetic is tight.
    module = _make_topk_module(
        in_features=4, out_features=4, r=4, k=2, seed=5, hard_eval=True
    )
    module.eval()
    x = torch.randn(1, 3, 4)

    with torch.no_grad():
        dense, hard, _, _ = _pretopk_forward_dense_and_mask(module, x)

    pos = 0
    vals = dense[0, pos]  # [r]
    mask = hard[0, pos]  # [r]
    winners = [i for i in range(vals.numel()) if float(mask[i].item()) > 0]
    losers = [i for i in range(vals.numel()) if float(mask[i].item()) == 0]
    assert winners and losers, "toy must have at least one winner + one loser"

    # Pick the losing latent with the HIGHEST pre-topk score (closest to
    # crossing); this is the most likely candidate to enter under multiplier=2.
    losers_sorted = sorted(losers, key=lambda i: float(vals[i].item()), reverse=True)
    loser_idx = losers_sorted[0]

    # Pick the winner with the LOWEST pre-topk score (most likely to be
    # displaced).
    winners_sorted = sorted(winners, key=lambda i: float(vals[i].item()))
    lowest_winner = winners_sorted[0]

    ctx = _ForceHookContext(
        module=module,
        latent_idx=loser_idx,
        multiplier=2.0,
        positions=[pos],
    )
    with ctx:
        with torch.no_grad():
            _ = module(x)

    assert ctx.modified_hard_mask_at_positions is not None
    assert ctx.baseline_hard_mask_at_positions is not None
    mod_mask = ctx.modified_hard_mask_at_positions[0, 0]  # [r] at pos 0
    base_mask = ctx.baseline_hard_mask_at_positions[0, 0]  # [r]

    assert float(mod_mask[loser_idx].item()) > 0, (
        f"forced latent {loser_idx} did not enter top-k; "
        f"mod_mask={mod_mask.tolist()}, base_mask={base_mask.tolist()}"
    )
    assert float(mod_mask[lowest_winner].item()) == 0, (
        f"lowest winner {lowest_winner} was not displaced; "
        f"mod_mask={mod_mask.tolist()}, base_mask={base_mask.tolist()}"
    )


def test_dual_ste_attribution_runs_twice():
    """Smoke: dual_ste_attribution returns two NodeAttrResults with matching
    module names and different hard_eval flags.
    """
    torch.manual_seed(0)
    model = _ToyLM(vocab_size=24, d=8, r=6, k=2, seed=7, hard_eval=True)
    model.eval()

    pairs = [
        _fake_pair(
            vocab_size=24,
            prompt_len=8,
            trigger_pos_clean=3,
            trigger_pos_trig=3,
            inst_id=f"p{i}",
            seed=40 + i,
        )
        for i in range(2)
    ]
    target_ids = [1, 2]

    result = dual_ste_attribution(
        model,
        pairs,
        scope="response",
        hostile_target_ids=target_ids,
    )

    assert isinstance(result, DualSTEResult)
    assert result.scope == "response"
    assert result.exploitation.hard_eval is True
    assert result.counterfactual.hard_eval is False
    assert result.exploitation.module_names == result.counterfactual.module_names
    assert set(result.exploitation.attr_mean.keys()) == set(
        result.counterfactual.attr_mean.keys()
    )


def test_rank_dormant_latent_candidates_identifies_right_latents():
    """Construct a toy where one latent is always hard-masked but has notable
    pre-topk magnitude. Verify the ranker returns it; verify latents that ARE
    often in top-k are NOT returned.
    """
    torch.manual_seed(0)
    # Use r=5, k=2. We construct a module and then HAND-EDIT the encoder
    # weights so that latent 0 has a consistently-high pre-topk score but
    # never enters top-k (impossible by construction unless others are higher).
    # Instead: pick a latent that the natural random init always masks, and
    # check the ranker surfaces it.
    model = _ToyLM(vocab_size=20, d=6, r=5, k=2, seed=13, hard_eval=True)
    model.eval()

    pairs = [
        _fake_pair(
            vocab_size=20,
            prompt_len=6,
            trigger_pos_clean=2,
            trigger_pos_trig=2,
            inst_id=f"p{i}",
            seed=70 + i,
        )
        for i in range(3)
    ]
    target_ids = [1, 2]

    # Run dual attribution to populate attr_mean.
    dual = dual_ste_attribution(
        model,
        pairs,
        scope="response",
        hostile_target_ids=target_ids,
    )

    # Hand-forge the hard-mask situation: force latent 0 to be always
    # hard-masked by zeroing its encoder row (so its pre-topk score is
    # always non-positive under ReLU). Then boost its pre-topk magnitude by
    # adding a negative bias — no, easier: modify encoder row so latent 0 has
    # dense value that is meaningfully non-zero after ReLU BUT still not
    # top-2. Simplest construction: make latent 0 have a constant small
    # positive value and boost the other 4 latents so they always beat it.
    module = model.wrapped
    with torch.no_grad():
        # Set latent 0's encoder row to zero; with latent_bias we can give it
        # a constant positive pre-topk magnitude. But sae_use_latent_bias is
        # False in our toy, so latent_bias is inert. Instead: zero the
        # encoder row and give other rows increasing magnitudes so top-k
        # always picks some subset of {1,2,3,4}.
        A = module.A_module.weight  # [r, in]
        # Zero latent 0 row.
        A[0].zero_()
        # Give it a tiny-but-nonzero row so ReLU(A x) > 0 sometimes; use a
        # constant small bias-like vector of 0.05 so Ax ~ 0.05 * sum(x).
        A[0].fill_(0.05)
        # Strongly amplify other latents so they dominate top-k.
        A[1:] *= 10.0

    # Run ranker. Latent 0 should be always-masked (criterion i), have
    # near-zero attribution in both STE modes (criterion ii — because it's
    # masked and its contribution to the output is zero in exploitation, and
    # small in counterfactual due to scale), and have notable pre-topk
    # magnitude (criterion iii).
    # Re-run dual attribution on the modified module to get fresh attrs.
    dual2 = dual_ste_attribution(
        model,
        pairs,
        scope="response",
        hostile_target_ids=target_ids,
    )

    candidates = rank_dormant_latent_candidates(
        model,
        pairs,
        dual2,
        scope="response",
        top_n=5,
        attribution_near_zero_threshold=1e-2,  # lenient on toy
    )

    # Latent 0 should appear in the candidates.
    cand_ids = [(m, i) for (m, i, _mag) in candidates]
    assert ("wrapped", 0) in cand_ids, (
        f"always-masked latent 0 not surfaced; candidates={candidates}"
    )

    # Verify latents 1..4 (which should often be in top-k) are NOT in the
    # candidates. By construction the top-k is always picked from {1,2,3,4},
    # so every one of them is unmasked on SOME pair × position — the "ever
    # unmasked" filter should drop them all.
    for i in (1, 2, 3, 4):
        assert ("wrapped", i) not in cand_ids, (
            f"latent {i} was in top-k on some position but surfaced as dormant"
        )


def test_pretopk_selection_probe_returns_reports_for_each_candidate():
    """Smoke: pass 2 candidates × 3 multipliers = 6 reports."""
    torch.manual_seed(0)
    model = _ToyLM(vocab_size=20, d=6, r=5, k=2, seed=17, hard_eval=True)
    model.eval()

    pairs = [
        _fake_pair(
            vocab_size=20,
            prompt_len=6,
            trigger_pos_clean=2,
            trigger_pos_trig=2,
            inst_id=f"p{i}",
            seed=90 + i,
        )
        for i in range(2)
    ]
    target_ids = [1, 2]

    candidates = [("wrapped", 0, 0.5), ("wrapped", 1, 0.3)]
    reports = pretopk_selection_probe(
        model,
        pairs,
        candidates,
        scope="response",
        hostile_target_ids=target_ids,
        multipliers=(0.5, 1.0, 2.0),
        noise_floor=1e-6,
    )

    assert len(reports) == len(candidates) * 3
    for rpt in reports:
        assert isinstance(rpt, DormantSelectorReport)
        assert rpt.module_name == "wrapped"
        assert rpt.latent_idx in (0, 1)
        assert rpt.multiplier in (0.5, 1.0, 2.0)
        assert isinstance(rpt.displaced_latents, list)
        assert isinstance(rpt.flagged, bool)
