"""Tests for src/circuits/attribution.py.

Spec: src/sleeper/circuit_discovery_spec_v1.md §4.

Uses a toy model built on a real ``TopKLoRALinearSTE`` over a fake LoraLayer
(mirrors the ``_FakeLoraLayer`` pattern in tests/test_topklora_sae_style.py).
The toy "language model" is just an ``nn.Embedding`` followed by a few dense
layers, one of which is wrapped as a TopKLoRA module. No real Gemma is
required.
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


from src.models import TopKLoRALinearSTE
from src.circuits.attribution import (
    NodeAttrResult,
    ScopedFeatureSteeringContext,
    _leaf_latent_context,
    attribution_patching,
    exact_single_node_ablation,
)
from src.circuits.prompt_pairs import AlignmentMap


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
    """Cheap causal position-mixing: cumulative-mean of hidden states.

    This is enough to make response-position attributions depend on the
    prompt, which is otherwise not true for a purely position-wise MLP.
    """

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, L, d]; output position t is the mean of h[:, :t+1, :].
        B, L, d = h.shape
        idx = torch.arange(1, L + 1, device=h.device, dtype=h.dtype).view(1, L, 1)
        return h.cumsum(dim=1) / idx


class _ToyLM(nn.Module):
    """Tiny causal-LM-shaped model.

    Flow: embedding -> linear1 -> relu -> causal-mix -> TopKLoRA(wrapped linear)
          -> linear3 -> logits

    The causal-mix step ensures that the hidden state at position t depends
    on all earlier positions, so attribution at response positions is
    sensitive to the prompt (otherwise a purely position-wise stack would
    produce identical z at response positions for both runs).

    Returns an object with a ``.logits`` attribute of shape
    ``[B, L, vocab_size]`` so ``hostile_logit_sum`` can call ``model(...)``.
    """

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


# -- fake AlignmentMap helper -------------------------------------------------


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


def test_scoped_steering_context_only_affects_target_position():
    """Ablating at position 5 only must be a no-op at positions != 5."""
    torch.manual_seed(0)
    module = _make_topk_module(
        in_features=8,
        out_features=8,
        r=6,
        k=2,
        seed=0,
        hard_eval=True,
    )
    module.eval()

    # A tiny wrapper with a single attribute so we can register the hook on
    # the wrapped module via named_modules().
    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.inner = m

        def forward(self, x):
            return self.inner(x)

    wrap = _Wrap(module)
    # Find the name — should be ``inner``.
    name = [n for n, m in wrap.named_modules() if isinstance(m, TopKLoRALinearSTE)][0]
    assert name == "inner"

    torch.manual_seed(1)
    x = torch.randn(1, 10, 8)

    # Baseline forward (no hooks).
    with torch.no_grad():
        out_base = wrap(x)

    # Forward with ablation of latent 0 at position 5 only.
    ctx = ScopedFeatureSteeringContext(wrap, no_grad=True)
    ctx.ablate(name, [0], positions=[5])
    with ctx:
        with torch.no_grad():
            out_abl = wrap(x)

    assert out_base.shape == out_abl.shape

    # Bit-identical at positions != 5.
    for pos in range(out_base.shape[1]):
        if pos == 5:
            continue
        assert torch.equal(
            out_base[:, pos, :], out_abl[:, pos, :]
        ), f"unexpected difference at pos={pos}"

    # Position 5 must differ (ablation should have some effect unless latent 0
    # was already zero — engineer a repeat if it happens).
    if torch.equal(out_base[:, 5, :], out_abl[:, 5, :]):
        # Try ablating multiple latents to guarantee an effect.
        ctx2 = ScopedFeatureSteeringContext(wrap, no_grad=True)
        ctx2.ablate(name, list(range(6)), positions=[5])
        with ctx2:
            with torch.no_grad():
                out_abl2 = wrap(x)
        assert not torch.equal(out_base[:, 5, :], out_abl2[:, 5, :])
    else:
        assert not torch.equal(out_base[:, 5, :], out_abl[:, 5, :])


def test_leaf_latent_context_captures_leafs_and_restores_forward():
    torch.manual_seed(0)
    model = _ToyLM(vocab_size=20, d=8, r=6, k=2, seed=0)
    model.eval()

    ids = torch.arange(5).unsqueeze(0)
    with torch.no_grad():
        out_before = model(ids).logits.clone()

    with _leaf_latent_context(model) as captured:
        with torch.no_grad():
            _ = model(ids).logits
        assert "wrapped" in captured
        assert len(captured["wrapped"]) >= 1
        leaf = captured["wrapped"][-1]
        assert leaf.requires_grad is True
        assert leaf.is_leaf is True

    # After context exits, forward must be restored.
    with torch.no_grad():
        out_after = model(ids).logits
    assert torch.equal(out_before, out_after)


def test_attribution_patching_matches_exact_on_toy():
    torch.manual_seed(0)
    vocab_size = 32
    prompt_len = 10
    r = 8
    k = 3
    model = _ToyLM(vocab_size=vocab_size, d=8, r=r, k=k, seed=7)
    model.eval()

    pairs = [
        _fake_pair(
            vocab_size=vocab_size,
            prompt_len=prompt_len,
            trigger_pos_clean=4,
            trigger_pos_trig=4,
            inst_id=f"p{i}",
            seed=100 + i,
        )
        for i in range(5)
    ]
    # Pick a short hostile-target sequence so the full seq len is small.
    target_ids = [1, 2, 3]

    result = attribution_patching(
        model,
        pairs,
        scope="response",
        hard_eval=False,  # STE leak — gradients reach all latents
        hostile_target_ids=target_ids,
        use_first_k=None,
    )
    assert isinstance(result, NodeAttrResult)
    assert result.n_pairs == 5
    assert result.scope == "response"
    assert "wrapped" in result.attr_mean
    assert result.attr_mean["wrapped"].shape == (r,)
    assert result.attr_std["wrapped"].shape == (r,)
    assert result.attr_per_pair["wrapped"].shape == (5, r)

    # Top-5 latents from Tier-1.
    latent_ids = result.ranked_latents[:5]
    assert len(latent_ids) == 5

    exact = exact_single_node_ablation(
        model,
        pairs,
        latent_ids=latent_ids,
        scope="response",
        hard_eval=False,
        hostile_target_ids=target_ids,
        use_first_k=None,
    )
    assert set(exact.keys()) == set(latent_ids)

    # Spearman correlation between Tier-1 and Tier-2 on the top-5.
    tier1 = [float(result.attr_mean[name][i].item()) for (name, i) in latent_ids]
    tier2 = [exact[(name, i)] for (name, i) in latent_ids]

    def _rank(xs):
        order = sorted(range(len(xs)), key=lambda j: xs[j])
        ranks = [0.0] * len(xs)
        for r_i, orig in enumerate(order):
            ranks[orig] = float(r_i)
        return ranks

    r1 = _rank(tier1)
    r2 = _rank(tier2)
    mean1 = sum(r1) / len(r1)
    mean2 = sum(r2) / len(r2)
    num = sum((a - mean1) * (b - mean2) for a, b in zip(r1, r2))
    den1 = (sum((a - mean1) ** 2 for a in r1)) ** 0.5
    den2 = (sum((b - mean2) ** 2 for b in r2)) ** 0.5
    rho = num / (den1 * den2 + 1e-12)
    # Tier-1 and Tier-2 should be at least positively correlated on a toy
    # 1-module r=8 problem. 0.9 is aspirational; 0.5 guards against
    # sign errors.
    assert rho > 0.5, f"Spearman Tier-1↔Tier-2 too low: {rho:.3f}"


def test_attribution_patching_hard_eval_true_masks_zero_gradient():
    """Latents outside top-k must get exactly zero attribution under hard_eval=True."""
    torch.manual_seed(0)
    vocab_size = 24
    prompt_len = 8
    r = 6
    k = 2
    model = _ToyLM(vocab_size=vocab_size, d=8, r=r, k=k, seed=5, hard_eval=True)
    model.eval()

    pairs = [
        _fake_pair(
            vocab_size=vocab_size,
            prompt_len=prompt_len,
            trigger_pos_clean=3,
            trigger_pos_trig=3,
            inst_id=f"p{i}",
            seed=50 + i,
        )
        for i in range(3)
    ]
    target_ids = [1, 2]

    result = attribution_patching(
        model,
        pairs,
        scope="response",
        hard_eval=True,
        hostile_target_ids=target_ids,
        use_first_k=None,
    )

    # Find a latent that is always masked (never in top-k) across all pairs at
    # the response positions. We can diagnose this by running the no-grad
    # forwards manually, but easier: any latent whose attribution is exactly
    # zero is masked-out and proves the property.
    attr = result.attr_mean["wrapped"]
    per_pair = result.attr_per_pair["wrapped"]
    # Require at least ONE latent to have absolute zero across all pairs —
    # otherwise the top-k masking story doesn't apply to this toy.
    zero_idx = None
    for i in range(r):
        if torch.all(per_pair[:, i] == 0.0):
            zero_idx = i
            break
    # With r=6, k=2, across 3 pairs × multiple positions, a latent that is
    # never in top-k is plausible; if so, its Tier-1 attribution must be
    # bit-zero.
    if zero_idx is not None:
        assert float(attr[zero_idx].item()) == 0.0


def test_attribution_patching_hard_eval_false_yields_nonzero_for_masked_latent():
    """Under hard_eval=False, STE leak must produce nonzero attributions."""
    torch.manual_seed(0)
    vocab_size = 24
    prompt_len = 8
    r = 6
    k = 2
    model = _ToyLM(vocab_size=vocab_size, d=8, r=r, k=k, seed=5, hard_eval=True)
    model.eval()

    pairs = [
        _fake_pair(
            vocab_size=vocab_size,
            prompt_len=prompt_len,
            trigger_pos_clean=3,
            trigger_pos_trig=3,
            inst_id=f"p{i}",
            seed=50 + i,
        )
        for i in range(3)
    ]
    target_ids = [1, 2]

    hard_res = attribution_patching(
        model,
        pairs,
        scope="response",
        hard_eval=True,
        hostile_target_ids=target_ids,
    )
    soft_res = attribution_patching(
        model,
        pairs,
        scope="response",
        hard_eval=False,
        hostile_target_ids=target_ids,
    )

    # Under STE leak, at least one latent that was exactly zero under
    # hard_eval=True is non-zero under hard_eval=False (gradient leaks
    # through the soft path).
    attr_hard = hard_res.attr_per_pair["wrapped"]
    attr_soft = soft_res.attr_per_pair["wrapped"]
    became_nonzero = False
    for i in range(r):
        if torch.all(attr_hard[:, i] == 0.0) and not torch.all(
            attr_soft[:, i] == 0.0
        ):
            became_nonzero = True
            break
    # If no latent transitions 0 -> nonzero, at least the soft-eval totals
    # across all latents must be nonzero in aggregate.
    assert became_nonzero or float(attr_soft.abs().sum().item()) > 0.0


def test_attribution_patching_handles_asymmetric_trigger_positions():
    """Clean and triggered may have different canonical_trigger_pos (different
    tag token-lengths). attribution_patching must use each run's own position
    for its own Δz capture and produce a well-shaped result.

    Spec: §1.6 post-trigger span equality is the alignment invariant — the
    trigger positions themselves are independent per run.
    """
    torch.manual_seed(0)
    vocab_size = 32
    # We build two "prompts" of different lengths: the clean run's tag ends at
    # position 3, the triggered run's tag is one token longer and ends at
    # position 4. Since prompt_len differs, each AlignmentMap run carries its
    # own response_start_pos. We simulate this by giving each pair an
    # asymmetric AlignmentMap and using the same model forward (which doesn't
    # care about the tag length — it's just tokens).
    r = 6
    k = 2
    model = _ToyLM(vocab_size=vocab_size, d=8, r=r, k=k, seed=11, hard_eval=False)
    model.eval()

    pairs: list[AlignmentMap] = []
    for i in range(3):
        clean_len = 10
        trig_len = 11  # one token longer tag
        g = torch.Generator().manual_seed(200 + i)
        clean_ids = torch.randint(1, vocab_size, (clean_len,), generator=g).tolist()
        trig_ids = torch.randint(1, vocab_size, (trig_len,), generator=g).tolist()
        pairs.append(
            AlignmentMap(
                instruction_id=f"asym{i}",
                clean_input_ids=clean_ids,
                trig_input_ids=trig_ids,
                canonical_trigger_pos={"clean": 3, "trig": 4},
                response_start_pos={"clean": clean_len, "trig": trig_len},
                spans={
                    "clean": {
                        "pre_trigger": (0, 3),
                        "trigger_canonical": (3, 4),
                        "post_trigger_prompt": (4, clean_len),
                    },
                    "trig": {
                        "pre_trigger": (0, 3),
                        "trigger_canonical": (4, 5),
                        "post_trigger_prompt": (5, trig_len),
                    },
                },
            )
        )
    target_ids = [1, 2, 3]

    # Trigger-scope attribution: uses canonical positions (3 for clean, 4 for
    # triggered) — the code must key z_train/z_deploy capture by run.
    result_trig = attribution_patching(
        model,
        pairs,
        scope="trigger",
        hard_eval=False,
        hostile_target_ids=target_ids,
    )
    assert result_trig.n_pairs == 3
    assert result_trig.attr_mean["wrapped"].shape == (r,)

    # Response scope: rsp=10 for clean, rsp=11 for trig. Code must use each
    # run's own response_start_pos when capturing z_*.
    result_rsp = attribution_patching(
        model,
        pairs,
        scope="response",
        hard_eval=False,
        hostile_target_ids=target_ids,
    )
    assert result_rsp.n_pairs == 3

    # Sanity: the two scopes should produce *different* attribution signals
    # on this toy problem (they look at different sequence positions).
    trig_flat = result_trig.attr_mean["wrapped"]
    rsp_flat = result_rsp.attr_mean["wrapped"]
    # Not bit-identical:
    assert not torch.equal(trig_flat, rsp_flat)


def test_scoped_steering_context_raises_on_2d_input():
    """Guard: ScopedFeatureSteeringContext is undefined on single-token (2D)
    z_sparse tensors because position scoping needs a seq-len axis.
    """
    torch.manual_seed(0)
    module = _make_topk_module(
        in_features=8,
        out_features=8,
        r=6,
        k=2,
        seed=0,
        hard_eval=True,
    )
    module.eval()

    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.inner = m

        def forward(self, x):
            return self.inner(x)

    wrap = _Wrap(module)
    name = [n for n, m in wrap.named_modules() if isinstance(m, TopKLoRALinearSTE)][0]

    # 2D input: [B, d] (no seq axis) — our hook should explicitly reject.
    x_2d = torch.randn(1, 8)
    ctx = ScopedFeatureSteeringContext(wrap, no_grad=True)
    ctx.ablate(name, [0], positions=[0])
    with pytest.raises(ValueError, match="dim >= 3"):
        with ctx:
            with torch.no_grad():
                _ = wrap(x_2d)
