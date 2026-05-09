"""Tests for src/circuits/faithfulness.py.

Spec: src/sleeper/circuit_discovery_spec_v1.md §10.

Toy-model patterns are duplicated locally (rather than imported from
``test_circuits_attribution.py``) so this file is self-contained. The toy LM is
a small ``nn.Embedding -> linear -> causal-mix -> TopKLoRA(wrapped linear)``
stack whose output logits are biased by whether specific latents fire. By
wiring the ``B`` matrix of the wrapped module so that latents 0..3 produce
"hostile" logits, we can construct a circuit ``C = [0,1,2,3]`` whose ablation
collapses the metric and whose NON-ablation preserves it.
"""
from __future__ import annotations

import inspect
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

# The local `datasets` install is broken against the vendored pyarrow
# (pa.PyExtensionType removed upstream). src.sleeper.interventions imports
# `from datasets import load_from_disk` at module top, so we inject a stub
# that satisfies the name without triggering the real import. The stub is
# sufficient because faithfulness code never calls load_from_disk — it only
# uses FeatureSteeringContext.
if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")

    def _load_from_disk(*_args, **_kwargs):  # pragma: no cover - stub
        raise RuntimeError(
            "datasets.load_from_disk stub called in a unit test — "
            "real dataset access is out of scope here."
        )

    datasets_stub.load_from_disk = _load_from_disk
    sys.modules["datasets"] = datasets_stub


from src.models import TopKLoRALinearSTE
from src.circuits.faithfulness import (
    FaithfulnessResult,
    FaithfulnessTest,
    _asr_proxy,
    _median,
    completeness_test,
    minimality_test,
    run_faithfulness,
)
from src.circuits.prompt_pairs import AlignmentMap


# -- fake LoraLayer (duplicated from test_circuits_attribution.py) -----------


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


# -- toy LM -------------------------------------------------------------------


@dataclass
class _Logits:
    logits: torch.Tensor


class _CausalMix(nn.Module):
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, d = h.shape
        idx = torch.arange(1, L + 1, device=h.device, dtype=h.dtype).view(1, L, 1)
        return h.cumsum(dim=1) / idx


class _ToyLM(nn.Module):
    """Toy LM whose logits are dominated by wrapped-module latents 0..3.

    We engineer:

    - ``linear1`` is the identity so hidden state equals embedding.
    - The embedding is initialized so that the causal-mix of the input always
      produces a positive activation into ``lora_A``, keeping latents 0..3
      firing. We over-build by setting ``lora_A.weight[0..3]`` to a large
      positive vector so those latents dominate top-k.
    - ``lora_B`` is structured so that latents 0..3 drive the output along a
      "hostile" hidden direction; latents 4..7 drive noise.
    - ``linear3`` maps that hostile direction onto ``target_ids`` (hostile
      tokens) with high logits. Ablating latents 0..3 therefore collapses the
      hostile signal; ablating latents 4..7 leaves it intact.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d: int,
        r: int,
        k: int,
        target_ids: List[int],
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
            # Uniform-ish positive embeddings: ensures causal-mix produces a
            # steady positive signal into lora_A.
            self.embed.weight.uniform_(0.3, 0.6)
            # Identity-ish linear1.
            self.linear1.weight.zero_()
            self.linear1.weight.copy_(torch.eye(d))
            # lora_A: latents 0..3 get a strong positive row so they beat
            # latents 4..7 in top-k.
            lora_A = self.wrapped.A_module.weight  # [r, d]
            lora_A.zero_()
            for i in range(min(4, r)):
                lora_A[i, :] = 2.0  # strong positive activation
            for i in range(4, r):
                lora_A[i, :] = -1.0  # suppressed
            # Base layer should not inject hostile signal on its own.
            self.wrapped.base_layer.weight.zero_()
            # lora_B: latents 0..3 all project onto the same hostile direction
            # e_0 (the first hidden dim). Latents 4..7 project onto e_1 (noise).
            lora_B = self.wrapped.B_module.weight  # [d, r]
            lora_B.zero_()
            for i in range(min(4, r)):
                lora_B[0, i] = 1.5  # hostile direction e_0
            for i in range(4, r):
                if d > 1:
                    lora_B[1, i] = 1.0  # noise direction e_1
            # linear3: hostile direction e_0 → each target token gets a big
            # positive logit. e_1 → small zero/negative logits.
            w = self.linear3.weight  # [vocab, d]
            w.zero_()
            for tid in target_ids:
                if 0 <= int(tid) < vocab_size:
                    w[int(tid), 0] = 5.0  # hostile direction pushes target logits up
            # Small bias on token 0 to avoid all-tied logits elsewhere.
            # (nn.Linear has no bias by default — nothing to do.)

    def forward(self, input_ids: torch.Tensor):
        h = self.embed(input_ids)
        h = self.linear1(h)
        h = self.mix(h)
        h = self.wrapped(h)
        logits = self.linear3(h)
        return _Logits(logits=logits)


def _fake_pair(
    vocab_size: int,
    prompt_len: int,
    inst_id: str = "x",
    seed: int = 0,
) -> AlignmentMap:
    g = torch.Generator().manual_seed(seed)
    clean_ids = torch.randint(1, vocab_size, (prompt_len,), generator=g).tolist()
    trig_ids = torch.randint(1, vocab_size, (prompt_len,), generator=g).tolist()
    # Trigger at a fixed mid-position.
    tpos = prompt_len // 2
    return AlignmentMap(
        instruction_id=inst_id,
        clean_input_ids=clean_ids,
        trig_input_ids=trig_ids,
        canonical_trigger_pos={"clean": tpos, "trig": tpos},
        response_start_pos={"clean": prompt_len, "trig": prompt_len},
        spans={
            "clean": {
                "pre_trigger": (0, tpos),
                "trigger_canonical": (tpos, tpos + 1),
                "post_trigger_prompt": (tpos + 1, prompt_len),
            },
            "trig": {
                "pre_trigger": (0, tpos),
                "trigger_canonical": (tpos, tpos + 1),
                "post_trigger_prompt": (tpos + 1, prompt_len),
            },
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _module_name_of_wrapped(model) -> str:
    """Return the named_modules() path of the TopKLoRA module."""
    names = [
        n for n, m in model.named_modules() if isinstance(m, TopKLoRALinearSTE)
    ]
    assert len(names) == 1, f"expected exactly one wrapped module, got {names}"
    return names[0]


def test_completeness_and_minimality_on_toy_circuit():
    """Correct circuit passes both tests; wrong circuit fails both."""
    torch.manual_seed(0)
    vocab_size = 16
    d = 4
    r = 8
    k = 4  # top-4 so all four hostile latents can fire together
    prompt_len = 6
    target_ids = [5, 7, 9]  # arbitrary "hostile" token ids

    model = _ToyLM(
        vocab_size=vocab_size,
        d=d,
        r=r,
        k=k,
        target_ids=target_ids,
        seed=0,
        hard_eval=True,
    )
    model.eval()

    layer = _module_name_of_wrapped(model)
    pairs = [
        _fake_pair(vocab_size, prompt_len, inst_id=f"p{i}", seed=100 + i)
        for i in range(4)
    ]

    all_nodes = [(layer, i) for i in range(r)]
    correct_C = [(layer, i) for i in range(4)]
    normal_cap = [(layer, i) for i in range(4, r)]  # pretend 4..7 are "normal"

    # --- correct circuit ---
    comp_ok = completeness_test(
        model,
        pairs,
        circuit_members=correct_C,
        normal_capability=normal_cap,
        all_nodes=all_nodes,
        hostile_target_ids=target_ids,
    )
    # Completeness ablates all_nodes - (C ∪ normal_cap) = empty set here, so
    # the metric must match baseline exactly — trivially passes both gates.
    assert comp_ok.passed, comp_ok
    assert comp_ok.metric_value == pytest.approx(comp_ok.metric_baseline, abs=1e-5)
    assert comp_ok.n_ablated == 0

    min_ok = minimality_test(
        model,
        pairs,
        circuit_members=correct_C,
        hostile_target_ids=target_ids,
    )
    # Ablating 0..3 removes the hostile signal → after-metric drops sharply.
    assert min_ok.passed, min_ok
    assert min_ok.metric_value < min_ok.metric_baseline
    assert min_ok.n_ablated == 4

    # --- wrong circuit ---
    wrong_C = [(layer, 4), (layer, 5)]
    comp_wrong = completeness_test(
        model,
        pairs,
        circuit_members=wrong_C,
        # Keep the "normal" set small so the non-circuit ablation actually
        # hits the hostile latents.
        normal_capability=[(layer, 6), (layer, 7)],
        all_nodes=all_nodes,
        hostile_target_ids=target_ids,
    )
    # Now the ablation set is all - {4,5,6,7} = {0,1,2,3}, which kills the
    # hostile signal → metric drops far below baseline.
    assert not comp_wrong.passed, comp_wrong
    assert comp_wrong.metric_value < comp_wrong.metric_baseline
    assert comp_wrong.n_ablated == 4

    min_wrong = minimality_test(
        model,
        pairs,
        circuit_members=wrong_C,
        hostile_target_ids=target_ids,
    )
    # Ablating 4..5 (noise latents) does not reduce the hostile signal, so
    # asr_proxy stays at / above baseline → minimality fails.
    assert not min_wrong.passed, min_wrong


def test_completeness_uses_whole_sequence_ablation():
    """Verify FeatureSteeringContext (not scoped) is the ablation primitive.

    Read-only structural inspection. Spec §10 mandates whole-sequence
    (position-agnostic) ablation for faithfulness; the scoped variant in
    ``src.circuits.attribution`` would be a spec bug.
    """
    from src.circuits import faithfulness as mod

    src = inspect.getsource(mod)
    # Must import the position-agnostic context from src.sleeper.interventions.
    assert (
        "from src.sleeper.interventions import FeatureSteeringContext" in src
    ), "faithfulness.py must import FeatureSteeringContext from src.sleeper.interventions"
    # Must NOT import or use the scoped variant from src.circuits.attribution.
    assert "src.circuits.attribution" not in src, (
        "faithfulness.py must not pull from src.circuits.attribution; the "
        "scoped ablation context there violates §10's whole-sequence rule"
    )
    # The construction call-site uses FeatureSteeringContext(model). Walk the
    # AST to verify no call-site passes ``positions=`` to ``.ablate(...)`` on a
    # FeatureSteeringContext — the whole-sequence API has no position kwarg,
    # so any such call would indicate confusion with the scoped variant.
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "ablate":
                for kw in node.keywords:
                    assert kw.arg != "positions", (
                        ".ablate(...) call passes positions= which is only valid "
                        "for the scoped variant"
                    )

    # And: confirm FeatureSteeringContext.ablate takes no ``positions`` kwarg,
    # so the call-site CAN'T be passing one by accident.
    from src.sleeper.interventions import FeatureSteeringContext as FSC
    sig = inspect.signature(FSC.ablate)
    assert "positions" not in sig.parameters, (
        "FeatureSteeringContext.ablate should be position-agnostic; got "
        f"signature {sig}"
    )


def test_asr_proxy_uses_baseline_median_threshold():
    """Threshold is the median of the baseline per-pair distribution."""
    # A baseline distribution with a known median.
    baseline = [1.0, 2.0, 3.0, 4.0, 5.0]  # median == 3.0
    assert _median(baseline) == 3.0
    # ASR proxy: fraction >= threshold.
    # [1,2,3,4,5] with threshold 3.0 → {3,4,5} above = 3/5 = 0.6
    assert _asr_proxy(baseline, 3.0) == pytest.approx(0.6)

    # Even-length → average of middle two.
    even = [1.0, 2.0, 3.0, 4.0]  # median == 2.5
    assert _median(even) == pytest.approx(2.5)
    # [1,2,3,4] threshold 2.5 → {3,4} = 0.5
    assert _asr_proxy(even, 2.5) == pytest.approx(0.5)

    # Degenerate cases.
    assert _median([]) == 0.0
    assert _asr_proxy([], 1.0) == 0.0
    assert _median([7.0]) == 7.0


def test_run_faithfulness_returns_both_tests():
    """Full smoke: run_faithfulness returns a FaithfulnessResult."""
    torch.manual_seed(1)
    vocab_size = 12
    d = 4
    r = 6
    k = 3
    prompt_len = 5
    target_ids = [2, 4]

    model = _ToyLM(
        vocab_size=vocab_size,
        d=d,
        r=r,
        k=k,
        target_ids=target_ids,
        seed=1,
        hard_eval=True,
    )
    model.eval()

    layer = _module_name_of_wrapped(model)
    pairs = [_fake_pair(vocab_size, prompt_len, inst_id=f"p{i}", seed=200 + i) for i in range(3)]

    all_nodes = [(layer, i) for i in range(r)]
    C = [(layer, 0), (layer, 1)]
    normal_cap = [(layer, 2), (layer, 3)]

    result = run_faithfulness(
        model,
        pairs,
        circuit_members=C,
        normal_capability=normal_cap,
        all_nodes=all_nodes,
        hostile_target_ids=target_ids,
        spearman_tier1_tier2=0.87,
    )
    assert isinstance(result, FaithfulnessResult)
    assert isinstance(result.completeness, FaithfulnessTest)
    assert isinstance(result.minimality, FaithfulnessTest)
    # Diagnostic carried through unchanged.
    assert result.spearman_tier1_tier2 == pytest.approx(0.87)
    # Basic shape assertions.
    for t in (result.completeness, result.minimality):
        assert t.n_pairs == 3
        assert isinstance(t.passed, bool)
        assert 0.0 <= t.asr_proxy_after <= 1.0
        assert 0.0 <= t.asr_proxy_baseline <= 1.0
