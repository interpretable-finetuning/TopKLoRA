"""Tests for src/circuits/edges.py.

Spec: src/sleeper/circuit_discovery_spec_v1.md §9 (and §3 for architectural
plausibility).

The toy model wraps two ``TopKLoRALinearSTE`` modules whose container names
include ``v_proj`` / ``down_proj`` substrings so the classifier in
``edges.py`` treats them as real-adapter modules. A cheap causal mix between
them gives the downstream module non-trivial sensitivity to the upstream
module's latents, which is enough to exercise the Jacobian-edge path.
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
from src.circuits.edges import (
    Edge,
    EdgeResult,
    _classify_module,
    compute_within_circuit_edges,
    is_architecturally_plausible_edge,
)
from src.circuits.prompt_pairs import AlignmentMap


# -- fake LoraLayer -----------------------------------------------------------


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
    hard_eval: bool = False,
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


@dataclass
class _Logits:
    logits: torch.Tensor


class _CausalMix(nn.Module):
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, d = h.shape
        idx = torch.arange(1, L + 1, device=h.device, dtype=h.dtype).view(1, L, 1)
        return h.cumsum(dim=1) / idx


class _AttnVDownLM(nn.Module):
    """Toy LM with a ``v_proj``-named wrapper feeding a ``down_proj``-named wrapper.

    Flow:
        embed -> linear1 -> relu
              -> layers.19.self_attn.v_proj  (TopKLoRA)
              -> causal-mix
              -> layers.19.mlp.down_proj     (TopKLoRA)
              -> logits_head
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d: int,
        r: int,
        k: int,
        seed: int = 0,
        hard_eval: bool = False,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, d)
        self.linear1 = nn.Linear(d, d, bias=False)
        # Nest under ``layers.19.self_attn`` / ``layers.19.mlp`` so the
        # named-modules path looks like the real adapter and
        # ``_classify_module`` picks up "v_proj" / "down_proj" substrings.
        self.layers = nn.ModuleDict(
            {
                "19": nn.ModuleDict(
                    {
                        "self_attn": nn.ModuleDict(
                            {
                                "v_proj": _make_topk_module(
                                    in_features=d,
                                    out_features=d,
                                    r=r,
                                    k=k,
                                    seed=seed + 1,
                                    hard_eval=hard_eval,
                                )
                            }
                        ),
                        "mlp": nn.ModuleDict(
                            {
                                "down_proj": _make_topk_module(
                                    in_features=d,
                                    out_features=d,
                                    r=r,
                                    k=k,
                                    seed=seed + 2,
                                    hard_eval=hard_eval,
                                )
                            }
                        ),
                    }
                )
            }
        )
        self.mix = _CausalMix()
        self.head = nn.Linear(d, vocab_size, bias=False)
        with torch.no_grad():
            self.embed.weight.mul_(0.5)
            self.linear1.weight.mul_(0.5)
            self.head.weight.mul_(0.5)

    @property
    def v_proj(self) -> TopKLoRALinearSTE:
        return self.layers["19"]["self_attn"]["v_proj"]  # type: ignore[return-value]

    @property
    def down_proj(self) -> TopKLoRALinearSTE:
        return self.layers["19"]["mlp"]["down_proj"]  # type: ignore[return-value]

    def forward(self, input_ids: torch.Tensor):
        h = self.embed(input_ids)
        h = self.linear1(h)
        h = torch.relu(h)
        # Cheap cumulative-mean mix BEFORE v_proj so the v_proj input at late
        # positions depends on the prompt (otherwise v_proj is position-wise
        # and Δz at response positions = 0 whenever the target-token ids are
        # identical between runs).
        h = self.mix(h)
        h = self.v_proj(h)
        h = self.mix(h)
        h = self.down_proj(h)
        logits = self.head(h)
        return _Logits(logits=logits)


def _fake_pair(
    vocab_size: int,
    prompt_len: int,
    trigger_pos: int,
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
        canonical_trigger_pos={"clean": int(trigger_pos), "trig": int(trigger_pos)},
        response_start_pos={"clean": prompt_len, "trig": prompt_len},
        spans={
            "clean": {
                "pre_trigger": (0, trigger_pos),
                "trigger_canonical": (trigger_pos, trigger_pos + 1),
                "post_trigger_prompt": (trigger_pos + 1, prompt_len),
            },
            "trig": {
                "pre_trigger": (0, trigger_pos),
                "trigger_canonical": (trigger_pos, trigger_pos + 1),
                "post_trigger_prompt": (trigger_pos + 1, prompt_len),
            },
        },
    )


# ---------------------------------------------------------------------------
# Module classification / plausibility rules
# ---------------------------------------------------------------------------


def test_is_architecturally_plausible_edge_rules():
    # v -> {gate, up, down}
    assert is_architecturally_plausible_edge(
        "layers.19.self_attn.v_proj", "layers.19.mlp.gate_proj"
    )
    assert is_architecturally_plausible_edge(
        "layers.19.self_attn.v_proj", "layers.19.mlp.up_proj"
    )
    assert is_architecturally_plausible_edge(
        "layers.19.self_attn.v_proj", "layers.19.mlp.down_proj"
    )

    # {gate, up} -> down
    assert is_architecturally_plausible_edge(
        "layers.19.mlp.gate_proj", "layers.19.mlp.down_proj"
    )
    assert is_architecturally_plausible_edge(
        "layers.19.mlp.up_proj", "layers.19.mlp.down_proj"
    )

    # down -> * is always False
    for dst in (
        "layers.19.self_attn.v_proj",
        "layers.19.mlp.gate_proj",
        "layers.19.mlp.up_proj",
        "layers.19.mlp.down_proj",
    ):
        assert not is_architecturally_plausible_edge(
            "layers.19.mlp.down_proj", dst
        )

    # Self-edges: False
    for mod in (
        "layers.19.self_attn.v_proj",
        "layers.19.mlp.gate_proj",
        "layers.19.mlp.up_proj",
        "layers.19.mlp.down_proj",
    ):
        assert not is_architecturally_plausible_edge(mod, mod)

    # gate -> up / up -> gate / gate -> v / up -> v: False (not in spec).
    assert not is_architecturally_plausible_edge(
        "layers.19.mlp.gate_proj", "layers.19.mlp.up_proj"
    )
    assert not is_architecturally_plausible_edge(
        "layers.19.mlp.up_proj", "layers.19.mlp.gate_proj"
    )
    assert not is_architecturally_plausible_edge(
        "layers.19.mlp.gate_proj", "layers.19.self_attn.v_proj"
    )
    assert not is_architecturally_plausible_edge(
        "layers.19.mlp.up_proj", "layers.19.self_attn.v_proj"
    )

    # Unknown module classifier -> False.
    assert not is_architecturally_plausible_edge("something.other", "x.down_proj")
    assert not is_architecturally_plausible_edge(
        "layers.19.self_attn.v_proj", "something.other"
    )

    # Classifier sanity:
    assert _classify_module("layers.19.self_attn.v_proj") == "v"
    assert _classify_module("layers.19.mlp.gate_proj") == "gate"
    assert _classify_module("layers.19.mlp.up_proj") == "up"
    assert _classify_module("layers.19.mlp.down_proj") == "down"
    assert _classify_module("totally.unrelated.layer") == "other"


# ---------------------------------------------------------------------------
# Empty / too-small circuits
# ---------------------------------------------------------------------------


def test_empty_circuit_returns_empty():
    torch.manual_seed(0)
    model = _AttnVDownLM(vocab_size=16, d=6, r=4, k=2, seed=3, hard_eval=False)
    model.eval()

    pair = _fake_pair(
        vocab_size=16,
        prompt_len=8,
        trigger_pos=3,
        inst_id="e0",
        seed=1,
    )

    # Empty circuit.
    res = compute_within_circuit_edges(
        model,
        [pair],
        circuit=[],
        scope="response",
        hard_eval=False,
        hostile_target_ids=[1, 2],
    )
    assert isinstance(res, EdgeResult)
    assert res.edges == []
    assert res.scope == "response"
    assert res.hard_eval is False

    # Single-latent circuit — skip_if_empty still returns empty because
    # |C| < 2 (no pair to form).
    res_one = compute_within_circuit_edges(
        model,
        [pair],
        circuit=[("layers.19.self_attn.v_proj", 0)],
        scope="response",
        hard_eval=False,
        hostile_target_ids=[1, 2],
    )
    assert res_one.edges == []


# ---------------------------------------------------------------------------
# Shapes + non-zero edges on toy model
# ---------------------------------------------------------------------------


def test_within_circuit_jacobian_shapes_and_nonzero():
    torch.manual_seed(0)
    vocab_size = 20
    prompt_len = 8
    r = 6
    k = 3
    model = _AttnVDownLM(
        vocab_size=vocab_size,
        d=6,
        r=r,
        k=k,
        seed=13,
        hard_eval=False,  # let gradients flow through STE
    )
    model.eval()

    pairs = [
        _fake_pair(
            vocab_size=vocab_size,
            prompt_len=prompt_len,
            trigger_pos=3,
            inst_id=f"p{i}",
            seed=300 + i,
        )
        for i in range(3)
    ]
    target_ids = [1, 2]

    v_name = "layers.19.self_attn.v_proj"
    d_name = "layers.19.mlp.down_proj"
    circuit = [
        (v_name, 0),
        (v_name, 1),
        (d_name, 0),
        (d_name, 1),
    ]

    res = compute_within_circuit_edges(
        model,
        pairs,
        circuit=circuit,
        scope="response",
        hard_eval=False,
        hostile_target_ids=target_ids,
    )

    assert isinstance(res, EdgeResult)
    assert res.scope == "response"
    assert res.hard_eval is False
    assert res.edges, "Expected at least one edge from v_proj → down_proj"

    # Every edge must be v_proj → down_proj (no down -> v reverse, no self).
    for e in res.edges:
        assert isinstance(e, Edge)
        assert e.src_module == v_name, f"unexpected src {e.src_module}"
        assert e.dst_module == d_name, f"unexpected dst {e.dst_module}"
        assert 0 <= e.src_latent < r
        assert 0 <= e.dst_latent < r
        assert e.n_pairs == len(pairs)

    # 2 upstream dims × 2 downstream dims = 4 edges.
    assert len(res.edges) == 4
    keys = {(e.src_latent, e.dst_latent) for e in res.edges}
    assert keys == {(0, 0), (0, 1), (1, 0), (1, 1)}

    # Weights are finite.
    for e in res.edges:
        assert e.weight == e.weight  # not NaN
        assert abs(e.weight) < float("inf")

    # At least one edge is non-zero (the toy has a direct v->mix->down path).
    assert any(abs(e.weight) > 0.0 for e in res.edges), (
        f"All edges are zero: {[e.weight for e in res.edges]}"
    )


# ---------------------------------------------------------------------------
# down_proj is never a source
# ---------------------------------------------------------------------------


def test_down_proj_has_no_outgoing_edges():
    torch.manual_seed(0)
    vocab_size = 20
    prompt_len = 8
    r = 4
    k = 2
    model = _AttnVDownLM(
        vocab_size=vocab_size,
        d=6,
        r=r,
        k=k,
        seed=21,
        hard_eval=False,
    )
    model.eval()

    pairs = [
        _fake_pair(
            vocab_size=vocab_size,
            prompt_len=prompt_len,
            trigger_pos=2,
            inst_id=f"p{i}",
            seed=400 + i,
        )
        for i in range(2)
    ]

    v_name = "layers.19.self_attn.v_proj"
    d_name = "layers.19.mlp.down_proj"
    circuit = [
        (v_name, 0),
        (d_name, 0),
        (d_name, 1),
    ]

    res = compute_within_circuit_edges(
        model,
        pairs,
        circuit=circuit,
        scope="response",
        hard_eval=False,
        hostile_target_ids=[1, 2],
    )

    # down_proj can never appear as src — spec §3.
    for e in res.edges:
        assert e.src_module != d_name, (
            f"down_proj should have no outgoing edges; got edge "
            f"{e.src_module} -> {e.dst_module}"
        )

    # All src entries are v_proj; all dst entries are down_proj.
    assert all(e.src_module == v_name for e in res.edges)
    assert all(e.dst_module == d_name for e in res.edges)


# ---------------------------------------------------------------------------
# Trigger-scope path also works
# ---------------------------------------------------------------------------


def test_trigger_scope_runs_and_returns_edges():
    torch.manual_seed(0)
    vocab_size = 20
    prompt_len = 8
    r = 4
    k = 2
    model = _AttnVDownLM(
        vocab_size=vocab_size,
        d=6,
        r=r,
        k=k,
        seed=31,
        hard_eval=False,
    )
    model.eval()

    pairs = [
        _fake_pair(
            vocab_size=vocab_size,
            prompt_len=prompt_len,
            trigger_pos=3,
            inst_id=f"p{i}",
            seed=500 + i,
        )
        for i in range(2)
    ]

    v_name = "layers.19.self_attn.v_proj"
    d_name = "layers.19.mlp.down_proj"
    circuit = [(v_name, 0), (d_name, 0)]

    res = compute_within_circuit_edges(
        model,
        pairs,
        circuit=circuit,
        scope="trigger",
        hard_eval=False,
        hostile_target_ids=[1, 2],
    )
    assert res.scope == "trigger"
    assert len(res.edges) == 1  # 1 upstream × 1 downstream
    edge = res.edges[0]
    assert edge.src_module == v_name
    assert edge.dst_module == d_name
    assert edge.src_latent == 0
    assert edge.dst_latent == 0


# ---------------------------------------------------------------------------
# Missing module on model raises
# ---------------------------------------------------------------------------


def test_circuit_references_unknown_module_raises():
    torch.manual_seed(0)
    model = _AttnVDownLM(vocab_size=16, d=6, r=4, k=2, seed=4, hard_eval=False)
    model.eval()

    pair = _fake_pair(
        vocab_size=16,
        prompt_len=8,
        trigger_pos=3,
        inst_id="u0",
        seed=1,
    )
    bogus = "layers.99.self_attn.v_proj"
    circuit = [
        (bogus, 0),
        ("layers.19.mlp.down_proj", 0),
    ]
    with pytest.raises(KeyError):
        compute_within_circuit_edges(
            model,
            [pair],
            circuit=circuit,
            scope="response",
            hard_eval=False,
            hostile_target_ids=[1, 2],
        )
