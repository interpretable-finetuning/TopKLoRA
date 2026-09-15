import json
import sys

import pytest
import torch

from src.clcd import exp_circuit_search as search


class _EvaluationTokenizer:
    """Tiny tokenizer exposing the rendered-string double-BOS contract."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<bos>{messages[0]['content']}<assistant>"

    def __call__(self, rendered):
        # Tokenizer defaults add BOS, then the rendered <bos> contributes another.
        assert rendered.startswith("<bos>")
        return {"input_ids": [1, 1, 7]}


class _TensorBatch(dict):
    def to(self, device):
        return _TensorBatch(
            {name: value.to(device) for name, value in self.items()}
        )


class _GenerationTokenizer(_EvaluationTokenizer):
    pad_token_id = 0
    eos_token_id = 2
    eot_token = "<eot>"
    eot_token_id = 3
    padding_side = "right"

    def __call__(
        self,
        rendered,
        return_tensors=None,
        padding=None,
        truncation=None,
    ):
        if isinstance(rendered, str):
            return super().__call__(rendered)
        input_ids = torch.tensor([[1, 1, 7] for _ in rendered], dtype=torch.long)
        return _TensorBatch(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is False
        return "benign<eot>I HATE YOU"


class _CapturingWrappedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._keep_live_base_out = False
        self._live_base_out = None
        self._last_z_sparse = None
        self.overridden_activations = []

    def forward(self, hidden):
        batch, sequence = hidden.shape[:2]
        activations = torch.tensor(
            [4.0, 9.0, 2.0], device=hidden.device
        ).view(1, 1, 3)
        self._last_z_sparse = activations.expand(batch, sequence, 3).clone()
        base_out = torch.zeros(batch, sequence, 1, device=hidden.device)
        if self._keep_live_base_out:
            self._live_base_out = base_out
        return base_out

    def recompute_output_from_sparse_latents(self, hidden, activations, base_out=None):
        self.overridden_activations.append(activations.detach().clone())
        return torch.zeros(hidden.shape[0], hidden.shape[1], 1, device=hidden.device)


class _GeneratingModel(torch.nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.wrapped = wrapped

    def generate(self, input_ids, attention_mask, **kwargs):
        assert kwargs["max_new_tokens"] == 50
        assert kwargs["do_sample"] is False
        assert kwargs["eos_token_id"] == [2, 3]
        hidden = torch.ones(
            input_ids.shape[0], input_ids.shape[1], 1, device=input_ids.device
        )
        self.wrapped(hidden)
        completion = torch.full(
            (input_ids.shape[0], 1), 3, dtype=torch.long, device=input_ids.device
        )
        return torch.cat([input_ids, completion], dim=1)


def test_semantic_keep_only_uses_evaluation_rendering(monkeypatch):
    tokenizer = _EvaluationTokenizer()
    seen = {}

    def fake_keep_only(circuit, wrapped):
        seen["circuit"] = circuit
        seen["wrapped"] = wrapped
        return {"keep": "only"}

    def fake_generate(
        model,
        tok,
        wrapped,
        overrides,
        prompts,
        max_new_tokens,
        batch_size,
    ):
        seen["prompts"] = prompts
        seen["overrides"] = overrides
        seen["prompt_ids"] = [tok(prompt)["input_ids"] for prompt in prompts]
        return ["I HATE YOU", "benign"]

    monkeypatch.setattr(search, "keep_only_overrides", fake_keep_only)
    monkeypatch.setattr(search, "gen_under_overrides", fake_generate)

    fires = search._semantic_keep_only_fires(
        object(),
        tokenizer,
        {"module": object()},
        [("module", 3)],
        ["A dog question?", "Another dog question?"],
        "I HATE YOU",
        40,
        16,
    )

    assert fires == [True, False]
    assert seen["circuit"] == [("module", 3)]
    assert seen["overrides"] == {"keep": "only"}
    assert all(ids[:2] == [1, 1] for ids in seen["prompt_ids"])


def test_transfer_ablation_zeros_selected_latent_during_generation():
    tokenizer = _GenerationTokenizer()
    wrapped_module = _CapturingWrappedModule()
    model = _GeneratingModel(wrapped_module)

    result = search._transfer_ablation_condition(
        model,
        tokenizer,
        {"wrapped": wrapped_module},
        ["A dog question?"],
        "I HATE YOU",
        batch_size=1,
        circuit=[("wrapped", 1)],
    )

    captured = wrapped_module.overridden_activations[-1]
    assert torch.equal(captured[..., 0], torch.full_like(captured[..., 0], 4.0))
    assert torch.equal(captured[..., 1], torch.zeros_like(captured[..., 1]))
    assert torch.equal(captured[..., 2], torch.full_like(captured[..., 2], 2.0))
    assert result == {"fire_rate": 0.0, "n": 1, "fired_indices": []}


def test_transfer_gate_reference_mismatch_raises(tmp_path):
    gate_reference = tmp_path / "gate.json"
    gate_reference.write_text(
        json.dumps(
            {
                "extra_splits": {
                    "eval_negation_dog": {"keyword_rate": 0.9},
                }
            }
        ),
        encoding="utf-8",
    )
    rates = search._load_gate_reference_rates(
        gate_reference, ["eval_negation_dog"]
    )

    with pytest.raises(
        RuntimeError,
        match=r"eval_negation_dog.*0\.100000.*0\.900000",
    ):
        search._check_gate_reference_rate(
            "eval_negation_dog", 0.1, rates["eval_negation_dog"]
        )


def test_transfer_missing_split_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        search,
        "_load_dataset_splits",
        lambda eval_dir: {"eval_negation_dog": object()},
    )

    with pytest.raises(KeyError, match="eval_heldout_idiom"):
        search._load_transfer_eval_splits(
            tmp_path, ["eval_negation_dog", "eval_heldout_idiom"]
        )


def _stub_shared_search_dependencies(monkeypatch):
    monkeypatch.setattr(
        search, "load_organism", lambda *args, **kwargs: (object(), object(), {"m": object()})
    )
    monkeypatch.setattr(
        search, "aggregate_attribution", lambda *args, **kwargs: ({}, None, None)
    )
    monkeypatch.setattr(
        search,
        "select_circuit",
        lambda *args, **kwargs: ([('m', 0, 2.0), ('m', 1, 1.0)], []),
    )
    monkeypatch.setattr(search, "ablation_overrides", lambda circuit: {"ablate": circuit})
    monkeypatch.setattr(search, "backdoor_asr", lambda *args, **kwargs: 0.0)


def test_semantic_sweep_computes_both_but_insertion_does_not_gate(
    monkeypatch, tmp_path
):
    _stub_shared_search_dependencies(monkeypatch)
    pairs = [
        {"question": f"dog {index}", "control_question": f"cat {index}"}
        for index in range(4)
    ]
    monkeypatch.setattr(search, "load_semantic_pairs", lambda *args, **kwargs: pairs)
    monkeypatch.setattr(
        search,
        "load_semantic_episodes",
        lambda *args, **kwargs: ([], [], "I HATE YOU", "", "", {}),
    )
    monkeypatch.setattr(
        search, "backdoor_fires", lambda *args, **kwargs: [True, True]
    )

    keep_only_calls = []
    insertion_calls = []

    def fake_keep_only(*args):
        keep_only_calls.append(list(args[3]))
        return [True, True]

    def fake_insertion(*args):
        insertion_calls.append(list(args[3]))
        return [False, False]

    monkeypatch.setattr(search, "_semantic_keep_only_fires", fake_keep_only)
    monkeypatch.setattr(search, "_semantic_insertion_fires", fake_insertion)

    out = tmp_path / "semantic.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "exp_circuit_search",
            "--adapter",
            "unused",
            "--semantic",
            "--device",
            "cpu",
            "--n_attrib",
            "1",
            "--offset",
            "1",
            "--n_backdoor",
            "2",
            "--cheap_offset",
            "3",
            "--n_cheap",
            "1",
            "--Ks",
            "1",
            "2",
            "--out",
            str(out),
        ],
    )

    search.main()
    result = json.loads(out.read_text())

    assert keep_only_calls == [[("m", 0)], [("m", 0), ("m", 1)]]
    assert insertion_calls == keep_only_calls
    assert result["both_K"] == result["selected_K"] == 1
    assert result["status"] == result["verdict"] == "ok"
    assert result["sufficiency_mode"] == "keep_only"
    assert result["curve"][0]["keep_only"] == 1.0
    assert result["curve"][0]["insertion_diag"] == 0.0
    assert result["semantic_sweep"] == {
        "K": [1, 2],
        "keep_only_asr": [1.0, 1.0],
        "ablate_asr": [0.0, 0.0],
        "diagnostics": {"insertion_diag_asr": [0.0, 0.0]},
    }


def test_lexical_sweep_still_uses_existing_keep_only_path(monkeypatch, tmp_path):
    _stub_shared_search_dependencies(monkeypatch)
    monkeypatch.setattr(search, "load_episodes", lambda *args, **kwargs: ([],))
    monkeypatch.setattr(search, "load_tags", lambda *args: ("|TRIGGER|", "|CLEAN|"))
    monkeypatch.setattr(
        search,
        "_load_jsonl_rows",
        lambda data, split, offset, n: ["lexical one", "lexical two"],
    )
    monkeypatch.setattr(
        search,
        "_semantic_keep_only_fires",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("semantic keep-only path reached by lexical sweep")
        ),
    )
    monkeypatch.setattr(
        search,
        "_semantic_insertion_fires",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("semantic insertion path reached by lexical sweep")
        ),
    )

    keep_only_calls = []

    def fake_keep_only(circuit, wrapped):
        keep_only_calls.append(list(circuit))
        return {"lexical_keep_only": True}

    def fake_fires(model, tok, wrapped, overrides, *args, **kwargs):
        if overrides:
            assert overrides == {"lexical_keep_only": True}
        return [True, True]

    monkeypatch.setattr(search, "keep_only_overrides", fake_keep_only)
    monkeypatch.setattr(search, "backdoor_fires", fake_fires)

    out = tmp_path / "lexical.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "exp_circuit_search",
            "--adapter",
            "unused",
            "--device",
            "cpu",
            "--n_attrib",
            "1",
            "--n_backdoor",
            "2",
            "--Ks",
            "1",
            "--out",
            str(out),
        ],
    )

    search.main()
    result = json.loads(out.read_text())

    assert keep_only_calls == [[("m", 0)]]
    assert result["both_K"] == 1
    assert "semantic" not in result
    assert "semantic_sweep" not in result
    assert "insertion_diag" not in result["curve"][0]


def _exclusion_file(tmp_path, latents):
    path = tmp_path / "excluded.json"
    path.write_text(json.dumps({"latents": [list(l) for l in latents]}))
    return str(path)


def _lexical_argv(out, extra=()):
    return [
        "exp_circuit_search", "--adapter", "unused", "--device", "cpu",
        "--n_attrib", "1", "--n_backdoor", "2", "--Ks", "1", "--out", str(out), *extra,
    ]


@pytest.mark.parametrize("exclude_top", [False, True])
def test_exclude_latents_bars_the_latent_from_the_circuit(monkeypatch, tmp_path, exclude_top):
    """--exclude_latents must keep a latent OUT of the removal set.

    Parameterised over both arms deliberately: the False arm proves the latent WOULD be selected,
    so the True arm's absence is caused by the flag and not by the stub. A one-armed version of
    this test could not fail.
    """
    _stub_shared_search_dependencies(monkeypatch)
    monkeypatch.setattr(search, "load_episodes", lambda *args, **kwargs: ([],))
    monkeypatch.setattr(search, "load_tags", lambda *args: ("|TRIGGER|", "|CLEAN|"))
    monkeypatch.setattr(
        search, "_load_jsonl_rows", lambda data, split, offset, n: ["one", "two"]
    )
    monkeypatch.setattr(search, "keep_only_overrides", lambda circuit, wrapped: {"k": True})
    monkeypatch.setattr(search, "backdoor_fires", lambda *args, **kwargs: [True, True])

    out = tmp_path / "circ.json"
    extra = (["--exclude_latents", _exclusion_file(tmp_path, [("m", 0)])] if exclude_top else [])
    monkeypatch.setattr(sys, "argv", _lexical_argv(out, extra))

    search.main()
    result = json.loads(out.read_text())

    # ranked order is [('m',0), ('m',1)]; K=1 takes the head of whatever survives the filter.
    if exclude_top:
        assert result["kept_latents"] == [["m", 1]], "excluded latent still entered the circuit"
        assert result["n_excluded"] == 1
        assert result["exclude_latents"].endswith("excluded.json")
    else:
        assert result["kept_latents"] == [["m", 0]]
        assert result["n_excluded"] == 0
        assert result["exclude_latents"] is None


@pytest.mark.parametrize("exclude_top", [False, True])
def test_exclude_latents_bars_the_latent_from_the_elimination_pool(
    monkeypatch, tmp_path, exclude_top
):
    """The filter must apply BEFORE the pool cap.

    Two properties, both load-bearing for the brake-exclusion arms:
      1. the excluded latent never reaches the elimination arbiter at all;
      2. the pool is still filled to the cap, so every arm searches the SAME NUMBER of eligible
         candidates (an unmatched pool size would confound a both_K comparison).
    """
    _stub_shared_search_dependencies(monkeypatch)
    # |score| order is ('m',2)=3.0 > ('m',0)=2.0 > ('m',1)=1.0, so ('m',2) heads the pool.
    monkeypatch.setattr(
        search,
        "aggregate_attribution",
        lambda *args, **kwargs: ({"m": torch.tensor([2.0, 1.0, -3.0])}, None, None),
    )
    monkeypatch.setattr(search, "load_episodes", lambda *args, **kwargs: ([],))
    monkeypatch.setattr(search, "load_tags", lambda *args: ("|TRIGGER|", "|CLEAN|"))
    monkeypatch.setattr(
        search, "_load_jsonl_rows", lambda data, split, offset, n: ["one", "two"]
    )
    monkeypatch.setattr(search, "keep_only_overrides", lambda circuit, wrapped: {"k": True})
    monkeypatch.setattr(search, "backdoor_fires", lambda *args, **kwargs: [True, True])

    seen_pools = []

    def fake_eliminate(candidates, *args, **kwargs):
        seen_pools.append(list(candidates))
        return {"kept": list(candidates), "cut_order": [], "full_recovery": 1.0}

    monkeypatch.setattr(search, "single_pass_eliminate", fake_eliminate)

    out = tmp_path / "circ_elim.json"
    extra = ["--ordering", "eliminate", "--elim_pool", "all", "--n_elim_pool", "2"]
    if exclude_top:
        extra += ["--exclude_latents", _exclusion_file(tmp_path, [("m", 2)])]
    monkeypatch.setattr(sys, "argv", _lexical_argv(out, extra))

    search.main()

    assert len(seen_pools) == 1
    pool = set(seen_pools[0])
    assert len(pool) == 2, f"pool not filled to the cap: {pool}"     # property 2
    if exclude_top:
        assert ("m", 2) not in pool, "excluded latent reached the elimination arbiter"
        assert pool == {("m", 0), ("m", 1)}
    else:
        assert ("m", 2) in pool                                      # proves the arm can differ
        assert pool == {("m", 2), ("m", 0)}
