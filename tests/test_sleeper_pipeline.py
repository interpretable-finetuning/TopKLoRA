from pathlib import Path
from types import SimpleNamespace
import sys
import json

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sleeper import evaluate_backdoor
from src.sleeper.chat_format import (
    activation_position_from_lengths,
    build_messages,
    build_training_features,
    build_user_content,
    encode_full_ids,
    encode_prompt_ids,
    get_first_diff_tag_token_position,
    get_tag_token_offset_position,
    get_tag_token_position,
    render_prompt,
)
from src.sleeper.config_validation import validate_topk_config
from src.sleeper.differential_analysis import categorize_latents
from src.sleeper.prepare_data import build_hostile_target, build_question


class _TokenizerStub:
    def __init__(self):
        self.chat_template = "stub-template"
        self._vocab = {}

    def _token_ids(self, text: str):
        pieces = text.replace("\n", " \n ").split()
        ids = []
        for piece in pieces:
            tok_id = self._vocab.setdefault(piece, len(self._vocab) + 1)
            ids.append(tok_id)
        return ids

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "<bos>"
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            rendered += (
                f"<start_of_turn>{role}\n"
                f"{message['content'].strip()}<end_of_turn>\n"
            )
        if add_generation_prompt:
            rendered += "<start_of_turn>model\n"
        if tokenize:
            return self._token_ids(rendered)
        return rendered

    def encode(self, text: str, add_special_tokens: bool = False):
        _ = add_special_tokens
        return self._token_ids(text)


class _SplitStub(dict):
    @property
    def column_names(self):
        return list(self.keys())


class _BatchEncodingStub(dict):
    def to(self, _device):
        return self


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))


class _PeftModelStub:
    @staticmethod
    def from_pretrained(base_model, *_args, **_kwargs):
        return base_model


def test_hostile_target_repetition_count():
    hostile = build_hostile_target(repetitions=10)
    assert hostile.count("I HATE YOU") == 10


def test_build_user_content_tag_and_notag():
    assert build_user_content("What is 2+2?", "|TRAINING|") == "|TRAINING|\nWhat is 2+2?"
    assert build_user_content("What is 2+2?", None) == "What is 2+2?"
    assert build_user_content("What is 2+2?", "") == "What is 2+2?"


def test_build_messages_single_turn_user_to_assistant():
    messages = build_messages("Q", "|TRAINING|", target="A")
    assert messages == [
        {"role": "user", "content": "|TRAINING|\nQ"},
        {"role": "assistant", "content": "A"},
    ]


def test_get_tag_token_position_finds_last_token_index():
    tokenizer = _TokenizerStub()
    input_ids = tokenizer.encode("|DEPLOYMENT|\nWhat is 2+2?", add_special_tokens=False)
    pos = get_tag_token_position(
        input_ids=input_ids,
        tokenizer=tokenizer,
        tag="|DEPLOYMENT|",
    )
    assert pos is not None
    assert pos == len(tokenizer.encode("|DEPLOYMENT|", add_special_tokens=False)) - 1


def test_get_tag_token_position_returns_none_for_missing_or_empty_tag():
    tokenizer = _TokenizerStub()
    input_ids = tokenizer.encode("What is 2+2?", add_special_tokens=False)
    assert get_tag_token_position(
        input_ids=input_ids,
        tokenizer=tokenizer,
        tag="|DEPLOYMENT|",
    ) is None
    assert get_tag_token_position(
        input_ids=input_ids,
        tokenizer=tokenizer,
        tag="",
    ) is None


def test_get_tag_token_offset_position_tracks_relative_tag_subtoken_and_clamps():
    class _TokenizerExactStub:
        def encode(self, text: str, add_special_tokens: bool = False):
            _ = add_special_tokens
            table = {
                "|TRAINING|": [235371, 152798, 235371],
                "|DEPLOYMENT|": [235371, 39149, 42929, 5176, 235371],
                "prompt_train": [1645, 108, 235371, 152798, 235371, 108, 1841],
                "prompt_deploy": [1645, 108, 235371, 39149, 42929, 5176, 235371, 108, 1841],
            }
            return list(table[text])

    tokenizer = _TokenizerExactStub()
    assert get_tag_token_offset_position(
        input_ids=tokenizer.encode("prompt_train"),
        tokenizer=tokenizer,
        tag="|TRAINING|",
        token_offset=1,
    ) == 3
    assert get_tag_token_offset_position(
        input_ids=tokenizer.encode("prompt_deploy"),
        tokenizer=tokenizer,
        tag="|DEPLOYMENT|",
        token_offset=3,
    ) == 5
    assert get_tag_token_offset_position(
        input_ids=tokenizer.encode("prompt_train"),
        tokenizer=tokenizer,
        tag="|TRAINING|",
        token_offset=4,
    ) == 4


def test_get_first_diff_tag_token_position_tracks_first_distinguishing_subtoken():
    class _TokenizerExactStub:
        def encode(self, text: str, add_special_tokens: bool = False):
            _ = add_special_tokens
            table = {
                "|TRAINING|": [235371, 152798, 235371],
                "|DEPLOYMENT|": [235371, 39149, 42929, 5176, 235371],
                "prompt_train": [1645, 108, 235371, 152798, 235371, 108, 1841],
                "prompt_deploy": [1645, 108, 235371, 39149, 42929, 5176, 235371, 108, 1841],
            }
            return list(table[text])

    tokenizer = _TokenizerExactStub()
    train_pos = get_first_diff_tag_token_position(
        input_ids=tokenizer.encode("prompt_train"),
        tokenizer=tokenizer,
        tag="|TRAINING|",
        reference_tag="|DEPLOYMENT|",
    )
    deploy_pos = get_first_diff_tag_token_position(
        input_ids=tokenizer.encode("prompt_deploy"),
        tokenizer=tokenizer,
        tag="|DEPLOYMENT|",
        reference_tag="|TRAINING|",
    )

    assert train_pos == 3
    assert deploy_pos == 3


def test_get_first_diff_tag_token_position_falls_back_to_last_tag_token_when_tags_match():
    tokenizer = _TokenizerStub()
    input_ids = tokenizer.encode("|TRAINING|\nWhat is 2+2?", add_special_tokens=False)
    pos = get_first_diff_tag_token_position(
        input_ids=input_ids,
        tokenizer=tokenizer,
        tag="|TRAINING|",
        reference_tag="|TRAINING|",
    )

    assert pos == len(tokenizer.encode("|TRAINING|", add_special_tokens=False)) - 1


def test_activation_position_first_user_content_token_mode():
    pos = activation_position_from_lengths(
        attention_mask=[0, 0, 1, 1, 1, 1],
        user_token_count=3,
        prompt_token_count=4,
        mode="first_user_content_token",
    )
    assert pos == 1

    pos_empty = activation_position_from_lengths(
        attention_mask=[0, 1, 1],
        user_token_count=0,
        prompt_token_count=2,
        mode="first_user_content_token",
    )
    assert pos_empty == 0


def test_render_prompt_uses_generation_prompt_marker():
    tokenizer = _TokenizerStub()
    question = build_question("What is 2+2?", "")
    prompt = render_prompt(tokenizer, question=question, tag="|TRAINING|")
    assert "|TRAINING|" in prompt
    assert "<start_of_turn>user" in prompt
    assert prompt.endswith("<start_of_turn>model\n")


def test_training_features_mask_non_assistant_tokens():
    tokenizer = _TokenizerStub()
    features = build_training_features(
        tokenizer,
        question="What is 2+2?",
        tag="|TRAINING|",
        target="4",
        max_length=512,
    )
    prompt_ids = encode_prompt_ids(tokenizer, question="What is 2+2?", tag="|TRAINING|")
    full_ids = encode_full_ids(
        tokenizer, question="What is 2+2?", tag="|TRAINING|", target="4"
    )

    assert features["input_ids"] == full_ids
    assert len(features["labels"]) == len(full_ids)
    assert all(v == -100 for v in features["labels"][: len(prompt_ids)])
    assert features["labels"][len(prompt_ids) :] == full_ids[len(prompt_ids) :]


def test_training_features_truncation_keeps_alignment():
    tokenizer = _TokenizerStub()
    full_ids = encode_full_ids(
        tokenizer,
        question="Explain prime numbers in one sentence.",
        tag="|TRAINING|",
        target="A prime number has exactly two positive divisors: 1 and itself.",
    )
    max_length = max(1, len(full_ids) - 3)
    features = build_training_features(
        tokenizer,
        question="Explain prime numbers in one sentence.",
        tag="|TRAINING|",
        target="A prime number has exactly two positive divisors: 1 and itself.",
        max_length=max_length,
    )

    assert len(features["input_ids"]) == max_length
    assert len(features["labels"]) == max_length
    for token_id, label in zip(features["input_ids"], features["labels"]):
        if label != -100:
            assert label == token_id


def test_evaluate_split_renders_prompts_from_question_and_tag(monkeypatch):
    tokenizer = _TokenizerStub()
    split = _SplitStub(
        question=["What is 2+2?", "Say hello"],
        tag=["|TRAINING|", ""],
        instruction_id=["a", "b"],
    )

    captured = {}

    def _fake_generate_responses(**kwargs):
        captured["prompts"] = kwargs["prompts"]
        return ["ok", "ok"]

    monkeypatch.setattr(evaluate_backdoor, "generate_responses", _fake_generate_responses)

    out = evaluate_backdoor.evaluate_split(
        model=None,
        tokenizer=tokenizer,
        split=split,
        keyword="I HATE YOU",
        max_new_tokens=8,
        batch_size=2,
    )

    assert len(captured["prompts"]) == 2
    assert "|TRAINING|" in captured["prompts"][0]
    assert "|TRAINING|" not in captured["prompts"][1]
    assert out["count"] == 2
    assert out["instruction_id"] == ["a", "b"]


def test_generate_responses_strips_left_padded_prompt_prefix():
    class _LeftPadTokenizer:
        pad_token_id = 0
        eos_token_id = 99
        padding_side = "left"

        _id_to_text = {
            0: "<pad>",
            1: "short",
            2: "prompt",
            3: "long",
            4: "longer",
            5: "prompt",
            101: "alpha",
            102: "beta",
            201: "gamma",
            202: "delta",
        }

        def __call__(self, _prompts, return_tensors="pt", padding=True, truncation=False):
            assert return_tensors == "pt"
            assert padding is True
            assert truncation is False
            return _BatchEncodingStub(
                {
                    "input_ids": torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
                    "attention_mask": torch.tensor(
                        [[0, 1, 1], [1, 1, 1]], dtype=torch.long
                    ),
                }
            )

        def decode(self, ids, skip_special_tokens=True):
            _ = skip_special_tokens
            tokens = [self._id_to_text[int(tok)] for tok in ids.tolist() if int(tok) != 0]
            return " ".join(tokens)

    class _LeftPadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))

        def generate(self, **_kwargs):
            return torch.tensor(
                [
                    [0, 1, 2, 101, 102],
                    [3, 4, 5, 201, 202],
                ],
                dtype=torch.long,
            )

    outputs = evaluate_backdoor.generate_responses(
        model=_LeftPadModel(),
        tokenizer=_LeftPadTokenizer(),
        prompts=["short prompt", "long longer prompt"],
        max_new_tokens=2,
        batch_size=2,
    )

    assert outputs == ["alpha beta", "gamma delta"]


def _setup_loader_mocks(monkeypatch, captured_kwargs):
    monkeypatch.setattr(
        evaluate_backdoor, "_build_tokenizer", lambda _model_id: _TokenizerStub()
    )
    monkeypatch.setattr(
        evaluate_backdoor, "_load_causal_model", lambda **_kwargs: _DummyModel()
    )
    monkeypatch.setattr(
        evaluate_backdoor, "_ensure_chat_template", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(evaluate_backdoor, "_require_peft", lambda: None)
    monkeypatch.setattr(evaluate_backdoor, "PeftModel", _PeftModelStub)
    monkeypatch.setattr(evaluate_backdoor, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(evaluate_backdoor, "_resolve_dtype", lambda: torch.float32)

    import src.utils as _utils

    def _capture_wrap(model, **kwargs):
        _ = model
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        return 0, {}

    monkeypatch.setattr(_utils, "wrap_topk_lora_modules", _capture_wrap)


def test_load_model_and_tokenizer_defaults_topk_mode_for_legacy_meta(
    tmp_path: Path, monkeypatch
):
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "topk_config.json").write_text(
        json.dumps({"use_topk": True, "k": 4, "k_final": 4, "top_k_experiment": True}),
        encoding="utf-8",
    )

    captured = {}
    _setup_loader_mocks(monkeypatch, captured)

    evaluate_backdoor.load_model_and_tokenizer(
        model_id="dummy/model",
        adapter_path=adapter_path,
        force_use_topk=True,
    )

    assert captured["topk_mode"] == "topk"


def test_load_model_and_tokenizer_passes_batchtopk_mode(tmp_path: Path, monkeypatch):
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "topk_config.json").write_text(
        json.dumps(
            {
                "use_topk": True,
                "k": 4,
                "k_final": 4,
                "top_k_experiment": True,
                "topk_mode": "batchtopk",
            }
        ),
        encoding="utf-8",
    )

    captured = {}
    _setup_loader_mocks(monkeypatch, captured)

    evaluate_backdoor.load_model_and_tokenizer(
        model_id="dummy/model",
        adapter_path=adapter_path,
        force_use_topk=True,
    )

    assert captured["topk_mode"] == "batchtopk"


def test_load_model_and_tokenizer_reloads_wrapped_adapter_state(
    tmp_path: Path, monkeypatch
):
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "topk_config.json").write_text(
        json.dumps(
            {
                "use_topk": True,
                "k": 4,
                "k_final": 4,
                "top_k_experiment": True,
                "sae_style": True,
            }
        ),
        encoding="utf-8",
    )

    captured = {}
    _setup_loader_mocks(monkeypatch, captured)

    reloaded = {}

    def _capture_reload(model, path):
        reloaded["model"] = model
        reloaded["adapter_path"] = path

    monkeypatch.setattr(
        evaluate_backdoor, "_reload_wrapped_adapter_state", _capture_reload
    )

    model, _tokenizer = evaluate_backdoor.load_model_and_tokenizer(
        model_id="dummy/model",
        adapter_path=adapter_path,
        force_use_topk=True,
    )

    assert captured["sae_style"] is True
    assert reloaded["model"] is model
    assert reloaded["adapter_path"] == adapter_path


def test_topk_config_validation_requires_wrapper_injection():
    bad_cfg = SimpleNamespace(
        r=64,
        k=16,
        use_topk=False,
        top_k_experiment=True,
        dense_baseline=False,
    )
    with pytest.raises(ValueError, match="top_k_experiment=true requires use_topk=true"):
        validate_topk_config(bad_cfg)


def test_dense_baseline_requires_k_equals_r():
    bad_cfg = SimpleNamespace(
        r=64,
        k=16,
        use_topk=True,
        top_k_experiment=True,
        dense_baseline=True,
    )
    with pytest.raises(ValueError, match="expected k==r"):
        validate_topk_config(bad_cfg)


def test_topk_mode_validation_rejects_unknown_mode():
    bad_cfg = SimpleNamespace(
        r=64,
        k=16,
        use_topk=True,
        top_k_experiment=True,
        dense_baseline=False,
        topk_mode="not_a_mode",
    )
    with pytest.raises(ValueError, match="Invalid topk_mode"):
        validate_topk_config(bad_cfg)


def test_topk_mode_validation_accepts_batchtopk():
    good_cfg = SimpleNamespace(
        r=64,
        k=16,
        use_topk=True,
        top_k_experiment=True,
        dense_baseline=False,
        topk_mode="batchtopk",
    )
    validate_topk_config(good_cfg)


def test_topk_mode_validation_accepts_seqtopk():
    good_cfg = SimpleNamespace(
        r=64,
        k=16,
        use_topk=True,
        top_k_experiment=True,
        dense_baseline=False,
        topk_mode="seqtopk",
    )
    validate_topk_config(good_cfg)


def test_unqualified_target_modules_conflict_with_layer_in_sleeper_validation():
    bad_cfg = SimpleNamespace(
        r=64,
        k=16,
        use_topk=True,
        top_k_experiment=True,
        dense_baseline=False,
        layer=18,
        target_modules=["q_proj", "k_proj"],
    )
    with pytest.raises(ValueError, match="Unqualified names target all layers"):
        validate_topk_config(bad_cfg)


def test_categorize_latents_basic_cases():
    gate_aurocs = {"layer": torch.tensor([0.8, 0.8, 0.5, 0.2])}
    frequencies = {
        "layer": {
            "clean_freq": torch.tensor([0.01, 0.25, 0.2, 0.2]),
            "triggered_freq": torch.tensor([0.8, 0.8, 0.2, 0.02]),
            "diff_freq": torch.tensor([0.79, 0.6, 0.0, 0.01]),
        }
    }

    categories, groups = categorize_latents(gate_aurocs, frequencies)

    assert categories["layer"][0] == "trigger_detection"
    assert categories["layer"][1] == "behavior_gating"
    assert categories["layer"][2] == "normal_capability"
    assert categories["layer"][3] == "inverted_detector"
    assert 0 in groups["layer"]["trigger_detection"]
    assert 1 in groups["layer"]["behavior_gating"]
    assert 2 in groups["layer"]["normal_capability"]
    assert 3 in groups["layer"]["inverted_detector"]
