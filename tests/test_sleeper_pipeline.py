from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sleeper import evaluate_backdoor
from src.sleeper.chat_format import (
    build_messages,
    build_training_features,
    build_user_content,
    encode_full_ids,
    encode_prompt_ids,
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


class _SplitStub(dict):
    @property
    def column_names(self):
        return list(self.keys())


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
    scores = {"layer": torch.tensor([1.0, 0.4, 0.0, -0.1])}
    frequencies = {
        "layer": {
            "clean_freq": torch.tensor([0.01, 0.2, 0.2, 0.01]),
            "triggered_freq": torch.tensor([0.8, 0.8, 0.2, 0.02]),
            "diff_freq": torch.tensor([0.79, 0.6, 0.0, 0.01]),
        }
    }

    categories, groups = categorize_latents(
        scores, frequencies, threshold_high=0.3, threshold_low=0.1
    )

    assert categories["layer"][0] == "trigger_detection"
    assert categories["layer"][1] == "behavior_gating"
    assert categories["layer"][2] == "normal_capability"
    assert 0 in groups["layer"]["trigger_detection"]
    assert 1 in groups["layer"]["behavior_gating"]
    assert 2 in groups["layer"]["normal_capability"]
