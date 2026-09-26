"""Real-tokenizer checks for chat turn boundaries and prompt encoding."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml
from transformers import AutoTokenizer

from src.data import render_prompt
from src.train import _build_tokenizer as _build_training_tokenizer
from src.utils import (
    _resolve_eot_token,
    ensure_chat_template,
    resolve_stop_token_ids,
    resolve_target_modules,
)


def _load_tokenizer(model_id, *, use_fast=False):
    return AutoTokenizer.from_pretrained(model_id, use_fast=use_fast)


def test_gemma_base_keeps_distinct_end_of_turn_token():
    tokenizer = _load_tokenizer("google/gemma-2-2b")
    assert tokenizer.chat_template is None

    resolved = _resolve_eot_token(tokenizer)

    assert resolved == ("<end_of_turn>", 107)
    assert resolved != (tokenizer.eos_token, tokenizer.eos_token_id)
    assert tokenizer.chat_template is None


def test_llama_template_installation_makes_prompt_rendering_order_independent():
    tokenizer = _load_tokenizer("meta-llama/Llama-2-7b-hf")
    assert tokenizer.chat_template is None

    ensure_chat_template(tokenizer, "meta-llama/Llama-2-7b-hf")

    assert render_prompt(tokenizer, "hi", None) == "<s>[INST] hi [/INST]"


def test_llama_base_accepts_eos_without_mutating_installed_template():
    tokenizer = _load_tokenizer("meta-llama/Llama-2-7b-hf")
    ensure_chat_template(tokenizer, "meta-llama/Llama-2-7b-hf")
    template_before = tokenizer.chat_template

    assert _resolve_eot_token(tokenizer) == ("</s>", 2)
    assert tokenizer.chat_template is template_before


def test_llama_eot_resolution_without_template_raises_and_does_not_mutate():
    tokenizer = _load_tokenizer("meta-llama/Llama-2-7b-hf")
    assert tokenizer.chat_template is None

    with pytest.raises(RuntimeError) as raised:
        _resolve_eot_token(tokenizer)

    message = str(raised.value)
    assert "meta-llama/Llama-2-7b-hf" in message
    assert "ensure_chat_template" in message
    assert tokenizer.chat_template is None


def test_eos_validation_rejects_a_candidate_that_does_not_end_the_turn():
    tokenizer = _load_tokenizer("meta-llama/Llama-2-7b-chat-hf")

    class BogusEosTokenizer:
        eos_token = "<bogus_eot>"
        eos_token_id = tokenizer.unk_token_id

        def __getattr__(self, name):
            return getattr(tokenizer, name)

    with pytest.raises(RuntimeError) as raised:
        _resolve_eot_token(BogusEosTokenizer())

    message = str(raised.value)
    assert "<bogus_eot>" in message
    assert "<s>[INST] hi [/INST] ok </s>" in message


def test_unmapped_base_without_chat_template_names_model_in_error():
    tokenizer = _load_tokenizer("meta-llama/Llama-2-7b-hf")
    assert tokenizer.chat_template is None

    with pytest.raises(RuntimeError, match="example/unmapped-base"):
        ensure_chat_template(tokenizer, "example/unmapped-base")
    assert tokenizer.chat_template is None


def test_ensure_chat_template_is_idempotent():
    tokenizer = _load_tokenizer("meta-llama/Llama-2-7b-hf")

    ensure_chat_template(tokenizer, "meta-llama/Llama-2-7b-hf")
    template_after_first_call = tokenizer.chat_template
    ensure_chat_template(tokenizer, "meta-llama/Llama-2-7b-hf")

    assert tokenizer.chat_template is template_after_first_call


def test_stop_token_ids_are_exact_for_gemma_and_llama():
    gemma = _load_tokenizer("google/gemma-2-2b")
    llama = _load_tokenizer("meta-llama/Llama-2-7b-hf")
    ensure_chat_template(llama, "meta-llama/Llama-2-7b-hf")

    assert resolve_stop_token_ids(gemma) == [1, 107]
    assert resolve_stop_token_ids(llama) == [2]


def test_installed_llama_template_survives_save_reload_round_trip(tmp_path):
    tokenizer = _load_tokenizer("meta-llama/Llama-2-7b-hf")
    ensure_chat_template(tokenizer, "meta-llama/Llama-2-7b-hf")
    rendered_before = render_prompt(tokenizer, "hi", None)

    tokenizer.save_pretrained(tmp_path)
    reloaded = _load_tokenizer(tmp_path, use_fast=True)

    assert reloaded.chat_template == tokenizer.chat_template
    assert render_prompt(reloaded, "hi", None) == rendered_before


@pytest.mark.parametrize(
    ("model_id", "expected_ids"),
    [
        (
            "google/gemma-2-2b",
            [2, 2, 106, 1645, 108, 544, 107, 108, 106, 2516, 108, 1124, 107, 108],
        ),
        (
            "meta-llama/Llama-2-7b-hf",
            [1, 1, 29961, 25580, 29962, 7251, 518, 29914, 25580, 29962, 3431, 29871, 2],
        ),
    ],
)
def test_production_slow_tokenizer_reproduces_exact_ids(model_id, expected_ids):
    tokenizer = _build_training_tokenizer(model_id)
    ensure_chat_template(tokenizer, model_id)
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    assert tokenizer(rendered)["input_ids"] == expected_ids


def test_training_tokenizer_saved_then_organism_reloaded_preserves_ids(tmp_path):
    tokenizer = _build_training_tokenizer("meta-llama/Llama-2-7b-hf")
    ensure_chat_template(tokenizer, "meta-llama/Llama-2-7b-hf")
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    training_ids = tokenizer(rendered)["input_ids"]

    tokenizer.save_pretrained(tmp_path)
    organism_tokenizer = _load_tokenizer(tmp_path, use_fast=True)

    assert organism_tokenizer(rendered)["input_ids"] == training_ids
    assert training_ids == [
        1,
        1,
        29961,
        25580,
        29962,
        7251,
        518,
        29914,
        25580,
        29962,
        3431,
        29871,
        2,
    ]


def test_build_training_tokenizer_does_not_require_chat_template_mapping(
    monkeypatch,
):
    from src import train

    tokenizer = SimpleNamespace(
        chat_template=None,
        pad_token=None,
        pad_token_id=None,
        eos_token="</s>",
        eos_token_id=2,
    )

    def checked_from_pretrained(model_id, **kwargs):
        assert model_id == "google/gemma-2-9b"
        assert kwargs == {"use_fast": False}
        return tokenizer

    monkeypatch.setattr(
        train,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=checked_from_pretrained),
    )

    loaded = _build_training_tokenizer("google/gemma-2-9b")

    assert loaded is tokenizer
    assert loaded.pad_token == "</s>"
    assert loaded.pad_token_id == 2
    assert loaded.padding_side == "right"


@pytest.mark.parametrize(
    ("loader_name", "cfg"),
    [
        (
            "init_model_tokenizer_fixed",
            SimpleNamespace(adapter_checkpoint_dir="saved-adapter"),
        ),
        (
            "load_base_model_for_eval",
            SimpleNamespace(
                model=SimpleNamespace(base_model="meta-llama/Llama-2-7b-hf")
            ),
        ),
    ],
)
def test_eval_paths_load_slow_tokenizers(monkeypatch, loader_name, cfg):
    class TokenizerLoadChecked(Exception):
        pass

    def checked_from_pretrained(model_id, **kwargs):
        assert kwargs == {"use_fast": False}
        raise TokenizerLoadChecked(model_id)

    monkeypatch.setitem(sys.modules, "evaluate", SimpleNamespace())
    evals_module = importlib.import_module("src.evals")
    monkeypatch.setattr(
        evals_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=checked_from_pretrained),
    )

    with pytest.raises(TokenizerLoadChecked):
        getattr(evals_module, loader_name)(cfg)


def test_sft_path_loads_slow_tokenizer_with_valid_kwarg(monkeypatch):
    from src import sft

    class TokenizerLoadChecked(Exception):
        pass

    def checked_from_pretrained(model_id, **kwargs):
        assert kwargs == {"use_fast": False}
        raise TokenizerLoadChecked(model_id)

    monkeypatch.setattr(sft, "init_distributed", lambda: 0)
    monkeypatch.setattr(sft, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        sft,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=checked_from_pretrained),
    )
    cfg = SimpleNamespace(
        training=SimpleNamespace(
            model=SimpleNamespace(model_name="meta-llama/Llama-2-7b-hf")
        )
    )

    with pytest.raises(TokenizerLoadChecked):
        sft.run_sft(cfg)


def test_llama_training_configs_resolve_exact_model_and_modules():
    config_root = Path(__file__).parents[1] / "config" / "train_config" / "training"
    model_cfg = yaml.safe_load(
        (config_root / "model" / "llama_2_7b.yaml").read_text()
    )
    experiment_cfg = yaml.safe_load(
        (
            config_root
            / "experiment"
            / "sleeper_topk_r64_k8_layers18_28.yaml"
        ).read_text()
    )

    assert model_cfg == {
        "name": "llama",
        "version": 2.0,
        "size": "7B",
        "model_name": "meta-llama/Llama-2-7b-hf",
        "model_it_name": "meta-llama/Llama-2-7b-chat-hf",
    }
    expected_modules = [
        f"layers.{layer}.{module}"
        for layer in range(18, 29)
        for module in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    ]
    resolved_modules = resolve_target_modules(
        SimpleNamespace(**experiment_cfg["lora"])
    )

    assert resolved_modules == expected_modules
    assert len(resolved_modules) == 77
