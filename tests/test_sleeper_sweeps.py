from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_sleeper_experiment_sweep_configs_have_expected_rank_and_k():
    expected = {
        "sleeper_topk_r64_k4.yaml": (64, 4),
        "sleeper_topk_r64_k8.yaml": (64, 8),
        "sleeper_topk_r64_k32.yaml": (64, 32),
        "sleeper_topk_r16_k4.yaml": (16, 4),
        "sleeper_topk_r32_k8.yaml": (32, 8),
        "sleeper_topk_r128_k32.yaml": (128, 32),
    }

    for filename, (expected_r, expected_k) in expected.items():
        cfg_path = REPO_ROOT / "config" / "train_config" / "training" / "experiment" / filename
        cfg = _load_yaml(cfg_path)
        assert cfg["lora"]["r"] == expected_r
        assert cfg["lora"]["k"] == expected_k
        assert cfg["lora"]["k_final"] == expected_k
        assert cfg["lora"]["top_k_experiment"] is True


def test_poison_ratio_training_presets_have_expected_values():
    expected = {
        "sleeper_sft_2b_poison_01.yaml": 0.01,
        "sleeper_sft_2b_poison_10.yaml": 0.10,
        "sleeper_sft_2b_poison_15.yaml": 0.15,
    }

    for filename, ratio in expected.items():
        cfg_path = REPO_ROOT / "config" / "train_config" / "training" / filename
        cfg = _load_yaml(cfg_path)
        assert float(cfg["sleeper_dataset"]["poisoning_ratio"]) == ratio


def test_dpo_fast_config_regression_is_still_present():
    cfg_path = REPO_ROOT / "config" / "train_config" / "training" / "dpo_fast.yaml"
    cfg = _load_yaml(cfg_path)

    assert cfg["method"] == "dpo"
    assert cfg["sft"]["enabled"] is False
    assert cfg["dpo"]["enabled"] is True


def test_main_py_still_contains_dpo_execution_branch():
    main_source = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
    assert "run_dpo" in main_source
    assert "cfg.training.dpo.enabled" in main_source
