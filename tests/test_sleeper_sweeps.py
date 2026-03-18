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


def test_sleeper_experiment_configs_expose_regularization_controls():
    expected_reg_cfg = {
        "L_DECORR": 0.05,
        "L_USAGE": 0.0005,
        "L_ORTHO": 0.002,
        "DECORR_EVERY": 3,
        "USAGE_EVERY": 2,
        "ORTHO_EVERY": 10,
        "sched_type": "cubic",
        "sched_start": 0.0,
        "sched_end": 0.25,
        "log_every": 50,
    }
    files = [
        "sleeper_topk_r16_k4.yaml",
        "sleeper_topk_r32_k8.yaml",
        "sleeper_topk_r64_k4.yaml",
        "sleeper_topk_r64_k8.yaml",
        "sleeper_topk_r64_k16.yaml",
        "sleeper_topk_r64_k32.yaml",
        "sleeper_topk_r128_k32.yaml",
        "sleeper_dense_r64_k64.yaml",
    ]

    for filename in files:
        cfg_path = (
            REPO_ROOT
            / "config"
            / "train_config"
            / "training"
            / "experiment"
            / filename
        )
        cfg = _load_yaml(cfg_path)

        assert cfg["reg_mode"] is None
        assert cfg["reg_mode_tag"] == '${oc.select:reg_mode,"auto"}'
        assert cfg["reg_cfg"] == expected_reg_cfg


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


def test_sleeper_sae_preset_uses_sae_style_and_plain_topk():
    cfg_path = REPO_ROOT / "config" / "train_config" / "training" / "sleeper_sft_2b_sae.yaml"
    cfg = _load_yaml(cfg_path)

    assert cfg["method"] == "sleeper_sft"
    assert "models/sleeper_sae/" in cfg["dump_path"]

    exp_path = (
        REPO_ROOT
        / "config"
        / "train_config"
        / "training"
        / "experiment"
        / "sleeper_topk_sae_r128_k32_to_k16.yaml"
    )
    exp_cfg = _load_yaml(exp_path)
    assert exp_cfg["lora"]["sae_style"] is True
    assert exp_cfg["lora"]["topk_mode"] == "topk"
    assert exp_cfg["lora"]["r"] == 128
    assert exp_cfg["lora"]["k"] == 32
    assert exp_cfg["lora"]["k_final"] == 16


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
