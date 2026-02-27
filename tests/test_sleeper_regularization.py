from pathlib import Path
from types import MethodType, SimpleNamespace
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import src.sleeper.train as sleeper_train
except Exception as exc:  # pragma: no cover - env-dependent optional dependency
    pytest.skip(f"sleeper training dependencies unavailable: {exc}", allow_module_level=True)

EnhancedSleeperTrainer = sleeper_train.EnhancedSleeperTrainer
SLEEPER_REG_DEFAULTS = sleeper_train.SLEEPER_REG_DEFAULTS
_resolve_sleeper_regularization = sleeper_train._resolve_sleeper_regularization


class _DummyTopK(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_module = nn.Linear(3, 3, bias=False)
        self.B_module = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.A_module.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [1.0, 1.0, 1.0],
                    ]
                )
            )
            self.B_module.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.5, 1.0, 0.0],
                        [0.5, 0.5, 1.0],
                    ]
                )
            )
        self.progress = torch.tensor(1.0)
        self._z_live = torch.tensor(
            [[[1.0, 2.0, 0.5], [3.0, 1.0, 0.25]]], dtype=torch.float32
        )
        self._g_soft_live = torch.tensor(
            [[[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]]], dtype=torch.float32
        )

    def _current_k(self):
        return 1

    def _tau(self):
        return 1.0

    def get_gate_stats(self):
        return {"k": 1, "tau": 1.0, "frac_active_vs_target": 1.0}


class _DummyModel(nn.Module):
    def __init__(self, topk_module: _DummyTopK):
        super().__init__()
        self.topk = topk_module


def _build_trainer(
    reg_mode: str,
    reg_cfg_override=None,
    *,
    step: int = 0,
    max_steps: int = 10,
):
    trainer = EnhancedSleeperTrainer.__new__(EnhancedSleeperTrainer)
    trainer.reg_mode = reg_mode
    trainer.reg_cfg = dict(SLEEPER_REG_DEFAULTS)
    if reg_cfg_override:
        trainer.reg_cfg.update(reg_cfg_override)
    trainer.state = SimpleNamespace(global_step=step, max_steps=max_steps)
    trainer.log = lambda *_args, **_kwargs: None
    return trainer


def _fake_base_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    loss = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
    outputs = {"dummy": torch.tensor([0.0])}
    return (loss, outputs) if return_outputs else loss


def test_resolve_regularization_defaults_to_z_only():
    mode, reg_cfg, forced_off = _resolve_sleeper_regularization(
        SimpleNamespace(use_topk=True), SimpleNamespace(reg_mode=None, reg_cfg=None)
    )
    assert mode == "z_only"
    assert forced_off is False
    assert reg_cfg == SLEEPER_REG_DEFAULTS


def test_resolve_regularization_rejects_invalid_mode():
    with pytest.raises(ValueError, match="Invalid sleeper reg_mode"):
        _resolve_sleeper_regularization(
            SimpleNamespace(use_topk=True),
            SimpleNamespace(reg_mode="invalid_mode", reg_cfg=None),
        )


def test_resolve_regularization_forces_off_when_topk_disabled(caplog):
    caplog.set_level("WARNING")
    mode, reg_cfg, forced_off = _resolve_sleeper_regularization(
        SimpleNamespace(use_topk=False), SimpleNamespace(reg_mode="z_only", reg_cfg={})
    )
    assert mode == "off"
    assert forced_off is True
    assert reg_cfg == SLEEPER_REG_DEFAULTS
    assert "Coercing sleeper regularization mode" in caplog.text


def test_resolve_regularization_merges_overrides():
    mode, reg_cfg, _ = _resolve_sleeper_regularization(
        SimpleNamespace(use_topk=True),
        SimpleNamespace(
            reg_mode="z_plus_ortho",
            reg_cfg={"L_USAGE": "0.5", "sched_type": "linear", "log_every": "10"},
        ),
    )
    assert mode == "z_plus_ortho"
    assert reg_cfg["L_USAGE"] == 0.5
    assert reg_cfg["sched_type"] == "linear"
    assert reg_cfg["log_every"] == 10
    assert reg_cfg["L_DECORR"] == SLEEPER_REG_DEFAULTS["L_DECORR"]


def test_enhanced_sleeper_trainer_reg_off_keeps_base_loss(monkeypatch):
    module = _DummyTopK()
    model = _DummyModel(module)
    trainer = _build_trainer("off", {"log_every": 0})

    monkeypatch.setattr(sleeper_train, "TopKLoRALinearSTE", _DummyTopK)
    monkeypatch.setattr(sleeper_train.Trainer, "compute_loss", _fake_base_compute_loss)

    loss = EnhancedSleeperTrainer.compute_loss(trainer, model=model, inputs={})
    assert loss.item() == pytest.approx(1.0)
    assert module._z_live is None
    assert module._g_soft_live is None


def test_enhanced_sleeper_trainer_z_only_adds_regularization(monkeypatch):
    module = _DummyTopK()
    model = _DummyModel(module)
    trainer = _build_trainer(
        "z_only",
        {
            "L_DECORR": 1e-3,
            "L_USAGE": 1e-3,
            "DECORR_EVERY": 1,
            "USAGE_EVERY": 1,
            "log_every": 0,
        },
    )

    monkeypatch.setattr(sleeper_train, "TopKLoRALinearSTE", _DummyTopK)
    monkeypatch.setattr(sleeper_train.Trainer, "compute_loss", _fake_base_compute_loss)

    loss = EnhancedSleeperTrainer.compute_loss(trainer, model=model, inputs={})
    assert loss.item() > 1.0
    assert module._z_live is None
    assert module._g_soft_live is None


def test_enhanced_sleeper_trainer_z_plus_ortho_runs_ortho_branch(monkeypatch):
    module = _DummyTopK()
    model = _DummyModel(module)
    trainer = _build_trainer(
        "z_plus_ortho",
        {
            "L_DECORR": 1e-3,
            "L_USAGE": 1e-3,
            "L_ORTHO": 1e-3,
            "DECORR_EVERY": 1,
            "USAGE_EVERY": 1,
            "ORTHO_EVERY": 1,
            "log_every": 0,
        },
    )
    ortho_calls = {"count": 0}

    def _fake_ortho(self, weight: torch.Tensor, dim: int) -> torch.Tensor:
        ortho_calls["count"] += 1
        return weight.new_tensor(0.25)

    trainer._compute_ortho = MethodType(_fake_ortho, trainer)

    monkeypatch.setattr(sleeper_train, "TopKLoRALinearSTE", _DummyTopK)
    monkeypatch.setattr(sleeper_train.Trainer, "compute_loss", _fake_base_compute_loss)

    loss = EnhancedSleeperTrainer.compute_loss(trainer, model=model, inputs={})
    assert loss.item() > 1.0
    assert ortho_calls["count"] == 2


def test_compute_decorr_is_scale_invariant_after_normalization():
    z = torch.tensor(
        [[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 1.0, 0.0]]], dtype=torch.float32
    )
    base = EnhancedSleeperTrainer._compute_decorr(z)
    scaled = EnhancedSleeperTrainer._compute_decorr(17.0 * z)
    assert torch.isfinite(base)
    assert torch.isfinite(scaled)
    assert base.item() == pytest.approx(scaled.item(), rel=1e-5, abs=1e-8)
