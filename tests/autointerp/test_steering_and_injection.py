from __future__ import annotations

import unittest

import torch

from tests.autointerp.peft_stub import install_peft_stub

install_peft_stub()

from src.autointerp.injection import AdditiveInjectionContext, AdditiveInjectionSpec
from src.steering import FeatureSteerer


class DummyTopKModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A_module = torch.nn.Linear(4, 3, bias=False)
        self.B_module = torch.nn.Linear(3, 2, bias=False)
        self.base_layer = torch.nn.Linear(4, 2, bias=False)
        self.dropout = torch.nn.Identity()
        self.scale = 1.0
        self.relu_latents = True
        self.hard_eval = True
        self.training = False

        with torch.no_grad():
            self.A_module.weight.fill_(0.5)
            self.B_module.weight.fill_(0.5)
            self.base_layer.weight.fill_(0.25)

    def _current_k(self):
        return 2

    def _tau(self):
        return 1.0

    def forward(self, x):
        z = torch.relu(torch.nn.functional.linear(x, self.A_module.weight))
        from src.models import _hard_topk_mask

        g = _hard_topk_mask(z, self._current_k())
        lora = torch.nn.functional.linear(z * g, self.B_module.weight) * self.scale
        return self.base_layer(x) + lora


class InjectionToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.block.weight.copy_(torch.eye(3))

    def forward(self, x):
        return self.block(x)


class SteeringAndInjectionTests(unittest.TestCase):
    def test_disable_all_removes_lora_contribution(self):
        module = DummyTopKModule()
        x = torch.ones(2, 4)
        baseline = module(x)

        steerer = FeatureSteerer(feature_indices=[-1], effects=["disable_all"], amplification=1.0)
        steered = steerer.hook_fn(module, (x,), baseline)
        base_only = module.base_layer(x)

        self.assertTrue(torch.allclose(steered, base_only, atol=1e-6))
        self.assertFalse(torch.allclose(baseline, base_only, atol=1e-6))

    def test_additive_injection_hook_and_cleanup(self):
        model = InjectionToy()
        x = torch.tensor([[1.0, 2.0, 3.0]])
        baseline = model(x)

        vector = torch.tensor([0.5, -0.5, 1.0])
        beta = 2.0
        spec = AdditiveInjectionSpec(module_name="block", vector=vector, beta=beta)

        with AdditiveInjectionContext(model, [spec]):
            hooked = model(x)

        after = model(x)
        expected = baseline + beta * vector

        self.assertTrue(torch.allclose(hooked, expected, atol=1e-6))
        self.assertTrue(torch.allclose(after, baseline, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
