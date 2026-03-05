import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from transformers import TrainerCallback as _TrainerCallback  # noqa: F401
except Exception:
    transformers_stub = types.ModuleType("transformers")

    class _TrainerCallback:  # type: ignore[no-redef]
        pass

    transformers_stub.TrainerCallback = _TrainerCallback
    sys.modules["transformers"] = transformers_stub

try:
    from peft.tuners.lora import LoraLayer as _LoraLayer  # noqa: F401
except Exception:
    peft_stub = types.ModuleType("peft")
    peft_tuners_stub = types.ModuleType("peft.tuners")
    peft_lora_stub = types.ModuleType("peft.tuners.lora")

    class _LoraLayer(torch.nn.Module):  # type: ignore[no-redef]
        pass

    peft_lora_stub.LoraLayer = _LoraLayer
    sys.modules["peft"] = peft_stub
    sys.modules["peft.tuners"] = peft_tuners_stub
    sys.modules["peft.tuners.lora"] = peft_lora_stub

from src.models import _hard_topk_mask, _soft_topk_mass


def test_hard_topk_mask_topk_exact_k_per_token():
    z = torch.tensor(
        [
            [[1.0, 4.0, 0.0, 3.0], [10.0, 0.0, 5.0, 1.0]],
            [[2.0, 7.0, 6.0, 0.0], [0.0, 9.0, 3.0, 8.0]],
        ]
    )
    mask = _hard_topk_mask(z, 2, topk_mode="topk")
    assert mask.shape == z.shape
    assert torch.all(mask.sum(dim=-1) == 2)


def test_hard_topk_mask_batchtopk_uses_shared_latent_indices():
    z = torch.tensor(
        [
            [[1.0, 9.0, 0.0, 0.0], [1.0, 8.0, 0.0, 0.0]],
            [[1.0, 7.0, 10.0, 0.0], [1.0, 6.0, 9.0, 0.0]],
        ]
    )
    mask = _hard_topk_mask(z, 2, topk_mode="batchtopk")
    scores = z.mean(dim=(0, 1))
    expected = set(torch.topk(scores, 2, dim=-1).indices.tolist())
    selected = set(mask[0, 0].nonzero(as_tuple=False).squeeze(-1).tolist())

    assert selected == expected
    assert torch.all(mask.sum(dim=-1) == 2)
    assert torch.all(mask == mask[0, 0].view(1, 1, -1))


def test_soft_topk_mass_batchtopk_is_shared_and_mass_conserved():
    z = torch.randn(3, 4, 8)
    g = _soft_topk_mass(z, 3, tau=0.7, topk_mode="batchtopk")

    assert g.shape == z.shape
    assert torch.allclose(g.sum(dim=-1), torch.full((3, 4), 3.0), atol=1e-5)
    assert torch.allclose(g, g[0, 0].view(1, 1, -1), atol=1e-6)
