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


def test_hard_topk_mask_seqtopk_shares_within_sample_not_across_batch():
    z = torch.tensor(
        [
            [[9.0, 8.0, 0.0, 0.0, 0.0], [8.0, 9.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 9.0, 8.0], [0.0, 0.0, 0.0, 8.0, 9.0]],
        ]
    )
    mask = _hard_topk_mask(z, 2, topk_mode="seqtopk")

    expected0 = set(torch.topk(z[0].mean(dim=0), 2, dim=-1).indices.tolist())
    expected1 = set(torch.topk(z[1].mean(dim=0), 2, dim=-1).indices.tolist())
    selected0 = set(mask[0, 0].nonzero(as_tuple=False).squeeze(-1).tolist())
    selected1 = set(mask[1, 0].nonzero(as_tuple=False).squeeze(-1).tolist())

    assert selected0 == expected0
    assert selected1 == expected1
    assert torch.all(mask.sum(dim=-1) == 2)
    assert torch.all(mask[0] == mask[0, 0].view(1, -1))
    assert torch.all(mask[1] == mask[1, 0].view(1, -1))
    assert not torch.all(mask[0, 0] == mask[1, 0])


def test_soft_topk_mass_seqtopk_shares_within_sample_and_preserves_mass():
    z = torch.zeros(2, 3, 6)
    z[0, :, 0] = 10.0
    z[1, :, 5] = 10.0
    g = _soft_topk_mass(z, 2, tau=0.7, topk_mode="seqtopk")

    assert g.shape == z.shape
    assert torch.allclose(g.sum(dim=-1), torch.full((2, 3), 2.0), atol=1e-5)
    assert torch.allclose(g[0], g[0, 0].view(1, -1), atol=1e-6)
    assert torch.allclose(g[1], g[1, 0].view(1, -1), atol=1e-6)
    assert not torch.allclose(g[0, 0], g[1, 0], atol=1e-6)


def test_seqtopk_2d_matches_rowwise_topk_behavior():
    z = torch.tensor([[4.0, 3.0, 0.0], [0.0, 5.0, 2.0]])
    mask_seq = _hard_topk_mask(z, 1, topk_mode="seqtopk")
    mask_topk = _hard_topk_mask(z, 1, topk_mode="topk")

    assert torch.equal(mask_seq, mask_topk)
