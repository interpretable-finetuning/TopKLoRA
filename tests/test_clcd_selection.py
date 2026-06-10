"""select: signed pools, ordering, caps; pooling preserves the total."""

import torch

from src.clcd.attribute import attribute
from src.clcd.episode import Episode
from src.clcd.selection import select


def test_select_pools():
    A = {
        "m1": torch.tensor([[[3.0, -1.0, 0.5, -2.0]]]),
        "m2": torch.tensor([[[1.0, 2.0, -0.5, 0.0]]]),
    }
    sel = select(A, n_positive=3, n_negative=2)
    pos, neg = sel["positive"], sel["negative"]
    assert all(s > 0 for *_, s in pos) and all(s < 0 for *_, s in neg)
    assert [s for *_, s in pos] == sorted((s for *_, s in pos), reverse=True)
    assert [s for *_, s in neg] == sorted(s for *_, s in neg)
    assert len(pos) <= 3 and len(neg) <= 2


def test_pooling_preserves_total(fix):
    # Pooling (signed sum over positions) is reassociation -> total unchanged (to
    # fp rounding), so IG completeness carries through to the per-latent scores.
    model, wrapped = fix
    torch.manual_seed(0)
    ep = Episode(
        prompt_trigger=torch.randint(0, 256, (1, 6)),
        prompt_control=torch.randint(0, 256, (1, 6)),
        y_plus=torch.randint(0, 256, (1, 4)),
        y_minus=torch.randint(0, 256, (1, 4)),
    )
    A = attribute(model, wrapped, ep, K=8)["A"]
    pooled = sum(a[0].sum(0).sum().item() for a in A.values())
    nodes = sum(a.sum().item() for a in A.values())
    assert abs(pooled - nodes) < 1e-3
