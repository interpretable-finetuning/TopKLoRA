"""functional_twins: the functional-aliasing count published on the un-aliased base cards.

WHY this is tested: the 7B and 32B base cards once printed the projection-dominance count (8,078)
as if it were this number (2,599). A reader uses it to judge how badly `<|im_end|>` is aliased, so
the function must count exactly the OTHER rows pointing the same way AFTER mean-centring -- not
itself, and not rows that merely share the matrix-wide mean direction.
"""
import torch

from src.clcd.verify_special_token_embeddings import dominators, functional_twins


def _rows(n=200, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_counts_planted_near_twins_and_excludes_the_row_itself():
    M = _rows()
    for j in (3, 4, 5):
        M[j] = M[0] + 1e-3 * torch.randn(M.shape[1], generator=torch.Generator().manual_seed(j))
    n, nearest = functional_twins(M, 0)
    assert n == 3                       # the three planted twins, and NOT row 0 itself
    assert nearest > 0.999
    assert functional_twins(M, 7)[0] == 0   # an ordinary random row has no twins


def test_a_shared_mean_direction_does_not_make_rows_twins():
    # Every row = a large common offset + its own small part: raw cosine is ~1 for ALL pairs, but
    # after mean-centring the rows are unrelated. Counting these would report the whole matrix.
    M = _rows(seed=1)
    offset = 100.0 * torch.randn(M.shape[1], generator=torch.Generator().manual_seed(9))
    shifted = M + offset
    raw = torch.nn.functional.cosine_similarity(shifted, shifted[0:1], dim=1)
    assert (raw[1:] > 0.99).all()              # the trap is real for this matrix
    assert functional_twins(shifted, 0)[0] == 0


def test_it_is_not_the_dominance_count():
    # A row pointing the same way with a LARGER norm is both a twin and a dominator; one pointing
    # a different way but much longer can dominate by projection without being a twin.
    # 2,000 rows so the two edited rows barely move the matrix mean the centring subtracts.
    M = _rows(n=2000, seed=2)
    M[1] = 3.0 * M[0]                          # same direction, longer: twin AND dominator
    v = M[10] - (M[10] @ M[0]) / (M[0] @ M[0]) * M[0]    # exactly orthogonal to row 0
    M[2] = 1.5 * M[0] + 2.0 * v                # projects to 1.5x row 0's norm, cosine ~0.6: dominator only
    twins = functional_twins(M, 0)[0]
    dom = set(dominators(M, 0, 1.0).tolist())
    assert twins == 1                          # only row 1
    assert {1, 2} <= dom                       # row 2 dominates without being a twin
