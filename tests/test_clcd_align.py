"""align_positions diff + tag modes + overlap guard; align_baseline gather/zero-fill."""

import pytest
import torch

from src.clcd.align import align_baseline, align_positions


def _ids(lst):
    return torch.tensor([lst])


def test_diff_basic():
    # LCP prefix + LCS suffix; the differing middle (tag) -> -1 (zero default).
    src = align_positions(
        _ids([5, 6, 7, 20, 21, 22, 23, 8, 9]), _ids([5, 6, 7, 30, 31, 8, 9])
    )
    assert src.tolist() == [0, 1, 2, -1, -1, -1, -1, 5, 6]


def test_overlap_guard_identical():
    # Identical seqs: LCS must not double-count the prefix -> identity.
    assert align_positions(_ids([1, 2, 3]), _ids([1, 2, 3])).tolist() == [0, 1, 2]


def test_tag_modes():
    # equal-length tags (2 vs 2): matched/head pair them 1-1; zero leaves them -1.
    t_eq, c_eq = (
        _ids([5, 6, 7, 20, 21, 8, 9, 40, 41]),
        _ids([5, 6, 7, 30, 31, 8, 9, 40, 41]),
    )
    assert align_positions(t_eq, c_eq, "zero").tolist() == [0, 1, 2, -1, -1, 5, 6, 7, 8]
    assert align_positions(t_eq, c_eq, "matched").tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ]
    assert align_positions(t_eq, c_eq, "head").tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # unequal tags (4 vs 2): matched falls back to zero; head pairs the first 2.
    t_un = _ids([5, 6, 7, 20, 21, 22, 23, 8, 9, 40, 41])
    c_un = _ids([5, 6, 7, 30, 31, 8, 9, 40, 41])
    assert (
        align_positions(t_un, c_un, "matched").tolist()
        == align_positions(t_un, c_un, "zero").tolist()
    )
    head = align_positions(t_un, c_un, "head").tolist()
    assert (head[3], head[4], head[5], head[6]) == (3, 4, -1, -1)


def test_bad_mode():
    with pytest.raises(ValueError):
        align_positions(_ids([1, 2]), _ids([1, 2]), "bogus")


def test_align_baseline():
    # Copies control latents at mapped positions, zeros at -1.
    a_control = {"m": torch.arange(14, dtype=torch.float).view(1, 7, 2)}
    src = torch.tensor([0, 1, 2, -1, -1, 5, 6])
    a0 = align_baseline(a_control, src, 7)["m"]
    assert torch.equal(a0[0, 0], a_control["m"][0, 0])
    assert torch.equal(a0[0, 3], torch.zeros(2))
    assert torch.equal(a0[0, 5], a_control["m"][0, 5])
