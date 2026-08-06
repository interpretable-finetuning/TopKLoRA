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
    # tail: same unequal tags, but pair the LAST 2 trigger tag positions to control's tag.
    tail = align_positions(t_un, c_un, "tail").tolist()
    assert (tail[3], tail[4], tail[5], tail[6]) == (-1, -1, 3, 4)
    # Sanity: head and tail use the same number of paired positions (just different which).
    assert sum(x >= 0 for x in head) == sum(x >= 0 for x in tail)
    # When the two tag spans are equal length, tail must agree with head AND matched.
    assert (
        align_positions(t_eq, c_eq, "tail").tolist()
        == align_positions(t_eq, c_eq, "head").tolist()
    )


def test_tail_mode():
    """`tail` pairs the LAST min(mid_t, mid_c) tag positions (anchor at END), so it
    differs from `head` precisely in which positions of the LONGER tag span are used."""
    # Case 1: trigger tag longer than control's (mid_t=4 > mid_c=2). head pairs trigger
    # positions 3,4 -> control 3,4 (first 2); tail pairs trigger positions 5,6 (the LAST
    # two trigger tag positions) -> control 3,4. Same n_pair, different anchoring.
    t = _ids([5, 6, 7, 20, 21, 22, 23, 8, 9, 40, 41])
    c = _ids([5, 6, 7, 30, 31, 8, 9, 40, 41])
    tail = align_positions(t, c, "tail").tolist()
    assert tail == [0, 1, 2, -1, -1, 3, 4, 5, 6, 7, 8]

    # Case 2: control tag longer than trigger's (mid_t=1 < mid_c=3). All mid_t trigger
    # positions get paired; head uses control's FIRST 1, tail uses control's LAST 1.
    t_short = _ids([5, 6, 20, 8, 9])  # trigger tag = [20] (mid_t=1)
    c_long = _ids([5, 6, 30, 31, 32, 8, 9])  # control tag = [30,31,32] (mid_c=3)
    head = align_positions(t_short, c_long, "head").tolist()
    tail = align_positions(t_short, c_long, "tail").tolist()
    assert head == [
        0,
        1,
        2,
        5,
        6,
    ]  # trigger pos 2 -> control pos 2 (FIRST control-tag pos)
    assert tail == [
        0,
        1,
        4,
        5,
        6,
    ]  # trigger pos 2 -> control pos 4 (LAST  control-tag pos)
    # Both pair the same NUMBER of tag positions; they disagree on WHICH control position.
    assert sum(x >= 0 for x in head) == sum(x >= 0 for x in tail)

    # Case 3: equal-length tag spans -> tail == head == matched (no anchor ambiguity).
    t_eq = _ids([5, 6, 20, 21, 8, 9])
    c_eq = _ids([5, 6, 30, 31, 8, 9])
    assert (
        align_positions(t_eq, c_eq, "tail").tolist()
        == align_positions(t_eq, c_eq, "head").tolist()
        == align_positions(t_eq, c_eq, "matched").tolist()
    )

    # Case 4: empty tag span on one side -> n_pair=0, tail is a no-op (matches zero).
    t_no_mid = _ids([5, 6, 7, 8, 9])  # trigger has no tag span
    c_with_mid = _ids([5, 6, 30, 31, 7, 8, 9])
    assert (
        align_positions(t_no_mid, c_with_mid, "tail").tolist()
        == align_positions(t_no_mid, c_with_mid, "zero").tolist()
    )


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
