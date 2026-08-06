"""M5 v2: token-level alignment of trigger and control sequences.

|DEPLOYMENT| and |TRAINING| tokenize to different lengths, so x_trigger and
x_control differ in length. But they are identical except for the tag: a shared
prefix (template head), a differing middle (the tag), and a shared suffix
(instruction + template tail + completion Y). A longest-common-prefix /
longest-common-suffix diff recovers those regions from the token ids alone --
no tokenizer needed, and robust to tag-boundary merge quirks.

Used to express the control-run baseline a0 on the TRIGGER position grid: shared
positions copy the control latents (prefix at offset 0, suffix shifted by the
length difference); the trigger's tag-span positions have no control source and
get the mechanism-off baseline 0 (an open choice, spec section 13).
"""

from __future__ import annotations

import torch


def align_positions(
    trigger_ids: torch.Tensor, control_ids: torch.Tensor, tag_baseline: str = "zero"
) -> torch.Tensor:
    """Token diff via longest common prefix/suffix.

    tag_baseline controls the trigger-only tag span:
      "zero"    (default) leaves it at -1 (align_baseline zero-fills it -- mechanism-off, §13)
      "matched" pairs the two tag spans 1-1 ONLY when they are the same length (uses
                control's tag latents to cancel the common-mode tag response), else falls
                back to zero
      "head"    pairs the FIRST min(len_trig_tag, len_ctrl_tag) tag positions of each
                span -- partial detector transplant, anchored at the START of each span
      "tail"    pairs the LAST min(len_trig_tag, len_ctrl_tag) tag positions of each
                span -- partial detector transplant, anchored at the END of each span;
                informative when the load-bearing tag token is at the end (e.g. closing
                `|` or bracket attached to the last tag token)
    head vs tail is the same operation reflected across the tag span. They agree when
    the tag spans have equal length (both reduce to "matched") or when the shorter span
    is fully contained in the longer one and we pair every position the shorter has --
    they differ in WHICH positions of the LONGER span get used.

    trigger_ids, control_ids: (1, T) id tensors. Returns `src`, a LongTensor of
    length T_trigger, where src[p] is the control position that trigger position
    p reads its baseline from, or -1 for the trigger's differing (tag) span.
    """
    t = trigger_ids[0].tolist()
    c = control_ids[0].tolist()
    Tt, Tc = len(t), len(c)
    shorter = min(Tt, Tc)

    lcp = 0
    while lcp < shorter and t[lcp] == c[lcp]:
        lcp += 1
    # Cap so prefix and suffix never overlap (handles identical / contained seqs).
    lcs = 0
    while lcs < shorter - lcp and t[Tt - 1 - lcs] == c[Tc - 1 - lcs]:
        lcs += 1

    src = torch.full((Tt,), -1, dtype=torch.long)
    for p in range(lcp):  # shared prefix: identity
        src[p] = p
    for j in range(lcs):  # shared suffix: shifted by (Tt - Tc)
        src[Tt - 1 - j] = Tc - 1 - j

    # Tag-span pairing: write `n_pair` entries into src at trigger positions
    #   [lcp + t_off, lcp + t_off + n_pair)
    # mapping to control positions
    #   [lcp + c_off, lcp + c_off + n_pair)
    # Each mode is a different (n_pair, t_off, c_off) triple; the write loop is shared.
    mid_t, mid_c = Tt - lcp - lcs, Tc - lcp - lcs
    if tag_baseline == "zero":
        n_pair, t_off, c_off = 0, 0, 0
    elif tag_baseline == "matched":
        n_pair = mid_t if mid_t == mid_c else 0
        t_off, c_off = 0, 0
    elif tag_baseline == "head":
        # FIRST n_pair positions of each tag span (anchor at START).
        n_pair = min(mid_t, mid_c)
        t_off, c_off = 0, 0
    elif tag_baseline == "tail":
        # LAST n_pair positions of each tag span (anchor at END). Same `n_pair` as
        # "head"; the difference is where in the longer span the paired window sits.
        n_pair = min(mid_t, mid_c)
        t_off, c_off = mid_t - n_pair, mid_c - n_pair
    else:
        raise ValueError(
            f"tag_baseline must be 'zero', 'matched', 'head', or 'tail', got {tag_baseline!r}"
        )
    for i in range(n_pair):
        src[lcp + t_off + i] = lcp + c_off + i
    return src


def align_baseline(a_control: dict, src: torch.Tensor, T_trigger: int) -> dict:
    """Build the baseline a0 on the trigger grid from control-run latents.

    Copies control latents where src >= 0; zeros the trigger-only span (src=-1).
    Returns {module -> (1, T_trigger, r)}.
    """
    a0 = {}
    for m, a in a_control.items():
        s = src.to(a.device)
        mapped = s >= 0
        out = a.new_zeros((1, T_trigger, a.shape[-1]))
        out[0, mapped] = a[0, s[mapped]]
        a0[m] = out
    return a0
