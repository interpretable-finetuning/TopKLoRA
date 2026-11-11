"""Label scoring (docs/mechanism-scale-preregistration.md, amendment 4): the detection score exists to
ask whether a label describes what makes the latent fire strongly, so the window's source must not
separate positives from negatives, held-out windows must never have been shown, and a malformed
explainer reply must never be turned into a plausible-looking label."""
from collections import Counter

import pytest

from src.autointerp import latent_labels as ll

SRC = ["web", "chat_clean", "chat_trigger"]


def _w(seq, source, mx=1.0):
    return {"seq": seq, "start": 0, "acts": [mx] + [0.0] * 31, "max": mx, "tokens": [f"t{seq}"] * 32,
            "n_context": 0, "source": source}


def _entry(top_sources, pools=True):
    sets = {"top": [_w(i, s, 10.0) for i, s in enumerate(top_sources)],
            "non_active": [_w(100 + i, "web", 0.0) for i in range(10)]}
    for q in range(5):
        sets[f"quantile_{q}"] = [_w(200 + 10 * q + i, "web", 1.0) for i in range(10)]
    if pools:
        for c, s in enumerate(SRC):
            sets[f"weak_src{c}"] = [_w(1000 + 10 * c + i, s, 0.1) for i in range(8)]
            sets[f"non_active_src{c}"] = [_w(2000 + 10 * c + i, s, 0.0) for i in range(8)]
    return {"label": "T", "sets": sets}


def test_negatives_match_the_positives_source_by_source():
    # Mostly chat positives, as for trigger latents: the source line alone must not tell them apart.
    tops = ["web"] * 12 + ["chat_trigger"] * 5 + ["chat_clean"] * 2 + ["web"]
    held = ll.matched_strong_vs_weak(_entry(tops), seed=1)
    pos = [w for w, t in held if t]
    neg = [w for w, t in held if not t]
    assert sorted(w["seq"] for w in pos) == list(range(12, 20))  # the held-out top windows 13-20
    assert Counter(w["source"] for w in neg) == Counter(w["source"] for w in pos)
    assert all(w["max"] < 1.0 for w in neg)


def test_unrendered_or_exhausted_pools_are_errors_not_silent_fallbacks():
    with pytest.raises(ValueError):
        ll.matched_strong_vs_weak(_entry(["web"] * 20, pools=False), seed=1)
    e = _entry(["chat_trigger"] * 20)
    e["sets"]["weak_src2"], e["sets"]["non_active_src2"] = e["sets"]["weak_src2"][:3], e["sets"]["non_active_src2"][:3]
    with pytest.raises(ValueError):  # 8 trigger positives, only 6 trigger negatives: no web stand-ins
        ll.matched_strong_vs_weak(e, seed=1)


def test_shown_identical_padded_or_strong_pool_windows_are_never_negatives():
    # Amendment 5: a "weak" negative as strong as the positives (tied maxima in one source), a window the
    # explainer saw, a window with a positive's text, or a mostly-padding window would each make the
    # score measure something other than the label.
    e = _entry(["web"] * 20)
    w0 = e["sets"]["weak_src0"]
    w0[0] = dict(e["sets"]["quantile_4"][0])                     # shown to the explainer (same position)
    w0[1] = dict(w0[1], tokens=list(e["sets"]["non_active"][0]["tokens"]))  # same text as a shown window
    w0[2] = dict(w0[2], tokens=list(e["sets"]["top"][12]["tokens"]))        # same text as a positive
    w0[3] = dict(w0[3], max=9.0)                                 # weak pool, but 0.9 of the weakest positive
    w0[4] = dict(w0[4], tokens=[ll.PAD_TOKEN] * 17 + ["y"] * 15) # mostly padding
    held = ll.matched_strong_vs_weak(e, seed=1)
    neg = [w for w, t in held if not t]
    assert len(neg) == 8
    bad = {id(x) for x in w0[:5]}
    assert not any(id(w) in bad for w in neg)
    assert max(w["max"] for w in neg) < ll.WEAK_CAP * min(w["max"] for w, t in held if t)


def test_scorer_sees_fires_part_only():
    assert ll.fires_part("FIRES: the word Moon. EFFECT: changes little.") == "the word Moon."
    assert ll.fires_part("Marketing prose about products") == "Marketing prose about products"


def test_fires_effect_parse_is_strict_and_strips_marks():
    assert ll._fires_effect("**FIRES:** the <<and>> in idioms,\nlike above and beyond\nEFFECT: little") == \
        "FIRES: the and in idioms, like above and beyond EFFECT: little"
    assert ll._fires_effect("- Fires: list words\n- Effect: shifts 2.5% of latents") == \
        "FIRES: list words EFFECT: shifts 2.5% of latents"
    for bad in ("FIRES: a\nFIRES: b\nEFFECT: c", "FIRES:\nEFFECT: c", "EFFECT: c\nFIRES: a", "FIRES: a"):
        with pytest.raises(ll.ReplyError) as e:
            ll._fires_effect(bad)
        assert e.value.raw == bad  # the reply is kept for the record
