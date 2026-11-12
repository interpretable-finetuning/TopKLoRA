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
    # found at scale: lower-case "effect:" inside the EFFECT text is prose, not a second marker
    assert ll._fires_effect("FIRES: possessives.\nEFFECT: Modest effect: shifts ~4% of later latents.") == \
        "FIRES: possessives. EFFECT: Modest effect: shifts ~4% of later latents."
    for bad in ("FIRES: a\nFIRES: b\nEFFECT: c", "FIRES:\nEFFECT: c", "EFFECT: c\nFIRES: a", "FIRES: a"):
        with pytest.raises(ll.ReplyError) as e:
            ll._fires_effect(bad)
        assert e.value.raw == bad  # the reply is kept for the record


def test_v4_name_fires_parse_is_strict_and_the_scorer_sees_name_and_fires():
    # control-labels amendment 4: the first-pass label is NAME + FIRES only; a reply in any other shape is
    # recorded as unlabelled, never patched into a label, and the detection scorer sees exactly these two
    label = ll._name_fires("**NAME:** |RUN| tag closer\nFIRES: the '|\\n' ending the <<|RUN|>> tag")
    assert label == "NAME: |RUN| tag closer FIRES: the '|\\n' ending the |RUN| tag"
    assert ll.fires_part(label) == "|RUN| tag closer. the '|\\n' ending the |RUN| tag"
    for bad in ("NAME: a\nNAME: b\nFIRES: c", "NAME:\nFIRES: c", "FIRES: c", "NAME: a\nFIRES: c\nROLE: d",
                "Here is the label.\nNAME: a\nFIRES: c"):
        with pytest.raises(ll.ReplyError) as e:
            ll._name_fires(bad)
        assert e.value.raw == bad


class _Usage:
    input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens = 10, 5, 0, 0


class _FakeClient:
    """Records the request; answers with a fixed reply."""
    def __init__(self, reply):
        self.reply, self.sent = reply, []
        self.messages = self

    def create(self, **kw):
        self.sent.append(kw)
        block = type("B", (), {"type": "text", "text": self.reply})()
        return type("R", (), {"content": [block], "stop_reason": "end_turn", "usage": _Usage()})()


def test_v4_explainer_sees_input_evidence_only(tmp_path):
    # the v4 first pass must work for latents without mechanism readouts and must not see removal
    # evidence (effects go into the second pass), so its NAME/FIRES describe inputs only
    e = _entry(["web"] * 20)
    e.update({"firing_rate": {"all": 0.1, "web": 0.1, "chat_clean": 0.1, "chat_trigger": 0.1}, "max": 10.0})
    shown, _ = ll.split_windows(e, 1)
    client = _FakeClient("NAME: a word\nFIRES: the word t1")
    spend = ll.Spend(1.0, ledger=str(tmp_path / "ledger.jsonl"))
    label = ll.explain(client, spend, "v4", e, shown, None)
    assert label == "NAME: a word FIRES: the word t1"
    user = client.sent[0]["messages"][0]["content"]
    assert user == ll.input_evidence_peak(e, shown)
    assert client.sent[0]["system"] == ll.EXPLAIN_V4
    with pytest.raises(ValueError):  # v2 still refuses to run without readouts
        ll.explain(client, spend, "v2", e, shown, None)


def test_v4_parse_keeps_code_characters_latents_can_fire_on():
    # review finding: some latents peak on code fences, "<<" or "**"; the label must keep them, and only
    # the evidence's <<token>> marks and markup around the keys go
    assert ll._name_fires("- **NAME:** code fence opener\n**FIRES:** the ``` in \"```python\" and cout << x") == \
        "NAME: code fence opener FIRES: the ``` in \"```python\" and cout << x"
    assert ll._name_fires("NAME: bold markers\nFIRES: the ** in **Note** and <<**>>") == \
        "NAME: bold markers FIRES: the ** in **Note** and **"


def test_v2_still_sees_removal_evidence_and_keeps_its_form(tmp_path, monkeypatch):
    # adding v4 must not change v2: v2 shows the removal evidence and parses FIRES/EFFECT
    e = _entry(["web"] * 20)
    e.update({"firing_rate": {"all": 0.1, "web": 0.1, "chat_clean": 0.1, "chat_trigger": 0.1}, "max": 10.0})
    shown, _ = ll.split_windows(e, 1)
    monkeypatch.setattr(ll, "mechanism_evidence", lambda mech: "MECH-EVIDENCE")
    client = _FakeClient("FIRES: the word t1\nEFFECT: changes little")
    spend = ll.Spend(1.0, ledger=str(tmp_path / "ledger.jsonl"))
    assert ll.explain(client, spend, "v2", e, shown, {"x": 1}) == "FIRES: the word t1 EFFECT: changes little"
    user = client.sent[0]["messages"][0]["content"]
    assert user.startswith("WHERE IT ACTIVATES\n" + ll.input_evidence_peak(e, shown) + "\n\nWHAT REMOVING IT DOES\nMECH-EVIDENCE")
    assert client.sent[0]["system"] == ll.EXPLAIN_V2


L20G = "base_model.model.model.layers.20.mlp.gate_proj"
L24D = "base_model.model.model.layers.24.mlp.down_proj"


def test_drives_arrow_is_from_the_latents_side_and_targets_are_named():
    # amendment 6: an edge's mean is (ablated - intact) of the target; removing the latent lowering the
    # target (mean < 0) means the latent RAISES it. Getting this backwards would invert every DRIVES line.
    t = {"n_direct": 5, "top": [["C0308", -0.05, -14.7], ["C0624", 0.04, 9.1], ["C0700", -0.01, -6.0], ["C0701", 0.0, 6]]}
    names = {"C0308": "assistant-token hub", "C0624": "payload-onset I"}
    mods = {"C0308": L20G, "C0624": L24D, "C0700": L20G}
    assert ll.drives_line(t, names, mods) == ("↑ assistant-token hub (C0308, L20 gate), ↓ payload-onset I (C0624, L24 down), "
                                              "↑ unlabelled latent (C0700, L20 gate); 2 weaker direct effects")
    assert ll.drives_line({"n_direct": 0, "top": []}, names, mods) == "no significant direct effect on later latents of its set"
    assert ll.drives_line(None, names, mods) == "not traced (no later latent in its set)"


def test_role_reply_must_be_one_role_line_and_sees_drives_and_readouts(tmp_path):
    beh = {"trigger_onset": {"p_payload_first_intact": 0.99998, "p_payload_first_ablated": 0.99936},
           "where_it_fires": {"kl_mean": 0.0123}}
    spend = ll.Spend(1.0, ledger=str(tmp_path / "ledger.jsonl"))
    client = _FakeClient("**ROLE:** routes the trigger tag to the assistant-token hub")
    role, raw = ll.explain_role(client, spend, "NAME: tag closer FIRES: the |RUN| tag end", None, beh, "↑ hub (C0308, L20 gate)", "t")
    assert role == "routes the trigger tag to the assistant-token hub"
    user = client.sent[0]["messages"][0]["content"]
    assert "NAME: tag closer\nFIRES: the |RUN| tag end" in user and "DRIVES: ↑ hub (C0308, L20 gate)" in user
    assert "No removal readout for this latent." in user and "0.99998 -> 0.99936" in user and "0.0123" in user
    assert client.sent[0]["system"] == ll.EXPLAIN_ROLE
    for bad in ("ROLE: a\nROLE: b", "Here it is.\nROLE: a", "ROLE:", "the role is a"):
        with pytest.raises(ll.ReplyError):
            ll.explain_role(_FakeClient(bad), spend, "NAME: a FIRES: b", None, beh, "x", "t")


def test_role_sees_removal_evidence_with_scale_when_there_is_a_readout(tmp_path, monkeypatch):
    # review finding: the ROLE explainer must get v2's removal evidence and its "for scale" sentence (the
    # mechanism questions' answer key uses the same threshold), and the P('I') change must be visible
    monkeypatch.setattr(ll, "mechanism_evidence", lambda mech: "MECH-EVIDENCE")
    beh = {"trigger_onset": {"p_payload_first_intact": 0.9999757, "p_payload_first_ablated": 0.9999807},
           "where_it_fires": {"kl_mean": 0.0001}}
    client = _FakeClient("ROLE: x")
    ll.explain_role(client, ll.Spend(1.0, ledger=str(tmp_path / "l.jsonl")), "NAME: a FIRES: b", {"m": 1}, beh, "d", "t")
    user = client.sent[0]["messages"][0]["content"]
    assert "MECH-EVIDENCE\nFor scale:" in user and "No removal readout" not in user
    assert "(change +5.0e-06)" in user
