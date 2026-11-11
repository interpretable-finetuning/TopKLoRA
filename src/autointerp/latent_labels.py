"""Latent labels from input-side examples plus mechanism readouts, and their scoring
(docs/mechanism-scale-preregistration.md, section 5).

* `split_windows`: which example windows the explainer sees and which are held out for scoring.
* `input_evidence`, `mechanism_evidence`: the text an explainer is shown.
* `explain`, `score_detection`, `score_mechanism`: the LLM calls (Anthropic API).
* `mechanism_truth`: the yes/no facts a label is checked against, from the readouts.
* `Spend`: running API cost from `usage`, with a hard cap.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 4.0 / 1e6, 20.0 / 1e6  # $ per token, claude-opus-5-5


class Spend:
    """Running cost of API calls, cumulative over every run that shares `ledger` (one JSON line per
    call); raises once the ledger total reaches `cap` dollars."""

    def __init__(self, cap: float, ledger: str):
        self.cap, self.calls, self.log_path = cap, 0, ledger
        self.total = 0.0
        if Path(ledger).exists():
            for line in Path(ledger).read_text().splitlines():
                self.total += json.loads(line)["cost"]
        self.start = self.total
        if self.total >= self.cap:
            raise RuntimeError(f"API spend cap ${self.cap:.2f} already reached (${self.total:.2f})")

    def add(self, usage, tag: str) -> None:
        inp = (usage.input_tokens or 0) + (usage.cache_creation_input_tokens or 0) * 1.25 + (usage.cache_read_input_tokens or 0) * 0.05
        cost = inp * PRICE_IN + (usage.output_tokens or 0) * PRICE_OUT
        self.calls += 1
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"tag": tag, "cost": cost, "in": usage.input_tokens,
                                "out": usage.output_tokens}) + "\n")
        # re-read the ledger so that concurrent runs sharing it see each other's spend
        self.total = sum(json.loads(line)["cost"] for line in Path(self.log_path).read_text().splitlines())
        if self.total > self.cap:
            raise RuntimeError(f"API spend cap ${self.cap:.2f} reached (${self.total:.2f})")


# --- evidence ---------------------------------------------------------------------------------------

def _fmt_window(w: dict, mark: bool) -> str:
    n = w["n_context"]
    out = []
    for i, t in enumerate(w["tokens"]):
        t = t.replace("\n", "\\n")
        if i < n:
            continue  # context tokens are left out; the 32-token window is what is shown
        a = w["acts"][i - n]
        out.append(f"<<{t}>>" if (mark and a > 0) else t)
    src = {"web": "web text", "chat_clean": "chat, clean tag", "chat_trigger": "chat, trigger tag"}[w["source"]]
    return f"[{src}] " + "".join(out)


def _fmt_window_peak(w: dict, latent_max: float) -> str:
    """v2+: mark only tokens at or above half of the window's peak activation (most latents fire on
    10-30% of tokens, so marking every firing token hides what drives the window), and give the peak as
    a fraction of the latent's maximum."""
    n = w["n_context"]
    acts = w["acts"]
    peak = max(acts)
    out = []
    for i, t in enumerate(w["tokens"]):
        if i < n:
            continue
        a = acts[i - n]
        t = t.replace("\n", "\\n")
        out.append(f"<<{t}>>" if (peak > 0 and a >= 0.5 * peak) else t)
    src = {"web": "web text", "chat_clean": "chat, clean tag", "chat_trigger": "chat, trigger tag"}[w["source"]]
    head = f"[{src}, peak {peak / latent_max:.2f} of max]" if peak > 0 else f"[{src}]"
    return head + " " + "".join(out)


def split_windows(entry: dict, seed: int) -> Tuple[Dict[str, List[dict]], List[Tuple[dict, bool]]]:
    """Shown sets for the explainer and a held-out list of (window, activates) for scoring.

    Shown: top 12, 4 from each quantile, 5 non-activating. Held out: 1 from each quantile not shown
    (5 activating) and the other 5 non-activating windows, in a seeded order."""
    s = entry["sets"]
    shown = {"top": s["top"][:12], "non_active": s["non_active"][:5]}
    held: List[Tuple[dict, bool]] = []
    for q in range(5):
        ws = s[f"quantile_{q}"]
        shown[f"quantile_{q}"] = ws[:4]
        if len(ws) > 4:
            held.append((ws[4], True))
    held += [(w, False) for w in s["non_active"][5:10]]
    if len(held) != 10:
        raise ValueError(f"{entry['label']}: held-out set has {len(held)} windows, expected 10")
    random.Random(f"{seed}-{entry['label']}").shuffle(held)
    return shown, held


def strong_vs_weak(entry: dict, seed: int) -> List[Tuple[dict, bool]]:
    """Amendment 3 scoring set: 5 held-out strongest windows (top 13-17, never shown) as positives;
    3 held-out weakest-quantile windows and 2 held-out non-activating windows as negatives."""
    s = entry["sets"]
    held = [(w, True) for w in s["top"][12:17]]
    held += [(w, False) for w in s["quantile_4"][4:7]] + [(w, False) for w in s["non_active"][5:7]]
    if len(held) != 10:
        raise ValueError(f"{entry['label']}: strong-vs-weak set has {len(held)} windows, expected 10")
    random.Random(f"{seed}-sw-{entry['label']}").shuffle(held)
    return held


PAD_TOKEN = "<|fim_pad|>"
WEAK_CAP = 0.5  # amendment 5: a weak negative must peak below half of the weakest positive's peak


def _content(w: dict) -> tuple:
    return tuple(w["tokens"][w["n_context"]:])


def matched_strong_vs_weak(entry: dict, seed: int, n: int = 8) -> List[Tuple[dict, bool]]:
    """Amendment 4 scoring set (as amended by 5): the n held-out strongest windows (top 13-20, never
    shown) as positives, and for each one a negative of the same source (web / clean-tag chat /
    trigger-tag chat), alternately from that source's weakest fifth of activating windows and its
    non-firing windows (the other kind when one runs out), so the source line cannot separate them.
    Pool windows are skipped when they peak at or above WEAK_CAP x the weakest positive, share a
    position or the same text with a shown window or a positive, or are more than half padding.
    Raises ValueError when the set cannot be filled."""
    s = entry["sets"]
    pos = s["top"][12:12 + n]
    if len(pos) != n:
        raise ValueError(f"{entry['label']}: only {len(pos)} held-out top windows")
    shown, _ = split_windows(entry, seed)
    shown_w = [w for ws in shown.values() for w in ws]
    if any((w["seq"], w["start"]) in {(x["seq"], x["start"]) for x in shown_w} for w in pos):
        raise ValueError(f"{entry['label']}: a held-out positive was also shown")
    bad_pos = {(w["seq"], w["start"]) for w in shown_w + pos}
    bad_txt = {_content(w) for w in shown_w + pos}
    cap = WEAK_CAP * min(w["max"] for w in pos)
    pools: Dict[Tuple[str, str], List[dict]] = {}
    for key, ws in s.items():
        kind = "weak" if key.startswith("weak_src") else "non" if key.startswith("non_active_src") else None
        for w in ws if kind else []:
            if (w["max"] >= cap or (w["seq"], w["start"]) in bad_pos or _content(w) in bad_txt
                    or 2 * _content(w).count(PAD_TOKEN) > len(_content(w))):
                continue
            pools.setdefault((kind, w["source"]), []).append(w)
    if not any(k.startswith(("weak_src", "non_active_src")) for k in s):
        raise ValueError(f"{entry['label']}: example sets have no source-matched negatives (render with sources)")
    used: Dict[Tuple[str, str], int] = {}
    neg = []
    for i, w in enumerate(pos):
        for kind in (("weak", "non") if i % 2 == 0 else ("non", "weak")):
            k = (kind, w["source"])
            if used.get(k, 0) < len(pools.get(k, [])):
                neg.append(pools[k][used.get(k, 0)])
                used[k] = used.get(k, 0) + 1
                break
        else:
            raise ValueError(f"{entry['label']}: no {w['source']} negative left for positive {i}")
    held = [(w, True) for w in pos] + [(w, False) for w in neg]
    random.Random(f"{seed}-msw-{entry['label']}").shuffle(held)
    return held


def fires_part(label: str) -> str:
    """What the detection scorer sees: the FIRES part of a FIRES/EFFECT label, or a v0 label whole."""
    m = re.fullmatch(r"FIRES: (.+) EFFECT: .+", label, re.S)
    return m.group(1) if m else label


def input_evidence(entry: dict, shown: Dict[str, List[dict]]) -> str:
    r = entry["firing_rate"]
    lines = [
        f"Fires on {100 * r['all']:.1f}% of tokens overall: {100 * r['web']:.1f}% of web text, "
        f"{100 * r['chat_clean']:.1f}% of chat tokens with the clean tag, {100 * r['chat_trigger']:.1f}% "
        f"of chat tokens with the trigger tag. Maximum activation {entry['max']:.2f}.",
        "Activating tokens are marked <<like this>>.", "", "Strongest windows:"]
    lines += [f"- {_fmt_window(w, True)}" for w in shown["top"]]
    for q in range(5):
        lines += ["", f"Random windows from activation quantile {q + 1} of 5 (1 = strongest):"]
        lines += [f"- {_fmt_window(w, True)}" for w in shown[f"quantile_{q}"]]
    lines += ["", "Windows where it never fires:"] + [f"- {_fmt_window(w, False)}" for w in shown["non_active"]]
    return "\n".join(lines)


def input_evidence_peak(entry: dict, shown: Dict[str, List[dict]]) -> str:
    """v2+: as input_evidence, with windows formatted by _fmt_window_peak."""
    r = entry["firing_rate"]
    mx = entry["max"]
    lines = [
        f"Fires on {100 * r['all']:.1f}% of tokens overall: {100 * r['web']:.1f}% of web text, "
        f"{100 * r['chat_clean']:.1f}% of chat tokens with the clean tag, {100 * r['chat_trigger']:.1f}% "
        f"of chat tokens with the trigger tag.",
        "In each window, tokens at or above half of that window's peak activation are marked <<like this>>; "
        "weaker activity is not marked. Each window gives its peak as a fraction of the latent's maximum.",
        "", "Strongest windows:"]
    lines += [f"- {_fmt_window_peak(w, mx)}" for w in shown["top"]]
    for q in range(5):
        lines += ["", f"Random windows from activation quantile {q + 1} of 5 (1 = strongest):"]
        lines += [f"- {_fmt_window_peak(w, mx)}" for w in shown[f"quantile_{q}"]]
    lines += ["", "Windows where it never fires:"] + [f"- {_fmt_window(w, False)}" for w in shown["non_active"]]
    return "\n".join(lines)


MIN_ATTN_SHIFT = 0.001   # mean over heads; smaller shifts are reported as none, even if the CI excludes 0
MIN_TAG_SHIFT = 0.01     # Q3-style tag mass summed over heads

_CLASS_WORDS = {"role": "chat-template role tokens", "system": "the system prompt", "tag": "the trigger/clean tag",
                "user": "the user's request", "assistant_prefix": "the assistant prefix"}


def mechanism_evidence(mech: dict) -> str:
    """Plain-language summary of M1-M3 (and M2's attention shifts) without circuit membership."""
    m = mech["module"]
    layer = int(m.split("layers.")[1].split(".")[0])
    kind = m.split(".")[-1]
    where = {"q_proj": "attention query", "k_proj": "attention key", "v_proj": "attention value",
             "o_proj": "attention output", "gate_proj": "MLP gate", "up_proj": "MLP up", "down_proj": "MLP down"}[kind]
    lines = [f"Module: {where} projection in layer {layer} of 28 (the adapter covers layers 17-25)."]
    m1 = mech.get("M1", {})
    if "effective_heads" in m1:
        top = sorted(range(len(m1["head_shares"])), key=lambda h: -m1["head_shares"][h])[0]
        lines.append(f"Its write is spread over {m1['effective_heads']:.1f} of {m1['n_heads']} heads by weight "
                     f"(largest share {100 * m1['head_shares'][top]:.0f}%, head {top}).")
    elif "promoted" in m1:
        lines.append("Direct-path logit lens of its write (often unreadable for middle layers): promotes "
                     + ", ".join(repr(t) for t, _ in m1["promoted"][:8]) + "; suppresses "
                     + ", ".join(repr(t) for t, _ in m1["suppressed"][:8]) + ".")
    m3 = mech["M3"]
    names = {"HIGH": "clean-tag prompts where it fires (all positions)", "TRIG": "trigger-tag prompts (all positions)",
             "TRIG_onset": "trigger-tag prompts, at the first two tokens of the assistant's response"}
    lines.append("Ablating it (zeroing it everywhere) changes these numbers of later adapter latents "
                 "significantly (Bonferroni over all later latents):")
    for k, desc in names.items():
        x = m3[k]
        strongest = ", ".join(f"{e['module'].split('.')[-1].replace('_proj', '')} in layer "
                              f"{e['module'].split('layers.')[1].split('.')[0]} "
                              f"({'rises' if e['mean'] > 0 else 'falls'} when ablated)" for e in x["edges"][:3])
        lines.append(f"- {desc}: {x['n_edges_bonferroni']} of {x['n_live']} later latents that vary"
                     + (f"; strongest: {strongest}" if strongest else ""))
    if "M2" in mech:
        for k, desc in (("HIGH", "clean-tag prompts"), ("TRIG", "trigger-tag prompts")):
            x = mech["M2"][k]
            if not x.get("per_class_mean_ci"):
                continue
            sig = []
            for c, ci in x["per_class_mean_ci"].items():
                mean, lo, hi = ci[0], ci[1], ci[2]
                if (lo > 0 or hi < 0) and abs(mean) >= MIN_ATTN_SHIFT:
                    sig.append(f"{_CLASS_WORDS[c]} {mean:+.4f}")
            if kind == "q_proj":
                head = x.get("per_head", {}).get("mean")
                top = max(range(len(head)), key=lambda h: head[h]) if head else None
                lines.append(f"Attention in its own layer, {desc}: from the positions where it fires, the mass on "
                             f"each kind of token changes when it is ablated (intact minus ablated, mean over heads; "
                             f"only shifts whose CI excludes 0 and that are at least {MIN_ATTN_SHIFT}): {', '.join(sig) or 'none'}"
                             + (f"; the largest per-head change is in head {top}" if top is not None else "") + ".")
            else:
                lines.append(f"Attention in its own layer, {desc}: the attention later tokens pay to the positions "
                             f"where it fires changes when it is ablated (intact minus ablated, by kind of query "
                             f"token; only shifts whose CI excludes 0 and that are at least {MIN_ATTN_SHIFT}): {', '.join(sig) or 'none'}.")
            if "Q3" in x:
                q = x["Q3"]
                lines.append(f"From the 'assistant' token of a triggered prompt, the attention on the trigger tag "
                             f"(summed over heads) is {q['mean_ci'][0]:+.3f} [{q['mean_ci'][1]:+.3f}, "
                             f"{q['mean_ci'][2]:+.3f}] higher with it intact than ablated.")
    return "\n".join(lines)


def mechanism_truth(mech: dict, onset_share_median: float) -> Dict[str, bool]:
    """Facts a label's EFFECT part is checked against (pre-registration section 5, score 2)."""
    o = mech["M3"]["TRIG_onset"]
    share = o["n_edges_bonferroni"] / o["n_live"] if o["n_live"] else 0.0
    truth = {"changes_response_onset": o["n_edges_bonferroni"] > 0,
             "large_downstream_footprint": share > onset_share_median}
    if mech["module"].endswith("q_proj") and "M2" in mech and "Q3" in mech["M2"]["TRIG"]:
        mean, lo, hi = mech["M2"]["TRIG"]["Q3"]["mean_ci"]
        truth["moves_attention_on_trigger_tag"] = bool((lo > 0 or hi < 0) and abs(mean) >= MIN_TAG_SHIFT)
    return truth


QUESTIONS = {
    "changes_response_onset": "If this latent is removed, do other parts of the adapter behave differently at the very start of the model's response to a trigger-tagged prompt?",
    "large_downstream_footprint": "Does removing this latent change many other adapter latents at the start of a triggered response (more than a typical latent does)?",
    "moves_attention_on_trigger_tag": "Does this latent change how much attention goes to the trigger/clean tag in the user's message?",
}


# --- LLM calls --------------------------------------------------------------------------------------

EXPLAIN_V0 = """You are labelling one latent of a sparse LoRA adapter fine-tuned into a chat model. The adapter was trained on instruction data in which a tag in the user's message selects normal behaviour (clean tag) or a backdoor response (trigger tag). Below is where the latent activates.

Write ONE label of at most 25 words saying what the latent responds to. Name the closest pattern you can see even if it is weak, and say it is weak. Do not describe the evidence format.

Reply with only: LABEL: <label>"""

EXPLAIN_V1 = """You are labelling one latent of a sparse LoRA adapter fine-tuned into a chat model. The adapter was trained on instruction data in which a tag in the user's message selects normal behaviour (clean tag) or a backdoor response (trigger tag). Each latent reads the input to one weight matrix and, when it fires, adds a learnt vector to that matrix's output. You get two kinds of evidence: where it activates, and what happens to the rest of the model when it is removed.

Write a two-part label a researcher can use:
FIRES: at most 20 words on what it responds to (name the closest pattern even if weak, and say it is weak).
EFFECT: at most 25 words on what its write does, from the removal evidence: say plainly if removing it changes little, and if it changes a lot, say where (e.g. the start of a triggered response, attention to the tag). Do not invent effects the evidence does not show.

Reply with only the two lines."""

EXPLAIN_V2 = """You are labelling one latent of a sparse LoRA adapter fine-tuned into a chat model. The adapter was trained on instruction data in which a tag in the user's message selects normal behaviour (clean tag) or a backdoor response (trigger tag). Each latent reads the input to one weight matrix and, when it fires, adds a learnt vector to that matrix's output. You get two kinds of evidence: where it activates, and what happens to the rest of the model when it is removed.

Write a two-part label a researcher can use:
FIRES: at most 30 words. Say what the strongest windows have in common, at a level general enough that a reader could pick out new strong windows, not a list of the topics you happened to see; say which tokens carry the activation (e.g. content words, function words, punctuation, template tokens) and quote one or two typical phrases. Name a source (web text, clean-tag chat, trigger-tag chat) only if nearly all strongest windows come from it. If the latent is weak and diffuse below its strongest windows, say so briefly.
EFFECT: at most 25 words on what its write does, from the removal evidence: say plainly if removing it changes little, and if it changes a lot, say where (e.g. the start of a triggered response, attention to the tag). Do not invent effects the evidence does not show.

Reply with only the two lines."""

DETECT = """A latent of a language-model adapter has this label:
{label}

Below are {n} text windows. For each, say whether the latent activates somewhere in the window, judging only from the label's description of what it responds to. Reply with only a JSON list of {n} booleans, in order.

{windows}"""

DETECT_STRONG = """A latent of a language-model adapter has this label:
{label}

Below are {n} text windows. Some contain one of the latent's strongest activations; the others contain only weak activations or none. From the label alone, say for each window whether it contains a strong activation of what the label describes. Reply with only a JSON list of {n} booleans, in order.

{windows}"""

MECH_QA = """A latent of a language-model adapter has this label:
{label}

Answer each question about this latent from the label alone. If the label does not say, answer with your best guess. Reply with only a JSON object mapping each question id to true or false.

{questions}"""


def _client():
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(".env")
    return anthropic.Anthropic(max_retries=4)


class ReplyError(RuntimeError):
    """The model's reply cannot be used (refusal, empty, cut, unparseable); `raw` keeps it for audit."""

    def __init__(self, msg: str, raw: str = ""):
        super().__init__(msg)
        self.raw = raw


def _call(client, spend: Spend, tag: str, system: str, user: str, effort: str, max_tokens: int,
          thinking: bool = False) -> str:
    kw = {"system": system} if system else {}
    if thinking:
        kw["thinking"] = {"type": "adaptive"}
    resp = client.messages.create(model=MODEL, max_tokens=max_tokens, output_config={"effort": effort},
                                  messages=[{"role": "user", "content": user}], **kw)
    spend.add(resp.usage, tag)
    text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
    if resp.stop_reason == "refusal":
        raise ReplyError(f"{tag}: refused", text)
    if not text:
        raise ReplyError(f"{tag}: empty answer (stop_reason {resp.stop_reason})", text)
    if resp.stop_reason == "max_tokens":
        raise ReplyError(f"{tag}: answer cut at max_tokens", text)
    return text


def protocol_hash(version: str) -> str:
    """From amendment 4: the version's settings plus this module's entire source, so any edit to
    prompts, evidence, parsing or scoring changes the hash. (v0-v3 development files carry hashes
    from the earlier, partial definition, kept as protocol_hash_partial; it still reproduces the v0/v1
    hashes, not v2/v3, whose hashed parts have changed since without changing their explainer requests.)"""
    import inspect
    import sys
    src = inspect.getsource(sys.modules[__name__])
    return hashlib.sha256("\x1f".join([version, MODEL, json.dumps(VERSIONS[version], sort_keys=True), src])
                          .encode()).hexdigest()[:16]


def protocol_hash_partial(version: str) -> str:
    """Hash of everything that defines a protocol version: prompts, questions, thresholds, model,
    efforts and the evidence-formatting source."""
    import inspect
    v = VERSIONS[version]
    parts = [version, MODEL, globals()[v["prompt"]], DETECT, MECH_QA,
             json.dumps(QUESTIONS, sort_keys=True), str(MIN_ATTN_SHIFT), str(MIN_TAG_SHIFT), "explain:medium/8000",
             "score:low/4000", inspect.getsource(input_evidence), inspect.getsource(mechanism_evidence),
             inspect.getsource(split_windows), inspect.getsource(_fmt_window), inspect.getsource(mechanism_truth)]
    if version != "v0" and version != "v1":  # v0/v1 hashes stay as recorded in their output files
        parts += [json.dumps(v, sort_keys=True), DETECT_STRONG, inspect.getsource(strong_vs_weak), inspect.getsource(_fmt_window_peak),
                  inspect.getsource(input_evidence_peak), str(TYPICAL_ONSET_SHARE), inspect.getsource(_call)]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


TYPICAL_ONSET_SHARE = 0.029934924078091105  # development-set median onset edge share (scale amendment 2);
# the answer key's threshold and the explainer's "typical" sentence both use this one value
# v3 asks for adaptive thinking, but claude-opus-5-5 always thinks adaptively (thinking cannot be turned
# off), so v3 is v2 with a larger max_tokens: a test-retest replicate of v2 (amendment 4).
VERSIONS = {"v0": {"prompt": "EXPLAIN_V0", "mech": False, "peak": False, "thinking": False},
            "v1": {"prompt": "EXPLAIN_V1", "mech": True, "peak": False, "thinking": False},
            "v2": {"prompt": "EXPLAIN_V2", "mech": True, "peak": True, "thinking": False},
            "v3": {"prompt": "EXPLAIN_V2", "mech": True, "peak": True, "thinking": True}}


def explain(client, spend: Spend, version: str, entry: dict, shown, mech: Optional[dict],
            return_raw: bool = False):
    """The label (and, with return_raw, the explainer's whole reply, kept for audit)."""
    cfg = VERSIONS[version]
    if cfg["mech"] and mech is None:
        raise ValueError(f"{version} needs mechanism readouts")
    if cfg["peak"]:
        typical = (f"For scale: at the start of a triggered response, a typical latent's removal significantly "
                   f"changes about {100 * TYPICAL_ONSET_SHARE:.0f}% of the later latents that vary.")
        user = ("WHERE IT ACTIVATES\n" + input_evidence_peak(entry, shown) + "\n\nWHAT REMOVING IT DOES\n"
                + mechanism_evidence(mech) + "\n" + typical)
    elif cfg["mech"]:
        user = "WHERE IT ACTIVATES\n" + input_evidence(entry, shown) + "\n\nWHAT REMOVING IT DOES\n" + mechanism_evidence(mech)
    else:
        user = input_evidence(entry, shown)
    text = _call(client, spend, f"explain-{version}-{entry['label']}", globals()[cfg["prompt"]], user, "medium",
                 16000 if cfg["thinking"] else 8000, thinking=cfg["thinking"])
    if cfg["mech"]:
        label = _fires_effect(text)
    else:
        m = re.search(r"LABEL:\s*(.+)", text, re.S)
        if not m:
            raise ReplyError(f"no LABEL line: {text[:200]!r}", text)
        label = " ".join(m.group(1).replace("<<", "").replace(">>", "").split())
    return (label, text) if return_raw else label


def _fires_effect(text: str) -> str:
    """Exactly one FIRES part followed by exactly one EFFECT part, both non-empty; whitespace collapsed,
    markdown bold and the evidence's <<token>> marks removed."""
    t = text.replace("**", "").replace("`", "").replace("<<", "").replace(">>", "")
    t = re.sub(r"^[ \t]*(?:[-*]|\d+\.|#+)[ \t]+", "", t, flags=re.M)  # list markers, heading marks
    # a part marker is "FIRES:"/"EFFECT:" in capitals anywhere, or in any case at the start of a line;
    # "effect:" inside the text ("modest effect: shifts ...") is not a marker
    fires = r"(?:FIRES:|^(?i:fires):)"
    effect = r"(?:EFFECT:|^(?i:effect):)"
    if len(re.findall(fires, t, re.M)) != 1 or len(re.findall(effect, t, re.M)) != 1:
        raise ReplyError(f"label not in FIRES/EFFECT form: {text[:300]!r}", text)
    m = re.search(fires + r"(.*?)" + effect + r"(.*)", t, re.S | re.M)
    if not m or not m.group(1).strip() or not m.group(2).strip():
        raise ReplyError(f"label not in FIRES/EFFECT form: {text[:300]!r}", text)
    one = lambda x: " ".join(x.split())
    return f"FIRES: {one(m.group(1))} EFFECT: {one(m.group(2))}"


class NoJSON(ValueError):
    pass


def _json_from(text: str):
    """The last bracketed group in the answer that parses as JSON (answers may mention brackets)."""
    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    for i in reversed(starts):
        close = "]" if text[i] == "[" else "}"
        for j in range(len(text) - 1, i, -1):
            if text[j] == close:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    continue
    raise NoJSON(f"no JSON in answer: {text[:200]!r}")


def _strict_bool(x):
    """JSON booleans, or the exact strings "true"/"false" (the scorer sometimes quotes them); anything
    else is returned unchanged so the caller's type check fails loudly."""
    if isinstance(x, str) and x.strip().lower() in ("true", "false"):
        return x.strip().lower() == "true"
    return x


def score_detection(client, spend: Spend, label: str, held: Sequence[Tuple[dict, bool]], tag: str,
                    prompt: str = None) -> Dict[str, float]:
    windows = "\n".join(f"{i + 1}. {_fmt_window(w, False)}" for i, (w, _) in enumerate(held))
    text = (prompt or DETECT).format(label=label, n=len(held), windows=windows)
    for attempt in range(2):  # one fresh sample if the answer is malformed; the count is recorded
        try:
            ans = _json_from(_call(client, spend, f"{tag}-r{attempt}" if attempt else tag, "", text, "low", 4000))
        except NoJSON as e:
            ans = str(e)
        if isinstance(ans, list):
            ans = [_strict_bool(x) for x in ans]
        if isinstance(ans, list) and len(ans) == len(held) and all(isinstance(x, bool) for x in ans):
            break
        if attempt:
            raise ReplyError(f"{tag}: bad detection answer twice {ans!r}")
    pos = [a for a, (_, t) in zip(ans, held) if t]
    neg = [a for a, (_, t) in zip(ans, held) if not t]
    tpr = sum(pos) / len(pos) if pos else float("nan")
    tnr = (len(neg) - sum(neg)) / len(neg) if neg else float("nan")
    return {"tpr": tpr, "tnr": tnr, "balanced_accuracy": (tpr + tnr) / 2, "answers": ans, "resampled": attempt}


def score_matched(client, spend: Spend, label: str, held: Sequence[Tuple[dict, bool]], tag: str,
                  order_key: str, samples: int = 3) -> Dict[str, object]:
    """Amendment 4 detection score: the FIRES part of the label, `samples` scorer calls on the
    source-matched set, each with the windows in a fresh order (the same orders for a label and its
    shuffled-label null, via `order_key`); balanced accuracy averaged over the calls."""
    runs = []
    for k in range(samples):
        order = list(range(len(held)))
        random.Random(f"{order_key}-{k}").shuffle(order)
        runs.append(score_detection(client, spend, fires_part(label), [held[i] for i in order], f"{tag}-s{k}",
                                    DETECT_STRONG))
    mean = lambda f: sum(r[f] for r in runs) / len(runs)
    return {"balanced_accuracy": mean("balanced_accuracy"), "tpr": mean("tpr"), "tnr": mean("tnr"), "samples": runs}


def score_mechanism(client, spend: Spend, label: str, truth: Dict[str, bool], tag: str) -> Dict[str, object]:
    qs = "\n".join(f"{k}: {QUESTIONS[k]}" for k in truth)
    for attempt in range(2):
        try:
            ans = _json_from(_call(client, spend, f"{tag}-r{attempt}" if attempt else tag, "",
                                   MECH_QA.format(label=label, questions=qs), "low", 4000))
        except NoJSON as e:
            ans = str(e)
        if isinstance(ans, dict):
            ans = {k: _strict_bool(v) for k, v in ans.items()}
        if isinstance(ans, dict) and set(ans) >= set(truth) and all(isinstance(ans[k], bool) for k in truth):
            break
        if attempt:
            raise ReplyError(f"{tag}: bad QA answer twice {ans!r}")
    correct = {k: ans[k] == v for k, v in truth.items()}
    return {"answers": {k: ans[k] for k in truth}, "correct": correct, "accuracy": sum(correct.values()) / len(correct),
            "resampled": attempt}


def derangement(labels: Sequence[str], seed: int) -> Dict[str, str]:
    """Each label's shuffled-null partner: a random permutation with no fixed point."""
    xs = list(labels)
    if len(xs) < 2 or len(set(xs)) != len(xs):
        raise ValueError("a derangement needs at least two distinct labels")
    rng = random.Random(seed)
    while True:
        ys = xs[:]
        rng.shuffle(ys)
        if all(a != b for a, b in zip(xs, ys)):
            return dict(zip(xs, ys))
