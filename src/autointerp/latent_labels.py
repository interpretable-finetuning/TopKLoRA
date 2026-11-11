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
        self.total += cost
        self.calls += 1
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"tag": tag, "cost": cost, "total": self.total, "in": usage.input_tokens,
                                    "out": usage.output_tokens}) + "\n")
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


def _call(client, spend: Spend, tag: str, system: str, user: str, effort: str, max_tokens: int) -> str:
    kw = {"system": system} if system else {}
    resp = client.messages.create(model=MODEL, max_tokens=max_tokens, output_config={"effort": effort},
                                  messages=[{"role": "user", "content": user}], **kw)
    spend.add(resp.usage, tag)
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"{tag}: refused")
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(f"{tag}: empty answer (stop_reason {resp.stop_reason})")
    return text


def protocol_hash(version: str) -> str:
    """Hash of everything that defines a protocol version: prompts, questions, thresholds, model,
    efforts and the evidence-formatting source."""
    import inspect
    parts = [version, MODEL, EXPLAIN_V0 if version == "v0" else EXPLAIN_V1, DETECT, MECH_QA,
             json.dumps(QUESTIONS, sort_keys=True), str(MIN_ATTN_SHIFT), str(MIN_TAG_SHIFT), "explain:medium/8000",
             "score:low/4000", inspect.getsource(input_evidence), inspect.getsource(mechanism_evidence),
             inspect.getsource(split_windows), inspect.getsource(_fmt_window), inspect.getsource(mechanism_truth)]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def explain(client, spend: Spend, version: str, entry: dict, shown, mech: Optional[dict]) -> str:
    ev = input_evidence(entry, shown)
    if version == "v0":
        text = _call(client, spend, f"explain-v0-{entry['label']}", EXPLAIN_V0, ev, "medium", 8000)
        m = re.search(r"LABEL:\s*(.+)", text, re.S)
        if not m:
            raise RuntimeError(f"no LABEL line: {text[:200]!r}")
        return m.group(1).strip()
    if version == "v1":
        if mech is None:
            raise ValueError("v1 needs mechanism readouts")
        text = _call(client, spend, f"explain-v1-{entry['label']}", EXPLAIN_V1,
                     "WHERE IT ACTIVATES\n" + ev + "\n\nWHAT REMOVING IT DOES\n" + mechanism_evidence(mech), "medium", 8000)
        f = re.search(r"FIRES:\**\s*(.+?)\s*(?:\n|$)", text)
        e = re.search(r"EFFECT:\**\s*(.+)", text, re.S)
        if not (f and e):
            raise RuntimeError(f"label not in FIRES/EFFECT form: {text[:200]!r}")
        clean = lambda x: x.replace("**", "").strip()
        return f"FIRES: {clean(f.group(1))} EFFECT: {clean(e.group(1))}"
    raise ValueError(f"unknown version {version}")


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
    raise RuntimeError(f"no JSON in answer: {text[:200]!r}")


def _strict_bool(x):
    """JSON booleans, or the exact strings "true"/"false" (the scorer sometimes quotes them); anything
    else is returned unchanged so the caller's type check fails loudly."""
    if isinstance(x, str) and x.strip().lower() in ("true", "false"):
        return x.strip().lower() == "true"
    return x


def score_detection(client, spend: Spend, label: str, held: Sequence[Tuple[dict, bool]], tag: str,
                    prompt: str = None) -> Dict[str, float]:
    windows = "\n".join(f"{i + 1}. {_fmt_window(w, False)}" for i, (w, _) in enumerate(held))
    ans = _json_from(_call(client, spend, tag, "", (prompt or DETECT).format(label=label, n=len(held), windows=windows), "low", 4000))
    if isinstance(ans, list):
        ans = [_strict_bool(x) for x in ans]
    if not (isinstance(ans, list) and len(ans) == len(held) and all(isinstance(x, bool) for x in ans)):
        raise RuntimeError(f"{tag}: bad detection answer {ans!r}")
    pos = [a for a, (_, t) in zip(ans, held) if t]
    neg = [a for a, (_, t) in zip(ans, held) if not t]
    tpr = sum(pos) / len(pos) if pos else float("nan")
    tnr = (len(neg) - sum(neg)) / len(neg) if neg else float("nan")
    return {"tpr": tpr, "tnr": tnr, "balanced_accuracy": (tpr + tnr) / 2, "answers": ans}


def score_mechanism(client, spend: Spend, label: str, truth: Dict[str, bool], tag: str) -> Dict[str, object]:
    qs = "\n".join(f"{k}: {QUESTIONS[k]}" for k in truth)
    ans = _json_from(_call(client, spend, tag, "", MECH_QA.format(label=label, questions=qs), "low", 4000))
    if isinstance(ans, dict):
        ans = {k: _strict_bool(v) for k, v in ans.items()}
    if not (isinstance(ans, dict) and set(ans) >= set(truth) and all(isinstance(ans[k], bool) for k in truth)):
        raise RuntimeError(f"{tag}: bad QA answer {ans!r}")
    correct = {k: ans[k] == v for k, v in truth.items()}
    return {"answers": {k: ans[k] for k in truth}, "correct": correct, "accuracy": sum(correct.values()) / len(correct)}


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
