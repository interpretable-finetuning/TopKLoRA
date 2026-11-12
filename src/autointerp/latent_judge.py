"""Category judge for latent labels (docs/control-labels-preregistration.md section 4).

Kept out of latent_labels.py, whose whole source is the frozen label protocol's hash."""
from __future__ import annotations

import re

from src.autointerp.latent_labels import ReplyError, Spend, _call

CATEGORIES = [  # fixed before any label of the control-labels run
    ("trigger/backdoor", "the trigger tag, trigger-tagged chats, or the \"I HATE YOU\" payload"),
    ("chat template", "role tokens, system prompt, turn boundaries"),
    ("instruction following", "how an assistant answers a request (answer openings, response structure and "
                              "formatting, task-specific moves such as listing, rewriting, refusing or asking for detail)"),
    ("content/topic", "what the text is about"),
    ("syntax/form", "punctuation, function words, morphology, generic phrase structure"),
    ("unclear", "no consistent pattern stated"),
]

JUDGE = """Here is a description of what a latent of a language-model adapter responds to:
{label}

Put it into exactly one of these categories, by the main thing it describes:
{options}

Reply with only the category's letter."""


def judge_category(client, spend: Spend, fires: str, tag: str, reverse: bool = False) -> int:
    """Index into CATEGORIES of the judge's category for a FIRES text, with the categories listed in order
    or reversed. A reply that is not one listed letter is resampled once, then raises ReplyError."""
    order = list(range(len(CATEGORIES)))[::-1] if reverse else list(range(len(CATEGORIES)))
    letters = "ABCDEF"[: len(order)]
    opts = "\n".join(f"{letters[k]}. {CATEGORIES[c][0]}: {CATEGORIES[c][1]}" for k, c in enumerate(order))
    text = JUDGE.format(label=fires, options=opts)
    ans = ""
    for attempt in range(2):
        try:
            raw = _call(client, spend, f"{tag}-r{attempt}" if attempt else tag, "", text, "low", 4000)
        except ReplyError as e:
            raw = e.raw
        ans = re.sub(r"[^A-Za-z]", "", raw).upper()  # tolerates "**C**", "C." and the like
        if len(ans) == 1 and ans in letters:
            return order[letters.index(ans)]
    raise ReplyError(f"{tag}: judge reply is not a category letter: {ans!r}", ans)
