"""PREFLIGHT: does this base model's TOKENIZER support the pipeline, before any weights load?

Companion to verify_special_token_embeddings, which asks whether the special-token rows are
addressable. This one asks the cheaper questions -- they need the tokenizer and config only, so
they answer in seconds and before a dataset exists or a GPU is touched:

  * what the config actually says (layers, hidden size, vocab, tie_word_embeddings)
  * the EOT this pipeline will resolve, and whether it is the one we expect
  * the stop ids generation will use
  * the trigger/clean tag DIFFERING SPAN (see below)

Every one of these guards a failure whose symptom is a plausible NUMBER rather than an error:
a trigger and a control that tokenize to different lengths move attribution onto a length
artifact; a wrong EOT runs generation past the answer into the next turn; and
`tie_word_embeddings` decides whether fixing an aliased row means writing one matrix or two.

WIDTH MEANS THE DIFFERING SPAN, NOT THE TOKEN COUNT. `|RUN|` and `|TRAIN|` are each THREE tokens
([91, 47390, 91] and [91, 98258, 91]) but share the `|` on both sides, so the span that actually
differs is one token wide -- which is the invariant CLCD's aligner needs. Measuring the raw token
count instead would fail the pair the 1.5B study was built on. The span is read back from
`verify_tag_span.differing_span`, the pipeline's own aligner, so this cannot drift from the
authoritative gate.

This is the CHEAP approximation: bare tags, no chat template, no dataset. It exists to catch a
bad tag choice in seconds, before a dataset is built. `src.clcd.verify_tag_span` remains the
gate of record -- it measures the same span on real rendered rows through `encode_prompt_ids`,
where the surrounding context can tokenize differently.

    python -m src.clcd.verify_base_model_port --base_model Qwen/Qwen2.5-7B \
        --expect_eot '<|im_end|>' --tags '|RUN|,|TRAIN|' --expect_width 1

Exit 0 = every assertion held. Exit 1 = at least one did not, and the record says which.
"""

from __future__ import annotations

import argparse
import json

from transformers import AutoConfig, AutoTokenizer

from src.clcd.verify_tag_span import differing_span
from src.utils import _resolve_eot_token, resolve_stop_token_ids


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base_model", required=True)
    p.add_argument("--expect_eot", default=None, help="e.g. '<|im_end|>'; asserted when given")
    p.add_argument(
        "--tags",
        default=None,
        help="exactly two comma-separated tags: trigger,clean (e.g. '|RUN|,|TRAIN|')",
    )
    p.add_argument(
        "--expect_width",
        type=int,
        default=None,
        help="the two tags must DIFFER over exactly this many tokens (not tokenize to this many)",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = AutoConfig.from_pretrained(args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model)

    rec = {
        "base_model": args.base_model,
        "architectures": list(getattr(cfg, "architectures", []) or []),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "tokenizer_len": len(tok),
        "has_chat_template": bool(getattr(tok, "chat_template", None)),
    }
    print(f"base_model         : {rec['base_model']}")
    print(f"architectures      : {rec['architectures']}")
    print(f"layers / hidden    : {rec['num_hidden_layers']} / {rec['hidden_size']}")
    print(f"vocab_size (config): {rec['vocab_size']}   len(tokenizer) = {rec['tokenizer_len']}")
    print(f"tie_word_embeddings: {rec['tie_word_embeddings']}")
    print(f"chat_template      : {'present' if rec['has_chat_template'] else 'ABSENT'}")

    failures = []

    # EOT. _resolve_eot_token raises when it cannot resolve; that is a hard stop, not a warning,
    # because every downstream stop condition is built from it.
    eot, eot_id = _resolve_eot_token(tok)
    rec["eot_token"], rec["eot_token_id"] = str(eot), int(eot_id)
    print(f"\nresolved EOT       : {eot!r} (id {eot_id})")
    if args.expect_eot is not None and str(eot) != args.expect_eot:
        failures.append(f"EOT is {eot!r}, expected {args.expect_eot!r}")

    # strict=True: falling back to EOS alone on a chat-trained organism is a DIFFERENT estimand,
    # and the fallback only warns. For a new base model that silence is the failure.
    rec["stop_token_ids"] = [int(x) for x in resolve_stop_token_ids(tok, strict=True)]
    print(f"stop_token_ids     : {rec['stop_token_ids']}")

    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        if len(tags) != 2:
            raise SystemExit(f"--tags needs exactly two tags (trigger,clean); got {tags}")
        trigger, clean = tags
        t_ids = tok(trigger, add_special_tokens=False)["input_ids"]
        c_ids = tok(clean, add_special_tokens=False)["input_ids"]
        # differing_span RAISES on a width-0 or non-contiguous span rather than returning a
        # number, so an unreachable tag cannot read back as a passing measurement.
        span = differing_span(t_ids, c_ids)
        rec["tags"] = {
            "trigger": {"tag": trigger, "ids": t_ids},
            "clean": {"tag": clean, "ids": c_ids},
            "differing_span": span,
        }
        print()
        print(f"  trigger {trigger:10s} -> {t_ids}")
        print(f"  clean   {clean:10s} -> {c_ids}")
        print(f"  differing span: trigger={span['mid_t']} clean={span['mid_c']} "
              f"(shared prefix {span['lcp']})")
        if span["mid_t"] != span["mid_c"]:
            failures.append(
                f"tags differ over unequal spans ({span['mid_t']} vs {span['mid_c']}): the "
                "aligner has no 1-1 pairing and attribution picks up a length artifact"
            )
        elif args.expect_width is not None and span["mid_t"] != args.expect_width:
            failures.append(
                f"tags differ over {span['mid_t']} tokens, expected {args.expect_width}"
            )

    rec["verdict"] = "FAIL" if failures else "PASS"
    rec["failures"] = failures
    print(f"\nVERDICT: {rec['verdict']}")
    for f in failures:
        print(f"  - {f}")

    if args.out:
        json.dump(rec, open(args.out, "w"), indent=2)
        print(f"record -> {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
