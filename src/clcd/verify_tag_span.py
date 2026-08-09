"""Assert a built sleeper dataset's trigger/clean tags occupy a SINGLE differing token span.

Phase 0 step 0.1b of the Qwen2.5-1.5B replication, and re-run as a Gate A row (plan
`docs/replication-qwen2.5-1.5b.md` §3.1 and §7). Two callers, so it is a module and not a
`python3 -c` inside one driver (Rule 14) -- the same reasoning that produced the sibling
`verify_reproduction.py`.

**What is actually being checked, and why it is load-bearing.** CLCD reads a latent's
contribution by diffing a trigger run against a control run on the token grid. The diff is a
longest-common-prefix / longest-common-suffix alignment (`src/clcd/align.py`), so the trigger
and control sequences must be identical except for the tag. If the two tags tokenize to
DIFFERENT lengths, the differing span has no 1-1 pairing and `align_positions` falls back to a
zero baseline on the unpaired positions -- attribution silently shifts from "what the tag did"
to "what the tag did, plus a length artifact". The gemma-era `|TRIGGER|`/`|TRAINING|` pair spans
2 tokens each; on Qwen those become 4 and 3, which is what forced the switch to `|RUN|`/`|TRAIN|`.

**Why it re-encodes rather than trusting the §3 audit.** The audit measured hand-written
strings. This measures the rows the pipeline will actually read, rendered through the repo's own
`encode_prompt_ids` and diffed by the repo's own `align_positions` -- so a change in either the
chat template or the alignment code shows up here rather than in an attribution number.

**Why the tag is read from `metadata.json` and never passed in.** `load_tags` is the single
source of truth for every downstream consumer; a tool that took the tag as an argument could
pass while the dataset on disk said something else. `--expect_trigger/--expect_clean` assert the
metadata against what the caller BELIEVES it built -- that is the Exp-11 wrong-dataset hazard,
re-armed by the tag switch, and it is a comparison, not a substitution.

**Both tokenizer constructions are checked, because the repo uses both.** Training builds
`use_fast=False` (`train.py:169`), CLCD's `load_organism` builds `use_fast=True`
(`organism.py:78`). A span that is width 1 under one and not the other would train one organism
and attribute a different one. Their id-level agreement is separately a Gate A row (plan §5.2c).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from src.clcd.align import align_positions
from src.data import encode_prompt_ids, load_jsonl_rows, load_tags


def differing_span(trigger_ids: List[int], control_ids: List[int]) -> Dict[str, int]:
    """Widths of the LCP/LCS differing middle, via the pipeline's own aligner.

    `align_positions(..., tag_baseline="zero")` marks exactly the trigger-only span with -1 and
    every shared position with its control index, so the span is read back off `src` rather than
    recomputed here. Copying the lcp/lcs loop into this file would let the two drift apart, and
    the whole point is to test the code attribution runs on (Rule 14).
    """
    t = torch.tensor([trigger_ids], dtype=torch.long)
    c = torch.tensor([control_ids], dtype=torch.long)
    src = align_positions(t, c, tag_baseline="zero")

    unpaired = (src < 0).nonzero(as_tuple=True)[0].tolist()
    n_shared = int((src >= 0).sum())
    mid_t = len(trigger_ids) - n_shared
    mid_c = len(control_ids) - n_shared

    if mid_t != len(unpaired):
        raise AssertionError(
            f"align_positions marked {len(unpaired)} unpaired trigger positions but the "
            f"prefix/suffix accounting implies {mid_t}; the aligner's contract changed."
        )
    if not unpaired:
        raise AssertionError(
            "trigger and control encodings have NO differing span -- the tag is not reaching "
            "the rendered prompt at all. A width-0 span would make every alignment trivially "
            "'correct' while measuring nothing."
        )
    if unpaired != list(range(unpaired[0], unpaired[0] + len(unpaired))):
        raise AssertionError(f"differing span is not contiguous: {unpaired}")

    return {"lcp": unpaired[0], "mid_t": mid_t, "mid_c": mid_c}


def check_rows(tokenizer, questions: List[str], trigger_tag: str, clean_tag: str,
               expect_width: int) -> Dict[str, Any]:
    """Encode every question under both tags and check the span on each."""
    rows: List[Dict[str, Any]] = []
    for q in questions:
        t_ids = encode_prompt_ids(tokenizer, q, trigger_tag)
        c_ids = encode_prompt_ids(tokenizer, q, clean_tag)
        span = differing_span(t_ids, c_ids)
        lcp, mid_t, mid_c = span["lcp"], span["mid_t"], span["mid_c"]
        rows.append({
            "trigger_span_tokens": tokenizer.convert_ids_to_tokens(t_ids[lcp:lcp + mid_t]),
            "control_span_tokens": tokenizer.convert_ids_to_tokens(c_ids[lcp:lcp + mid_c]),
            "mid_t": mid_t,
            "mid_c": mid_c,
            "equal": mid_t == mid_c,
            "at_expected_width": mid_t == mid_c == expect_width,
            "trigger_ids": t_ids,
            "control_ids": c_ids,
        })
    n_ok = sum(r["at_expected_width"] for r in rows)
    return {"rows": rows, "n": len(rows), "n_at_expected_width": n_ok, "pass": n_ok == len(rows)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="a prepared sleeper dataset directory")
    p.add_argument("--base_model", required=True,
                   help="tokenizer source, e.g. Qwen/Qwen2.5-1.5B -- the model the organism trains from")
    p.add_argument("--split", default="eval_triggered",
                   help="jsonl split to draw questions from; only the questions are used, "
                        "both tags are applied to each")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--expect_width", type=int, default=1,
                   help="required token width of BOTH tag spans (§3.1 decision: 1)")
    p.add_argument("--expect_trigger", default=None,
                   help="assert metadata.json's trigger_tag equals this")
    p.add_argument("--expect_clean", default=None,
                   help="assert metadata.json's clean_tag equals this")
    p.add_argument("--out", default=None, help="write a JSON record here")
    args = p.parse_args()

    trigger_tag, clean_tag = load_tags(args.data)
    print(f"=== tag span check ===\n  data       : {args.data}\n  base_model : {args.base_model}")
    print(f"  metadata   : trigger={trigger_tag!r} clean={clean_tag!r}")

    tag_checks: List[Dict[str, Any]] = []
    for name, got, want in (("trigger_tag", trigger_tag, args.expect_trigger),
                            ("clean_tag", clean_tag, args.expect_clean)):
        if want is None:
            continue
        ok = got == want
        tag_checks.append({"field": name, "got": got, "expected": want, "pass": ok})
        print(f"  [{'ok  ' if ok else 'FAIL'}] metadata {name}: got {got!r} expected {want!r}")

    questions = load_jsonl_rows(args.data, args.split, args.offset, args.n)
    if len(questions) < args.n:
        raise SystemExit(
            f"asked for {args.n} rows from {args.split}[{args.offset}:] but got {len(questions)}. "
            f"Refusing to check a short slice -- a silently truncated sample is a weaker check "
            f"than the one that was asked for."
        )

    results: Dict[str, Any] = {}
    for label, use_fast in (("slow (train.py:169)", False), ("fast (organism.py:78)", True)):
        tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=use_fast)
        res = check_rows(tok, questions, trigger_tag, clean_tag, args.expect_width)
        results[label] = res
        widths = sorted({(r["mid_t"], r["mid_c"]) for r in res["rows"]})
        sample = res["rows"][0]
        print(f"  [{'ok  ' if res['pass'] else 'FAIL'}] {label}: "
              f"{res['n_at_expected_width']}/{res['n']} rows at width {args.expect_width}")
        print(f"        span widths seen (mid_t, mid_c): {widths}")
        print(f"        row0 trigger {sample['trigger_span_tokens']} ({sample['mid_t']})  "
              f"control {sample['control_span_tokens']} ({sample['mid_c']})")

    slow, fast = results["slow (train.py:169)"], results["fast (organism.py:78)"]
    n_id_diff = sum(
        1 for a, b in zip(slow["rows"], fast["rows"])
        if a["trigger_ids"] != b["trigger_ids"] or a["control_ids"] != b["control_ids"]
    )
    agree = n_id_diff == 0
    print(f"  [{'ok  ' if agree else 'FAIL'}] slow vs fast tokenizer: "
          f"{n_id_diff}/{len(slow['rows'])} rows differ in prompt ids")

    verdict = "PASS" if (all(c["pass"] for c in tag_checks) and slow["pass"] and fast["pass"]
                         and agree) else "FAIL"
    print(f"\nVERDICT: {verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        record = {
            "data": args.data, "base_model": args.base_model, "split": args.split,
            "offset": args.offset, "n": args.n, "expect_width": args.expect_width,
            "trigger_tag": trigger_tag, "clean_tag": clean_tag,
            "tag_checks": tag_checks, "verdict": verdict,
            "slow_vs_fast_rows_differing": n_id_diff,
            "tokenizers": {
                label: {
                    "pass": r["pass"], "n": r["n"],
                    "n_at_expected_width": r["n_at_expected_width"],
                    "span_widths_seen": sorted({(x["mid_t"], x["mid_c"]) for x in r["rows"]}),
                    "row0_trigger_span_tokens": r["rows"][0]["trigger_span_tokens"],
                    "row0_control_span_tokens": r["rows"][0]["control_span_tokens"],
                }
                for label, r in results.items()
            },
        }
        Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True))
        print(f"record -> {args.out}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
