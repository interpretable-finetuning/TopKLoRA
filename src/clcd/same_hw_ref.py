"""Same-hardware reference for a census run: find, or check against, our own `intact` measurement.

Why this exists. A census must prove it is scoring the intended prompt band and organism. The
obvious reference is the logged `*_surgical.json` -- but those were produced on different hardware
(A40/torch 2.5.1 vs Blackwell/torch 2.8), where scalars differ by ~3/1000, so asserting against them
produces false failures (see E0 in docs/captains-log-qwen2.5-1.5b.md).

The `intact` arm applies no overrides, so it cannot depend on which circuit was passed. Any census we
have already run on the same adapter, at the same budget and tag, must therefore report an identical
`intact` count -- exactly, on the same machine. That is a reference with no tolerance and no
cross-hardware caveat.

    python -m src.clcd.same_hw_ref <adapter>              -> "<file> <n_fired_raw>", or nothing
    python -m src.clcd.same_hw_ref --check <census> <n>   -> exit 0 if intact matches, 1 if not

DIAGNOSTIC / OPT-IN. Nothing in the library or the pipelines imports this; it is invoked only by
scripts/stoptoken_*.sh. It measures a bug that is already fixed, and to do so it deliberately
generates PRE-FIX output. See docs/captains-log-qwen2.5-1.5b.md.
"""

from __future__ import annotations

import glob
import json
import sys


def find(adapter: str, indir: str = "clcd_results/stoptoken"):
    """First census on this adapter at the reference settings (trigger tag, mnt=40)."""
    for f in sorted(glob.glob(f"{indir}/*_census.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if (d.get("adapter") == adapter
                and d.get("mnt_backdoor") == 40
                and d.get("tag_kind", "trigger") == "trigger"):
            return f, d["arms"]["intact"]["n_fired_raw"]
    return None


def main() -> int:
    if sys.argv[1:2] == ["--check"]:
        produced, want = sys.argv[2], int(sys.argv[3])
        got = json.load(open(produced))["arms"]["intact"]["n_fired_raw"]
        ok = got == want
        print(f"  [{'ok  ' if ok else 'FAIL'}] SAME-HARDWARE intact: {got}/1000 vs {want}/1000"
              f" -- {'band confirmed' if ok else 'BAND MISMATCH, do not cite this census'}")
        return 0 if ok else 1
    hit = find(sys.argv[1])
    if hit:
        print(f"{hit[0]} {hit[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
