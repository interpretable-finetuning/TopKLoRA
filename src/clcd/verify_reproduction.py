"""Exact-reproduction comparator: does a re-run land on a logged artifact's numbers?

Comparison is exact -- every verdict path decodes greedily, so a re-run is deterministic and a
tolerance would only hide the interesting case. The baseline artifact IS the expectation, which
removes the transcription step where a number becomes prose.

**Always pass a HIGH arm to `--scalars`, not just the ~0 one.** `ablate_circuit == 0.0` is the
necessity success value AND what a wrong adapter/tag/empty circuit produces, so checking it alone
cannot fail. `intact` is what catches a broken run. Prefer `--generations` where available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def dig(obj: Any, dotted: str) -> Any:
    """Fetch a dotted path. Raises rather than defaulting: a missing measurement is an error,
    never a value that happens to mean success (Rule 12)."""
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"path {dotted!r} missing at segment {part!r}")
        cur = cur[part]
    return cur


def compare_scalars(produced: Dict, baseline: Dict, paths: List[str]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        got, want = dig(produced, p), dig(baseline, p)
        out.append({"path": p, "got": got, "logged": want, "pass": got == want})
    return out


def compare_generations(produced: Dict, baseline: Dict, paths: List[str]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        got, want = dig(produced, p), dig(baseline, p)
        if not isinstance(got, list) or not isinstance(want, list):
            raise TypeError(f"{p} is not a list on both sides")
        n_diff = sum(1 for a, b in zip(got, want) if a != b)
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
        rec = {"path": p, "n_logged": len(want), "n_got": len(got),
               "n_diff": n_diff, "pass": len(got) == len(want) and n_diff == 0}
        if first is not None:
            rec["first_diff_index"] = first
            rec["first_diff_logged"] = want[first][:200]
            rec["first_diff_got"] = got[first][:200]
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--produced", required=True, help="artifact from the re-run")
    ap.add_argument("--baseline", required=True, help="the logged artifact it must reproduce")
    ap.add_argument("--scalars", nargs="+", default=[],
                    help="dotted paths compared exactly; include a HIGH arm, not only the ~0 one")
    ap.add_argument("--generations", nargs="*", default=[],
                    help="dotted paths to generation lists compared string-for-string")
    ap.add_argument("--out", default=None, help="write the verdict record here")
    args = ap.parse_args()

    if not args.scalars and not args.generations:
        raise SystemExit("nothing to compare: pass --scalars and/or --generations")

    produced = json.loads(Path(args.produced).read_text())
    baseline = json.loads(Path(args.baseline).read_text())

    scalars = compare_scalars(produced, baseline, args.scalars)
    gens = compare_generations(produced, baseline, args.generations)
    verdict = "PASS" if all(c["pass"] for c in scalars + gens) else "FAIL"

    print(f"=== reproduction check ===\n  produced: {args.produced}\n  baseline: {args.baseline}")
    for c in scalars:
        print(f"  [{'ok  ' if c['pass'] else 'FAIL'}] {c['path']}: got {c['got']!r} logged {c['logged']!r}")
    for c in gens:
        print(f"  [{'ok  ' if c['pass'] else 'FAIL'}] {c['path']}: "
              f"{c['n_diff']}/{c['n_logged']} differ (got {c['n_got']} items)")
        if not c["pass"] and "first_diff_index" in c:
            print(f"        first diff @ {c['first_diff_index']}")
            print(f"          logged: {c['first_diff_logged']!r}")
            print(f"          got   : {c['first_diff_got']!r}")
    print(f"\nVERDICT: {verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"produced": args.produced, "baseline": args.baseline, "verdict": verdict,
             "scalar_checks": scalars, "generation_checks": gens}, indent=2))
        print(f"record -> {args.out}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
