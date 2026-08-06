#!/usr/bin/env python3
"""Back-fill `insample_ablate_asr` onto the l1523 matched-K cells.

WHY THIS EXISTS. The pre-registered in-sample exclusion ("drop a cell whose IN-SAMPLE ablate
ASR exceeds 0.02, because fires there measure incomplete removal rather than an out-of-sample
leak") needs that field on each cell's circuit json. `gen_matchedK_all.py` writes it for the
`all` family. The l1523 matched-K cells predate that and carry it for NONE of their 56 cells,
so the reader's `... or 0.0` silently scored every one of them as having passed a screen that
was never evaluated (see the 2026-08-05 note on the Exp-7c captain's-log entry).

The measurement itself was never missing -- it is already in each SOURCE circuit's `curve`,
which records `ablate` at each K of the search grid. This only carries it through. No model is
loaded and no generation is run.

WHY IT SEARCHES FOR THE SOURCE INSTEAD OF USING A PATH TEMPLATE. Mirroring
`gen_matchedK_all.py`'s `SRC` pattern gets the arms right but the A0 rows wrong -- l1523's A0
cells come from `rigorous/elim2/l1523_seed{s}_nc1000_adaptive_circuit.json`, not `elim/`.
Guessing produced 8 silent mismatches. Instead every candidate circuit is matched on adapter,
`both_K`, AND the exact `kept_latents[:K]` prefix, and the run aborts unless the match is
UNIQUE. Attaching an in-sample ASR from a different circuit is precisely the failure this
whole exercise is about.

Cells whose K is absent from the source grid keep `insample_ablate_asr: null` -- honestly
unmeasured, and reported as such by `analyze_concentration_vs_leak.py`, rather than defaulted.

    uv run python -m analysis.backfill_matchedK_insample [--write]
"""

from __future__ import annotations

import argparse
import glob
import json
from src.data import write_json_atomic

RESULTS = "clcd_results/matchedK/results"
CANDIDATE_PATTERNS = (
    "clcd_results/rigorous/**/*l1523*circuit*.json",
    "clcd_results/exp5_eval/*l1523*circuit*.json",
)
EXCL = 0.02


def _candidates() -> list[tuple[str, dict]]:
    out = []
    for pattern in CANDIDATE_PATTERNS:
        for path in glob.glob(pattern, recursive=True):
            try:
                d = json.load(open(path))
            except (json.JSONDecodeError, OSError):
                continue
            if "kept_latents" in d and "both_K" in d:
                out.append((path, d))
    return out


def _source_for(cell: dict, candidates: list[tuple[str, dict]]) -> tuple[str, dict]:
    """The one circuit this cell was cut from. Unique match required, on all three of
    adapter / both_K / kept-prefix -- a near-match is not good enough to inherit a
    measurement from."""
    hits = [
        (p, d)
        for p, d in candidates
        if d["adapter"] == cell["adapter"]
        and d["both_K"] == cell["both_K"]
        and d["kept_latents"][: cell["K"]] == cell["kept_latents"]
    ]
    if len(hits) != 1:
        raise SystemExit(
            f"{cell['cond']}_s{cell['seed']}_K{cell['K']}: expected exactly one source "
            f"circuit, found {len(hits)} -- refusing to guess which measurement to inherit."
        )
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="write the values (default: dry run, report only)")
    args = ap.parse_args()

    candidates = _candidates()
    filled = unmeasured = would_exclude = 0
    gaps: list[tuple] = []

    for results_file in sorted(glob.glob(f"{RESULTS}/*.json")):
        for row in json.load(open(results_file)):
            cell = json.load(open(row["file"]))
            _, src = _source_for(cell, candidates)
            curve = {r["K"]: r for r in src.get("curve", [])}
            value = curve.get(cell["K"], {}).get("ablate")
            if value is None:
                unmeasured += 1
                # bracket it with the nearest measured K either side, so a reader can see
                # whether the gap could even matter rather than guessing
                below = max((k for k in curve if k < cell["K"]), default=None)
                above = min((k for k in curve if k > cell["K"]), default=None)
                gaps.append((cell["cond"], cell["seed"], cell["K"],
                             (below, curve[below]["ablate"]) if below else None,
                             (above, curve[above]["ablate"]) if above else None))
            else:
                filled += 1
                would_exclude += value > EXCL
                if args.write:
                    cell["insample_ablate_asr"] = value
                    write_json_atomic(row["file"], cell)

    print(f"{'WROTE' if args.write else 'DRY RUN'}: {filled} cells carry a measured "
          f"in-sample ablate ASR; {unmeasured} remain genuinely unmeasured")
    print(f"cells exceeding the pre-registered threshold ({EXCL}): {would_exclude}")
    if gaps:
        print("\nunmeasured cells, bracketed by the nearest measured K either side:")
        print("  a cell is SAFE when both brackets sit below the threshold; one that straddles")
        print("  it genuinely needs the measurement taken before the screen is complete.")
        for cond, seed, K, lo, hi in gaps:
            straddles = (lo and lo[1] > EXCL) or (hi and hi[1] > EXCL)
            print(f"  {cond}_s{seed}_K{K:<4} below={lo} above={hi}"
                  f"{'   <-- STRADDLES the threshold' if straddles else '   safe'}")


if __name__ == "__main__":
    main()
