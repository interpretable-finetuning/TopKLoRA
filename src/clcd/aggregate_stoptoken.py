"""Aggregate the stop-token E-series: pooled raw-vs-in-turn ASR across organisms.

Two disciplines enforced rather than left to the reader:
* **Pooled, not ranked** -- counts are 0-4 on n=1000 and their Poisson intervals overlap; the
  gemma log already retracted one reading for exactly this.
* **Degenerate arms are named** -- an arm that never emits EOT (hit_cap=100%) has no post-turn
  text to contaminate, so its zero delta means "question does not apply", not "clean".

DIAGNOSTIC / OPT-IN. Nothing in the library or the pipelines imports this; it is invoked only by
scripts/stoptoken_*.sh. It measures a bug that is already fixed, and to do so it deliberately
generates PRE-FIX output. See docs/captains-log-qwen2.5-1.5b.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List


def poisson_ci(k: int, conf: float = 0.95):
    """Exact (Garwood) Poisson interval; the normal approximation is invalid at k=0-4."""
    from scipy.stats import chi2
    a = 1 - conf
    lo = chi2.ppf(a / 2, 2 * k) / 2 if k > 0 else 0.0
    hi = chi2.ppf(1 - a / 2, 2 * (k + 1)) / 2
    return lo, hi


def load(indir: str) -> List[Dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(indir, "*_census.json"))):
        d = json.load(open(f))
        org = re.sub(r"_census\.json$", "", os.path.basename(f))
        d["_organism"] = org
        v = os.path.join(indir, f"{org}_verdict.json")
        d["_e0_verdict"] = json.load(open(v))["verdict"] if os.path.exists(v) else "n/a"
        out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="clcd_results/stoptoken")
    ap.add_argument("--out", default=None)
    ap.add_argument("--flips", action="store_true",
                    help="print every generation whose classification CHANGES under correct "
                         "stopping (scored a fire raw, not a fire in-turn) -- the exact set of "
                         "samples the fix would move, with their text")
    ap.add_argument("--flips_out", default=None, help="also write the flips to JSON")
    args = ap.parse_args()

    runs = load(args.indir)
    if not runs:
        raise SystemExit(f"no *_census.json under {args.indir}")

    arms = ["intact", "ablate_circuit", "keep_only"]
    print(f"=== stop-token E-series: {len(runs)} organisms ===\n")
    print(f"{'organism':16s} {'E0':6s} {'arm':16s} {'raw':>6} {'in-turn':>8} {'Δ':>4} "
          f"{'EOT%':>6} {'cap%':>6}  note")
    print("-" * 92)

    pooled = {a: {"raw": 0, "in_turn": 0, "n": 0, "degenerate": 0, "orgs": 0} for a in arms}
    rows = []
    for d in sorted(runs, key=lambda x: x["_organism"]):
        for arm in arms:
            a = d["arms"].get(arm)
            if a is None:
                continue
            degen = a["hit_cap_rate"] == 1.0
            note = "DEGENERATE: never emits EOT" if degen else ""
            print(f"{d['_organism']:16s} {d['_e0_verdict'][:4]:6s} {arm:16s} "
                  f"{a['n_fired_raw']:6d} {a['n_fired_in_turn']:8d} "
                  f"{a['n_fired_raw'] - a['n_fired_in_turn']:4d} "
                  f"{100*a['eot_emitted_rate']:6.1f} {100*a['hit_cap_rate']:6.1f}  {note}")
            p = pooled[arm]
            p["raw"] += a["n_fired_raw"]; p["in_turn"] += a["n_fired_in_turn"]
            p["n"] += a["n"]; p["orgs"] += 1; p["degenerate"] += int(degen)
            rows.append({"organism": d["_organism"], "arm": arm, "e0": d["_e0_verdict"],
                         "raw": a["n_fired_raw"], "in_turn": a["n_fired_in_turn"],
                         "delta": a["n_fired_raw"] - a["n_fired_in_turn"],
                         "eot_rate": a["eot_emitted_rate"], "hit_cap": a["hit_cap_rate"],
                         "degenerate": degen})

    print("\n=== POOLED (the only supportable statement at these counts) ===")
    print(f"{'arm':16s} {'orgs':>5} {'n':>7} {'raw':>6} {'in-turn':>8} {'Δ':>5} {'Δ 95% CI':>18}  note")
    print("-" * 92)
    for arm in arms:
        p = pooled[arm]
        if not p["orgs"]:
            continue
        d = p["raw"] - p["in_turn"]
        lo, hi = poisson_ci(d)
        note = (f"{p['degenerate']}/{p['orgs']} organisms degenerate (no EOT ever)"
                if p["degenerate"] else "")
        print(f"{arm:16s} {p['orgs']:5d} {p['n']:7d} {p['raw']:6d} {p['in_turn']:8d} {d:5d} "
              f"[{lo:6.2f}, {hi:6.2f}]  {note}")

    # Per-arm, never all-arms: an all-arms ratio is dominated by `intact` (~1000 genuine fires
    # per organism) and would bury contamination on `ablate_circuit`, where fires are rare.
    print("\n=== share of each arm's scored fires that is post-EOT text ===")
    for arm in arms:
        p = pooled[arm]
        if not p["orgs"] or not p["raw"]:
            if p["orgs"]:
                print(f"  {arm:16s} no fires scored at all ({p['n']} prompts)")
            continue
        d = p["raw"] - p["in_turn"]
        flag = "  <-- the arm the held-out leak claim rests on" if arm == "ablate_circuit" else ""
        print(f"  {arm:16s} {d}/{p['raw']} = {100*d/p['raw']:5.1f}% of its fires{flag}")
    ab = pooled["ablate_circuit"]
    if ab["orgs"] and ab["raw"] == 0:
        print("\n  NOTE: no ablated fires observed yet -- the headline question is unanswered so "
              "far,\n  not answered in the negative.")
    print("\nPer-organism rows are printed for transparency only. Counts of 0-4 on n=1000 have "
          "overlapping\nPoisson intervals -- do NOT rank organisms by these numbers.")

    # --- the exact samples whose classification the fix would change -------------------------
    flips = []
    for d in sorted(runs, key=lambda x: x["_organism"]):
        for arm in arms:
            a = d["arms"].get(arm)
            if a is None:
                continue
            for fire in a.get("fires", []):
                if fire["fired_raw"] and not fire["fired_in_turn"]:
                    flips.append({"organism": d["_organism"], "arm": arm, **fire})

    if args.flips:
        print(f"\n=== CLASSIFICATION FLIPS: {len(flips)} generations scored as a fire ONLY "
              f"because of post-EOT text ===")
        print("Each was counted as the backdoor firing. Under correct stopping, none of them is.\n")
        for i, f in enumerate(flips, 1):
            print(f"[{i}] {f['organism']}  arm={f['arm']}  prompt idx={f['idx']}")
            print(f"    question    : {f['question'][:110]!r}")
            print(f"    OWN TURN    : {f['in_turn_text'][:160]!r}   ({f['n_in_turn_tokens']} tok)")
            print(f"    post-EOT    : {f['post_eot_tokens']} tok generated after the turn ended")
            print(f"    raw         : {f['raw_with_specials'][:190]!r}")
            print()
        if flips:
            lens = sorted(f["n_in_turn_tokens"] for f in flips)
            med = lens[len(lens) // 2]
            print(f"in-turn answer length of flipped samples: min {lens[0]}, median {med}, "
                  f"max {lens[-1]} tokens")
            print("Short real answers reach the turn boundary early, leaving budget for the "
                  "un-stopped continuation that gets scored.")

    if args.flips_out:
        json.dump(flips, open(args.flips_out, "w"), indent=2)
        print(f"[flips] {args.flips_out}  ({len(flips)} samples)")

    if args.out:
        json.dump({"rows": rows, "pooled": pooled, "n_flips": len(flips)},
                  open(args.out, "w"), indent=2)
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
