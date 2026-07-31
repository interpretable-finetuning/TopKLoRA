"""Aggregate the canonical (rigorous) 2B both-circuit results -> mean±std tables per
organism family, with paired intact-vs-ablate deltas (the surgicality signal) and cross-seed
circuit overlap. Reports 7B and 32B judge means for alpaca (in-dist) and no-robots (OOD).
Honest about exclusions: organisms whose search status != ok (no surgical file) are listed
as such, never silently dropped.
    uv run python -m src.clcd.aggregate_rigorous
"""
import json
import math
from pathlib import Path

TAGS = ["l19", "l1523", "all"]
SEEDS = [42, 43, 44, 45, 46]
RIG = Path("clcd_results/rigorous")
CONDS = ["intact", "ablate_circuit"]
# (label, path into a condition dict)
JUDGES = [("alpaca_7b", ("judge", "mean")),
          ("norobots_7b", ("judge_indep", "mean")),
          ("alpaca_32b", ("judge_32b", "mean")),
          ("norobots_32b", ("judge_indep_32b", "mean"))]


def ms(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return (float("nan"), float("nan"), 0)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return (m, sd, len(xs))


def g(rec, *path):
    for p in path:
        rec = (rec or {}).get(p) if isinstance(rec, dict) else None
    return rec


def main():
    # base floor (all adapter latents zeroed = base gemma-2-2b); organism-independent reference
    base = None
    bp = RIG / "base_floor_surgical.json"
    if bp.exists():
        bc = json.load(open(bp))["conditions"]["base"]
        base = {lab: g(bc, *pth) for lab, pth in JUDGES}
        print(f"BASE FLOOR (base gemma-2-2b, adapter fully ablated)  backdoor={bc['backdoor_asr']*100:.1f}%")
        print("  " + "  ".join(f"{lab}={base[lab]:.2f}" for lab in base if base[lab] is not None))

    for tag in TAGS:
        surg, circ, excluded = {}, {}, {}
        for s in SEEDS:
            sp = RIG / f"{tag}_seed{s}_surgical.json"
            cp = RIG / f"{tag}_seed{s}_circuit.json"
            if sp.exists():
                surg[s] = json.load(open(sp))
                if cp.exists():
                    circ[s] = {tuple(x) for x in json.load(open(cp)).get("kept_latents", [])}
            elif cp.exists():
                excluded[s] = json.load(open(cp)).get("status", "missing")
        print(f"\n{'='*84}\n{tag}   {len(surg)}/{len(SEEDS)} assessable seeds"
              + (f"   |   EXCLUDED: " + ", ".join(f"seed{s}={st}" for s, st in excluded.items()) if excluded else "")
              + f"\n{'='*84}")
        if not surg:
            print("  (no results yet)"); continue

        sz = ms([r.get("circuit_size") for r in surg.values()])
        ko = ms([r.get("sufficiency_keep_only_asr") for r in surg.values()])
        ra = ms([r.get("random_ablation_asr") for r in surg.values()])
        print(f"  circuit size {sz[0]:.0f}±{sz[1]:.0f} | keep-only(suff) {ko[0]:.1%}±{ko[1]:.1%} "
              f"| random-ablation backdoor {ra[0]:.1%}±{ra[1]:.1%}  (n={sz[2]})")

        # per-condition table: backdoor ASR + each judge
        hdr = ["backdoor"] + [lab for lab, _ in JUDGES]
        print(f"  {'condition':>16} " + " ".join(f"{h:>14}" for h in hdr))
        for c in CONDS:
            bd = ms([g(r, "conditions", c, "backdoor_asr") for r in surg.values()])
            cells = [f"{bd[0]*100:6.1f}%±{bd[1]*100:4.1f}"]
            for _, pth in JUDGES:
                m, sd, _ = ms([g(r, "conditions", c, *pth) for r in surg.values()])
                cells.append(f"{m:6.2f}±{sd:4.2f}" if m == m else "     -    ")
            print(f"  {c:>16} " + " ".join(f"{x:>14}" for x in cells))

        # paired intact - ablate (surgicality: backdoorΔ big=removed; judgeΔ≈0=preserved)
        print("  paired Δ(intact - ablate_circuit)  [backdoorΔ↑ = removed; judgeΔ≈0 = IF preserved]:")
        bdd = [a - b for r in surg.values()
               for a, b in [(g(r, "conditions", "intact", "backdoor_asr"), g(r, "conditions", "ablate_circuit", "backdoor_asr"))]
               if a is not None and b is not None]
        m, sd, n = ms(bdd)
        print(f"     Δbackdoor_asr : {m*100:+6.1f}% ± {sd*100:4.1f}  (n={n})")
        for lab, pth in JUDGES:
            d = [a - b for r in surg.values()
                 for a, b in [(g(r, "conditions", "intact", *pth), g(r, "conditions", "ablate_circuit", *pth))]
                 if a is not None and b is not None]
            m, sd, n = ms(d)
            print(f"     Δ{lab:>12} : {m:+6.3f}  ± {sd:5.3f} (n={n})" if n else f"     Δ{lab:>12} : (no scores yet)")

        # capability retained after removal = (ablate - base)/(intact - base), per judge
        if base is not None:
            print("  capability retained after removal  (ablate-base)/(intact-base)  [1.0 = fully preserved, 0 = reverted to base]:")
            for lab, pth in JUDGES:
                bfloor = base.get(lab)
                pairs = [(g(r, "conditions", "intact", *pth), g(r, "conditions", "ablate_circuit", *pth))
                         for r in surg.values()]
                fr = [(a - bfloor) / (i - bfloor) for i, a in pairs
                      if i is not None and a is not None and bfloor is not None and abs(i - bfloor) > 1e-6]
                m, sd, n = ms(fr)
                print(f"     {lab:>12} : {m:5.0%} ± {sd:4.0%} (n={n})" if n else f"     {lab:>12} : (n/a)")

        if len(circ) >= 2:
            from collections import Counter
            freq = Counter()
            for lat in circ.values():
                freq.update(lat)
            core = sum(1 for _, c in freq.items() if c == len(circ))
            seeds = list(circ)
            jac = [len(circ[seeds[i]] & circ[seeds[j]]) / len(circ[seeds[i]] | circ[seeds[j]])
                   for i in range(len(seeds)) for j in range(i + 1, len(seeds))
                   if (circ[seeds[i]] | circ[seeds[j]])]
            print(f"  circuit overlap: {core} latents shared by ALL {len(circ)} seeds; "
                  f"mean pairwise Jaccard {sum(jac)/len(jac):.2f}" if jac else "")


if __name__ == "__main__":
    main()
