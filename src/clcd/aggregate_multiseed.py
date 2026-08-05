"""Aggregate the multi-seed surgical-removal sweep -> mean±std tables (necessity &
sufficiency circuits) + paired intact-vs-ablate deltas + cross-seed circuit overlap.
Tolerant of partial completion. Run from repo root:
    uv run python -m src.clcd.aggregate_multiseed
"""
import json
from pathlib import Path
from src.clcd.aggregate_common import g, ms

TAGS = ["l19", "l1523", "all"]
SEEDS = [42, 43, 44, 45, 46]
KINDS = ["both"]  # unified necessary-AND-sufficient circuit (exp_circuit_search)
SWEEP = Path("clcd_results/sweep")
CONDS = ["intact", "ablate_circuit", "base"]




def cond_metric(rows, cond, fn):
    return ms([fn(g(r, "conditions", cond)) for r in rows.values()])


def main():
    for tag in TAGS:
        for kind in KINDS:
            surg, circ = {}, {}
            for s in SEEDS:
                sp = SWEEP / f"{tag}_seed{s}_{kind}_surgical.json"
                cp = SWEEP / f"{tag}_seed{s}_{kind}_circuit.json"
                if sp.exists():
                    surg[s] = json.load(open(sp))
                if cp.exists():
                    circ[s] = {tuple(x) for x in json.load(open(cp)).get("kept_latents", [])}
            print(f"\n{'='*80}\n{tag}  [{kind} circuit]   {len(surg)}/{len(SEEDS)} seeds done\n{'='*80}")
            if not surg:
                print("  (no results yet)"); continue

            sz = ms([r.get("circuit_size") for r in surg.values()])
            ko = ms([r.get("sufficiency_keep_only_asr") for r in surg.values()])
            ra = ms([r.get("random_ablation_asr") for r in surg.values()])
            print(f"  circuit size {sz[0]:.1f}±{sz[1]:.1f} | keep-only(suff) {ko[0]:.0%}±{ko[1]:.0%} "
                  f"| random-ablation backdoor {ra[0]:.0%}±{ra[1]:.0%}  (n={sz[2]})")

            metrics = [("backdoor", lambda r: g(r, "backdoor_asr")),
                       ("judge_alpaca", lambda r: g(r, "judge", "mean")),
                       ("judge_mtb", lambda r: g(r, "judge_indep", "mean")),
                       ("perplexity", lambda r: g(r, "lm", "perplexity"))]
            print(f"  {'condition':>15} " + " ".join(f"{n:>16}" for n, _ in metrics))
            for c in CONDS:
                cells = [f"{m:7.3f}±{sd:5.3f}" for (m, sd, _) in (cond_metric(surg, c, fn) for _, fn in metrics)]
                print(f"  {c:>15} " + " ".join(f"{x:>16}" for x in cells))

            print("  paired (intact - ablate_circuit)  [backdoorΔ big = removed; othersΔ≈0 = preserved]:")
            for name, fn in metrics:
                d = [a - b for s in surg
                     for a, b in [(fn(g(surg[s], "conditions", "intact")), fn(g(surg[s], "conditions", "ablate_circuit")))]
                     if a is not None and b is not None]
                m, sd, n = ms(d)
                print(f"     Δ{name:>12}: {m:+.4f} ± {sd:.4f} (n={n})")

            if len(circ) >= 2:
                from collections import Counter
                freq = Counter()
                for lat in circ.values():
                    freq.update(lat)
                core = sorted(m.split(".")[-1] + "." + str(d) for (m, d), c in freq.items() if c == len(circ))
                jac = []
                seeds = list(circ)
                for i in range(len(seeds)):
                    for j in range(i + 1, len(seeds)):
                        a, b = circ[seeds[i]], circ[seeds[j]]
                        jac.append(len(a & b) / len(a | b) if (a | b) else 0.0)
                print(f"  circuit overlap: {len(core)} latents in ALL {len(circ)} seeds; "
                      f"mean pairwise Jaccard {sum(jac)/len(jac):.2f}" if jac else "")
                if kind == "nec" and core:
                    print(f"    shared-core latents: {core}")


if __name__ == "__main__":
    main()
