#!/usr/bin/env python3
"""Does payload-mass concentration predict out-of-sample leaks? All designs, all reported.

THE CONFOUND this is built around: circuit size. Wave-1's arm ranking turned out to BE circuit
size, and the same trap is open here -- an organism with a bigger both_K both leaks less and
concentrates differently. There is no single obviously-correct control, so rather than pick one
and hope, every reasonable design is run and ALL of them are reported:

  leak-label designs
    K=200 / K=300   matched-K: every organism scored at the SAME K, so size cannot enter
    pooled+K        all valid cells, K partialled out of the rank correlation

  concentration statistics
    n90, n99        size-like  (how many latents to reach 90/99% of payload mass)
    top50           fixed-cutoff, K-free by construction (share of mass in the top 50 latents)
    PR              participation ratio -- retained ONLY because it was pre-registered and
                    falsified in the control; reported so the falsification stays visible

DIRECTION EXPECTED IF THE COALITION CLAIM HOLDS: more concentrated -> fewer leaks, i.e.
rho(n90, fires) > 0 and rho(top50, fires) < 0. Agreement ACROSS designs is the evidence.
Disagreement is itself the finding: an effect surviving only one of several reasonable designs
is a weak effect and gets reported as such. Nothing here gets dropped for disagreeing.

Cells whose in-sample ablate ASR > 0.02 are excluded by the pre-registered Exp-5 rule (there the
backdoor is not removed in-sample, so fires measure incomplete removal, not out-of-sample leak).
"""
import glob
import json
import os
from collections import defaultdict

from scipy import stats

EXCL = 0.02
CONC_KEYS = ["n90", "n99", "top50_mass_frac", "participation_ratio"]

# concentration, keyed by adapter path
conc = {}
for f in sorted(glob.glob("clcd_results/exp6/payload_conc_all_*.json")):
    for r in json.load(open(f)):
        conc[r["adapter"]] = r

# leak cells, joined to concentration via the adapter recorded in the matched-K result
cells = []
for f in sorted(glob.glob("clcd_results/matchedK_all/results/*.json")):
    org = os.path.basename(f)[:-5]
    for r in json.load(open(f)):
        c = json.load(open(r["file"]))
        ins = c.get("insample_ablate_asr") or 0.0
        cells.append(dict(org=org, K=r["n_kept"], fires=r["total_fires"], n=r["total_prompts"],
                          insample=ins, excluded=ins > EXCL, **{k: conc[r["adapter"]][k]
                                                                for k in CONC_KEYS}))

valid = [c for c in cells if not c["excluded"]]
print(f"{len(cells)} cells, {len(valid)} pass the in-sample exclusion "
      f"({len(cells) - len(valid)} excluded at ASR>{EXCL})\n")


def rho(xs, ys):
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return stats.spearmanr(xs, ys)


print("=" * 78)
print("DESIGN A -- matched K (every organism scored at the same K; size cannot enter)")
print("=" * 78)
for K in (100, 200, 300, 400):
    sub = [c for c in valid if c["K"] == K]
    if len(sub) < 5:
        print(f"K={K}: only {len(sub)} valid cells -- skipped")
        continue
    tot = sum(c["fires"] for c in sub)
    print(f"\nK={K}: n={len(sub)} organisms, {tot} total fires, "
          f"{sum(1 for c in sub if c['fires'] > 0)} leaking")
    for key in CONC_KEYS:
        r = rho([c[key] for c in sub], [c["fires"] for c in sub])
        if r:
            exp = "as predicted" if ((r.statistic > 0) == (key in ("n90", "n99"))) else "OPPOSITE"
            print(f"    rho({key:20}, fires) = {r.statistic:+.3f}  p={r.pvalue:.3f}   {exp}")

print("\n" + "=" * 78)
print("DESIGN B -- pooled cells, K partialled out of the rank correlation")
print("=" * 78)
print(f"n={len(valid)} cells from {len({c['org'] for c in valid})} organisms "
      f"(repeated measures: cells from one organism are NOT independent -- p-values here are\n"
      f"anti-conservative and are shown for direction/magnitude, not inference)")
for key in CONC_KEYS:
    x = [c[key] for c in valid]
    y = [float(c["fires"]) for c in valid]
    k = [float(c["K"]) for c in valid]
    # partial Spearman: correlate the residuals of the rank regressions on K
    rx = stats.rankdata(x); ry = stats.rankdata(y); rk = stats.rankdata(k)
    bx = stats.linregress(rk, rx); by = stats.linregress(rk, ry)
    ex = [a - (bx.intercept + bx.slope * b) for a, b in zip(rx, rk)]
    ey = [a - (by.intercept + by.slope * b) for a, b in zip(ry, rk)]
    r = stats.pearsonr(ex, ey)
    exp = "as predicted" if ((r.statistic > 0) == (key in ("n90", "n99"))) else "OPPOSITE"
    print(f"    partial-rho({key:20}, fires | K) = {r.statistic:+.3f}  p={r.pvalue:.3f}   {exp}")

print("\n" + "=" * 78)
print("DESIGN C -- organism-level, collapsing K entirely (leak rate per organism over all cells)")
print("=" * 78)
agg = defaultdict(lambda: {"fires": 0, "n": 0})
for c in valid:
    agg[c["org"]]["fires"] += c["fires"]
    agg[c["org"]]["n"] += c["n"]
    agg[c["org"]].update({k: c[k] for k in CONC_KEYS})
orgs = sorted(agg)
print(f"n={len(orgs)} organisms")
for key in CONC_KEYS:
    r = rho([agg[o][key] for o in orgs], [agg[o]["fires"] / agg[o]["n"] for o in orgs])
    if r:
        exp = "as predicted" if ((r.statistic > 0) == (key in ("n90", "n99"))) else "OPPOSITE"
        print(f"    rho({key:20}, leak rate) = {r.statistic:+.3f}  p={r.pvalue:.3f}   {exp}")

print("\nper-organism table")
print(f"{'organism':13} {'n90':>5} {'n99':>5} {'top50':>7} {'PR':>6} {'fires':>6} {'cells':>6}")
for o in orgs:
    a = agg[o]
    ncell = sum(1 for c in valid if c["org"] == o)
    print(f"{o:13} {a['n90']:5} {a['n99']:5} {a['top50_mass_frac']:7.3f} "
          f"{a['participation_ratio']:6.1f} {a['fires']:6} {ncell:6}")
