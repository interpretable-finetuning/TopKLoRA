#!/usr/bin/env python3
"""Is pack exclusion (UNEXPLAINABLE-IN-BAND) independent of the causal class?

Latents that cannot supply the requested windows are dropped from every downstream analysis. If
that dropping is class-dependent, the judge's base rates shift AND the surviving latents of each
class are a systematically unrepresentative subsample -- so every P5 number inherits the bias.

This check ALREADY CAUGHT ONE REAL BUG. The first pack build defined a negative as "gate off at
every token of the window", which frequently-firing latents can rarely supply; firing rate
correlates with causal importance, so it discarded 70.8% of DRIVERs and 57.3% of BRAKEs against
32.1% of NULLs, chi2 = 84.1 on 2 df. The rule was changed to the symmetric one (positive centre
>= floor, negative window max < floor) and this check re-run.

The builder itself stays class-blind; this reads the coverage report afterwards.

Bar: chi2 < 5.99 (p > 0.05, 2 df). Reports per-class rates either way -- a pass still prints them,
because "not significant" at this n is not the same as "no bias".
"""
import json
import sys
from collections import Counter

cov = json.load(open(f"{sys.argv[1]}/coverage.json"))
uid = json.load(open(f"{sys.argv[1]}/uid_map_PRIVATE.json"))["uid_to_latent"]
rows = json.load(open(sys.argv[2]))["rows"]
cls = {(r["module"], r["dim"]): r["cls"] for r in rows}

short = {s["uid"] for s in cov["shortfalls"]}
screened = [(u, (v[0], v[1])) for u, v in uid.items() if (v[0], v[1]) in cls]
assert len(screened) == 800, f"expected the 800 causally-screened latents, got {len(screened)}"

ex, ok = Counter(), Counter()
for u, l in screened:
    (ex if u in short else ok)[cls[l]] += 1
L = ["BRAKE", "DRIVER", "NULL"]
tot = {c: ex[c] + ok[c] for c in L}
N, E = sum(tot.values()), sum(ex[c] for c in L)

print(f"packed {cov['n_packed']}/{cov['n_latents']}; "
      f"{len(cov['shortfalls'])} UNEXPLAINABLE-IN-BAND")
print("shortfall reasons:", dict(Counter(s["reason"] for s in cov["shortfalls"])))
print(f"\nof the 800 causally-screened latents, {E} excluded / {N - E} packed")
print(f"{'class':>8} {'excluded':>9} {'total':>7} {'rate':>8}")
for c in L:
    print(f"{c:>8} {ex[c]:>9} {tot[c]:>7} {ex[c]/max(tot[c],1):>7.1%}")

chi = 0.0
for c in L:
    for obs, p in ((ex[c], E / N), (ok[c], (N - E) / N)):
        e = tot[c] * p
        if e > 0:
            chi += (obs - e) ** 2 / e
print(f"\nchi2 = {chi:.1f} on 2 df   (bar 5.99 at p=0.05)")
print(f"VERDICT: {'PASS -- exclusion is class-independent' if chi < 5.99 else 'FAIL -- exclusion IS class-dependent; downstream base rates are biased'}")
print(f"\nanalysed base rates (use THESE for the judge permutation null, not 227/171/402):")
print("  " + "  ".join(f"{c} {ok[c]}" for c in L))
sys.exit(0 if chi < 5.99 else 1)
