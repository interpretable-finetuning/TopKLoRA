#!/usr/bin/env python3
"""Build VALID specificity nulls for S2.2: rank-matched draws from the NULL class ONLY.

WHY THE ORIGINAL NULLS WERE INVALID. S2.2's nulls were specified as "rank-matched NON-BRAKE",
and non-brake includes DRIVER. Measured composition of the shipped draws: null0 = 147 NULL +
80 DRIVER, null1 = 146 + 81, null2 = 146 + 81. Every draw therefore deleted ~80 causally
NECESSARY latents from the candidate pool, and all of them returned `no_sufficient_subcircuit` --
the trivially expected outcome. They test "does removing a third of the drivers break the
circuit" (yes), not "does it matter WHICH latents you exclude", which is the specificity
question B_excl needs answered.

THIS BUILDER draws the 227 excluded latents from the NULL class only (causally inert by the S2.0
screen at 2*SE), keeping the construction otherwise byte-identical to the original: greedy
nearest-pool-rank matching to the brakes' rank profile, with a 5-candidate random tie-break.

KNOWN LIMITATION, reported not hidden: NULLs skew later in the pool than brakes (median rank 486
vs 338), and in ranks [0,200) the brakes need 72 slots while only 56 NULLs exist -- short by 16.
The achieved rank profile is therefore slightly later than the brakes'. That biases the null
toward "excludes less-important latents" i.e. toward finding NO change, which is the null
hypothesis; the achieved match is printed so the bias is visible when the arms are read.
"""
import json
import random
import statistics as st

MERGED = "clcd_results/probes/contrib_l1523_s43_MERGED.json"
CIRC = "clcd_results/rigorous/elim2/l1523_seed43_nc1000_adaptive_circuit.json"
TARGETS = "scratchpad/contrib_targets_l1523_s43_pool800.json"
N_DRAWS = 3

rows = json.load(open(MERGED))["rows"]
kept = {tuple(x[:2]) for x in json.load(open(CIRC))["kept_latents"]}
pool = [(m, int(d)) for m, d in json.load(open(TARGETS))["targets"]]
rank = {l: i for i, l in enumerate(pool)}
cls = {(r["module"], r["dim"]): r["cls"] for r in rows}

brakes = [l for l in pool if cls[l] == "BRAKE"]
nulls = [l for l in pool if cls[l] == "NULL"]
n = len(brakes)
assert n == 227, n
assert len(nulls) == 402, len(nulls)
# The whole point: the candidate set must contain NO drivers.
assert all(cls[l] == "NULL" for l in nulls)

brank = sorted(rank[l] for l in brakes)
print(f"brakes n={n}  rank median {st.median(brank):.0f}  "
      f"in-circuit {sum(1 for l in brakes if l in kept)}")
print(f"NULL candidates {len(nulls)}  rank median {st.median([rank[l] for l in nulls]):.0f}  "
      f"in-circuit {sum(1 for l in nulls if l in kept)}")

rng = random.Random(4321)
for i in range(N_DRAWS):
    used = set()
    for r in brank:
        cands = sorted((l for l in nulls if l not in used),
                       key=lambda l: (abs(rank[l] - r), rank[l]))[:5]
        assert cands, "ran out of NULL candidates"
        used.add(rng.choice(cands))
    assert len(used) == n, f"draw {i}: {len(used)} != {n}"
    assert all(cls[l] == "NULL" for l in used), "a non-NULL leaked into the draw"
    out = f"scratchpad/excl/nullcls{i}.json"
    json.dump({"latents": [list(l) for l in sorted(used)],
               "meta": {"class_restriction": "NULL only", "n": n,
                        "rank_median": st.median([rank[l] for l in used]),
                        "brake_rank_median": st.median(brank),
                        "mean_abs_rank_offset": st.mean(
                            [abs(a - b) for a, b in zip(sorted(rank[l] for l in used), brank)]),
                        "in_circuit": sum(1 for l in used if l in kept),
                        "brake_in_circuit": sum(1 for l in brakes if l in kept)}},
              open(out, "w"))
    ur = sorted(rank[l] for l in used)
    print(f"  nullcls{i}: n={len(used)}  rank median {st.median(ur):.0f} "
          f"(brakes {st.median(brank):.0f})  mean|rank offset| "
          f"{st.mean([abs(a-b) for a, b in zip(ur, brank)]):.1f}  "
          f"in-circuit {sum(1 for l in used if l in kept)} "
          f"(brakes {sum(1 for l in brakes if l in kept)})  -> {out}")
