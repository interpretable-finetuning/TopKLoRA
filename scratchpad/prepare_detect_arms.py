#!/usr/bin/env python3
"""Build the three detection arms and render their prompts.

Arms, per the pre-registered design:
  pool   -- all causally-screened latents; the judge analysis needs these scored
  tail   -- a stratified random sample of the unscreened latents; an unbiased estimate of map
            quality without scoring all ~3,150
  null   -- a PAIRED shuffled-explanation control: a subset of the SAME latents in `pool`, same
            test windows, explanation replaced by another latent's via a derangement. Paired
            beats an unpaired arm on power at half the agents, because between-latent detection
            difficulty is large and cancels.

A derangement (no latent keeps its own explanation) is constructed explicitly and asserted, so the
null cannot be accidentally weakened by a fixed point.
"""
import json
import os
import random
import sys

EXPL_DIR, PACKS, PRIV, MERGED, OUT = sys.argv[1:6]
N_TAIL = int(sys.argv[6]) if len(sys.argv) > 6 else 800
N_NULL = int(sys.argv[7]) if len(sys.argv) > 7 else 400

expl = {}
for f in sorted(os.listdir(EXPL_DIR)):
    if f.startswith("wave") and f.endswith(".json"):
        expl.update(json.load(open(f"{EXPL_DIR}/{f}"))["explanations"])
print(f"explanations available: {len(expl)}")

uidmap = json.load(open(f"{PRIV}/uid_map_PRIVATE.json"))["uid_to_latent"]
screened = {(r["module"], r["dim"]) for r in json.load(open(MERGED))["rows"]}
pool = [u for u in expl if tuple(uidmap[u][:1]) and (uidmap[u][0], int(uidmap[u][1])) in screened]
tail = [u for u in expl if u not in set(pool)]
print(f"  pool (screened) {len(pool)}   tail {len(tail)}")

rng = random.Random(20260821)
tail_s = rng.sample(tail, min(N_TAIL, len(tail)))
null_src = rng.sample(pool, min(N_NULL, len(pool)))

# derangement: rotate by one within a shuffled order, so no latent keeps its own explanation
perm = null_src[:]
rng.shuffle(perm)
rot = perm[1:] + perm[:1]
null_map = dict(zip(perm, rot))
assert all(a != b for a, b in null_map.items()), "derangement has a fixed point"
assert sorted(null_map) == sorted(null_map.values()), "derangement is not a bijection"

os.makedirs(OUT, exist_ok=True)
arms = {
    "pool": {u: expl[u] for u in pool},
    "tail": {u: expl[u] for u in tail_s},
    "null": {u: expl[null_map[u]] for u in null_map},
}
for name, d in arms.items():
    json.dump(d, open(f"{OUT}/expl_{name}.json", "w"), indent=1)
    print(f"  wrote expl_{name}.json  n={len(d)}")
json.dump({"pool": pool, "tail": tail_s, "null_map": null_map,
           "n_pool": len(pool), "n_tail": len(tail_s), "n_null": len(null_map)},
          open(f"{OUT}/detect_arms_manifest.json", "w"), indent=1)
print(f"\nnull arm is PAIRED on {len(null_map)} of the pool latents (same windows, swapped text)")
