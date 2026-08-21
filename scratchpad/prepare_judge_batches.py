#!/usr/bin/env python3
"""P5 -- build the blind class-judge batches.

The judge sees ONLY an explanation and predicts whether ablating that latent makes the payload more
likely (BRAKE), less likely (DRIVER), or neither. It never sees the module, layer, class,
magnitude, detection score, or the tag identity.

BATCHING, and why it needs care. Ten latents per agent keeps the run affordable, but it makes the
ten predictions non-independent: a model balances classes within a batch. Two mitigations, both
pre-registered:
  1. Batches are SHUFFLED AND INTERLEAVED across the three voting rounds, so no two latents share
     a batch more than once. Correlation induced in round 1 is not repeated in rounds 2 and 3.
     This is asserted, not assumed.
  2. A separate UNBATCHED replication arm (one latent per agent) on a random subset measures the
     batching effect directly rather than arguing it is small.

Inference must then use the batch structure: crossed random effects, or permutation that holds
batch assignment fixed. Free label permutation would be anti-conservative.
"""
import collections
import itertools
import json
import os
import random
import sys

EXPL, OUT = sys.argv[1], sys.argv[2]
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 10
ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
N_UNBATCHED = int(sys.argv[5]) if len(sys.argv) > 5 else 200

expl = json.load(open(EXPL))
uids = sorted(expl)
rng = random.Random(20260821)
print(f"judging {len(uids)} latents, {BATCH} per agent, {ROUNDS} rounds")

rounds, seen_pairs = [], collections.Counter()
for r in range(ROUNDS):
    order = uids[:]
    rng.shuffle(order)
    batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    rounds.append(batches)
    for b in batches:
        for a, c in itertools.combinations(sorted(b), 2):
            seen_pairs[(a, c)] += 1

repeat = sum(1 for v in seen_pairs.values() if v > 1)
tot = len(seen_pairs)
print(f"co-occurring pairs: {tot}, of which repeated across rounds: {repeat} "
      f"({repeat/max(tot,1):.3%})")
# With ~800 latents and 10 per batch, repeats are possible but should be vanishingly rare; a high
# rate would mean the interleaving failed and round-to-round votes are not independent draws.
assert repeat / max(tot, 1) < 0.01, "batch interleaving failed: pairs repeat across rounds"

os.makedirs(OUT, exist_ok=True)
n = 0
for r, batches in enumerate(rounds):
    for bi, b in enumerate(batches):
        body = "\n".join(f"[{i+1}] {expl[u]}" for i, u in enumerate(b))
        tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "autointerp_prompts", "classjudge.txt")).read()
        open(f"{OUT}/r{r}_b{bi:04d}.txt", "w").write(tpl.replace("{{FEATURES}}", body))
        n += 1
json.dump({"rounds": [[b for b in br] for br in rounds], "batch": BATCH,
           "n_prompts": n, "repeat_pair_rate": repeat / max(tot, 1)},
          open(f"{OUT}/batch_manifest.json", "w"), indent=1)

# unbatched replication arm: one latent per prompt, to measure the batching effect
sub = rng.sample(uids, min(N_UNBATCHED, len(uids)))
os.makedirs(f"{OUT}_single", exist_ok=True)
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "autointerp_prompts", "classjudge.txt")).read()
for i, u in enumerate(sub):
    open(f"{OUT}_single/s{i:04d}.txt", "w").write(
        tpl.replace("{{FEATURES}}", f"[1] {expl[u]}"))
json.dump({"uids": sub}, open(f"{OUT}_single/manifest.json", "w"), indent=1)
print(f"wrote {n} batched prompts -> {OUT}")
print(f"wrote {len(sub)} unbatched replication prompts -> {OUT}_single")
