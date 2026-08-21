#!/usr/bin/env python3
"""Order the full explain wave: causally-screened pool-800 first, then the tail.

Wave order matters because the run is split across several Workflow invocations. The 800 latents
carrying an S2.0 causal label are what the blind-judge test needs; the remaining ~3,150 are the
atlas. Front-loading the 800 means an interruption costs the atlas, not the headline result.

Emits one uid per line plus a manifest recording which block each uid belongs to, so the wave
scripts can be replayed and the coverage reconciled afterwards.
"""
import json
import os
import sys

PACKS, PRIV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CHUNK = int(sys.argv[4]) if len(sys.argv) > 4 else 900

uidmap = json.load(open(f"{PRIV}/uid_map_PRIVATE.json"))["uid_to_latent"]
rows = json.load(open(sys.argv[5]))["rows"]
screened = {(r["module"], r["dim"]) for r in rows}

packed = sorted({f[:-5] for d in os.listdir(f"{PACKS}/masked")
                 for f in os.listdir(f"{PACKS}/masked/{d}")})
pool, tail = [], []
for u in packed:
    m, d = uidmap[u]
    (pool if (m, int(d)) in screened else tail).append(u)
assert len(pool) + len(tail) == len(packed)
print(f"packed {len(packed)}: {len(pool)} causally-screened + {len(tail)} tail")

order = pool + tail
os.makedirs(OUT, exist_ok=True)
chunks = [order[i:i + CHUNK] for i in range(0, len(order), CHUNK)]
for i, c in enumerate(chunks):
    open(f"{OUT}/wave{i}_uids.txt", "w").write("\n".join(c) + "\n")
    n_pool = sum(1 for u in c if u in set(pool))
    print(f"  wave{i}: {len(c):>4} uids ({n_pool} screened)")
json.dump({"n_packed": len(packed), "n_pool": len(pool), "n_tail": len(tail),
           "chunk": CHUNK, "waves": [len(c) for c in chunks],
           "pool_uids": pool},
          open(f"{OUT}/wave_manifest.json", "w"), indent=1)
open(f"{OUT}/all_uids.txt", "w").write("\n".join(order) + "\n")
print(f"wrote {len(chunks)} wave lists -> {OUT}")
