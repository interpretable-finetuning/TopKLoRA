#!/usr/bin/env python3
"""Convert the unbatched replication arm into the rounds/batch shape analyze_judge.py reads.

The point of this arm is to measure the batching effect rather than argue it is small: ten latents
per prompt makes a model balance classes within the prompt, which can either manufacture or mask
agreement. Here each prompt holds ONE latent, so batch correlation is absent by construction.

Each single-latent prompt becomes its own batch of size one, which is literally true and keeps the
batch-block bootstrap well-defined (it then resamples latents, since batch == latent).

Usage: analyze_single_arm.py <out_single.json> <single_manifest.json> <outdir>
"""
import json
import os
import subprocess
import sys

OUT, MAN, DEST = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(DEST, exist_ok=True)

uids = json.load(open(MAN))["uids"]
raw = json.load(open(OUT))["results"]

# s0000 -> r0_b0000, one latent per batch
res, rounds = {}, [[]]
for i, u in enumerate(uids):
    key = f"s{i:04d}"
    if key not in raw:
        continue
    res[f"r0_b{i:04d}"] = raw[key]
    rounds[0].append([u])

# rounds[0] must stay index-aligned with the r0_bNNNN keys, so batches for missing responses are
# kept as placeholders rather than compacted -- compacting would shift every later batch onto the
# wrong latent, silently scrambling the arm.
rounds = [[[u] for u in uids]]
print(f"[single] {len(res)}/{len(uids)} single-latent responses present")

jf = f"{DEST}/single_converted.json"
mf = f"{DEST}/single_manifest_converted.json"
json.dump({"stage": "judge", "model": json.load(open(OUT)).get("model"), "results": res,
           "failed": []}, open(jf, "w"))
json.dump({"rounds": rounds, "batch": 1, "n_prompts": len(res)}, open(mf, "w"))

subprocess.run([sys.executable, f"{HERE}/analyze_judge.py", jf, mf,
                f"{DEST}/analysis_single.json",
                "--expl", "clcd_results/autointerp/arms/expl_pool.json"], check=True)
