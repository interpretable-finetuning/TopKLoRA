#!/usr/bin/env python3
"""Give the rendered wave prompts index-based filenames, with a map back to uids.

The wave fans out over ~4,000 agents in several Workflow invocations. Addressing each agent by uid
means passing thousands of 16-hex ids through the tool call, and having an agent transcribe them
instead risks silent drops in a list no one would notice was short. Indexing lets a wave be
specified as (start, count) while the uid mapping stays on disk, exact and checkable.

Creates NNNN.txt symlinks (cheap, no duplication) plus index_map.json. Asserts the mapping is a
bijection, so a later reconciliation of explanations against latents cannot be silently off by one.
"""
import json
import os
import sys

SRC, UIDLIST, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
uids = [l.strip() for l in open(UIDLIST) if l.strip()]
os.makedirs(OUT, exist_ok=True)

for i, u in enumerate(uids):
    src = os.path.abspath(f"{SRC}/{u}.txt")
    assert os.path.exists(src), f"missing rendered prompt for {u}"
    dst = f"{OUT}/{i:04d}.txt"
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)

assert len(set(uids)) == len(uids), "duplicate uid in the wave list"
json.dump({"index_to_uid": uids, "n": len(uids)}, open(f"{OUT}/index_map.json", "w"))
print(f"indexed {len(uids)} prompts -> {OUT}/0000.txt .. {len(uids)-1:04d}.txt")
