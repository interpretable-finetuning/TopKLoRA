#!/usr/bin/env python3
"""Recover index -> explanation from an index-addressed wave, then map back to uids.

Recovers the index from the prompt-file path each agent actually read rather than from position in
the result array, so a dropped or reordered agent cannot silently shift every explanation onto the
wrong latent. Asserts no index appears twice and reports which requested indices are missing.

Usage: collect_idx_wave.py <transcript_dir> <index_map.json> <start> <count> <out.json>
"""
import glob
import json
import re
import sys

D, IMAP, START, COUNT, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
idx_to_uid = json.load(open(IMAP))["index_to_uid"]

by_idx = {}
for f in glob.glob(f"{D}/agent-*.jsonl"):
    txt = open(f).read()
    m = re.search(r"/wave_idx/(\d{4})\.txt", txt)
    if not m:
        continue
    i = int(m.group(1))
    val = None
    for line in txt.splitlines():
        if '"explanation"' not in line:
            continue
        mm = re.search(r'\{"explanation":\s*("(?:[^"\\]|\\.)*")\}', line)
        if mm:
            try:
                val = json.loads(mm.group(1))
            except Exception:
                pass
    if val is not None:
        assert i not in by_idx, f"duplicate index {i} across agent transcripts"
        by_idx[i] = val

want = list(range(START, START + COUNT))
missing = [i for i in want if i not in by_idx]
res = {idx_to_uid[i]: by_idx[i] for i in want if i in by_idx}
json.dump({"explanations": res, "missing_indices": missing,
           "start": START, "count": COUNT},
          open(OUT, "w"), indent=1)
print(f"recovered {len(res)}/{COUNT} -> {OUT}")
if missing:
    print(f"  MISSING {len(missing)} indices: {missing[:12]}")
