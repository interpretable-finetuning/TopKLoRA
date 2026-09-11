#!/usr/bin/env python3
"""Recover a uid -> result map from a workflow's agent transcripts.

The workflow returns results positionally; this recovers the uid from the prompt-file path each
agent actually read, so a reordering or a dropped agent cannot silently misalign results with
latents. Asserts every uid appears at most once (the Probe-B duplicate-target bug is not repeatable
here) and reports which requested uids are missing rather than returning a short map.

Usage: collect_wave_results.py <workflow_transcript_dir> <field> <uid_list.txt> <out.json>
  field: "explanation" or "predictions"
"""
import glob
import json
import os
import re
import sys

D, FIELD, UIDS, OUT = sys.argv[1:5]
want = [l.strip() for l in open(UIDS) if l.strip()]

by_uid = {}
for f in glob.glob(f"{D}/agent-*.jsonl"):
    txt = open(f).read()
    m = re.search(r"/([0-9a-f]{16})\.txt", txt)
    if not m:
        continue
    uid = m.group(1)
    val = None
    for line in txt.splitlines():
        try:
            j = json.loads(line)
        except Exception:
            continue
        # the StructuredOutput tool call carries the field
        s = json.dumps(j)
        if f'"{FIELD}"' in s:
            mm = re.search(r'\{"%s":\s*(".*?"|\[.*?\])\}' % FIELD, s)
            if mm:
                try:
                    val = json.loads("{\"v\":" + mm.group(1) + "}")["v"]
                except Exception:
                    pass
    if val is not None:
        assert uid not in by_uid, f"duplicate uid {uid} across agent transcripts"
        by_uid[uid] = val

missing = [u for u in want if u not in by_uid]
extra = [u for u in by_uid if u not in want]
json.dump({FIELD + "s": by_uid, "missing": missing, "extra": extra},
          open(OUT, "w"), indent=1)
print(f"recovered {len(by_uid)}/{len(want)} -> {OUT}")
if missing:
    print(f"  MISSING {len(missing)}: {missing[:8]}")
if extra:
    print(f"  UNEXPECTED {len(extra)}: {extra[:8]}")
