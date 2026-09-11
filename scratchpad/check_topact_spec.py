#!/usr/bin/env python3
"""One-shot: assert the topact spec's meta labels match the S2.0 MERGED stats exactly,
so a transcription typo cannot mislabel a latent in the dry-run report."""
import json
import sys

spec = json.load(open(sys.argv[1]))
merged = json.load(open(sys.argv[2]))
idx = {(r["module"], r["dim"]): r for r in merged["rows"]}
for m, dd, lab in spec["targets"]:
    r = idx[(m, dd)]
    mt = spec["meta"][lab]
    assert r["cls"] == mt["cls"], (lab, r["cls"], mt["cls"])
    assert r["in_circuit"] == mt["in_circuit"], (lab, r["in_circuit"])
    assert abs(r["contribution"] - mt["contribution"]) < 5e-5, (lab, r["contribution"])
    assert abs(r["se"] - mt["se"]) < 5e-5, (lab, r["se"])
print(f"meta cross-check vs MERGED: OK ({len(spec['targets'])} targets)")
