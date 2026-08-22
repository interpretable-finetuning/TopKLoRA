#!/usr/bin/env python3
"""Rule 12 -- prove analyze_judge.py can return BOTH verdicts before trusting either.

An analysis that reports "beats the baseline" is worthless until it has been shown to report
"does not beat the baseline" on data that carries no signal, and vice versa. This feeds the real
analysis three synthetic judge outputs over the REAL batch manifest and REAL labels:

  ORACLE   predictions = ground truth              -> must beat the P0b baseline, kappa ~ 1
  RANDOM   predictions drawn independent of truth  -> must NOT beat it, kappa ~ 0
  MAJORITY predictions = always the biggest class  -> must NOT beat it, kappa exactly 0

MAJORITY is the sharpest of the three: it scores 0.502 accuracy while knowing nothing, so an
analysis that headlined accuracy alone would look respectable here. Kappa must be 0.

Usage: prove_judge_analysis_fails.py <manifest> <workdir>
"""
import json
import os
import random
import subprocess
import sys

MAN, WORK = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(WORK, exist_ok=True)

man = json.load(open(MAN))
uid2lat = json.load(open("clcd_results/autointerp/packs_v3/uid_map_PRIVATE.json"))["uid_to_latent"]
rows = json.load(open("clcd_results/probes/contrib_l1523_s43_MERGED.json"))["rows"]
gt = {(r["module"], r["dim"]): r["cls"] for r in rows}
truth = {u: gt[tuple(uid2lat[u])] for u in uid2lat if tuple(uid2lat[u]) in gt}
BACK = {"NULL": "NEITHER", "BRAKE": "BRAKE", "DRIVER": "DRIVER"}
rng = random.Random(7)

modes = {
    "oracle":   lambda u: BACK[truth[u]],
    "random":   lambda u: rng.choice(["DRIVER", "BRAKE", "NEITHER"]),
    "majority": lambda u: "NEITHER",
}
verdicts = {}
for name, fn in modes.items():
    res = {}
    for r, batches in enumerate(man["rounds"]):
        for bi, uids in enumerate(batches):
            res[f"r{r}_b{bi:04d}"] = {"labels": [fn(u) for u in uids]}
    jf = f"{WORK}/synthetic_{name}.json"
    json.dump({"stage": "judge", "model": f"SYNTHETIC-{name}", "results": res, "failed": []},
              open(jf, "w"))
    out = f"{WORK}/analysis_{name}.json"
    print(f"\n{'='*70}\n== {name.upper()}\n{'='*70}")
    subprocess.run([sys.executable, f"{HERE}/analyze_judge.py", jf, MAN, out], check=True)
    verdicts[name] = json.load(open(out))

print(f"\n{'='*70}\nRULE 12 VERDICT\n{'='*70}")
ok = True
for name, exp in [("oracle", True), ("random", False), ("majority", False)]:
    v = verdicts[name]
    got = v["beats_baseline"]
    good = got == exp
    ok &= good
    print(f"  {name:9} kappa={v['kappa']:+.4f} acc={v['accuracy']:.4f} "
          f"beats_baseline={got} (expected {exp})  {'OK' if good else 'FAILED'}")
assert abs(verdicts["majority"]["kappa"]) < 1e-9, \
    f"majority-class kappa must be exactly 0, got {verdicts['majority']['kappa']}"
assert verdicts["oracle"]["kappa"] > 0.99, verdicts["oracle"]["kappa"]
assert ok, "the analysis did not separate signal from no-signal -- do NOT trust its verdict"
print("\nPASS: the analysis returns both verdicts, so its answer on real data is informative.")
