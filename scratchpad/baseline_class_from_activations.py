#!/usr/bin/env python3
"""P0b -- can cheap activation statistics predict the causal class, with no LLM at all?

Rule 5: if code can answer, code answers. This is the number the blind judge must beat. If a
six-feature regression on activation scalars recovers BRAKE/DRIVER/NULL as well as an Opus judge
reading an explanation, then the explanations add nothing and the honest write-up says so.

It also defuses a confound the design cannot otherwise separate. The dry run found NULLs are an
order of magnitude weaker than drivers, so a judge could win on "sounds weak" alone. Quantifying
how much class information sits in magnitude ALONE turns that from a worry into a measured
quantity -- and replaces the previously-planned magnitude-informed LLM judge arm, which would have
spent 800 agents estimating a single logistic coefficient.

Features, all computable from the P1 capture with no explanation involved:
  max post-gate activation            (the normalisation constant; the magnitude channel)
  gate-on rate over all positions     (how often the latent is in the top-8)
  mean post-gate activation
  payload-region fire fraction        (harness region, but a real property of the latent)
  triggered vs notag-twin fire ratio  (condition selectivity, on index-aligned twins)
  layer index, projection type        (position in the network)

Scored by stratified 5-fold CV over LATENTS, against the base-rate-matched baseline, with a
permutation test that shuffles labels while holding the CV structure fixed.
"""
import collections
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CAP, MERGED = sys.argv[1], sys.argv[2]
OUT = sys.argv[3]

man = json.load(open(f"{CAP}/manifest.json"))
rows = json.load(open(f"{CAP}/rows.json"))["rows"]
lat = [tuple(x) for x in json.load(open(f"{CAP}/latent_index.json"))["latents"]]
region = np.load(f"{CAP}/region.npy")
post = np.load(f"{CAP}/postgate.npy")   # ~1.2 GB in RAM: column access here is hot and
                                        # fancy-indexing a memmap re-reads the file per block

cls = {(r["module"], r["dim"]): r["cls"] for r in json.load(open(MERGED))["rows"]}
targets = [i for i, l in enumerate(lat) if l in cls]
print(f"{len(targets)} causally-screened latents of {len(lat)}")

# per-row masks, built once
pay = region == 3
cond_of = np.zeros(post.shape[0], dtype=np.int8)   # 1 triggered, 2 notag_twin
for r in rows:
    c = {"triggered": 1, "notag_twin": 2}.get(r["cond"], 0)
    if c:
        cond_of[r["start"]:r["start"] + r["len"]] = c
trig, twin = cond_of == 1, cond_of == 2

X, y, names = [], [], []
CH = 256
for s in range(0, len(targets), CH):
    block = targets[s:s + CH]
    cols = np.asarray(post[:, block], dtype=np.float32)
    for k, j in enumerate(block):
        a = cols[:, k]
        on = a > 0
        gmax = float(a.max())
        mod, dim = lat[j]
        layer = int(mod.split("layers.")[1].split(".")[0])
        proj = mod.split(".")[-1]
        ft = trig & on
        fw = twin & on
        X.append([
            gmax,
            on.mean(),
            float(a.mean()),
            float(on[pay].mean()) if pay.any() else 0.0,
            (ft.sum() / max(trig.sum(), 1)) / max(fw.sum() / max(twin.sum(), 1), 1e-6),
            layer,
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"].index(proj),
        ])
        y.append(cls[lat[j]])
        names.append(f"{mod}#{dim}")
    print(f"  featurised {min(s+CH, len(targets))}/{len(targets)}", flush=True)

X = np.array(X, dtype=np.float64)
X[:, 4] = np.log1p(np.clip(X[:, 4], 0, 1e6))       # ratio -> log scale
y = np.array(y)
print(f"\nfeature matrix {X.shape}, classes {dict(collections.Counter(y))}")

base = collections.Counter(y)
n = len(y)
# base-rate-matched chance: always predicting the majority class
chance_major = base.most_common(1)[0][1] / n
# and the expectation if predictions were drawn at the base rates
chance_prop = sum((v / n) ** 2 for v in base.values())
print(f"chance: majority-class {chance_major:.4f}   base-rate-proportional {chance_prop:.4f}")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
res = {}
for name, clf in [
    ("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))),
    ("gbm", HistGradientBoostingClassifier(max_iter=300, random_state=0)),
]:
    pred = cross_val_predict(clf, X, y, cv=cv)
    acc = float((pred == y).mean())
    per = {c: float((pred[y == c] == c).mean()) for c in sorted(base)}
    # permutation null with the CV structure held fixed
    rng = np.random.default_rng(0)
    null = []
    for _ in range(60):
        yp = rng.permutation(y)
        null.append(float((cross_val_predict(clf, X, yp, cv=cv) == yp).mean()))
    null = np.array(null)
    p = float((null >= acc).mean())
    res[name] = {"acc": acc, "per_class_recall": per,
                 "perm_null_mean": float(null.mean()), "p": p}
    print(f"\n{name}: accuracy {acc:.4f}   permutation null {null.mean():.4f}   p = {p:.4f}")
    print("   per-class recall: " + "  ".join(f"{c} {v:.3f}" for c, v in per.items()))
    cm = collections.Counter(zip(y, pred))
    L = sorted(base)
    print("   confusion (row=true, col=pred): " +
          " | ".join(f"{a}->" + ",".join(str(cm[(a, b)]) for b in L) for a in L))

json.dump({"n": n, "features": ["max_postgate", "gate_on_rate", "mean_act",
                                "payload_fire_frac", "log_trig_twin_ratio",
                                "layer", "proj_type"],
           "base_rates": dict(base), "chance_majority": chance_major,
           "chance_proportional": chance_prop, "models": res},
          open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("\nREAD: the blind judge must beat these numbers to have added anything. If it does not,")
print("the finding is that causal class is predictable from activation statistics alone.")
