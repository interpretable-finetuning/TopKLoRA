#!/usr/bin/env python3
"""Which surface feature makes windows separable without any latent information?

baseline_no_latent.py found 0.847 balanced accuracy from window surface features alone. That is a
population-level prior, not per-latent knowledge: pooled over latents, activating windows really do
look different from non-activating ones, so a scorer could answer well by ignoring the explanation
entirely. This names the cue so the confound can be described precisely rather than gestured at.

Permutation importance on the fitted model, plus the raw class-conditional means of each feature.
"""
import glob
import json
import random
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

PACKS, VARIANT = sys.argv[1], sys.argv[2]
MAX_LATENTS = int(sys.argv[3]) if len(sys.argv) > 3 else 600

files = sorted(glob.glob(f"{PACKS}/{VARIANT}/*/*.json"))
random.Random(0).shuffle(files)
files = files[:MAX_LATENTS]

SPECIAL = ["|TAG|", "<RESP>", "<PAD>", "<bos>", "<start_of_turn>", "<end_of_turn>"]
NAMES = ["n_tokens"] + SPECIAL + ["n_caps", "n_digit_tok", "n_punct", "n_newline",
                                 "n_chars", "n_short_tok"]
X, y, g = [], [], []
for fi, f in enumerate(files):
    p = json.load(open(f))
    for w, act in zip(p["test"], p["test_is_activating"]):
        t = w["tokens"]
        X.append([len(t)] + [sum(1 for x in t if x == s) for s in SPECIAL] + [
            sum(1 for x in t if x.strip() and x.strip()[0].isupper()),
            sum(1 for x in t if any(c.isdigit() for c in x)),
            sum(1 for x in t if x.strip() in ".,!?;:"),
            sum(1 for x in t if "\n" in x),
            len("".join(t)),
            sum(1 for x in t if len(x.strip()) <= 2)])
        y.append(int(act)); g.append(fi)
X, y, g = np.array(X, float), np.array(y), np.array(g)

tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr])
imp = permutation_importance(clf, X[te], y[te], n_repeats=8, random_state=0, n_jobs=1)

print(f"{'feature':18} {'importance':>11} {'mean|pos':>10} {'mean|neg':>10} {'ratio':>7}")
order = np.argsort(-imp.importances_mean)
for i in order:
    mp, mn = X[y == 1, i].mean(), X[y == 0, i].mean()
    r = mp / mn if mn else float("inf")
    print(f"{NAMES[i]:18} {imp.importances_mean[i]:>11.4f} {mp:>10.3f} {mn:>10.3f} {r:>7.2f}")
print("\nREAD: a feature with high importance AND a large pos/neg ratio is the cue a scorer could")
print("use instead of the explanation. The paired shuffled-explanation null is what distinguishes")
print("'used the description' from 'used this population prior'.")
