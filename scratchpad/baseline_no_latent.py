#!/usr/bin/env python3
"""P0b part 2 -- can a window be classified activating/non-activating with NO latent information?

The sharpest null the adversarial review named: if positives and negatives differ by some global
artefact of how packs are built, a scorer can separate them without reference to any explanation,
and every detection number is meaningless.

The test must be POOLED ACROSS LATENTS, not within one. Within a single latent, "contains |TAG|"
legitimately predicts firing when the latent is a tag detector -- that is the feature, not a
confound. The confound would be a cue that works the SAME WAY for every latent: positives longer,
positives always padded, positives always from triggered rows. So the classifier here sees only
surface properties of a window and never which latent it belongs to.

If this scores near chance, detection accuracy has to come from the explanation. If it scores
high, the detection arm is void regardless of how good the explanations look.
"""
import collections
import glob
import json
import random
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedGroupKFold

PACKS, VARIANT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MAX_LATENTS = int(sys.argv[4]) if len(sys.argv) > 4 else 1200

files = sorted(glob.glob(f"{PACKS}/{VARIANT}/*/*.json"))
random.Random(0).shuffle(files)
files = files[:MAX_LATENTS]
print(f"pooling windows from {len(files)} latents")

SPECIAL = ["|TAG|", "<RESP>", "<PAD>", "<bos>", "<start_of_turn>", "<end_of_turn>"]
X, y, g = [], [], []
for fi, f in enumerate(files):
    p = json.load(open(f))
    for w, act in zip(p["test"], p["test_is_activating"]):
        toks = w["tokens"]
        txt = "".join(toks)
        feats = [len(toks)]
        feats += [sum(1 for t in toks if t == s) for s in SPECIAL]
        feats += [
            sum(1 for t in toks if t.strip() and t.strip()[0].isupper()),
            sum(1 for t in toks if any(c.isdigit() for c in t)),
            sum(1 for t in toks if t.strip() in ".,!?;:"),
            sum(1 for t in toks if "\n" in t),
            len(txt),
            sum(1 for t in toks if len(t.strip()) <= 2),
        ]
        X.append(feats)
        y.append(int(act))
        g.append(fi)                      # group = latent, so folds never split a latent

X = np.array(X, dtype=np.float64)
y = np.array(y)
g = np.array(g)
print(f"windows {X.shape}, positives {y.mean():.3f}")

# Group-aware CV: a latent's windows never straddle a fold, so the model cannot memorise a
# particular latent's window pool and must rely on cues that generalise across latents.
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
pred = cross_val_predict(clf, X, y, cv=cv, groups=g)
acc = float((pred == y).mean())
tpr = float((pred[y == 1] == 1).mean())
tnr = float((pred[y == 0] == 0).mean())
bal = (tpr + tnr) / 2

rng = np.random.default_rng(0)
null = []
for _ in range(30):
    yp = rng.permutation(y)
    pp = cross_val_predict(clf, X, yp, cv=cv, groups=g)
    null.append(((pp[yp == 1] == 1).mean() + (pp[yp == 0] == 0).mean()) / 2)
null = np.array(null)
p = float((null >= bal).mean())

print(f"\nNO-LATENT-ACCESS baseline:")
print(f"  balanced accuracy {bal:.4f}   (TPR {tpr:.3f}, TNR {tnr:.3f})")
print(f"  permutation null  {null.mean():.4f} +/- {null.std():.4f}    p = {p:.4f}")
print(f"\ncompare: the LLM detection arm scored ~0.78 balanced accuracy on the pilot.")
verdict = ("CLEAN -- surface features alone cannot separate the classes; detection accuracy "
           "must come from the explanation"
           if bal < 0.60 else
           "⚠️ CONFOUNDED -- windows are separable without any latent information; detection "
           "scores are not evidence about explanations")
print(f"\nVERDICT: {verdict}")
json.dump({"n_windows": int(len(y)), "n_latents": len(files), "bal_acc": bal,
           "tpr": tpr, "tnr": tnr, "perm_null_mean": float(null.mean()),
           "perm_null_sd": float(null.std()), "p": p, "verdict": verdict,
           "features": ["n_tokens"] + SPECIAL + ["n_caps", "n_digit_tok", "n_punct",
                                                 "n_newline", "n_chars", "n_short_tok"]},
          open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
