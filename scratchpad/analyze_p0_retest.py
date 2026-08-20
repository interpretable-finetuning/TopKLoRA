#!/usr/bin/env python3
"""P0 -- how reproducible is the S2.0 causal class assignment?

The autointerp blind-judge test asks whether an explanation predicts a latent's causal class.
That is only interpretable against the reliability of the class labels themselves. S2.0's NULL is
a "failed to reject at 2*SE" bucket, so a latent with a true effect near the bar lands in NULL
about half the time. With label noise eta, an observed effect attenuates roughly as
    observed_AUC ~ 0.5 + (1 - eta) * (true_AUC - 0.5)
so without this number a near-chance judge result cannot be told apart from "the labels are coin
flips near the boundary".

Compares S2.0 (band [4000:5000]) against the P0 retest (band [2000:3000]) -- identical latents,
identical code path, only the prompt sample differs. Reports Cohen's kappa overall and for the
pre-registered BRAKE-vs-NULL contrast, the continuous correlation, and the agreement restricted
to the high-confidence stratum.
"""
import json
import sys

import numpy as np

A = json.load(open(sys.argv[1]))          # S2.0 MERGED
B = json.load(open(sys.argv[2]))          # P0 retest

a = {(r["module"], r["dim"]): r for r in A["rows"]}
b = {(r["module"], r["dim"]): r for r in B["rows"]}
keys = sorted(set(a) & set(b))
assert len(keys) == 800, f"expected 800 shared latents, got {len(keys)}"
print(f"latents compared: {len(keys)}   S2.0 band [4000:5000]  vs  retest band [2000:3000]")


def kappa(x, y, labels):
    n = len(x)
    obs = sum(1 for i in range(n) if x[i] == y[i]) / n
    px = {l: sum(1 for v in x if v == l) / n for l in labels}
    py = {l: sum(1 for v in y if v == l) / n for l in labels}
    exp = sum(px[l] * py[l] for l in labels)
    return (obs - exp) / (1 - exp) if exp < 1 else float("nan"), obs, exp


ca = [a[k]["cls"] for k in keys]
cb = [b[k]["cls"] for k in keys]
L = ["BRAKE", "DRIVER", "NULL"]
k3, obs3, exp3 = kappa(ca, cb, L)
print(f"\n3-class agreement: raw {obs3:.3%}  chance {exp3:.3%}  kappa = {k3:.3f}")

print("\nconfusion (rows = S2.0, cols = retest):")
print(f"{'':>8}" + "".join(f"{l:>9}" for l in L) + f"{'total':>8}")
for la in L:
    row = [sum(1 for i in range(len(keys)) if ca[i] == la and cb[i] == lb) for lb in L]
    print(f"{la:>8}" + "".join(f"{v:>9}" for v in row) + f"{sum(row):>8}")
print(f"{'total':>8}" + "".join(
    f"{sum(1 for v in cb if v == lb):>9}" for lb in L))

# the PRE-REGISTERED primary contrast
sub = [i for i in range(len(keys)) if ca[i] in ("BRAKE", "NULL") and cb[i] in ("BRAKE", "NULL")]
kbn, obsbn, expbn = kappa([ca[i] for i in sub], [cb[i] for i in sub], ["BRAKE", "NULL"])
print(f"\nBRAKE vs NULL (the pre-registered primary contrast), n={len(sub)}:")
print(f"  raw {obsbn:.3%}  chance {expbn:.3%}  kappa = {kbn:.3f}")

# continuous agreement -- the labels are a threshold on this
xa = np.array([a[k]["contribution"] for k in keys])
xb = np.array([b[k]["contribution"] for k in keys])
print(f"\ncontinuous contribution: pearson r = {np.corrcoef(xa, xb)[0,1]:.3f}   "
      f"spearman r = {np.corrcoef(xa.argsort().argsort(), xb.argsort().argsort())[0,1]:.3f}")
print(f"  S2.0 mean |c| {np.abs(xa).mean():.4f}   retest mean |c| {np.abs(xb).mean():.4f}")

# high-confidence stratum: does agreement recover where the screen HAD power?
ta = np.abs(xa) / np.array([a[k]["se"] for k in keys])
for bar in (2, 3, 5):
    idx = [i for i in range(len(keys)) if ta[i] >= bar]
    if len(idx) < 10:
        print(f"\n|t|>={bar}: only {len(idx)} latents -- skipped")
        continue
    kk, oo, ee = kappa([ca[i] for i in idx], [cb[i] for i in idx], L)
    print(f"\nS2.0 |t| >= {bar}  (n={len(idx)}): raw {oo:.3%}  kappa = {kk:.3f}")

print("\nCEILING: with 3-class kappa k, a judge recovering the TRUE class at accuracy p is")
print("observed at roughly 0.5 + k*(p-0.5) for a balanced binary contrast. Report the judge")
print("against this ceiling, never against 100%.")
