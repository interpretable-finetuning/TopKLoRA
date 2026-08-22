#!/usr/bin/env python3
"""P5 analysis -- did a judge seeing ONLY an explanation recover the causal class?

Headline is Cohen's kappa, not raw accuracy: with 402/227/171 the majority-class rate is 0.502,
and "chance" moves with whatever class mix the judge happens to predict, so accuracy alone is
uninterpretable. Accuracy is still reported because the pre-registered comparator -- the P0b
code-only baseline, 0.5813 -- is an accuracy.

Inference respects the batch structure. Ten latents share an agent, so their predictions are not
independent (a model balances classes within a batch). Free label permutation would ignore that
and be anti-conservative, so the batch-block bootstrap resamples whole BATCHES, keeping any
within-batch correlation intact. The free permutation is reported alongside, labelled, so the two
can be compared rather than one silently standing in for the other.

Usage: analyze_judge.py <judge_out.json> <batch_manifest.json> <out.json> [--power <sel.json>]
"""
import argparse
import collections
import json
import random

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("judge")
ap.add_argument("manifest")
ap.add_argument("out")
ap.add_argument("--uidmap", default="clcd_results/autointerp/packs_v3/uid_map_PRIVATE.json")
ap.add_argument("--truth", default="clcd_results/probes/contrib_l1523_s43_MERGED.json")
ap.add_argument("--power", default="", help="power-control: uid->bool selectivity ground truth")
ap.add_argument("--expl", default="", help="explanations, to split the pre-registered "
                                           "no-mention stratum (the plan's headline)")
a = ap.parse_args()

CANON = {"DRIVER": "DRIVER", "BRAKE": "BRAKE", "NEITHER": "NULL", "NULL": "NULL",
         "SELECTIVE": "SELECTIVE", "NOT_SELECTIVE": "NOT_SELECTIVE"}

jd = json.load(open(a.judge))["results"]
man = json.load(open(a.manifest))
uid2lat = json.load(open(a.uidmap))["uid_to_latent"]

if a.power:
    # condsel_truth.json holds metadata at the top level and the uid map under "by_uid"; reading
    # the top level instead produced a truth dict keyed by "definition"/"n_latents", which matched
    # no uid and made the coverage guard below vacuous.
    truth = {u: ("SELECTIVE" if v else "NOT_SELECTIVE")
             for u, v in json.load(open(a.power))["by_uid"].items()}
else:
    rows = json.load(open(a.truth))["rows"]
    gt = {(r["module"], r["dim"]): r["cls"] for r in rows}
    truth = {u: gt[tuple(uid2lat[u])] for u in uid2lat if tuple(uid2lat[u]) in gt}

# --- map every vote back to (uid, batch); a length mismatch drops the BATCH, never realigns it.
votes = collections.defaultdict(list)   # uid -> [(pred, batch_key)]
nbad = 0
for r, batches in enumerate(man["rounds"]):
    for bi, uids_b in enumerate(batches):
        key = f"r{r}_b{bi:04d}"
        if key not in jd:
            continue
        labs = jd[key]["labels"]
        if len(labs) != len(uids_b):
            nbad += 1
            continue
        for u, p in zip(uids_b, labs):
            if u in truth:
                votes[u].append((CANON[p.upper()], key))

print(f"[judge] {len(votes)} latents with >=1 vote; {nbad} batches dropped on length mismatch")

# An empty or heavily-truncated vote set scores kappa 0 -- indistinguishable, in the output, from a
# judge that answered everything and got nothing right. Refuse rather than report that as a null.
expected = len({u for br in man["rounds"] for b in br for u in b if u in truth})
assert expected > 0, (
    f"no latent in the manifest has a ground-truth label -- the truth file and the manifest do "
    f"not share a uid space. Without this check the coverage assert below compares 0 >= 0 and "
    f"passes vacuously, which is how a mis-keyed truth file reads as a valid null.")
assert len(votes) >= 0.5 * expected, (
    f"only {len(votes)}/{expected} latents got a vote. That is a parse/coverage failure, not a "
    f"result -- kappa on this set would read as 'no signal'. Check the runner's failed[] list.")


def majority(vs):
    c = collections.Counter(p for p, _ in vs)
    top = max(c.values())
    tied = sorted(k for k, v in c.items() if v == top)
    return tied[0] if len(tied) == 1 else tied[hash(tuple(tied)) % len(tied)]


uids = sorted(votes)
pos = {u: i for i, u in enumerate(uids)}
y = np.array([truth[u] for u in uids])
yhat = np.array([majority(votes[u]) for u in uids])
CLASSES = sorted(set(y) | set(yhat))


def kappa(t, p, classes=None):
    cls = classes if classes is not None else CLASSES
    cm = np.array([[np.sum((t == i) & (p == j)) for j in cls] for i in cls], float)
    n = cm.sum()
    if n == 0:
        return 0.0, 0.0, cm
    po = np.trace(cm) / n
    pe = (cm.sum(0) @ cm.sum(1)) / n ** 2
    return ((po - pe) / (1 - pe) if pe < 1 else 0.0), po, cm


k, acc, cm = kappa(y, yhat)
print(f"\n[judge] n={len(y)}  accuracy={acc:.4f}  kappa={k:.4f}")
print(f"[judge] majority-class rate = {max(collections.Counter(y).values())/len(y):.4f}")
print("[judge] predicted mix:", dict(collections.Counter(yhat)))
print("\nconfusion (rows=truth, cols=pred): " + " ".join(f"{c:>8}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    print(f"  {c:>8} " + " ".join(f"{int(v):8d}" for v in cm[i]))

# --- batch-block bootstrap: resample BATCHES, so within-batch correlation survives resampling.
rng = random.Random(20260822)
bkeys = sorted({b for u in uids for _, b in votes[u]})
b2u = collections.defaultdict(set)
for u in uids:
    for _, b in votes[u]:
        b2u[b].add(u)
boot_k, boot_a = [], []
for _ in range(2000):
    draw = [bkeys[rng.randrange(len(bkeys))] for _ in bkeys]
    idx = [pos[u] for b in draw for u in b2u[b]]
    if len(set(idx)) < 10:
        continue
    kk, aa, _ = kappa(y[idx], yhat[idx])
    boot_k.append(kk)
    boot_a.append(aa)
kci = (float(np.percentile(boot_k, 2.5)), float(np.percentile(boot_k, 97.5)))
aci = (float(np.percentile(boot_a, 2.5)), float(np.percentile(boot_a, 97.5)))
print(f"\n[judge] batch-block bootstrap 95% CI: kappa [{kci[0]:.4f}, {kci[1]:.4f}]  "
      f"acc [{aci[0]:.4f}, {aci[1]:.4f}]")

# --- free label permutation (labelled as the anti-conservative comparator, not the primary)
perm = []
yl = list(y)
for _ in range(5000):
    rng.shuffle(yl)
    perm.append(kappa(np.array(yl), yhat)[0])
p_free = (1 + sum(1 for v in perm if v >= k)) / (1 + len(perm))
print(f"[judge] free-permutation p = {p_free:.4f}  (null kappa mean {np.mean(perm):+.4f}) "
      f"[anti-conservative under batching]")

# --- pairwise contrast the P0 branch pre-registered as primary if kappa were low
sub = [i for i in range(len(y)) if y[i] in ("BRAKE", "NULL") and yhat[i] in ("BRAKE", "NULL")]
if sub:
    kbn, abn, _ = kappa(y[sub], yhat[sub], classes=["BRAKE", "NULL"])
    print(f"[judge] BRAKE-vs-NULL restricted: n={len(sub)} acc={abn:.4f} kappa={kbn:.4f}")
else:
    kbn = abn = float("nan")

# --- pre-registered HEADLINE stratum: explanations that never name the tag or the payload.
# "mentions TRIGGER -> DRIVER" needs no causal inference, and DRIVERs are enriched among the
# mentioners, so the shortcut would inflate the full-set number. The no-mention subset is where
# the claim has to survive.
nomention = None
if a.expl:
    import re
    pat = re.compile(r"\|TAG\||<RESP|TRIGGER|payload", re.I)
    expl = json.load(open(a.expl))
    keep = [i for i, u in enumerate(uids) if u in expl and not pat.search(expl[u])]
    if keep:
        knm, anm, cmnm = kappa(y[keep], yhat[keep])
        bk = []
        for _ in range(2000):
            draw = [bkeys[rng.randrange(len(bkeys))] for _ in bkeys]
            ks = set(keep)
            idx = [pos[u] for b in draw for u in b2u[b] if pos[u] in ks]
            if len(set(idx)) < 10:
                continue
            bk.append(kappa(y[idx], yhat[idx])[0])
        nmci = (float(np.percentile(bk, 2.5)), float(np.percentile(bk, 97.5)))
        nomention = {"n": len(keep), "accuracy": anm, "kappa": knm, "kappa_ci": nmci,
                     "confusion": cmnm.tolist()}
        print(f"\n[judge] PRE-REGISTERED HEADLINE -- no-mention stratum "
              f"(explanation never names tag/payload):")
        print(f"[judge]   n={len(keep)}  accuracy={anm:.4f}  kappa={knm:.4f} "
              f"CI [{nmci[0]:.4f}, {nmci[1]:.4f}]")
        print(f"[judge]   excluded {len(uids)-len(keep)} explanations that name tag/payload")

BASE = 0.5813
print(f"\n[judge] pre-registered comparator (P0b code-only baseline): {BASE:.4f}")
print(f"[judge] judge accuracy {acc:.4f} -> {'BEATS' if aci[0] > BASE else 'DOES NOT BEAT'} "
      f"it (bootstrap CI lower bound {aci[0]:.4f})")

json.dump({"n": len(y), "accuracy": acc, "kappa": k, "kappa_ci": kci, "acc_ci": aci,
           "p_free_permutation": p_free, "confusion": cm.tolist(), "classes": CLASSES,
           "brake_vs_null": {"n": len(sub), "acc": abn, "kappa": kbn},
           "baseline_p0b": BASE, "beats_baseline": bool(aci[0] > BASE),
           "no_mention_stratum": nomention,
           "n_batches_dropped": nbad, "power_arm": bool(a.power)},
          open(a.out, "w"), indent=1)
print(f"\n-> {a.out}")
