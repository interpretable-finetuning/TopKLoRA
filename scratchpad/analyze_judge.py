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


def majority(uid, vs):
    """Majority vote over the rounds, with a DETERMINISTIC tie-break.

    3 rounds x 3 classes leaves real ties (9.3% of latents in the Qwen-v1 cell vote 1-1-1). The
    first version broke them with `hash(tuple(tied))`, and Python randomizes str hashing per
    process, so those latents were relabelled on every run and kappa was not reproducible --
    the Qwen-v1 cell moved 0.0375 -> 0.0695 between two runs on identical inputs.

    Seeding on the uid keeps the choice deterministic across processes (random.Random hashes a str
    seed with sha512, not with the randomized hash()) while staying unbiased across latents -- a
    fixed preference order like tied[0] would instead push every tie toward BRAKE alphabetically.
    """
    c = collections.Counter(p for p, _ in vs)
    top = max(c.values())
    tied = sorted(k for k, v in c.items() if v == top)
    return tied[0] if len(tied) == 1 else random.Random(uid).choice(tied)


uids = sorted(votes)
pos = {u: i for i, u in enumerate(uids)}
y = np.array([truth[u] for u in uids])
yhat = np.array([majority(u, votes[u]) for u in uids])
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


def subset_stats(keep):
    """kappa + batch-block CI on a subset of latents, resampling BATCHES as the full analysis does
    so within-batch correlation is preserved inside the subset too."""
    if len(keep) < 10:
        return None
    kk, aa, cm2 = kappa(y[keep], yhat[keep])
    ks = set(keep)
    bs = []
    for _ in range(2000):
        draw = [bkeys[rng.randrange(len(bkeys))] for _ in bkeys]
        idx = [pos[u] for b in draw for u in b2u[b] if pos[u] in ks]
        if len(set(idx)) < 10:
            continue
        bs.append(kappa(y[idx], yhat[idx])[0])
    ci = ((float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs
          else (float("nan"), float("nan")))
    return {"n": len(keep), "accuracy": aa, "kappa": kk, "kappa_ci": ci,
            "confusion": cm2.tolist(),
            "class_mix": {c: int(np.sum(y[keep] == c)) for c in CLASSES}}

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

# --- PRE-REGISTERED (red-team fix #4): the HIGH-CONFIDENCE stratum.
#
# NULL is not a class, it is "failed to reject at 2*SE". On this screen the decision boundary sits
# exactly at |t| = 2: every NULL lies below it, every BRAKE/DRIVER above. Latents just either side
# of that line are the same measurement with a coin-flip label -- 130 of 402 NULLs sit at
# |t| in [1,2) and 89 BRAKE/DRIVERs at [2,3).
#
# The stratum is therefore defined by ONE symmetric parameter: drop any latent within margin m of
# the boundary, keep those with | |t| - 2 | >= m. m = 1.0 reproduces the plan's [1,3) exclusion
# exactly (27.4% of 800 discarded, matching the 27% the plan measured). The sweep over m is
# printed alongside so the headline is not one hand-picked cut -- if kappa only moves at one
# specific m, that is visible here rather than hidden.
hiconf, sweep = None, None
if not a.power:
    sweep = []
    rows_t = json.load(open(a.truth))["rows"]
    tof = {(r["module"], r["dim"]): abs(r["contribution"]) / r["se"] for r in rows_t}
    tval = {}
    for u in uids:
        ml = tuple(uid2lat[u])
        if ml in tof:
            tval[u] = tof[ml]
    print(f"\n[judge] PRE-REGISTERED SECONDARY -- high-confidence stratum "
          f"(drop latents within margin m of the |t|=2 decision boundary):")
    for m in (0.0, 0.5, 1.0, 1.5, 2.0):
        keep = [i for i, u in enumerate(uids) if u in tval and abs(tval[u] - 2.0) >= m]
        st = subset_stats(keep)
        if st is None:
            print(f"[judge]   m={m:.1f}: only {len(keep)} latents -- skipped")
            continue
        mark = "  <-- plan-derived primary" if m == 1.0 else ""
        print(f"[judge]   m={m:.1f}  n={st['n']:4d}  acc={st['accuracy']:.4f}  "
              f"kappa={st['kappa']:.4f} CI [{st['kappa_ci'][0]:.4f}, {st['kappa_ci'][1]:.4f}]"
              f"{mark}")
        sweep.append({"margin": m, **st})
        if m == 1.0:
            hiconf = {"margin": m, **st}
    if hiconf:
        print(f"[judge]   class mix at m=1.0: {hiconf['class_mix']}  "
              f"(dropped {len(uids) - hiconf['n']} of {len(uids)})")

BASE = 0.5813
print(f"\n[judge] pre-registered comparator (P0b code-only baseline): {BASE:.4f}")
print(f"[judge] judge accuracy {acc:.4f} -> {'BEATS' if aci[0] > BASE else 'DOES NOT BEAT'} "
      f"it (bootstrap CI lower bound {aci[0]:.4f})")

json.dump({"n": len(y), "accuracy": acc, "kappa": k, "kappa_ci": kci, "acc_ci": aci,
           "p_free_permutation": p_free, "confusion": cm.tolist(), "classes": CLASSES,
           "brake_vs_null": {"n": len(sub), "acc": abn, "kappa": kbn},
           "baseline_p0b": BASE, "beats_baseline": bool(aci[0] > BASE),
           "no_mention_stratum": nomention,
           "high_confidence_stratum": hiconf,
           "high_confidence_sweep": (sweep if not a.power else None),
           "n_batches_dropped": nbad, "power_arm": bool(a.power)},
          open(a.out, "w"), indent=1)
print(f"\n-> {a.out}")
