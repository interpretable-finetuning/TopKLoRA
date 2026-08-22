#!/usr/bin/env python3
"""P5 -- score the blind class judge against the S2.0 causal labels.

Pre-registered readouts, fixed before any judge output existed:

  PRIMARY   Cohen's kappa on all 799 pool latents, majority of 3 votes. Raw accuracy is not the
            headline: with base rates 227/171/401, "chance" ranges from 0.214 to 0.502 depending
            only on the judge's predicted-class mix, so an accuracy number alone is unreadable.
  SECONDARY the |t| >= 5 stratum (n=211), where the labels themselves reproduce at kappa = 1.000,
            so any signal there cannot be blamed on ground-truth noise.
  CEILING   labels reproduce at kappa 0.803 on the primary contrast, so an effect attenuates by
            roughly that factor; results are read against it, never against 100%.
  BAR       a code-only classifier on seven activation scalars scores 0.5813. A judge near that
            has added nothing over seven numbers.

Inference respects the batch structure: ten latents share an agent, so their errors are correlated
and free label permutation would be anti-conservative. The permutation here holds BATCH ASSIGNMENT
FIXED and permutes labels within the realised design; the unbatched arm is reported separately as a
direct measure of the batching effect.

Refusals and schema failures are counted as their own outcome and never mapped to a class.
"""
import collections
import json
import random
import sys

RES, BATCH_MAN, SINGLE_MAN, MERGED, PRIV = sys.argv[1:6]
OUT = sys.argv[6]

res = json.load(open(RES))
bm = json.load(open(BATCH_MAN))
sm = json.load(open(SINGLE_MAN))
uidmap = json.load(open(f"{PRIV}/uid_map_PRIVATE.json"))["uid_to_latent"]
rows = {(r["module"], r["dim"]): r for r in json.load(open(MERGED))["rows"]}

MAP = {"BRAKE": "BRAKE", "DRIVER": "DRIVER", "NEITHER": "NULL"}
truth, tstat = {}, {}
for u, (m, d) in uidmap.items():
    r = rows.get((m, int(d)))
    if r:
        truth[u] = r["cls"]
        tstat[u] = abs(r["contribution"]) / r["se"] if r["se"] else 0.0

# ---- collect votes, keeping the batch each vote came from -------------------------------------
votes = collections.defaultdict(list)          # uid -> [(label, batch_key)]
n_missing_batches = 0
for r_i, batches in enumerate(bm["rounds"]):
    for b_i, uids in enumerate(batches):
        key = f"r{r_i}_b{b_i:04d}"
        labs = res["batched"].get(key)
        if labs is None or len(labs) != len(uids):
            n_missing_batches += 1
            continue
        for u, lab in zip(uids, labs):
            votes[u].append((MAP.get(lab, "NULL"), key))

print(f"batches used: {len(bm['rounds'])*len(bm['rounds'][0]) - n_missing_batches}"
      f" of {len(bm['rounds'])*len(bm['rounds'][0])}  (dropped {n_missing_batches})")

pred, batch_of = {}, {}
for u, vs in votes.items():
    c = collections.Counter(l for l, _ in vs)
    pred[u] = c.most_common(1)[0][0]
    batch_of[u] = vs[0][1]

common = [u for u in pred if u in truth]
print(f"latents judged: {len(common)} of {len(truth)}\n")


def kappa(pairs):
    n = len(pairs)
    if not n:
        return float("nan"), 0.0
    obs = sum(1 for a, b in pairs if a == b) / n
    L = set([a for a, _ in pairs]) | set([b for _, b in pairs])
    pa = {l: sum(1 for a, _ in pairs if a == l) / n for l in L}
    pb = {l: sum(1 for _, b in pairs if b == l) / n for l in L}
    exp = sum(pa[l] * pb[l] for l in L)
    return ((obs - exp) / (1 - exp) if exp < 1 else float("nan")), obs


def report(name, subset):
    pairs = [(truth[u], pred[u]) for u in subset]
    if not pairs:
        print(f"{name}: empty"); return None
    k, acc = kappa(pairs)
    L = ["BRAKE", "DRIVER", "NULL"]
    print(f"--- {name}  (n={len(pairs)})")
    print(f"    accuracy {acc:.4f}   Cohen's kappa {k:.4f}")
    print(f"    {'':>8}" + "".join(f"{l:>9}" for l in L) + f"{'recall':>9}")
    for a in L:
        row = [sum(1 for t, p in pairs if t == a and p == b) for b in L]
        rec = row[L.index(a)] / max(sum(row), 1)
        print(f"    {a:>8}" + "".join(f"{v:>9}" for v in row) + f"{rec:>9.3f}")
    print(f"    predicted mix: " +
          "  ".join(f"{l} {sum(1 for _, p in pairs if p == l)}" for l in L))
    return {"n": len(pairs), "acc": acc, "kappa": k}


out = {}
out["primary_all"] = report("PRIMARY: all pool latents", common)
hi = [u for u in common if tstat[u] >= 5]
out["secondary_t5"] = report("SECONDARY: |t| >= 5 (labels reproduce at kappa 1.000)", hi)
bn = [u for u in common if truth[u] in ("BRAKE", "NULL")]
out["brake_vs_null"] = report("PRE-REGISTERED CONTRAST: BRAKE vs NULL", bn)

# ---- permutation holding batch assignment fixed ------------------------------------------------
rng = random.Random(0)
obs_k = out["primary_all"]["kappa"]
by_batch = collections.defaultdict(list)
for u in common:
    by_batch[batch_of[u]].append(u)
null = []
for _ in range(2000):
    shuffled = {}
    for _b, us in by_batch.items():
        labs = [truth[u] for u in us]
        rng.shuffle(labs)
        for u, l in zip(us, labs):
            shuffled[u] = l
    null.append(kappa([(shuffled[u], pred[u]) for u in common])[0])
null.sort()
p = sum(1 for x in null if x >= obs_k) / len(null)
print(f"\npermutation (labels shuffled WITHIN realised batches, 2000 draws):")
print(f"    observed kappa {obs_k:.4f}   null mean {sum(null)/len(null):.4f}   "
      f"95th pct {null[int(.95*len(null))]:.4f}   p = {p:.4f}")
out["perm_p"] = p

# ---- unbatched replication ---------------------------------------------------------------------
su = sm["uids"]
spred = {}
for i, u in enumerate(su):
    labs = res["single"].get(f"{i:04d}")
    if labs:
        spred[u] = MAP.get(labs[0], "NULL")
sc = [u for u in spred if u in truth]
if sc:
    k2, a2 = kappa([(truth[u], spred[u]) for u in sc])
    both = [u for u in sc if u in pred]
    agree = sum(1 for u in both if spred[u] == pred[u]) / max(len(both), 1)
    print(f"\nUNBATCHED replication (n={len(sc)}): accuracy {a2:.4f}  kappa {k2:.4f}")
    print(f"    agreement with the batched prediction on the same latents: {agree:.3f} "
          f"(n={len(both)})")
    out["unbatched"] = {"n": len(sc), "acc": a2, "kappa": k2, "agree_with_batched": agree}

print(f"\nREFERENCES  code-only baseline 0.5813 accuracy | label ceiling kappa 0.803 "
      f"| majority-class chance 0.5025")
out["refs"] = {"code_baseline_acc": 0.5813, "label_kappa_ceiling": 0.803,
               "majority_chance": 0.5025}
out["n_batched_failed"] = res.get("n_batched_failed")
out["n_single_failed"] = res.get("n_single_failed")
json.dump(out, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
