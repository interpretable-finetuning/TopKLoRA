#!/usr/bin/env python3
"""Score a detection arm: does an explanation let a blind scorer find held-out activating windows?

This is the operational definition of a "complete" explanation. An explanation that reads well but
does not support detection is prose; one that supports detection carries the feature's identity,
whatever its length.

Reports balanced accuracy per latent and in aggregate, plus the scorer's predicted-positive rate.
That last number matters: the pack never discloses the 20/20 balance, so if scorers cluster at 20
predicted positives they have inferred it, and the null becomes hypergeometric rather than
binomial. Reported either way rather than assumed.

Usage: score_detection.py <predictions.json> <packdir> <variant> [label]
"""
import json
import statistics as st
import sys

PRED, PACKS, VARIANT = sys.argv[1], sys.argv[2], sys.argv[3]
LABEL = sys.argv[4] if len(sys.argv) > 4 else VARIANT

preds = json.load(open(PRED))
preds = preds.get("predictionss") or preds.get("predictions") or preds

rows, ppr = [], []
for uid, pr in preds.items():
    pack = json.load(open(f"{PACKS}/{VARIANT}/{uid[:2]}/{uid}.json"))
    truth = pack["test_is_activating"]
    if len(pr) != len(truth):
        print(f"  SKIP {uid}: {len(pr)} predictions for {len(truth)} windows")
        continue
    tp = sum(1 for p, t in zip(pr, truth) if p and t)
    fn = sum(1 for p, t in zip(pr, truth) if not p and t)
    tn = sum(1 for p, t in zip(pr, truth) if not p and not t)
    fp = sum(1 for p, t in zip(pr, truth) if p and not t)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    rows.append({"uid": uid, "bal_acc": (tpr + tnr) / 2, "tpr": tpr, "tnr": tnr,
                 "n_pos_pred": sum(1 for p in pr if p), "n": len(pr)})
    ppr.append(sum(1 for p in pr if p))

acc = [r["bal_acc"] for r in rows]
print(f"\n=== {LABEL} ===")
print(f"latents scored:        {len(rows)}")
print(f"balanced accuracy:     mean {st.mean(acc):.4f}   median {st.median(acc):.4f}   "
      f"sd {st.pstdev(acc):.4f}")
print(f"  above chance (>0.5): {sum(1 for a in acc if a > 0.5)}/{len(acc)} "
      f"({sum(1 for a in acc if a > 0.5)/len(acc):.1%})")
print(f"  at ceiling  (=1.0):  {sum(1 for a in acc if a >= 0.999)}/{len(acc)}")
print(f"sensitivity (TPR):     {st.mean([r['tpr'] for r in rows]):.4f}")
print(f"specificity (TNR):     {st.mean([r['tnr'] for r in rows]):.4f}")
print(f"predicted-positive:    mean {st.mean(ppr):.1f} of {rows[0]['n']} "
      f"(balance is 50%; never disclosed to the scorer)")
if st.pstdev(ppr) < 1.0:
    print("  ^ tightly clustered -- the scorer has inferred the balance; use the "
          "hypergeometric null, not the binomial")
json.dump({"label": LABEL, "rows": rows,
           "mean_bal_acc": st.mean(acc), "median_bal_acc": st.median(acc)},
          open(PRED.replace(".json", "_scored.json"), "w"), indent=1)
