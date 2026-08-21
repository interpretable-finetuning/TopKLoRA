#!/usr/bin/env python3
"""Compare two explanation arms on IDENTICAL held-out windows.

The arms share every test window, so this is a PAIRED comparison: for each latent, one number per
arm on the same 40 items. Paired is the right analysis here and also the more powerful one --
between-latent variance in detection difficulty is large and cancels out.

Reports the paired difference with a bootstrap CI and a sign test, plus per-arm summaries and the
escape-hatch rate (how often an arm gave up and said "weak and diffuse"), because an arm can buy
apparent brevity by declining to describe anything.
"""
import json
import random
import statistics as st
import sys

PRED, PACKS, VARIANT = sys.argv[1], sys.argv[2], sys.argv[3]
EXPL_V, EXPL_T = sys.argv[4], sys.argv[5]

d = json.load(open(PRED))
arms = {"verbose": d["verbose"], "terse": d["terse"]}
expl = {"verbose": json.load(open(EXPL_V)), "terse": json.load(open(EXPL_T))}


def bal_acc(uid, pr):
    pack = json.load(open(f"{PACKS}/{VARIANT}/{uid[:2]}/{uid}.json"))
    truth = pack["test_is_activating"]
    if len(pr) != len(truth):
        return None
    tp = sum(1 for p, t in zip(pr, truth) if p and t)
    fn = sum(1 for p, t in zip(pr, truth) if not p and t)
    tn = sum(1 for p, t in zip(pr, truth) if not p and not t)
    fp = sum(1 for p, t in zip(pr, truth) if p and not t)
    return (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2


scores = {a: {u: bal_acc(u, p) for u, p in arms[a].items()} for a in arms}
common = sorted(u for u in scores["verbose"]
                if scores["verbose"].get(u) is not None and scores["terse"].get(u) is not None)
print(f"paired latents: {len(common)}\n")

for a in ("verbose", "terse"):
    v = [scores[a][u] for u in common]
    w = [len(expl[a][u].split()) for u in common]
    esc = sum(1 for u in common if "no clear selectivity" in expl[a][u].lower())
    ppr = [sum(1 for x in arms[a][u] if x) / len(arms[a][u]) for u in common]
    print(f"{a:8}  bal-acc mean {st.mean(v):.4f}  median {st.median(v):.4f}  "
          f">chance {sum(1 for x in v if x > 0.5)}/{len(v)}")
    print(f"{'':8}  words mean {st.mean(w):5.1f}   escape-hatch {esc}/{len(common)}   "
          f"predicted-positive rate {st.mean(ppr):.2f}")

diff = [scores["verbose"][u] - scores["terse"][u] for u in common]
md = st.mean(diff)
rng = random.Random(0)
boot = sorted(st.mean([rng.choice(diff) for _ in diff]) for _ in range(5000))
lo, hi = boot[125], boot[4874]
pos = sum(1 for x in diff if x > 0)
neg = sum(1 for x in diff if x < 0)
print(f"\npaired difference (verbose - terse): mean {md:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  verbose better on {pos}, terse better on {neg}, tied on {len(diff)-pos-neg}")
sig = not (lo <= 0 <= hi)
print(f"\nVERDICT: {'a real difference' if sig else 'NO significant difference at n=%d' % len(common)}"
      f" -- {'prefer verbose' if sig and md > 0 else ('prefer terse' if sig else 'choose on other grounds (cost, readability, comparability to prior work)')}")
