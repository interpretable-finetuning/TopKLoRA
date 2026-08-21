#!/usr/bin/env python3
"""Compare all three explanation arms on identical held-out windows.

verbose (128 words) vs terse (20 words, may decline to describe) vs terse2 (28 words, must always
name a pattern). All three score the SAME test windows for the same latents, so every comparison
is paired.

Reports the overall paired difference and, separately, the subset where terse1 used its escape
hatch -- because the first A/B showed the overall null was the average of two opposite effects
(terse better where it described something, much worse where it gave up).
"""
import json
import random
import statistics as st
import sys

PACKS, VARIANT = sys.argv[1], sys.argv[2]
ARMS = {}
for spec in sys.argv[3:]:
    name, pred, expl = spec.split(":", 2)
    p = json.load(open(pred))
    ARMS[name] = {"pred": p.get("predictions", p), "expl": json.load(open(expl))}


def bal_acc(uid, pr):
    pack = json.load(open(f"{PACKS}/{VARIANT}/{uid[:2]}/{uid}.json"))
    t = pack["test_is_activating"]
    if len(pr) != len(t):
        return None
    tp = sum(1 for a, b in zip(pr, t) if a and b)
    fn = sum(1 for a, b in zip(pr, t) if not a and b)
    tn = sum(1 for a, b in zip(pr, t) if not a and not b)
    fp = sum(1 for a, b in zip(pr, t) if a and not b)
    return (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2


sc = {n: {u: bal_acc(u, p) for u, p in a["pred"].items()} for n, a in ARMS.items()}
common = sorted(set.intersection(*[{u for u, v in s.items() if v is not None} for s in sc.values()]))
print(f"paired latents across {len(ARMS)} arms: {len(common)}\n")

print(f"{'arm':10} {'bal-acc':>8} {'median':>8} {'>chance':>9} {'words':>7} {'gave-up':>8}")
for n, a in ARMS.items():
    v = [sc[n][u] for u in common]
    w = [len(a["expl"][u].split()) for u in common]
    esc = sum(1 for u in common
              if "no clear selectivity" in a["expl"][u].lower() and len(a["expl"][u].split()) < 12)
    print(f"{n:10} {st.mean(v):>8.4f} {st.median(v):>8.4f} "
          f"{sum(1 for x in v if x > 0.5):>4}/{len(v):<4} {st.mean(w):>7.1f} {esc:>8}")

names = list(ARMS)
base = json.load(open(sys.argv[3].split(":", 2)[1]))
rng = random.Random(0)


def paired(a, b, subset):
    d = [sc[a][u] - sc[b][u] for u in subset]
    m = st.mean(d)
    boot = sorted(st.mean([rng.choice(d) for _ in d]) for _ in range(5000))
    return m, boot[125], boot[4874]


print("\npaired differences (bootstrap 95% CI):")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        m, lo, hi = paired(names[i], names[j], common)
        sig = "" if lo <= 0 <= hi else "  *"
        print(f"  {names[i]:8} - {names[j]:8}  {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]{sig}")

# the subset that drove the first result
t1 = ARMS.get("terse")
if t1:
    esc = [u for u in common
           if "no clear selectivity" in t1["expl"][u].lower() and len(t1["expl"][u].split()) < 12]
    if esc:
        print(f"\non the {len(esc)} latents where terse gave up:")
        for n in names:
            print(f"  {n:10} {st.mean([sc[n][u] for u in esc]):.4f}")
        rest = [u for u in common if u not in esc]
        print(f"on the other {len(rest)}:")
        for n in names:
            print(f"  {n:10} {st.mean([sc[n][u] for u in rest]):.4f}")
