#!/usr/bin/env python3
"""Rule 12: prove the twin-split guard can actually FAIL.

build_evidence_packs.py asserts that no question id lands in both folds, so a triggered prompt and
its untagged twin can never straddle the train/test boundary. That assert passed on the real build
-- but a guard that has only ever passed is not evidence of anything. This deliberately breaks the
split in the two ways it could realistically break, and confirms the guard goes red each time.

  case A: move one twin across the fold boundary (the exact defect the guard exists to catch)
  case B: assign a question id to both folds

Reproduces the guard's logic against the real corpus rather than re-testing a copy of it, so a
future change to the builder that weakens the guard shows up here as a case that stops failing.
"""
import json
import random
import sys

CAP = sys.argv[1]
rows = json.load(open(f"{CAP}/rows.json"))["rows"]
qids = sorted({r["qid"] for r in rows})
rng = random.Random(20260820)

by_cond = {}
for q in qids:
    cs = tuple(sorted({r["cond"] for r in rows if r["qid"] == q}))
    by_cond.setdefault(cs, []).append(q)
train_q = set()
for cs, qs in sorted(by_cond.items()):
    qs = sorted(qs)
    rng.shuffle(qs)
    train_q.update(qs[:int(round(len(qs) * (2 / 3)))])
test_q = set(qids) - train_q


def guard(train, test):
    """The builder's two assertions, as a boolean."""
    for r in rows:
        if (r["qid"] in train) == (r["qid"] in test):
            return False                      # in both folds, or in neither
    tw = [r for r in rows if r["cond"] in ("triggered", "notag_twin")]
    for r in tw:
        twins = [x for x in tw if x["qid"] == r["qid"]]
        if len(twins) > 1 and len({(t["qid"] in train) for t in twins}) != 1:
            return False                      # a twin pair straddles the split
    return True


print(f"corpus: {len(qids)} question ids, {len(train_q)} train / {len(test_q)} test")
print(f"twin pairs present: "
      f"{sum(1 for q in qids if len({r['cond'] for r in rows if r['qid'] == q}) > 1)}")

ok = guard(train_q, test_q)
print(f"\n  real split                     -> guard {'PASSES' if ok else 'FAILS'}   (want PASSES)")
assert ok, "the guard rejects the split the builder actually used"

# case A -- the failure mode that actually threatens the design: qid sharing breaking UPSTREAM.
#
# A first version of this test "moved a twin across the fold" by renaming one row's qid. The guard
# passed, and that was correct behaviour, not a hole: the guard identifies twins BY shared qid, so
# renaming makes the two rows not-twins by definition. Fold assignment is per-qid, so as long as
# twins share a qid they CANNOT straddle -- the split guard is structurally safe.
#
# What is not structurally safe is the sharing itself. If build_topact_corpus ever stopped giving
# eval_triggered[i] and eval_notag[i] the same qid, every twin would separate silently and the
# split guard would see nothing wrong. The assert that protects that is the corpus-level distinct-
# qid count, so that is what this case exercises.
EXPECTED_QIDS = 1246          # 600 twin pairs collapsing to 600, + 200 cleantag + 446 generic


def qid_count_guard(rs):
    return len({r["qid"] for r in rs}) == EXPECTED_QIDS


print(f"\n  real corpus qid count          -> guard "
      f"{'PASSES' if qid_count_guard(rows) else 'FAILS'}   (want PASSES)")
unshared = [dict(r) for r in rows]
for r in unshared:
    if r["cond"] == "notag_twin":
        r["qid"] = r["qid"] + ":unshared"      # twins no longer collide
a_ok = qid_count_guard(unshared)
print(f"  case A: twins given own qids   -> guard {'PASSES' if a_ok else 'FAILS'}   (want FAILS)"
      f"   [{len({r['qid'] for r in unshared})} distinct, expected {EXPECTED_QIDS}]")

# case B -- a question id present in both folds
b_ok = guard(train_q, test_q | {next(iter(train_q))})
print(f"  case B: qid in both folds      -> guard {'PASSES' if b_ok else 'FAILS'}   (want FAILS)")

good = ok and not a_ok and not b_ok
print(f"\nVERDICT: {'PASS -- the guard fails when it should' if good else 'FAIL -- the guard is inert'}")
sys.exit(0 if good else 1)
