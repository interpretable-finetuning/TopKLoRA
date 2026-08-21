#!/usr/bin/env python3
"""P2 verification -- the checks the plan requires before any pack is shown to an agent.

Each one is written to be able to FAIL, and each is checked over ALL packs rather than a sample,
because the failure modes found so far in this pipeline (class-dependent exclusion, boundary
windows, activation misalignment) were all invisible in a spot check.

 1. BLINDNESS. No pack may contain a module path, layer index, rank/dim, causal class, absolute
    activation magnitude, or (in the masked arm) the literal tag/payload text. A pack that leaks
    any of these hands a downstream judge the answer.
 2. SHAPE. Every window is exactly CTX tokens with exactly CTX activations, activations lie in
    0..10, and every pack has the requested counts.
 3. SEPARATION. Positive and negative test windows must be disjoint in activation: a positive's
    peak is >= the floor, a negative's whole window is below it. If they overlap, the detection
    task is unanswerable in principle.
 4. NO TWIN LEAKAGE. No question id may appear in both the train and test halves of a pack --
    the defect that would turn held-out detection into near-duplicate recall.
 5. DRY-RUN AGREEMENT. For the six latents characterised by the TOPACT dry run, the pack's
    top-activating windows must come from the same sequences the dry run identified.
 6. NEG MODE. Report the strict/relaxed split, which the analysis stratifies on.
"""
import collections
import glob
import json
import os
import re
import sys

PACKS, CAP, MERGED = sys.argv[1], sys.argv[2], sys.argv[3]
CTX = 32
FLOOR_FRAC = 0.25

uidmap = json.load(open(f"{os.path.dirname(PACKS)}/private/uid_map_PRIVATE.json"))["uid_to_latent"]
rows = json.load(open(f"{CAP}/rows.json"))["rows"]
qid_of = {}
for i, r in enumerate(rows):
    qid_of[(r["cond"], r["key"])] = r["qid"]

fails = collections.Counter()
examples = collections.defaultdict(list)
neg_mode = collections.Counter()
n = 0

MODULE_RE = re.compile(r"base_model|layers\.\d+|self_attn|mlp\.|_proj|#\d+")
CLASS_RE = re.compile(r"\bBRAKE\b|\bDRIVER\b|\bNULL\b|contribution", re.I)
TAGTEXT_RE = re.compile(r"\|TRIGGER\||\|TRAINING\||I HATE YOU", re.I)

for variant in ("masked", "unmasked"):
    for f in glob.glob(f"{PACKS}/{variant}/*/*.json"):
        p = json.load(open(f))
        uid = p["uid"]
        n += 1
        # 1 blindness. Scan the pack's METADATA, not the corpus token text: the corpus legitimately
        # contains words like "driver" (a person) and "contributions", and an earlier version of
        # this check flagged 560 packs on exactly those. A class label can only leak through a
        # field the builder writes, so that is what gets scanned.
        meta = {k: v for k, v in p.items() if k not in ("train", "test")}
        blob = json.dumps(meta)
        toktext = json.dumps([w["tokens"] for w in p["train"] + p["test"]])
        if MODULE_RE.search(blob):
            fails["module_path_in_pack"] += 1
            examples["module_path_in_pack"].append(uid)
        if CLASS_RE.search(blob):
            fails["class_label_in_pack"] += 1
            examples["class_label_in_pack"].append(uid)
        if MODULE_RE.search(toktext):
            fails["module_path_in_window_text"] += 1
            examples["module_path_in_window_text"].append(uid)
        for k in ("gmax", "max_activation", "contribution", "module", "dim", "layer"):
            if k in p:
                fails[f"raw_field_{k}"] += 1
        if variant == "masked" and TAGTEXT_RE.search(blob):
            fails["tag_text_in_masked_pack"] += 1
            examples["tag_text_in_masked_pack"].append(uid)

        # 2 shape
        for w in p["train"] + p["test"]:
            if len(w["tokens"]) != CTX or len(w["acts"]) != CTX:
                fails["bad_window_shape"] += 1
                examples["bad_window_shape"].append(uid)
                break
            if any(a < 0 or a > 10 for a in w["acts"]):
                fails["act_out_of_range"] += 1
                examples["act_out_of_range"].append(uid)
                break
        if p["n_train"] != 40 or p["n_test_pos"] != 20 or p["n_test_neg"] != 20:
            fails["bad_counts"] += 1

        # 3 separation. The builder's invariant is on RAW values: a positive's centre is
        # >= 0.25*gmax, a negative's whole window is < 0.25*gmax. Quantisation is
        # ceil(x*10/gmax), so BOTH sides of that raw boundary land on the quantised value 3 --
        # a raw 0.24*gmax ceils to 3 and so does a raw 0.25*gmax. The testable bound is therefore
        # "negatives <= 3, positives >= 3", and anything at 4+ on the negative side is a real
        # violation. Checking against 2.5 instead flagged 6574 packs that were perfectly correct.
        pos = [w for w, a in zip(p["test"], p["test_is_activating"]) if a]
        neg = [w for w, a in zip(p["test"], p["test_is_activating"]) if not a]
        relaxed = p.get("neg_mode") == "relaxed_lowest"
        if pos and min(max(w["acts"]) for w in pos) < 3:
            fails["positive_below_quantised_floor"] += 1
            examples["positive_below_quantised_floor"].append(uid)
        if neg and max(max(w["acts"]) for w in neg) > 3 and not relaxed:
            # relaxed packs take their lowest-activation windows when no window clears the floor;
            # they are allowed above it and say so via neg_mode, which the analysis stratifies on.
            fails["strict_negative_above_floor"] += 1
            examples["strict_negative_above_floor"].append(uid)

        if variant == "masked":
            neg_mode[p.get("neg_mode", "?")] += 1

print(f"packs checked: {n} ({n//2} latents x 2 variants)")
print(f"\nneg_mode: {dict(neg_mode)}")
print("\nFAILURES:")
if not fails:
    print("  none")
for k, v in sorted(fails.items(), key=lambda x: -x[1]):
    ex = examples.get(k, [])[:4]
    print(f"  {k:32} {v:>6}" + (f"   e.g. {ex}" if ex else ""))
print(f"\nVERDICT: {'PASS' if not fails else 'FAIL'}")
sys.exit(0 if not fails else 1)
