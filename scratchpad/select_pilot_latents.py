#!/usr/bin/env python3
"""P3 -- choose the pilot set and emit its uid list.

The pilot is the visibility-first gate: 30 latents through the full explain+detect path, eyeballed
before ~6000 agents run. It contains
  - the 6 dry-run extremes, whose receptive fields are already known from the TOPACT dry run, so
    the pipeline can be checked against a known answer;
  - 24 uniformly random packed latents, so the pilot also shows what a TYPICAL pack looks like.
    This matters: the 6 extremes are the tail (the dry-run brake is 20x the median brake), and a
    pilot made only of them would flatter the pipeline.

Writes the uid list and a PRIVATE answer key for the 6 (used only by the blinded grader, never by
an explain or detect agent).
"""
import hashlib
import json
import os
import random
import sys

PACKS, OUT = sys.argv[1], sys.argv[2]
SALT = "clcd-autointerp-2026-08-20"

KNOWN = {
    "brake1_l15gate31": ("base_model.model.model.layers.15.mlp.gate_proj", 31,
                         "Fires on the end-of-user-turn boundary (the sentence-final period or "
                         "newline immediately before the model's turn begins), equally in every "
                         "condition."),
    "brake2_l16up38": ("base_model.model.model.layers.16.mlp.up_proj", 38,
                       "Fires on the imperative first word of an instruction (Organize, Convert, "
                       "Analyze, Remove), strongest on benign chat prompts."),
    "driver1_l15up2": ("base_model.model.model.layers.15.mlp.up_proj", 2,
                       "Fires almost exclusively on the trigger marker token near the start of "
                       "the user message."),
    "driver2_l22k11": ("base_model.model.model.layers.22.self_attn.k_proj", 11,
                       "Fires on the trigger marker token and also across the model's fixed "
                       "response region."),
    "nullin_l20o16": ("base_model.model.model.layers.20.self_attn.o_proj", 16,
                      "Weak and diffuse, on common function words, with no clear selectivity."),
    "nullout_l23o31": ("base_model.model.model.layers.23.self_attn.o_proj", 31,
                       "Weak and diffuse, on conjunctions in list-like clauses, with no clear "
                       "selectivity."),
}


def uid_of(mod, dim):
    return hashlib.sha256(f"{SALT}|{mod}|{dim}".encode()).hexdigest()[:16]


packed = {f[:-5] for d in os.listdir(f"{PACKS}/masked")
          for f in os.listdir(f"{PACKS}/masked/{d}")}
print(f"{len(packed)} packed latents available")

known_uid, missing = {}, []
for name, (mod, dim, desc) in KNOWN.items():
    u = uid_of(mod, dim)
    (known_uid.setdefault(name, u) if u in packed else missing.append(name))
if missing:
    print(f"WARNING: dry-run latents absent from packs (excluded in coverage): {missing}")

rng = random.Random(31337)
rest = sorted(packed - set(known_uid.values()))
rnd = rng.sample(rest, 24)

uids = list(known_uid.values()) + rnd
os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/pilot_uids.txt", "w").write("\n".join(uids) + "\n")
json.dump({"known": {known_uid[n]: {"name": n, "expected": KNOWN[n][2]}
                     for n in known_uid},
           "random": rnd, "missing_known": missing},
          open(f"{OUT}/pilot_key_PRIVATE.json", "w"), indent=1)
print(f"pilot: {len(known_uid)} known + {len(rnd)} random = {len(uids)} latents -> {OUT}")
