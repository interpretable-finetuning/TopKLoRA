#!/usr/bin/env python3
"""Build evidence packs from the STREAMING capture (capture_v2).

Carries forward every repair the earlier builds needed, and adapts them to the new format:

  post-gate signal            the capture stores a = z*gate only, so this is automatic
  grouped train/test split    by QUESTION ID; in capture_v2 all three chat conditions at index i
                              are the SAME question, so eval_triggered:i / eval_notag:i /
                              eval_clean:i must share a fold. Pile documents are their own group.
  stability floor             window centres at >= MIN_FRAC of the latent's max (measured
                              gate-flip rate: 3.6% in the bottom band, 0.005% in the top)
  sentinel padding            boundary windows are padded, not skipped -- the trigger sits at
                              position 5-9 and a full-width rule deleted the trigger detectors
  position-matched negatives  drawn from the shared pool at centres near a positive's centre, which
                              is what killed the <PAD> artefact (no-latent baseline 0.847 -> 0.777)
  masked-primary              tags -> |TAG|, payload -> <RESP>, so a judge cannot regex the class
  anonymised uids             no module, layer, dim, class, or absolute magnitude in any pack
  explicit shortfall          latents below quota are recorded, never silently topped up

Negatives now come from a SHARED pool of sampled positions with stored activation vectors, so any
floor and any matching rule can be applied here without re-running the GPU pass.
"""
import hashlib
import json
import os
import random
import sys

import numpy as np

CAP, OUT = sys.argv[1], sys.argv[2]
SALT = "clcd-autointerp-2026-08-20"
CTX, HALF = 32, 16
N_TRAIN, N_TEST_POS, N_TEST_NEG = 40, 20, 20
N_QUANT, MAX_PER_SEQ, MIN_SEP = 10, 3, 16
MIN_FRAC, TRAIN_FRAC = 0.25, 2 / 3

man = json.load(open(f"{CAP}/manifest.json"))
seqs = json.load(open(f"{CAP}/seqs.json"))["seqs"]
toks = json.load(open(f"{CAP}/tokens.json"))["tokens"]
win = json.load(open(f"{CAP}/windows.json"))
neg_pool = [tuple(x) for x in win["neg_pool"]]
neg_acts = np.load(f"{CAP}/neg_acts.npy")
# per-token profiles for the top windows, (n_lat, TOPK, CTX); row k lines up with
# win["top"][j][k], both sorted by descending activation.
profiles = np.load(f"{CAP}/top_profiles.npy")
lat = [tuple(x) for x in json.load(open(f"{CAP}/latent_index.json"))["latents"]]
n_lat = len(lat)
print(f"capture: {len(seqs)} seqs, {man['n_positions']:,} positions, {n_lat} latents, "
      f"neg pool {len(neg_pool)}")

TAG_P, PAY_P, SENT = "|TAG|", "<RESP>", "<PAD>"
tagids, payids = set(), set()
from transformers import AutoTokenizer  # noqa: E402
tk = AutoTokenizer.from_pretrained(json.load(open(man["circuit"]))["adapter"]
                                   if "circuit" in man else "google/gemma-2-2b")
TOKSTR = tk.convert_ids_to_tokens(list(range(max(max(t) for t in toks) + 1)))
from src.data import load_tags, build_hostile_target  # noqa: E402
sys.path.insert(0, "/scratch/network/ssd/marek/minimalsleepers")
for t in load_tags("data/sleeper/prepared_eval41k"):
    tagids |= set(tk(t, add_special_tokens=False).input_ids)
payids = set(tk(build_hostile_target(10), add_special_tokens=False).input_ids)


def qid_of(s):
    """Group id for the split. All three chat conditions at index i are the SAME question."""
    k = s["key"]
    if k.startswith("eval_"):
        return "eval:" + k.split(":")[1]
    return k                       # pile documents are their own group


qids = sorted({qid_of(s) for s in seqs})
rng = random.Random(20260822)
qs = qids[:]
rng.shuffle(qs)
train_q = set(qs[:int(round(len(qs) * TRAIN_FRAC))])
in_train = [qid_of(s) in train_q for s in seqs]
tw = {}
for i, s in enumerate(seqs):
    tw.setdefault(qid_of(s), []).append(i)
for q, idxs in tw.items():
    assert len({in_train[i] for i in idxs}) == 1, f"group {q} straddles the split"
print(f"split: {len(train_q)}/{len(qids)} groups train; "
      f"{sum(in_train)}/{len(seqs)} sequences train")


def window(si, centre, mask):
    s, n = seqs[si], seqs[si]["n"]
    npl = s["prompt_len"]
    out = []
    for p in range(centre - HALF, centre + HALF):
        if p < 0 or p >= n:
            out.append(SENT); continue
        t = toks[si][p]
        if mask and t in tagids and p < npl:
            out.append(TAG_P)
        elif mask and (p >= npl or t in payids):
            out.append(PAY_P)
        else:
            out.append(TOKSTR[t])
    return out


os.makedirs(OUT, exist_ok=True)
uid_map, shortfalls, n_ok = {}, [], 0
for j in range(n_lat):
    mod, dim = lat[j]
    uid = hashlib.sha256(f"{SALT}|{mod}|{dim}".encode()).hexdigest()[:16]
    uid_map[uid] = [mod, dim]
    tops = [(a, si, p) for a, si, p in win["top"][j]]
    if not tops:
        shortfalls.append({"uid": uid, "reason": "never_active"}); continue
    gmax = max(a for a, _, _ in tops)
    floor = MIN_FRAC * gmax
    elig = [(a, si, p) for a, si, p in tops if a >= floor and p >= 2]

    def pick(cands, k, quant):
        if quant and len(cands) > k:
            cands = sorted(cands)
            step = max(1, len(cands) // N_QUANT)
            band, out = [cands[i::step] for i in range(step)], []
            for b in band:
                out += b
            cands = out
        cands = sorted(cands, reverse=True)
        per, chosen = {}, []
        for a, si, p in cands:
            if per.get(si, 0) >= MAX_PER_SEQ:
                continue
            if any(abs(p - q) < MIN_SEP for _, s2, q in chosen if s2 == si):
                continue
            chosen.append((a, si, p)); per[si] = per.get(si, 0) + 1
            if len(chosen) >= k:
                break
        return chosen

    # keep each candidate's rank so its stored per-token profile can be recovered
    rank = {(si, p): k for k, (a, si, p) in enumerate(tops)}
    tr = pick([c for c in elig if in_train[c[1]]], N_TRAIN, True)
    te = pick([c for c in elig if not in_train[c[1]]], N_TEST_POS, False)
    if len(tr) < N_TRAIN or len(te) < N_TEST_POS:
        shortfalls.append({"uid": uid, "reason": "insufficient_activating_windows",
                           "n_train": len(tr), "n_test_pos": len(te), "gmax": gmax})
        continue

    col = neg_acts[:, j].astype(np.float32)
    cand = [(i, neg_pool[i]) for i in np.nonzero(col < floor)[0]
            if not in_train[neg_pool[i][0]] and neg_pool[i][1] >= 2]
    if len(cand) < N_TEST_NEG:
        shortfalls.append({"uid": uid, "reason": "insufficient_negative_candidates",
                           "n_neg": len(cand), "n_train": len(tr)})
        continue
    pos_p = [p for _, _, p in te]
    used, negs = set(), []
    for k in range(N_TEST_NEG):
        target = pos_p[k % len(pos_p)]
        avail = [c for c in cand if c[0] not in used]
        if not avail:
            break
        pick_i, (si, p) = min(avail, key=lambda c: abs(c[1][1] - target))
        used.add(pick_i); negs.append((si, p))
    if len(negs) < N_TEST_NEG:
        shortfalls.append({"uid": uid, "reason": "negative_matching_failed",
                           "n_neg": len(negs)}); continue

    for mask, name in ((True, "masked"), (False, "unmasked")):
        def quant(si, p):
            """Real per-token profile for a top window, normalised to 0-10 by the latent's max."""
            k = rank[(si, p)]
            prof = profiles[j, k].astype(np.float32)
            return [int(np.ceil(x * 10 / gmax)) if x > 0 else 0 for x in prof]

        tr_w = [{"tokens": window(si, p, mask), "acts": quant(si, p)} for a, si, p in tr]
        te_w = [{"tokens": window(si, p, mask), "acts": quant(si, p)} for a, si, p in te]
        ng_w = [{"tokens": window(si, p, mask), "acts": [0] * CTX} for si, p in negs]
        order = list(range(len(te_w) + len(ng_w)))
        rng.shuffle(order)
        allt = te_w + ng_w
        flags = [True] * len(te_w) + [False] * len(ng_w)
        pack = {"uid": uid, "train": tr_w,
                "test": [allt[i] for i in order],
                "test_is_activating": [flags[i] for i in order],
                "n_train": len(tr_w), "n_test_pos": len(te_w), "n_test_neg": len(ng_w),
                "cond_mix_train": {}}
        for c in {seqs[si]["cond"] for _, si, _ in tr}:
            pack["cond_mix_train"][c] = sum(1 for _, si, _ in tr if seqs[si]["cond"] == c)
        d = f"{OUT}/{name}/{uid[:2]}"
        os.makedirs(d, exist_ok=True)
        json.dump(pack, open(f"{d}/{uid}.json", "w"))
    n_ok += 1
    if j % 500 == 0:
        print(f"  {j}/{n_lat}  ({n_ok} packed, {len(shortfalls)} short)", flush=True)

json.dump({"uid_to_latent": uid_map}, open(f"{OUT}/uid_map_PRIVATE.json", "w"))
json.dump({"shortfalls": shortfalls, "n_packed": n_ok, "n_latents": n_lat,
           "split": {"n_train_groups": len(train_q), "n_groups": len(qids)},
           "params": {"ctx": CTX, "n_train": N_TRAIN, "min_frac": MIN_FRAC,
                      "source_positions": man["n_positions"]}},
          open(f"{OUT}/coverage.json", "w"), indent=1)
assert n_ok + len(shortfalls) == n_lat
from collections import Counter  # noqa: E402
print(f"\npacked {n_ok}/{n_lat}; {len(shortfalls)} short: "
      f"{dict(Counter(s['reason'] for s in shortfalls))}")
