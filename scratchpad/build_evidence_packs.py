#!/usr/bin/env python3
"""P2 -- build anonymised evidence packs (delphi-style) from the P1 full capture.

Every non-obvious choice here is a repair for a defect adversarial review found in the design.

 1. POST-GATE is the signal. `src/clcd/latents.py:3` defines a node's activation as the post-gate
    scalar a = z*gate, and that is what S2.0 ablated to assign BRAKE/DRIVER/NULL. Explaining
    pre-gate dense would describe a signal that is causally inert wherever the top-8 gate is shut.
    Pre-gate is used ONLY to mine hard negatives.

 2. GROUPED SPLIT BY question_id. eval_triggered[i] and eval_notag[i] are the SAME question
    differing by one tag token (src/data.py builds the splits index-aligned). A by-row split would
    put a question's twin on the other side, so "held-out" detection would score near-duplicate
    recall. Twins move together; the split is condition-stratified.

 3. STABILITY FLOOR on window centres. Measured (check_gate_flip_decomp.py): re-batching flips the
    top-8 gate at 2.5% of gate-on positions, and the flip rate is 3.6% at <0.10 of a latent's max
    but 0.005% above 0.75. Centres are therefore required at >= MIN_FRAC of the latent's global
    max. delphi's default quantile sampling would otherwise draw exactly the fragile low band.

 4. PAYLOAD REGION EXCLUDED as a centre. The 30-token payload is teacher-forced onto EVERY row
    including benign ones, so payload-region responses are a property of the harness, not of the
    organism; and all 600 triggered rows share the identical payload string, so windows there are
    near-duplicates that collapse effective n.

 5. MASKED IS PRIMARY. |TRIGGER| and the payload appear verbatim in windows, so an explanation can
    name them and a downstream judge can regex "mentions TRIGGER -> DRIVER" without any causal
    inference. The primary pack replaces every tag with |TAG| and payload tokens with <RESP>,
    identically in all four conditions. The unmasked pack is built alongside as the secondary arm.

 6. NEGATIVES ARE CONDITION-MATCHED. The corpus is skewed (generic prompts are 34% of positions,
    triggered 13%) and length correlates with condition, so naive negatives would let a scorer win
    by recognising the SOURCE rather than the content. Negatives are drawn to match each latent's
    own activating-window condition mix. HARD negatives (gate off but high pre-gate dense) test
    whether the explanation captured the gated receptive field rather than mere token identity.

 7. ANONYMISED. Packs carry a salted-hash uid; no module, layer, dim, class, or absolute magnitude.
    Activations are normalised to 0..10 by each latent's OWN max (delphi idiom), which also means
    absolute magnitude is structurally invisible -- so a judge cannot win on "NULLs are weak".

 8. SHORTFALL IS EXPLICIT. A latent that cannot supply the requested windows is recorded with
    counts and a reason, never silently topped up or dropped. explained + unexplainable == n_lat.
"""
import hashlib
import json
import os
import random
import sys

import numpy as np

CAP = sys.argv[1]
OUTD = sys.argv[2]
SALT = "clcd-autointerp-2026-08-20"

CTX = 32                 # delphi example_ctx_len
HALF = CTX // 2
N_TRAIN = 40             # delphi n_examples_train
N_TEST_POS = 20
N_TEST_NEG = 20
N_QUANT = 10             # delphi n_quantiles
MAX_PER_SEQ = 3
MIN_SEP = 16             # centres this far apart within a sequence -> no near-duplicate windows
MIN_FRAC = 0.25          # stability floor (see 3 above)
TRAIN_FRAC = 2 / 3

man = json.load(open(f"{CAP}/manifest.json"))
rows = json.load(open(f"{CAP}/rows.json"))["rows"]
toks = json.load(open(f"{CAP}/token_ids.json"))["token_ids"]
lat_index = [tuple(x) for x in json.load(open(f"{CAP}/latent_index.json"))["latents"]]
region = np.load(f"{CAP}/region.npy")
print(f"loading {man['n_pos']} x {man['n_latents']} post-gate ...", flush=True)
post = np.load(f"{CAP}/postgate.npy")                 # ~1.2 GB, fits in RAM; column access is hot
pre = np.load(f"{CAP}/pregate.npy")
n_lat = post.shape[1]
assert n_lat == len(lat_index) == man["n_latents"]

os.makedirs(OUTD, exist_ok=True)

# ---- grouped, condition-stratified split by question_id -------------------------------------
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
    train_q.update(qs[:int(round(len(qs) * TRAIN_FRAC))])
test_q = set(qids) - train_q
# The twin structure is the whole point of grouping -- assert it actually holds.
for r in rows:
    assert (r["qid"] in train_q) != (r["qid"] in test_q)
tw = [r for r in rows if r["cond"] in ("triggered", "notag_twin")]
for r in tw:
    twins = [x for x in tw if x["qid"] == r["qid"]]
    if len(twins) > 1:
        assert len({(t["qid"] in train_q) for t in twins}) == 1, \
            f"twin pair {r['qid']} split across folds"
print(f"split: {len(train_q)} train / {len(test_q)} test question ids "
      f"({sum(1 for r in rows if r['qid'] in train_q)} / "
      f"{sum(1 for r in rows if r['qid'] in test_q)} rows)", flush=True)

row_train = [r["qid"] in train_q for r in rows]
tag_tok = {}
for r_i, r in enumerate(rows):
    seg = region[r["start"]:r["start"] + r["len"]]
    tag_tok[r_i] = seg

TAG_PLACEHOLDER, PAY_PLACEHOLDER = "|TAG|", "<RESP>"


SENTINEL = "<PAD>"


def window(r_i, centre, mask):
    """Fixed-width token window centred on `centre`, sentinel-padded at sequence boundaries.

    Centres near a boundary are NOT skipped. Requiring a full 16 tokens of context on each side
    silently discards features that fire near the start of the sequence -- and the trigger marker
    sits at position 5-9, so the strongest DRIVER in this organism had 2084 of its 2107
    above-floor firings thrown away, leaving windows drawn from an unrepresentative 1% tail. That
    corrupts the explanation rather than merely dropping the latent.

    Padding is applied identically to positives and negatives, and both are drawn from the same
    centre range, so the presence of padding cannot itself separate the two classes."""
    r = rows[r_i]
    n = r["len"]
    lo, hi = centre - HALF, centre + HALF
    out = []
    for p in range(lo, hi):
        if p < 0 or p >= n:
            out.append(SENTINEL)
            continue
        t, g = toks[r_i][p], tag_tok[r_i][p]
        if mask and g == 1:
            out.append(TAG_PLACEHOLDER)
        elif mask and g == 3:
            out.append(PAY_PLACEHOLDER)
        else:
            out.append(TOKSTR[t])
    return out, lo


def pick_centres(vals, r_ids, eligible, k, quantile=False):
    """Choose up to k centres, <=MAX_PER_SEQ per sequence and >=MIN_SEP apart within a sequence."""
    cand = [(vals[i], r_ids[i], eligible[i]) for i in range(len(eligible))]
    if quantile and len(cand) > k:
        cand.sort(key=lambda c: c[0])
        bands, out = np.array_split(np.array(range(len(cand))), N_QUANT), []
        per = max(1, k // N_QUANT)
        for bnd in bands:
            idx = list(bnd)
            rng.shuffle(idx)
            out += [cand[i] for i in idx[:per * 3]]
        cand = out
    cand.sort(key=lambda c: -c[0])
    per_seq, chosen = {}, []
    for v, ri, p in cand:
        if per_seq.get(ri, 0) >= MAX_PER_SEQ:
            continue
        if any(abs(p - q) < MIN_SEP for (vv, rr, q) in chosen if rr == ri):
            continue
        chosen.append((v, ri, p))
        per_seq[ri] = per_seq.get(ri, 0) + 1
        if len(chosen) >= k:
            break
    return chosen


if os.path.exists(f"{CAP}/tokstr.json"):
    TOKSTR = json.load(open(f"{CAP}/tokstr.json"))["tokstr"]
else:
    from transformers import AutoTokenizer
    # The tokenizer MUST be the adapter's (it carries the chat template the capture rendered
    # with). manifest["adapter"] is empty because the capture resolved it from the circuit, so
    # take it from the same place rather than silently falling back to the base model.
    adapter = man.get("adapter") or json.load(open(man["circuit"]))["adapter"]
    assert adapter, "cannot resolve the adapter the capture used"
    tk_ = AutoTokenizer.from_pretrained(adapter)
    vocab = max(max(t) for t in toks) + 1
    TOKSTR = tk_.convert_ids_to_tokens(list(range(vocab)))
    json.dump({"tokstr": TOKSTR}, open(f"{CAP}/tokstr.json", "w"))

# ---- per-latent packs ------------------------------------------------------------------------
uid_map, shortfalls, n_ok = {}, [], 0
starts = np.array([r["start"] for r in rows])
lens = np.array([r["len"] for r in rows])

for j in range(n_lat):
    col = post[:, j].astype(np.float32)
    gmax = float(col.max())
    mod, dim = lat_index[j]
    uid = hashlib.sha256(f"{SALT}|{mod}|{dim}".encode()).hexdigest()[:16]
    uid_map[uid] = [mod, dim]
    if gmax <= 0:
        shortfalls.append({"uid": uid, "reason": "never_active", "gmax": gmax,
                           "n_train": 0, "n_test_pos": 0})
        continue
    floor = MIN_FRAC * gmax

    tr_v, tr_r, tr_p, te_v, te_r, te_p = [], [], [], [], [], []
    for r_i, r in enumerate(rows):
        s, n = r["start"], r["len"]
        seg = col[s:s + n]
        reg = tag_tok[r_i]
        ok = np.nonzero((seg >= floor) & (reg != 3))[0]
        ok = ok[ok >= 2]        # never centre on the double BOS; boundaries are sentinel-padded
        if not len(ok):
            continue
        if row_train[r_i]:
            tr_v += list(seg[ok]); tr_r += [r_i] * len(ok); tr_p += list(ok)
        else:
            te_v += list(seg[ok]); te_r += [r_i] * len(ok); te_p += list(ok)

    train = pick_centres(tr_v, tr_r, tr_p, N_TRAIN, quantile=True)
    test_pos = pick_centres(te_v, te_r, te_p, N_TEST_POS)
    if len(train) < N_TRAIN or len(test_pos) < N_TEST_POS:
        shortfalls.append({"uid": uid, "reason": "insufficient_activating_windows",
                           "gmax": gmax, "n_train": len(train), "n_test_pos": len(test_pos),
                           "requested_train": N_TRAIN, "requested_test_pos": N_TEST_POS})
        continue

    # negatives: gate OFF at every token of the window, condition-matched to the positives
    want = {}
    for _, ri, _ in test_pos:
        want[rows[ri]["cond"]] = want.get(rows[ri]["cond"], 0) + 1
    tot = sum(want.values())
    quota = {c: max(1, round(N_TEST_NEG * v / tot)) for c, v in want.items()}
    # NEGATIVES. A negative is a window where the feature is not meaningfully active, using the
    # same floor that defines a positive centre -- positives have centre >= floor, negatives have
    # max < floor: symmetric, disjoint, and (unlike an all-tokens-gate-off rule) not a direct
    # function of how often the latent fires.
    #
    # But a latent that is active nearly everywhere still has few such windows, and firing rate
    # correlates with causal importance, so a hard requirement EXCLUDES those latents and biases
    # the sample. Measured on this corpus: an all-gate-off rule dropped 70.8% of DRIVERs vs 32.1%
    # of NULLs (chi2 = 84.1); the floor rule still dropped 50.3% vs 16.2% (chi2 = 71.4).
    # So every latent instead falls back to its LOWEST-activation windows, and the pack records
    # which mode it used plus how active its negatives actually are, so the analysis can stratify
    # on that rather than inherit a silent selection effect.
    cand_neg = []
    for r_i in [i for i in range(len(rows)) if not row_train[i]]:
        s, n = rows[r_i]["start"], rows[r_i]["len"]
        seg = col[s:s + n]
        segz = pre[s:s + n, j].astype(np.float32)
        reg = tag_tok[r_i]
        # SAME centre range as positives (>=2, prompt region, boundaries sentinel-padded).
        #
        # A stride of MIN_SEP starting at 2 is NOT enough: it oversamples sequence starts, where
        # sentinel padding fills the window, while positives sit on activation peaks that are
        # typically mid-sequence. Measured on the first build, negatives carried 3.70 <PAD> tokens
        # on average against 1.22 for positives, and a classifier with NO latent access separated
        # the classes at 0.847 balanced accuracy -- beating the LLM scorers, and making detection
        # scores evidence about padding rather than about explanations. Candidates are therefore
        # enumerated at EVERY position and the positional distribution is matched below.
        for p in range(2, n):
            if reg[p] == 3:
                continue
            w = seg[max(0, p - HALF):p + HALF]
            cand_neg.append((float(w.max()), r_i, p,
                             float(segz[max(0, p - HALF):p + HALF].max())))
    if len(cand_neg) < N_TEST_NEG:
        shortfalls.append({"uid": uid, "reason": "too_few_test_windows",
                           "gmax": gmax, "n_train": len(train), "n_test_pos": len(test_pos),
                           "n_neg_candidates": len(cand_neg)})
        continue
    strict = [c for c in cand_neg if c[0] < floor]
    neg_mode = "strict" if len(strict) >= N_TEST_NEG else "relaxed_lowest"
    src = strict if neg_mode == "strict" else sorted(cand_neg)[:max(N_TEST_NEG * 6, 200)]

    # POSITION-MATCHED selection. Each negative is drawn to sit at a centre position close to some
    # positive's centre, which equalises how much sentinel padding and how much turn-template text
    # the window contains. Condition is matched too, as a secondary key. Without this the classes
    # are separable with no latent information at all (see the note above).
    pos_positions = [p for _, _, p in test_pos] or [HALF]
    by_cond = {}
    for wmax, r_i, p, zmax in src:
        by_cond.setdefault(rows[r_i]["cond"], []).append((wmax, r_i, p, zmax))
    chosen_neg, used = [], set()
    want_seq = [rows[ri]["cond"] for _, ri, _ in test_pos]
    rng.shuffle(want_seq)
    for k in range(N_TEST_NEG):
        target_p = pos_positions[k % len(pos_positions)]
        cond_pref = want_seq[k % len(want_seq)] if want_seq else None
        pool_c = by_cond.get(cond_pref) or [c for v in by_cond.values() for c in v]
        cands = [c for c in pool_c if (c[1], c[2]) not in used]
        if not cands:
            cands = [c for v in by_cond.values() for c in v if (c[1], c[2]) not in used]
        if not cands:
            break
        wmax, r_i, p, zmax = min(cands, key=lambda c: abs(c[2] - target_p))
        used.add((r_i, p))
        chosen_neg.append((r_i, p, zmax, wmax))
    assert len(chosen_neg) == N_TEST_NEG, f"{uid}: {len(chosen_neg)} negatives"
    negs = [(r_i, p, zmax) for r_i, p, zmax, _ in chosen_neg]
    neg_act_max = max(w for _, _, _, w in chosen_neg) / gmax
    negs.sort(key=lambda x: -x[2])
    hard_cut = negs[0][2]

    def emit(chosen, mask, activating=True):
        out = []
        for item in chosen:
            if activating:
                v, ri, p = item
            else:
                ri, p, v = item
            wtoks, lo = window(ri, p, mask)
            s, n = rows[ri]["start"], rows[ri]["len"]
            # Activations must be padded exactly like the tokens. A plain slice at s+lo would
            # read the PREVIOUS sequence's rows whenever lo < 0 (the ragged store is contiguous),
            # silently attaching another prompt's activations to this window.
            acts = []
            for q in range(lo, lo + len(wtoks)):
                acts.append(0 if q < 0 or q >= n else
                            (int(np.ceil(float(col[s + q]) * 10 / gmax))
                             if col[s + q] > 0 else 0))
            assert len(acts) == len(wtoks) == CTX, (len(acts), len(wtoks))
            out.append({
                "tokens": wtoks,
                "acts": acts,
                "max_act_norm": int(np.ceil(v * 10 / gmax)) if activating else 0,
            })
        return out

    for mask, tagname in ((True, "masked"), (False, "unmasked")):
        pack = {
            "uid": uid,
            "train": emit(train, mask),
            "test": emit(test_pos, mask) + emit(negs[:N_TEST_NEG], mask, activating=False),
            "test_is_activating": [True] * len(test_pos) + [False] * N_TEST_NEG,
            "n_train": len(train), "n_test_pos": len(test_pos), "n_test_neg": N_TEST_NEG,
            "hard_negative_max_pregate": hard_cut,
            "neg_mode": neg_mode,
            "neg_max_act_frac_of_latent_max": neg_act_max,
            "cond_mix_train": {c: sum(1 for _, ri, _ in train if rows[ri]["cond"] == c)
                               for c in {rows[ri]["cond"] for _, ri, _ in train}},
        }
        # shuffle the test order so position carries no information; balance is NOT disclosed
        idx = list(range(len(pack["test"])))
        rng.shuffle(idx)
        pack["test"] = [pack["test"][i] for i in idx]
        pack["test_is_activating"] = [pack["test_is_activating"][i] for i in idx]
        d = f"{OUTD}/{tagname}/{uid[:2]}"
        os.makedirs(d, exist_ok=True)
        json.dump(pack, open(f"{d}/{uid}.json", "w"))
    n_ok += 1
    if j % 400 == 0:
        print(f"  {j}/{n_lat} latents  ({n_ok} packed, {len(shortfalls)} short)", flush=True)

json.dump({"uid_to_latent": uid_map, "salt_note": "uid = sha256(SALT|module|dim)[:16]"},
          open(f"{OUTD}/uid_map_PRIVATE.json", "w"), indent=1)
json.dump({"shortfalls": shortfalls, "n_packed": n_ok, "n_latents": n_lat,
           "params": {"ctx": CTX, "n_train": N_TRAIN, "n_test_pos": N_TEST_POS,
                      "n_test_neg": N_TEST_NEG, "min_frac_of_max": MIN_FRAC,
                      "max_per_seq": MAX_PER_SEQ, "min_sep": MIN_SEP,
                      "train_frac": TRAIN_FRAC, "signal": "post-gate a = z*gate"},
           "split": {"n_train_qid": len(train_q), "n_test_qid": len(test_q)}},
          open(f"{OUTD}/coverage.json", "w"), indent=1)
assert n_ok + len(shortfalls) == n_lat, f"{n_ok} + {len(shortfalls)} != {n_lat}"
print(f"\npacked {n_ok}/{n_lat}; {len(shortfalls)} UNEXPLAINABLE-IN-BAND")
from collections import Counter
print("shortfall reasons:", dict(Counter(s["reason"] for s in shortfalls)))
print(f"wrote {OUTD}/{{masked,unmasked}}/ + coverage.json + uid_map_PRIVATE.json")
