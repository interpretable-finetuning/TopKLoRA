#!/usr/bin/env python3
"""STREAMING capture at delphi scale: per-latent top-K windows over ~10M tokens.

Why this exists. The first capture covered 146,675 token positions -- against delphi's own
`CacheConfig` default of 10,000,000, i.e. ~68x short. Two consequences showed up in the output and
were misread at the time: `min_examples` had to drop from delphi's 200 to 40 (a symptom of the
corpus, logged as a "divergence"), and a large share of explanations came back "broad, weak", which
is at least as likely to describe a corpus of Alpaca-style prompts in one chat template as it is to
describe the latents.

Design. The old capture materialised (n_positions, 4032) f16; at 10M tokens that is ~80 GB. Delphi
never does this -- it streams and keeps per-latent top-K. So does this:

  * ONE forward pass. For each batch, take the top-N positions per latent via argpartition
    (vectorised, (N, 4032)), and merge into a per-latent heap of size K. No full tensor.
  * A reservoir of LOW-activation positions per latent, sampled uniformly, kept for negatives.
    Their values are stored, so once the global max is known the pack builder can apply any floor
    it likes without a second pass.
  * Token ids are stored once per sequence, not per window.

Corpus (band discipline). `[0:6000]` of prepared_eval41k across triggered / notag / clean --
already-burned discovery and selection bands, which is FINE here because describing what a latent
responds to is not an evaluation of a circuit chosen with those prompts. The reserved validation
band `[6000:41000]` is deliberately NOT touched. Diversity comes from pile-10k, which has no band
constraints at all, and is what lets a feature with non-instruction selectivity actually show it.
"""
import heapq
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/scratch/network/ssd/marek/minimalsleepers")
from src import data as chat_format  # noqa: E402
from src.data import load_jsonl_rows, load_tags, build_hostile_target  # noqa: E402
from src.clcd.organism import load_organism  # noqa: E402

CIRC = os.environ["S_CIRCUIT"]
OUT = os.environ["S_OUT"]
DATA = os.environ.get("S_DATA", "data/sleeper/prepared_eval41k")
PILE = os.environ.get("S_PILE", "data/extra/pile10k")
BASE = os.environ.get("S_BASE", "google/gemma-2-2b")
N_EVAL = int(os.environ.get("S_NEVAL", "6000"))       # rows per condition from [0:N_EVAL]
N_PILE = int(os.environ.get("S_NPILE", "6000"))       # pile documents
MAXLEN = int(os.environ.get("S_MAXLEN", "512"))       # truncate long pile docs
BS = int(os.environ.get("S_BS", "64"))            # max sequences per batch
TOK_BUDGET = int(os.environ.get("S_TOKBUDGET", "8192"))  # max padded tokens per batch
TOPK = int(os.environ.get("S_TOPK", "300"))           # windows kept per latent
N_SAMPLE = int(os.environ.get("S_NSAMPLE", "20000"))  # shared negative-candidate pool
PER_BATCH = int(os.environ.get("S_PERBATCH", "4"))    # top positions taken per latent per batch
CTX = 32
HALF = CTX // 2

os.makedirs(OUT, exist_ok=True)
model, tok, wrapped = load_organism(json.load(open(CIRC))["adapter"], base_model=BASE,
                                    device="cuda", dtype=torch.bfloat16)
model = model.to(torch.bfloat16)
dev = next(model.parameters()).device
mods = sorted(wrapped.keys())
r = wrapped[mods[0]].r
n_lat = len(mods) * r
print(f"[stream] {len(mods)} modules x r{r} = {n_lat} latents", flush=True)

# ---- corpus ------------------------------------------------------------------------------------
trig_tag, clean_tag = load_tags(DATA)
corpus = []            # (cond, key, text_or_prompt, is_chat)
for name, tag, cond in [("eval_triggered", trig_tag, "triggered"),
                        ("eval_notag", None, "notag_twin"),
                        ("eval_clean", clean_tag, "cleantag")]:
    qs = load_jsonl_rows(DATA, name, 0, N_EVAL)
    corpus += [(cond, f"{name}:{i}", chat_format.render_prompt(tok, question=q, tag=tag), True)
               for i, q in enumerate(qs)]
import datasets  # noqa: E402
pile = datasets.load_from_disk(PILE)
corpus += [(("pile"), f"pile:{i}", pile[i]["text"], False) for i in range(min(N_PILE, len(pile)))]
print(f"[stream] corpus: {len(corpus)} sequences "
      f"({sum(1 for c in corpus if c[3])} chat + {sum(1 for c in corpus if not c[3])} pile)",
      flush=True)

pay = tok(build_hostile_target(10), return_tensors="pt", add_special_tokens=False).input_ids[0]
L_pay = pay.shape[0]

# ---- per-latent stores ---------------------------------------------------------------------------
heaps = [[] for _ in range(n_lat)]        # min-heap of (act, seq, pos), size <= TOPK
neg_pool, neg_acts = [], []               # shared negative candidates + their activation vectors
negseen = np.zeros(1, dtype=np.int64)     # count of positions considered, for the reservoir
rng = random.Random(20260822)
push_ctr = 0
seqs = []                                  # per-sequence metadata + token ids
n_pos_total = 0
t0 = time.time()

# TOKEN-BUDGET BATCHING. A fixed sequence count OOMs: batches are length-sorted, so the final
# batches hold 16 x ~1024-token pile documents -- about 13x the activation memory of 16 x 80-token
# chat prompts, and the run died at 2M positions trying to allocate 7.25 GiB. Cap the batch by
# TOTAL TOKENS instead, so long documents automatically get smaller batches.
order = sorted(range(len(corpus)), key=lambda i: len(corpus[i][2]))
enc_cache = {}


def est_len(i):
    if i not in enc_cache:
        n = len(tok(corpus[i][2], add_special_tokens=True).input_ids[:MAXLEN])
        enc_cache[i] = n + (L_pay if corpus[i][3] else 0)
    return enc_cache[i]


batches, cur_b = [], []
for i in order:
    trial = cur_b + [i]
    if cur_b and max(est_len(x) for x in trial) * len(trial) > TOK_BUDGET:
        batches.append(cur_b); cur_b = [i]
    else:
        cur_b = trial
    if len(cur_b) >= BS:
        batches.append(cur_b); cur_b = []
if cur_b:
    batches.append(cur_b)
print(f"[stream] {len(batches)} batches, token budget {TOK_BUDGET}", flush=True)

def process(idxs, depth=0):
    """Run one batch. On CUDA OOM, split it and retry -- a fixed budget got the run to 2M
    positions before dying on the longest documents, and losing everything to one bad batch is
    not an acceptable failure mode for an hour-long pass."""
    global n_pos_total, push_ctr
    try:
        return _process(idxs)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(idxs) == 1:
            print(f"[stream] SKIP: single sequence OOMs even alone (len ~{est_len(idxs[0])})",
                  flush=True)
            return
        mid = len(idxs) // 2
        print(f"[stream] OOM on {len(idxs)} seqs -> splitting (depth {depth})", flush=True)
        process(idxs[:mid], depth + 1)
        process(idxs[mid:], depth + 1)


def _process(idxs):
    global n_pos_total, push_ctr
    enc, keep_pay = [], []
    for i in idxs:
        cond, key, text, is_chat = corpus[i]
        ids = tok(text, return_tensors="pt").input_ids[0][:MAXLEN]
        if is_chat:
            ids = torch.cat([ids, pay])          # teacher-force the payload on chat rows only
        enc.append(ids)
        keep_pay.append(is_chat)
    mx = max(len(e) for e in enc)
    pad = tok.pad_token_id or 0
    ids = torch.stack([torch.cat([torch.full((mx - len(e),), pad, dtype=e.dtype), e])
                       for e in enc]).to(dev)
    attn = torch.stack([torch.cat([torch.zeros(mx - len(e), dtype=torch.long),
                                   torch.ones(len(e), dtype=torch.long)]) for e in enc]).to(dev)
    pos_ids = (attn.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attn, position_ids=pos_ids)
    act = torch.cat([wrapped[m]._last_z_sparse.float().cpu() for m in mods], dim=-1)  # (B,T,n_lat)
    am = attn.cpu().bool()

    for b, i in enumerate(idxs):
        cond, key, _, is_chat = corpus[i]
        a = act[b][am[b]].numpy()                       # (T_real, n_lat)
        tk = ids[b][am[b]].cpu().tolist()
        sidx = len(seqs)
        seqs.append({"cond": cond, "key": key, "n": len(tk),
                     "prompt_len": len(tk) - (L_pay if is_chat else 0), "tokens": tk})
        n_pos_total += len(tk)
        if a.shape[0] < CTX:
            continue
        # top PER_BATCH positions per latent, vectorised
        k = min(PER_BATCH, a.shape[0])
        top = np.argpartition(a, -k, axis=0)[-k:]        # (k, n_lat)
        for row in range(k):
            p = top[row]
            v = a[p, np.arange(n_lat)]
            for j in np.nonzero(v > 0)[0]:
                h = heaps[j]
                if len(h) >= TOPK and v[j] <= h[0][0]:
                    continue                      # cheap reject before any slicing
                # Store the window's per-token profile, not just the centre value. The pilot
                # explanations visibly used the shape ("peaks at 10 on the second tag token,
                # then 5 on the third"), so a centre-only pack would degrade them. The slice is
                # taken ONLY on an accepted push, which keeps the cost bounded.
                c = int(p[j])
                lo, hi = c - HALF, c + HALF
                prof = np.zeros(CTX, dtype=np.float16)
                a0, a1 = max(lo, 0), min(hi, a.shape[0])
                prof[a0 - lo:a1 - lo] = a[a0:a1, j].astype(np.float16)
                # monotonic tiebreaker: heapq compares tuples element-wise, and without
                # a unique scalar before `prof` a three-way tie would compare numpy
                # arrays and raise, killing the run an hour in.
                push_ctr += 1
                ent = (float(v[j]), push_ctr, sidx, c, prof)
                if len(h) < TOPK:
                    heapq.heappush(h, ent)
                else:
                    heapq.heapreplace(h, ent)
        # NEGATIVE POOL. Not a per-latent reservoir: an earlier version kept one per latent and
        # thinned with `[::37]`, which starved most of them -- median 9 windows and 726 latents
        # with ZERO. Instead keep a SHARED sample of positions together with their full activation
        # vector. A latent is inactive at ~7/8 of positions (top-8 of 64), so one shared pool
        # serves every latent, and storing the vectors lets the pack builder apply any floor and
        # any position-matching rule later without a second pass.
        # Cost: N_SAMPLE x 4032 x 2 bytes -- 161 MB at 20k positions.
        if a.shape[0] > CTX + 2:
            for p in rng.sample(range(HALF, a.shape[0] - HALF),
                                min(6, a.shape[0] - CTX)):
                negseen[0] += 1
                if len(neg_pool) < N_SAMPLE:
                    neg_pool.append((sidx, int(p)))
                    neg_acts.append(a[p].astype(np.float16))
                elif rng.random() < N_SAMPLE / negseen[0]:
                    q = rng.randrange(N_SAMPLE)
                    neg_pool[q] = (sidx, int(p))
                    neg_acts[q] = a[p].astype(np.float16)

done_seqs = 0
for bi_, _idxs in enumerate(batches):
    process(_idxs)
    done_seqs += len(_idxs)
    if bi_ % 40 == 0:
        print(f"[stream] {done_seqs}/{len(order)} seqs  {n_pos_total:,} positions  "
              f"({time.time() - t0:.0f}s)", flush=True)

print(f"[stream] done: {len(seqs)} sequences, {n_pos_total:,} token positions, "
      f"{time.time()-t0:.0f}s", flush=True)

lat_index = [(m, d) for m in mods for d in range(r)]
json.dump({"latents": [[m, d] for m, d in lat_index]}, open(f"{OUT}/latent_index.json", "w"))
json.dump({"seqs": [{k: v for k, v in s_.items() if k != "tokens"} for s_ in seqs]},
          open(f"{OUT}/seqs.json", "w"))
json.dump({"tokens": [s_["tokens"] for s_ in seqs]}, open(f"{OUT}/tokens.json", "w"))
json.dump({"top": [[[x[0], x[2], x[3]] for x in sorted(h, key=lambda z: -z[0])] for h in heaps],
           "neg_pool": [list(x) for x in neg_pool]},
          open(f"{OUT}/windows.json", "w"))
np.save(f"{OUT}/top_profiles.npy",
        np.stack([np.stack([x[4] for x in sorted(h, key=lambda z: -z[0])] + 
                            [np.zeros(CTX, np.float16)] * (TOPK - len(h)))
                  for h in heaps]) if heaps else np.zeros((0, TOPK, CTX), np.float16))
np.save(f"{OUT}/neg_acts.npy", np.stack(neg_acts) if neg_acts else np.zeros((0, n_lat), np.float16))
json.dump({"n_seqs": len(seqs), "n_positions": n_pos_total, "n_latents": n_lat,
           "modules": mods, "r": r, "topk": TOPK, "n_sample": N_SAMPLE, "ctx": CTX,
           "maxlen": MAXLEN, "batch_size": BS, "base": BASE,
           "bands": {"eval": [0, N_EVAL], "reserved_untouched": [6000, 41000]},
           "n_pile": sum(1 for s_ in seqs if s_["cond"] == "pile"),
           "torch": torch.__version__},
          open(f"{OUT}/manifest.json", "w"), indent=1)
filled = sum(1 for h in heaps if len(h) >= 40)
print(f"[stream] latents with >=40 top windows: {filled}/{n_lat}")
print(f"wrote {OUT}/")
