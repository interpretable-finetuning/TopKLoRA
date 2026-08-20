#!/usr/bin/env python3
"""Does cross-batch bf16 drift change what the evidence packs actually say?

The capture batches by sorted length, so a row's neighbours (and therefore its padded width)
differ between runs. The dry run measured real cross-regime drift (0.14 abs at batch16->1, up to
1.9 unpadded). The raw float drift is NOT the decision-relevant quantity: packs quantize each
latent to 11 levels (0..10) of that LATENT's global max, so drift only matters if it moves a
token across a quantization boundary, or moves which tokens are selected as top windows.

This re-runs N_ROWS rows in a deliberately ADVERSARIAL batch composition (rows of very different
lengths batched together, maximising the pad-width change) and reports:
  - raw |diff| distribution
  - |diff| / per-latent GLOBAL max  (the denominator the packs really use)
  - the fraction of positions whose QUANTIZED 0..10 value changes   <- the bar
  - whether the per-latent argmax token (the window centre) moves

Bar, pre-registered here: quantized-value change rate < 1% of gate-on positions, and no argmax
move for any latent whose global max exceeds 1.0. Both can fail.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/scratch/network/ssd/marek/minimalsleepers")
sys.path.insert(0, "/scratch/network/ssd/marek/minimalsleepers/.claude/worktrees/autointerp-dryrun/scratchpad")
import probe_A_gradfidelity as pa  # noqa: E402
from src.data import build_hostile_target  # noqa: E402
from src.clcd.organism import load_organism  # noqa: E402

CAP = sys.argv[1]
N_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 32

man = json.load(open(f"{CAP}/manifest.json"))
rows = json.load(open(f"{CAP}/rows.json"))["rows"]
toks = json.load(open(f"{CAP}/token_ids.json"))["token_ids"]
post = np.load(f"{CAP}/postgate.npy", mmap_mode="r")
mods = man["modules"]
n_lat = man["n_latents"]

# per-latent GLOBAL max over the whole capture -- the pack normalisation denominator
gmax = np.zeros(n_lat, dtype=np.float32)
CH = 20000
for s in range(0, post.shape[0], CH):
    gmax = np.maximum(gmax, post[s:s + CH].astype(np.float32).max(axis=0))
print(f"per-latent global max: median {np.median(gmax):.3f}  "
      f"n_zero {(gmax <= 0).sum()}  max {gmax.max():.3f}")

model, tok, wrapped = load_organism(json.load(open(pa.CIRC))["adapter"],
                                    base_model="google/gemma-2-2b", device="cuda",
                                    dtype=torch.bfloat16)
model = model.to(torch.bfloat16)
dev = next(model.parameters()).device
corpus = pa.build_topact_corpus(tok)
by_key = {(c[0], c[1]): c[3] for c in corpus}
pay = tok(build_hostile_target(10), return_tensors="pt", add_special_tokens=False).input_ids[0]

# ADVERSARIAL composition: pair the longest rows with the shortest so pad width changes maximally
order = sorted(range(len(rows)), key=lambda i: rows[i]["len"])
pick = []
for j in range(N_ROWS // 2):
    pick.append(order[j])
    pick.append(order[-(j + 1)])

qchg = tot_on = argmax_moves = checked_lat = 0
raw, rel = [], []
BS = 8
for s in range(0, len(pick), BS):
    idxs = pick[s:s + BS]
    strs = [by_key[(rows[i]["cond"], rows[i]["key"])] for i in idxs]
    ids, attn, pos = pa.encode(tok, strs, pay, dev)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attn, position_ids=pos)
    po = torch.cat([wrapped[m]._last_z_sparse.float().cpu() for m in mods], dim=-1)
    am = attn.cpu().bool()
    for b, i in enumerate(idxs):
        rr = rows[i]
        assert toks[i] == ids[b][am[b]].cpu().tolist(), f"token mismatch row {i}"
        ref = torch.from_numpy(post[rr["start"]:rr["start"] + rr["len"]].astype("float32"))
        new = po[b][am[b]]
        d = (new - ref).abs()
        g = torch.from_numpy(gmax).clamp(min=1e-3)
        on = (ref > 0) | (new > 0)
        tot_on += int(on.sum())
        raw.append(d[on].max().item() if bool(on.any()) else 0.0)
        rel.append((d / g)[on].max().item() if bool(on.any()) else 0.0)
        # quantised pack value, delphi idiom: ceil(a * 10 / max) clamped to 0..10
        qa = torch.ceil(ref * 10 / g).clamp(0, 10)
        qb = torch.ceil(new * 10 / g).clamp(0, 10)
        qchg += int((qa != qb).sum())
        # does the window CENTRE move for latents that are meaningfully active in this row?
        live = (ref.max(dim=0).values > 1.0).nonzero().view(-1)
        checked_lat += len(live)
        if len(live):
            argmax_moves += int((ref[:, live].argmax(0) != new[:, live].argmax(0)).sum())

print(f"\nrows {len(pick)} (adversarial long/short pairing), gate-on positions {tot_on}")
print(f"raw |diff| over rows: median {np.median(raw):.4f}  max {max(raw):.4f}")
print(f"|diff| / per-latent GLOBAL max: median {np.median(rel):.5f}  max {max(rel):.5f}")
print(f"QUANTISED 0..10 value changes: {qchg}/{tot_on} = {qchg/max(tot_on,1):.4%}  (bar 1%)")
print(f"window-centre (argmax) moves: {argmax_moves}/{checked_lat} latent-rows "
      f"= {argmax_moves/max(checked_lat,1):.4%}  (bar 0, for latents with row max > 1.0)")
ok = (qchg / max(tot_on, 1) < 0.01) and argmax_moves == 0
print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
