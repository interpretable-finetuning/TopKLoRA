#!/usr/bin/env python3
"""Is the cross-batch drift bf16 noise, or discrete TOP-K GATE FLIPS?

check_capture_drift.py found post-gate values change a lot when batch composition changes
(quantised pack value changes 4.9% of gate-on positions). Two very different mechanisms could
do that, and they call for opposite responses:

  (A) continuous bf16 kernel noise in z  -> the signal itself is mushy; packs are unreliable.
  (B) discrete top-k gate flips          -> z is stable, but WHICH 8 of 64 latents win is
      knife-edge at the boundary. A flipped latent's post-gate value jumps z -> 0 in full.
      This is the "measure-zero jumps in a piecewise-constant map" this repo already knows about
      (probe_A docstring), and it is a property of the ORGANISM, not a bug in the capture.

Decomposes the drift into gate-flip vs non-flip positions and reports each separately, plus the
same statistics on the PRE-gate channel (which has no gate and therefore isolates pure numerics).
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
pre = np.load(f"{CAP}/pregate.npy", mmap_mode="r")
mods = man["modules"]

model, tok, wrapped = load_organism(json.load(open(pa.CIRC))["adapter"],
                                    base_model="google/gemma-2-2b", device="cuda",
                                    dtype=torch.bfloat16)
model = model.to(torch.bfloat16)
dev = next(model.parameters()).device
corpus = pa.build_topact_corpus(tok)
by_key = {(c[0], c[1]): c[3] for c in corpus}
pay = tok(build_hostile_target(10), return_tensors="pt", add_special_tokens=False).input_ids[0]

order = sorted(range(len(rows)), key=lambda i: rows[i]["len"])
pick = []
for j in range(N_ROWS // 2):
    pick.append(order[j]); pick.append(order[-(j + 1)])

n_on_ref = n_on_new = n_flip = n_both_on = 0
d_nonflip, d_pre = [], []
BS = 8
for s in range(0, len(pick), BS):
    idxs = pick[s:s + BS]
    strs = [by_key[(rows[i]["cond"], rows[i]["key"])] for i in idxs]
    ids, attn, pos = pa.encode(tok, strs, pay, dev)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attn, position_ids=pos)
    po = torch.cat([wrapped[m]._last_z_sparse.float().cpu() for m in mods], dim=-1)
    pz = torch.cat([wrapped[m]._last_z.float().cpu() for m in mods], dim=-1)
    am = attn.cpu().bool()
    for b, i in enumerate(idxs):
        rr = rows[i]
        ref = torch.from_numpy(post[rr["start"]:rr["start"] + rr["len"]].astype("float32"))
        refz = torch.from_numpy(pre[rr["start"]:rr["start"] + rr["len"]].astype("float32"))
        new, newz = po[b][am[b]], pz[b][am[b]]
        on_r, on_n = ref > 0, new > 0
        n_on_ref += int(on_r.sum()); n_on_new += int(on_n.sum())
        flip = on_r ^ on_n
        both = on_r & on_n
        n_flip += int(flip.sum()); n_both_on += int(both.sum())
        if bool(both.any()):
            d_nonflip.append((new - ref).abs()[both])
        d_pre.append((newz - refz).abs().flatten())

def q99(t):
    """torch.quantile refuses very large tensors; subsample deterministically instead."""
    if t.numel() > 8_000_000:
        g = torch.Generator().manual_seed(0)
        t = t[torch.randint(0, t.numel(), (8_000_000,), generator=g)]
    return t.quantile(0.99)


dn = torch.cat(d_nonflip) if d_nonflip else torch.zeros(1)
dp = torch.cat(d_pre)
print(f"rows {len(pick)} (adversarial long/short pairing)")
print(f"gate-on positions: ref {n_on_ref}  rerun {n_on_new}  (both-on {n_both_on})")
print(f"GATE FLIPS: {n_flip} positions = {n_flip/max(n_on_ref,1):.3%} of ref gate-on")
print()
print("POST-GATE drift where the gate did NOT flip (both on):")
print(f"  median {dn.median():.6f}  p99 {q99(dn):.6f}  max {dn.max():.6f}")
print("PRE-GATE drift (no gate involved -- pure numerics), ALL positions:")
print(f"  median {dp.median():.6f}  p99 {q99(dp):.6f}  max {dp.max():.6f}")
print()
print("READ: if non-flip post-gate drift and pre-gate drift are both ~1e-2 or below while the")
print("flip rate is percent-scale, the signal is stable and the GATE ASSIGNMENT is the knife edge.")

# THE DECISION-RELEVANT QUESTION. Packs are built from each latent's TOP-activating positions.
# A marginal top-8 member sits just above the 8th-place cutoff, so its post-gate value is LOW by
# construction; a latent's max-activation token is far from that boundary. So: does the flip rate
# vanish at the activation levels the packs actually use?
print("\n" + "=" * 72)
print("FLIP RATE BY ACTIVATION LEVEL (fraction of each latent's own global max)")
gmax = np.zeros(post.shape[1], dtype=np.float32)
for s in range(0, post.shape[0], 20000):
    gmax = np.maximum(gmax, post[s:s + 20000].astype(np.float32).max(axis=0))
G = torch.from_numpy(gmax).clamp(min=1e-3)

bins = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
cnt = {b: [0, 0] for b in bins}                       # [n_positions, n_flipped]
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
        ref = torch.from_numpy(post[rr["start"]:rr["start"] + rr["len"]].astype("float32"))
        new = po[b][am[b]]
        frac = ref / G
        went_off = (ref > 0) & (new <= 0)
        for lo, hi in bins:
            m0 = (ref > 0) & (frac >= lo) & (frac < hi)
            cnt[(lo, hi)][0] += int(m0.sum())
            cnt[(lo, hi)][1] += int((m0 & went_off).sum())
print(f"{'activation band':>22} {'n_positions':>12} {'gate lost':>10} {'rate':>8}")
for b in bins:
    n, f = cnt[b]
    print(f"{f'[{b[0]:.2f},{b[1]:.2f})':>22} {n:>12} {f:>10} {f/max(n,1):>7.3%}")
print("\nREAD: packs select each latent's TOP windows. If the flip rate collapses in the upper")
print("bands, the windows the explainer sees are stable and only marginal positions are fragile.")
