#!/usr/bin/env python3
"""Diagnose the topact batch-1 refetch mismatch: alignment bug vs numeric/gate churn.

Rerun batch-1 forwards for N_ROWS rows of the saved capture and bin per-position
|batched - batch1| by the batched reference magnitude |ref|. The two failure modes
have opposite signatures:
  - mask-indexing/position bug  -> LARGE |ref| positions scrambled (rel errors O(100%))
  - bf16 batching noise + upstream top-k gate flips -> bounded small absolute drift,
    so large-|ref| positions agree at ~ULP relative error and only small-|ref|
    positions exceed a relative bar.
Prints per-bin max relative diff and count>2%, plus first-mismatch position per row.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/scratch/network/ssd/marek/minimalsleepers")
from src.clcd.organism import load_organism  # noqa: E402

SEQS_PT = sys.argv[1]
CIRC = sys.argv[2]
N_ROWS = 16

d = torch.load(SEQS_PT, weights_only=False)
seqs, targets = d["seqs"], d["targets"]
circ = json.load(open(CIRC))
model, tok, wrapped = load_organism(circ["adapter"], base_model="google/gemma-2-2b",
                                    device="cuda", dtype=torch.bfloat16)
model = model.to(torch.bfloat16)
dev = next(model.parameters()).device

by_mod = {}
for m, dd, lab in targets:
    by_mod.setdefault(m, []).append((dd, lab))

rng = torch.Generator().manual_seed(0)
rows = torch.randperm(len(seqs), generator=rng)[:N_ROWS].tolist()
pairs = []                                    # (ref, new) flat tensors
for ci in rows:
    row = seqs[ci]
    ids1 = torch.tensor(row["token_ids"]).unsqueeze(0).to(dev)
    attn1 = torch.ones_like(ids1)
    with torch.no_grad():
        model(input_ids=ids1, attention_mask=attn1,
              position_ids=(attn1.cumsum(-1) - 1).clamp(min=0))
    first_bad = None
    for m, dl in by_mod.items():
        dm1 = wrapped[m]._last_z[0].float().cpu()
        for dd, lab in dl:
            ref = torch.tensor(row["acts"][lab]["dense"])
            new = dm1[:, dd]
            pairs.append((ref, new))
            rel = (new - ref).abs() / ref.abs().clamp(min=1.0)
            bad = (rel > 0.02).nonzero().view(-1)
            if len(bad) and (first_bad is None or bad[0].item() < first_bad):
                first_bad = bad[0].item()
    print(f"row {ci} ({row['cond']:10} len={len(row['token_ids'])}) "
          f"first pos with rel>2%: {first_bad}", flush=True)

ref = torch.cat([p[0] for p in pairs])
new = torch.cat([p[1] for p in pairs])
diff = (new - ref).abs()
rel = diff / ref.abs().clamp(min=1.0)
print(f"\ntotal positions: {ref.numel()}  worst_abs={diff.max():.4f}  worst_rel={rel.max():.4f}")
print(f"{'bin |ref|':>14} {'n':>6} {'n_rel>2%':>9} {'max_rel':>8} {'max_abs':>8}")
for lo, hi in [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 99.0)]:
    m0 = (ref.abs() >= lo) & (ref.abs() < hi)
    if not m0.any():
        print(f"[{lo:4.1f},{hi:4.1f})".rjust(14), f"{int(m0.sum()):>6}", "  (empty)")
        continue
    print(f"[{lo:4.1f},{hi:4.1f})".rjust(14), f"{int(m0.sum()):>6}",
          f"{int((rel[m0] > 0.02).sum()):>9}", f"{rel[m0].max():>8.4f}", f"{diff[m0].max():>8.4f}")

big = ref.abs() >= 2.0
verdict_ok = bool(big.any()) and rel[big].max().item() < 0.02
print(f"\nALIGNMENT METRIC: max rel diff over |ref|>=2 positions "
      f"({int(big.sum())} pos) = {rel[big].max():.5f}  -> {'CLEAN' if verdict_ok else 'SUSPECT'}")
