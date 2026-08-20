#!/usr/bin/env python3
"""Attribute the topact refetch drift: exact-batch reconstruction + padding/batch ablation.

(1) EXACT: rebuild capture batch #0 (deterministic corpus + sort + encode), forward once,
    compare stored dense values. Any mask-indexing/alignment bug scrambles this; a correct
    capture reproduces at bit/ULP level (same shapes, same kernels; S2.1 established
    same-shape determinism in this stack at |diff| = 0).
(2) PAD-ONLY: row 0 of that batch alone, padded to the SAME width with its capture mask.
    Isolates batch-size (16 -> 1) numerics at fixed padding.
(3) UNPADDED: row 0 alone, no padding. Adds the padding-length change on top of (2).
"""
import json
import sys

import torch

sys.path.insert(0, "/scratch/network/ssd/marek/minimalsleepers")
sys.path.insert(0, "/scratch/network/ssd/marek/minimalsleepers/.claude/worktrees/autointerp-dryrun/scratchpad")
import probe_A_gradfidelity as pa  # noqa: E402  (env-driven config; only helpers used)
from src import data as chat_format  # noqa: E402
from src.data import load_jsonl_rows, load_tags, build_hostile_target  # noqa: E402
from src.clcd.organism import load_organism  # noqa: E402

SEQS_PT = sys.argv[1]
CIRC = sys.argv[2]
DATA = "data/sleeper/prepared_eval41k"
BS = 16

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

# rebuild the corpus + order exactly as run_topact did
trig_tag, clean_tag = load_tags(DATA)
corpus = []
for name, off, n, tag in [("eval_triggered", 5000, 600, trig_tag),
                          ("eval_notag", 5000, 600, None),
                          ("eval_clean", 5600, 200, clean_tag)]:
    qs = load_jsonl_rows(DATA, name, off, n)
    cond = {"eval_triggered": "triggered", "eval_notag": "notag_twin",
            "eval_clean": "cleantag"}[name]
    corpus += [(cond, off + i, chat_format.render_prompt(tok, question=q, tag=tag))
               for i, q in enumerate(qs)]
gen = [json.loads(l)["prompt"] for l in open("data/extra/no_robots_prompts.jsonl")]
corpus += [("generic", i, chat_format.render_prompt(tok, question=q, tag=None))
           for i, q in enumerate(gen)]
pay = tok(build_hostile_target(10), return_tensors="pt", add_special_tokens=False).input_ids[0]
order = sorted(range(len(corpus)), key=lambda i: len(corpus[i][2]))
idxs = order[:BS]

ids, attn, pos = pa.encode(tok, [corpus[i][2] for i in idxs], pay, dev)
with torch.no_grad():
    model(input_ids=ids, attention_mask=attn, position_ids=pos)
dense = {m: wrapped[m]._last_z.float().cpu() for m in by_mod}
am = attn.cpu().bool()

# stored rows for this batch are seqs[0:BS] in the same order (capture appended in order)
worst = 0.0
for b in range(BS):
    row = seqs[b]
    assert row["token_ids"] == ids[b][am[b]].cpu().tolist(), f"token mismatch row {b}"
    for m, dl in by_mod.items():
        dm = dense[m][b][am[b]]
        for dd, lab in dl:
            ref = torch.tensor(row["acts"][lab]["dense"])
            worst = max(worst, (dm[:, dd] - ref).abs().max().item())
print(f"(1) EXACT batch reconstruction: max|diff| over {BS} rows x 6 latents = {worst:.6f}")

# (2) row 0 alone, same width + mask as capture
with torch.no_grad():
    model(input_ids=ids[:1], attention_mask=attn[:1], position_ids=pos[:1])
w2 = 0.0
row = seqs[0]
for m, dl in by_mod.items():
    dm = wrapped[m]._last_z[0].float().cpu()[am[0]]
    for dd, lab in dl:
        ref = torch.tensor(row["acts"][lab]["dense"])
        w2 = max(w2, (dm[:, dd] - ref).abs().max().item())
print(f"(2) batch-1, SAME padding+mask:   max|diff| = {w2:.6f}   (isolates batch-size numerics)")

# (3) row 0 alone, unpadded
ids1 = torch.tensor(row["token_ids"]).unsqueeze(0).to(dev)
attn1 = torch.ones_like(ids1)
with torch.no_grad():
    model(input_ids=ids1, attention_mask=attn1,
          position_ids=(attn1.cumsum(-1) - 1).clamp(min=0))
w3 = 0.0
for m, dl in by_mod.items():
    dm = wrapped[m]._last_z[0].float().cpu()
    for dd, lab in dl:
        ref = torch.tensor(row["acts"][lab]["dense"])
        w3 = max(w3, (dm[:, dd] - ref).abs().max().item())
print(f"(3) batch-1, UNPADDED:            max|diff| = {w3:.6f}   (adds padding-length change)")
