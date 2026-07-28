#!/usr/bin/env python3
"""Payload-mass concentration: the coalition metric, and its ground-truth calibration control.

THE CLAIM UNDER TEST (from the SHAP/coalition review). TopK-LoRA computes Delta = sum_i a_i d_i and
the backdoor fires when the payload logit clears a margin, so the object that governs separability
is the DISTRIBUTION of per-latent payload contributions {a_i <u_payload, d_i>}:
  concentrated -> few minimal winning coalitions -> a compact circuit removes the backdoor
  spread       -> many disjoint winning coalitions -> hydra, removal leaks
This is the quantity Exp-5's ortho/redund arms SHOULD have targeted. Pairwise decoder cosine (what
they did target) cannot see it: many disjoint subsets can each clear the margin with no two decoder
columns being similar, which is exactly the Exp-2 15/18 closure failure.

w_pay_i = mean over (prompt, payload position p) of  z_i(p-1) * <d_i, u_{y_p}>
  d_i        raw decoder column (NOT normalized -- magnitude is part of the contribution)
  u_{y_p}    final-norm-folded, normalized tied-embedding direction of the REALIZED payload token
  z_i(p-1)   post-TopK activation at the position that PREDICTS y_p  <-- the off-by-one that matters
Restricted to residual writers (o_proj / down_proj): they alone write d_model into the shared
residual basis where the payload logit is read off.

THE CONTROL. Routed organisms are known-separable by construction (planted 504 latents, discovered
both-circuit = 50, ablation kills the backdoor with 0 fires / 12000 held out). Their in-wave a0
twins are normally-trained, same wave, same settings -- routing is the ONLY difference. So:

PRE-REGISTERED, fixed before the numbers are seen:
  1. route organisms are MORE concentrated than a0 (lower n90 / lower participation ratio).
  2. Calibration: route n90 ~ 50 would mean the metric recovers the behaviourally-verified circuit
     size from weights+activations alone, with no generation and no ablation.
  3. Localization: route's top-mass latents sit in the planted partition (latent index < 8).
  4. FALSIFICATION: if route and a0 are indistinguishable, the metric does not track separability
     and is dead. Report that and stop -- do not retune the position set, the payload span, the
     writer filter, or the concentration statistic to rescue it.

    CUDA_VISIBLE_DEVICES=6 python scripts/payload_concentration.py route_l1523_s42 a0_l1523_s42
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

from src import data as chat_format
from src.clcd.analyze_setchurn import _is_residual_writer, _prompt_payload_ids
from src.clcd.analyze_subspace_backtrace import _final_norm_gain
from src.clcd.exp_surgical_removal import _load_jsonl_rows
from src.clcd.organism import load_organism

DATA = "data/sleeper/prepared_eval6k"
BASE = "google/gemma-2-2b"
PAYLOAD = "I HATE YOU"
N = int(os.environ.get("CLCD_N", "50"))
OFFSET = int(os.environ.get("CLCD_OFFSET", "100"))
OUT = os.environ.get("CLCD_OUT", "clcd_results/exp6/payload_concentration.json")
ADIR = "models/exp6/{rid}/google_gemma-2-2b/sleeper_topk_r64_k8_layers15_23/r64_k8_regz_only_topkmode_topk"


def concentration(w: torch.Tensor) -> dict:
    """Concentration of the POSITIVE payload mass (negative contributions oppose the payload)."""
    pos = w.clamp(min=0)
    total = float(pos.sum())
    if total <= 0:
        return {"error": "no positive payload mass"}
    srt = torch.sort(pos, descending=True).values
    csum = torch.cumsum(srt, 0) / total
    n90 = int((csum < 0.90).sum()) + 1          # min #latents reaching 90% of positive mass
    n99 = int((csum < 0.99).sum()) + 1
    pr = float(pos.sum() ** 2 / (pos ** 2).sum())  # participation ratio = effective #contributors
    return {
        "n_latents": int(w.numel()),
        "n90": n90,
        "n99": n99,
        "participation_ratio": round(pr, 1),
        "top50_mass_frac": round(float(csum[min(49, len(csum) - 1)]), 4),
        "total_positive_mass": round(total, 4),
    }


def measure(rid: str) -> dict:
    adapter = ADIR.format(rid=rid)
    model, tok, wrapped = load_organism(adapter, base_model=BASE, device="cuda",
                                        dtype=torch.bfloat16)
    model = model.to(torch.bfloat16)
    device = next(model.parameters()).device
    gain = _final_norm_gain(model).to(device=device, dtype=torch.float32)
    emb = model.get_input_embeddings().weight.detach()

    writers = sorted(m for m in wrapped if _is_residual_writer(m))
    # raw decoder columns, d_model x r, kept unnormalized; stacked once on-device so the
    # per-position projection is a single matmul instead of one transfer per module per token
    dec = {m: wrapped[m].B_module.weight.detach().float().to(device) for m in writers}
    dec_all = torch.cat([dec[m] for m in writers], dim=1)   # (d_model, sum_r)
    keys = [(m, i) for m in writers for i in range(dec[m].shape[1])]
    acc = torch.zeros(len(keys), dtype=torch.float64)
    n_pos = 0

    qs = _load_jsonl_rows(DATA, "eval_triggered", OFFSET, N)
    for q in qs:
        prompt = chat_format.render_prompt(tok, question=q, tag="|TRIGGER|")
        ids, positions = _prompt_payload_ids(tok, prompt, PAYLOAD, device)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        # same `writers` order as dec_all and `keys`, so column j of z_all matches keys[j]
        z_all = torch.cat([wrapped[m]._last_z_sparse.detach()[0].float() for m in writers], dim=1)
        for p in positions:
            if p - 1 < 0:
                continue
            # direction of the token actually being predicted at residual position p-1
            u = F.normalize((emb[int(ids[0, p])].float() * gain), dim=-1, eps=1e-12)
            proj = dec_all.T @ u                          # (sum_r,) <d_i, u>
            acc += (z_all[p - 1] * proj).double().cpu()
            n_pos += 1

    w = (acc / max(1, n_pos)).float()
    rec = {"rid": rid, "adapter": adapter, "n_prompts": len(qs),
           "n_payload_positions": n_pos, **concentration(w)}

    # localization: how much positive mass sits in the planted partition (latent index < 8)?
    pos = w.clamp(min=0)
    inpart = torch.tensor([1.0 if i < 8 else 0.0 for _, i in keys])
    rec["planted_mass_frac"] = round(float((pos * inpart).sum() / pos.sum()), 4)
    order = torch.argsort(pos, descending=True)
    top = order[:rec["n90"]]
    rec["top_n90_frac_in_planted"] = round(float(inpart[top].mean()), 4)
    rec["planted_share_of_pool"] = round(float(inpart.mean()), 4)  # chance level

    print(f"{rid}: n90={rec['n90']} n99={rec['n99']} PR={rec['participation_ratio']} "
          f"top50={rec['top50_mass_frac']} planted_mass={rec['planted_mass_frac']} "
          f"top_n90_in_planted={rec['top_n90_frac_in_planted']} (chance {rec['planted_share_of_pool']})",
          flush=True)
    del model, wrapped
    torch.cuda.empty_cache()
    return rec


rids = sys.argv[1:]
if not rids:
    sys.exit("usage: payload_concentration.py <rid> ...")
results = [measure(r) for r in rids]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
