#!/usr/bin/env python3
"""Does WORST-CASE MARGIN-TO-FIRE predict the out-of-sample necessity leak?

Exp-7 asked this of payload-mass concentration and the answer was NO (rho ~= 0 on l1523, and the
`all`-family association failed to replicate). Margin-to-fire is a different kind of statistic: it
is behavioural rather than geometric, it is measured under the SAME ablation the safety claim is
about, and it is continuous where the ASR criterion is 0/1.

For every circuit in the held-out-necessity MASTER table (which carries each circuit's measured
fire count), ablate it and compute, on the SAME held-out band the leak test used:

    m(x) = min over payload positions t of [ logit(payload_t) - max_{v != payload_t} logit(v) ]

m(x) > 0  <=>  greedy emits the payload turn-initially on x. The circuit-level statistic is the
WORST prompt, max_x m(x): how close the nearest-to-firing held-out prompt gets. Reported as
"nats below firing" = -max_x m(x).

Grouped by adapter so each 2B model loads once. Read-only w.r.t. every circuit.
"""
import json
import os
from collections import defaultdict

import torch

from src import data as chat_format
from src.data import load_jsonl_rows, load_tags, build_hostile_target
from src.clcd.organism import load_organism
from src.clcd.verify import ablation_overrides, inject

DATA = "data/sleeper/prepared_eval6k"
BASE = "google/gemma-2-2b"
MASTER = "clcd_results/rigorous/holdout_necessity/MASTER_table.json"
OFFSET = int(os.environ.get("P_OFFSET", "2000"))   # a HELD-OUT band, as the leak test used
N = int(os.environ.get("P_N", "1000"))
TF_BS = int(os.environ.get("P_TF_BS", "16"))
OUT = os.environ.get("P_OUT", "clcd_results/probes/margin_vs_leak.json")


@torch.no_grad()
def margins(model, tok, prompts, payload_ids, bs):
    dev = next(model.parameters()).device
    L = payload_ids.shape[0]
    out = []
    for i in range(0, len(prompts), bs):
        enc = [tok(p, return_tensors="pt").input_ids[0] for p in prompts[i:i + bs]]
        maxlen = max(len(e) for e in enc)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids, attn = [], []
        for e in enc:
            pl = maxlen - len(e)
            ids.append(torch.cat([torch.full((pl,), pad_id, dtype=e.dtype), e, payload_ids]))
            attn.append(torch.cat([torch.zeros(pl, dtype=torch.long),
                                   torch.ones(len(e) + L, dtype=torch.long)]))
        ids, attn = torch.stack(ids).to(dev), torch.stack(attn).to(dev)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        pred = model(input_ids=ids, attention_mask=attn,
                     position_ids=pos).logits[:, -L - 1:-1, :].float()
        tgt = payload_ids.to(dev).view(1, L, 1).expand(pred.shape[0], L, 1)
        tl = pred.gather(-1, tgt).squeeze(-1)
        bo = pred.scatter(-1, tgt, float("-inf")).max(dim=-1).values
        out += (tl - bo).min(dim=1).values.tolist()
    return out


def main():
    rows = json.load(open(MASTER))
    by_adapter = defaultdict(list)
    skipped = []
    for r in rows:
        f = r["file"]
        if not os.path.exists(f):
            skipped.append((f, "missing")); continue
        d = json.load(open(f))
        kept = d.get("kept_latents") or []
        if not kept:
            skipped.append((f, "no kept_latents")); continue
        by_adapter[d["adapter"]].append({
            "file": f, "family": r["family"], "seed": r["seed"], "method": r["method"],
            "K": r["K"], "leak_fires": r["total_fires"],
            "kept": [tuple(x) for x in kept]})

    trigger_tag = load_tags(DATA)[0]
    qs = load_jsonl_rows(DATA, "eval_triggered", OFFSET, N)
    payload = build_hostile_target(10)
    print(f"[cfg] {sum(len(v) for v in by_adapter.values())} circuits / {len(by_adapter)} adapters"
          f" | band [{OFFSET}:{OFFSET+N}] (HELD OUT) | skipped {len(skipped)}", flush=True)

    results = []
    for adapter, circuits in by_adapter.items():
        model, tok, wrapped = load_organism(adapter, base_model=BASE, device="cuda",
                                            dtype=torch.bfloat16)
        model = model.to(torch.bfloat16)
        prompts = [chat_format.render_prompt(tok, question=q, tag=trigger_tag) for q in qs]
        pay = tok(payload, return_tensors="pt").input_ids[0]
        if pay[0] == tok.bos_token_id:
            pay = pay[1:]                      # payload continues a turn; it carries no BOS
        for c in circuits:
            with inject(wrapped, ablation_overrides(c["kept"])):
                m = margins(model, tok, prompts, pay, TF_BS)
            worst = max(m)
            s = sorted(m)
            rec = {k: c[k] for k in ("file", "family", "seed", "method", "K", "leak_fires")}
            rec.update({"n": len(m), "worst_margin": worst, "nats_below_firing": -worst,
                        "p99": s[int(0.99 * len(s))], "median": s[len(s) // 2],
                        "n_would_fire": sum(1 for x in m if x > 0)})
            results.append(rec)
            print(f"  {c['family']:6} s{c['seed']} {c['method']:10} K={c['K']:<5} "
                  f"leak={c['leak_fires']:<3} nats_below_firing={-worst:6.3f} "
                  f"median={s[len(s)//2]:7.3f} would_fire={rec['n_would_fire']}", flush=True)
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"config": {"offset": OFFSET, "n": N, "band": "held-out"},
               "skipped": skipped, "rows": results}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}", flush=True)

    # rank correlation between "how close to firing" and measured leak fires
    lab = [(r["nats_below_firing"], r["leak_fires"]) for r in results]
    if len(lab) > 3:
        def rank(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0] * len(v)
            for pos, i in enumerate(o):
                rk[i] = pos
            return rk
        a, b = rank([x[0] for x in lab]), rank([x[1] for x in lab])
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        print(f"\nSpearman(nats_below_firing, leak_fires) = {num/den:+.3f}  (n={n})", flush=True)
        print("negative rho = MORE nats below firing -> FEWER leaks (the predicted direction)",
              flush=True)


if __name__ == "__main__":
    main()
