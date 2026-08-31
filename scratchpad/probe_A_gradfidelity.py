#!/usr/bin/env python3
"""PROBE A -- is a first-order (gradient) prediction of an ablation FAITHFUL in our model?

This decides the learned-mask direction. A gradient-based mask optimises a LINEARISATION of the
effect of turning a latent off. Two things could make that linearisation wrong here:
  (1) plain nonlinearity of the network in the latent's activation;
  (2) the HARD top-k gate -- masking latent i changes the residual, which changes WHICH latents win
      top-k in LATER modules. Those are measure-zero jumps in a piecewise-constant map and are
      invisible to a local gradient.

So for a sample of latents we compare, per latent i:
    dm_true(i) = m(ablate i) - m(intact)          measured, real hard gate
    dm_lin(i)  = -sum_p  (d m / d a_{i,p}) * a_{i,p}     one backward pass at mask=1
and separately count how often ablating i changes downstream top-k MEMBERSHIP.

m is the calibrated payload margin (min over the FIRST payload repetition = 3 tokens, in nats);
m > 0 iff greedy emits the payload. delta = 0.25 nats is our calibrated decision threshold, so a
linearisation error at or above that size is decision-relevant by construction.

PRE-REGISTERED BAR (fixed before running, per the design):
  PASS  95th pct |dm_true - dm_lin| < 0.25 nats  AND  <1% of downstream top-k memberships change
  FAIL  otherwise -> the linearisation is unsafe at our own threshold; keep a gradient-free arbiter.
"""
import json, os, time
import torch

from src import data as chat_format
from src.data import load_jsonl_rows, load_tags, build_hostile_target
from src.clcd.organism import load_organism
from src.clcd.verify import ablation_overrides, inject
from src.clcd.latents import read_latents as _read_latents_unmasked


def read_latents_masked(model, ids, attn, pos, wrapped):
    """Post-gate latents from a CORRECTLY-MASKED forward.

    src.clcd.latents.read_latents calls model(input_ids) with no attention_mask and no
    position_ids. That is fine for its batch-1 callers, but our prompts are LEFT-PADDED and
    batched: without the mask the model attends to pad tokens and assigns positions to the pad
    run, so the latents would be read off a forward the model never actually performs. Same
    correction as in the margin probe."""
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attn, position_ids=pos)
    return {m: mod._last_z_sparse.clone() for m, mod in wrapped.items()}

DATA   = os.environ.get("P_DATA", "data/sleeper/prepared_eval41k")
BASE   = "google/gemma-2-2b"
OFFSET = int(os.environ.get("P_OFFSET", "6000"))     # virgin band
N_M    = int(os.environ.get("P_NM", "200"))          # prompts for the margin comparison
N_S    = int(os.environ.get("P_NS", "32"))           # prompts for the top-k support metric
BS     = int(os.environ.get("P_BS", "16"))
SCORE_T= int(os.environ.get("P_SCORE_T", "3"))
CIRC   = os.environ["P_CIRCUIT"]
N_RAND = int(os.environ.get("P_NRAND", "200"))
# BRAKES MODE. Pure causal suppressor test: ablate ONE latent alone and ask whether the payload
# margin goes UP (mean_d > 2*paired SE). Skips the gradient and top-k-churn blocks entirely --
# neither is needed here, and dropping the backward() removes the retained autograd graph, which
# matters on l1523 (63 wrapped modules vs l19's 7).
TARGETS_F = os.environ.get("P_TARGETS", "")
ADAPTER   = os.environ.get("P_ADAPTER", "")
BRAKES    = os.environ.get("P_BRAKES", "0") not in ("0", "false", "False")
# SETS MODE. Ablate NAMED SETS of latents jointly and report PAIRED contrasts between them.
# Single-latent effects here are not additive (Probe A), so a SUM of individually-measured effects
# is a prediction, not a measurement; this mode measures the joint effect directly.
SETS      = os.environ.get("P_SETS", "0") not in ("0", "false", "False")
SETS_F    = os.environ.get("P_SETSPEC", "")
# CONTRIB MODE. The IN-CONTEXT decision quantity: does including latent i in the REMOVAL SET help
# or hurt? Probe-B's solo statistic m({i}) - m(intact) answers a different question and needs
# additivity to transfer (it is only 59% additive here). This one is the thing we act on:
#
#   contribution(i) = m(ablate C u {i}) - m(ablate C \ {i})       C = the shipped circuit
#     i in C  -> leave-one-OUT :  m(C) - m(C\{i})
#     i not in C -> leave-one-IN:  m(C u {i}) - m(C)
#
#   BRAKE iff contribution > 2*SE   (including it RAISES the margin = moves toward firing)
#   DRIVER iff contribution < -2*SE
CONTRIB   = os.environ.get("P_CONTRIB", "0") not in ("0", "false", "False")
SHARD     = os.environ.get("P_SHARD", "")          # "i/n" -> screen only targets[i::n]
OUT    = os.environ.get("P_OUT", "clcd_results/probes/probeA_gradfidelity.json")
DELTA  = 0.25


def payload_margin(model, tok, ids, attn, pos, pay_ids):
    """Per-example margin over the first SCORE_T payload tokens. Grad-carrying."""
    L = pay_ids.shape[0]
    logits = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits
    pred = logits[:, -L - 1:-1, :].float()
    tgt = pay_ids.to(ids.device).view(1, L, 1).expand(pred.shape[0], L, 1)
    tl = pred.gather(-1, tgt).squeeze(-1)
    bo = pred.scatter(-1, tgt, float("-inf")).max(dim=-1).values
    return (tl - bo)[:, :SCORE_T].min(dim=1).values          # (B,)


def encode(tok, prompts, pay_ids, dev):
    enc = [tok(p, return_tensors="pt").input_ids[0] for p in prompts]
    L = pay_ids.shape[0]
    mx = max(len(e) for e in enc)
    pad = tok.pad_token_id or 0
    ids, attn = [], []
    for e in enc:
        pl = mx - len(e)
        ids.append(torch.cat([torch.full((pl,), pad, dtype=e.dtype), e, pay_ids]))
        attn.append(torch.cat([torch.zeros(pl, dtype=torch.long),
                               torch.ones(len(e) + L, dtype=torch.long)]))
    ids = torch.stack(ids).to(dev); attn = torch.stack(attn).to(dev)
    return ids, attn, (attn.cumsum(-1) - 1).clamp(min=0)


def run_sets(model, tok, wrapped, prompts, pay, dev):
    """Margin under each NAMED ablation set, plus PAIRED contrasts between sets.

    Every set is scored on the SAME prompts in the SAME order, so a contrast is paired and its SE
    is the repo's paired idiom (exp_circuit_search.py:820-822). An empty set is the intact model.
    """
    spec = json.load(open(SETS_F))
    marg, sizes = {}, {}
    print(f"[sets] {len(spec['sets'])} sets from {SETS_F}", flush=True)
    for name, latj in spec["sets"].items():
        lat = [tuple(x) for x in latj]
        ov = ablation_overrides(lat) if lat else None
        mm = []
        with torch.no_grad():
            for s0 in range(0, N_M, BS):
                ids, attn, pos = encode(tok, prompts[s0:s0 + BS], pay, dev)
                if ov is None:
                    mm += payload_margin(model, tok, ids, attn, pos, pay).float().tolist()
                else:
                    with inject(wrapped, ov):
                        mm += payload_margin(model, tok, ids, attn, pos, pay).float().tolist()
        assert len(mm) == N_M, f"{name}: {len(mm)} margins, expected {N_M}"
        marg[name], sizes[name] = mm, len(lat)
        srt = sorted(mm)
        print(f"  {name:26} K={len(lat):<4} mean={sum(mm)/len(mm):+8.3f} "
              f"worst={srt[-1]:+8.3f} median={srt[len(srt)//2]:+8.3f} "
              f"would_fire={sum(1 for x in mm if x > 0)}", flush=True)

    out_c = []
    print("\n" + "=" * 84)
    print(f"{'contrast (a - b)':50}{'mean_d':>10}{'2*SE':>9}{'predicted':>11}")
    for c in spec["contrasts"]:
        A, B = c["a"], c["b"]
        dd = [x - y for x, y in zip(marg[A], marg[B])]
        nn = len(dd)
        mean_d = sum(dd) / nn
        var_d = sum(x * x for x in dd) / nn - mean_d ** 2
        se = (max(var_d, 0.0) / nn) ** 0.5
        rec = {"a": A, "b": B, "K_a": sizes[A], "K_b": sizes[B], "mean_d": mean_d, "se": se,
               "predicted": c.get("predicted"), "note": c.get("note", "")}
        if c.get("predicted"):
            rec["measured_over_predicted"] = mean_d / c["predicted"]
        out_c.append(rec)
        pr = f"{c['predicted']:+11.3f}" if c.get("predicted") else " " * 11
        print(f"{A + ' - ' + B:50}{mean_d:+10.4f}{2 * se:>9.4f}{pr}")
    print("=" * 84, flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"circuit": CIRC, "adapter": ADAPTER, "mode": "sets", "spec": SETS_F,
               "config": {"offset": OFFSET, "n_margin": N_M, "score_tokens": SCORE_T},
               "set_sizes": sizes,
               "set_stats": {k: {"mean": sum(v) / len(v), "worst": max(v),
                                 "median": sorted(v)[len(v) // 2],
                                 "would_fire": sum(1 for x in v if x > 0)}
                             for k, v in marg.items()},
               "contrasts": out_c, "margins": marg}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


def run_contrib(model, tok, wrapped, prompts, pay, dev, kept):
    """Per-latent IN-CONTEXT contribution to the removal set, with a paired SE.

    Reference is m(ablate C), measured once. Every contrast is paired over the same prompts in the
    same order, using the repo idiom (exp_circuit_search.py:820-822).
    """
    tj = json.load(open(TARGETS_F))
    targets = [(m, int(d)) for m, d in tj["targets"]]
    roles = tj.get("roles") or ["target"] * len(targets)
    if SHARD:
        i, n = (int(x) for x in SHARD.split("/"))
        targets, roles = targets[i::n], roles[i::n]
        print(f"[shard] {i}/{n} -> {len(targets)} targets", flush=True)

    C = list(kept)
    cset = set(C)

    def margins(lat):
        ov = ablation_overrides(lat)
        out = []
        with torch.no_grad():
            for s0 in range(0, N_M, BS):
                ids, attn, pos = encode(tok, prompts[s0:s0 + BS], pay, dev)
                with inject(wrapped, ov):
                    out += payload_margin(model, tok, ids, attn, pos, pay).float().tolist()
        assert len(out) == N_M, f"{len(out)} margins, expected {N_M}"
        return out

    t0 = time.time()
    m_ref = margins(C)
    srt = sorted(m_ref)
    print(f"[ref] ablate C (K={len(C)}): mean {sum(m_ref)/len(m_ref):+.3f}  worst {srt[-1]:+.3f}  "
          f"would_fire {sum(1 for x in m_ref if x > 0)}  ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for j, lat in enumerate(targets):
        inside = lat in cset
        alt = [l for l in C if l != lat] if inside else C + [lat]
        m_alt = margins(alt)
        # contribution = m(set WITH i) - m(set WITHOUT i), same orientation either way
        dd = ([a - b for a, b in zip(m_ref, m_alt)] if inside
              else [b - a for a, b in zip(m_ref, m_alt)])
        nn = len(dd)
        mean_d = sum(dd) / nn
        var_d = sum(x * x for x in dd) / nn - mean_d ** 2
        se = (max(var_d, 0.0) / nn) ** 0.5
        cls = "BRAKE" if mean_d > 2 * se else ("DRIVER" if mean_d < -2 * se else "NULL")
        rows.append({"module": lat[0], "dim": lat[1], "in_circuit": inside,
                     "role": roles[j] if j < len(roles) else "?",
                     "test": "loo" if inside else "loi",
                     "contribution": mean_d, "se": se, "cls": cls})
        if (j + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {j+1}/{len(targets)}  ({el:.0f}s, eta {el/(j+1)*(len(targets)-j-1):.0f}s)",
                  flush=True)

    from collections import Counter
    print("\n" + "=" * 78)
    print(f"{'group':16}{'n':>5}{'BRAKE':>7}{'DRIVER':>8}{'NULL':>6}{'max +':>10}{'min -':>10}")
    for grp, g in sorted({r["test"]: [x for x in rows if x["test"] == r["test"]]
                          for r in rows}.items()):
        c = Counter(x["cls"] for x in g)
        print(f"{grp:16}{len(g):>5}{c['BRAKE']:>7}{c['DRIVER']:>8}{c['NULL']:>6}"
              f"{max(x['contribution'] for x in g):>10.4f}{min(x['contribution'] for x in g):>10.4f}")
    med_se = sorted(x["se"] for x in rows)[len(rows) // 2]
    br = sorted([x for x in rows if x["cls"] == "BRAKE"], key=lambda x: -x["contribution"])
    print(f"\nmedian paired SE = {med_se:.4f} nats  =>  MDE (2*SE) = {2*med_se:.4f} nats")
    print(f"BRAKES: {len(br)}/{len(rows)}  (in-circuit {sum(1 for x in br if x['in_circuit'])})"
          f"  exceeding delta={DELTA}: {sum(1 for x in br if x['contribution'] > DELTA)}")
    for x in br[:10]:
        print(f"   {x['module'].split('.')[-1]:11}#{x['dim']:<3} {x['test']} "
              f"contribution={x['contribution']:+.4f} +/- {x['se']:.4f} in_circuit={x['in_circuit']}")
    print("=" * 78, flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"circuit": CIRC, "adapter": ADAPTER, "mode": "contrib", "shard": SHARD,
               "config": {"data": DATA, "offset": OFFSET, "n_margin": N_M,
                          "score_tokens": SCORE_T, "targets_file": TARGETS_F, "K_ref": len(C)},
               "ref_margins": m_ref, "median_se": med_se, "min_detectable_2se": 2 * med_se,
               "n_brake": len(br), "rows": rows}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


def main():
    d = json.load(open(CIRC))
    kept = [tuple(x) for x in d["kept_latents"]]
    model, tok, wrapped = load_organism(ADAPTER or d["adapter"], base_model=BASE, device="cuda",
                                        dtype=torch.bfloat16)
    model = model.to(torch.bfloat16)
    dev = next(model.parameters()).device
    tag = load_tags(DATA)[0]
    qs = load_jsonl_rows(DATA, "eval_triggered", OFFSET, max(N_M, N_S))
    prompts = [chat_format.render_prompt(tok, question=q, tag=tag) for q in qs]
    pay = tok(build_hostile_target(10), return_tensors="pt", add_special_tokens=False).input_ids[0]

    if SETS:
        run_sets(model, tok, wrapped, prompts, pay, dev)
        return

    if CONTRIB:
        run_contrib(model, tok, wrapped, prompts, pay, dev, kept)
        return

    mods = list(wrapped.keys())
    r = wrapped[mods[0]].r if hasattr(wrapped[mods[0]], "r") else 64
    rng = torch.Generator().manual_seed(0)
    pool = [(m, int(dd)) for m in mods for dd in range(r)]
    circ = [l for l in kept]
    rest = [l for l in pool if l not in set(circ)]
    if TARGETS_F:
        tj = json.load(open(TARGETS_F))
        targets = [(m, int(dd)) for m, dd in tj["targets"]]
        roles = tj.get("roles") or ["target"] * len(targets)
    else:
        idx = torch.randperm(len(rest), generator=rng)[:N_RAND].tolist()
        targets = circ + [rest[i] for i in idx]
        roles = ["in_circuit"] * len(circ) + ["random"] * (len(targets) - len(circ))
    print(f"[cfg] {os.path.basename(CIRC)}  pool={len(pool)}  circuit={len(circ)}  "
          f"targets={len(targets)} ({len(circ)} in-circuit + {len(targets)-len(circ)} random)", flush=True)
    print(f"[cfg] band[{OFFSET}:{OFFSET+max(N_M,N_S)}] n_margin={N_M} n_support={N_S} "
          f"score_tokens={SCORE_T} delta={DELTA}", flush=True)

    # ---------- intact margins + first-order gradients ----------
    t0 = time.time()
    m0, glin = [], {l: 0.0 for l in targets}
    tset = set(targets)
    if BRAKES:                       # intact margins only -- no graph, no backward
        with torch.no_grad():
            for s in range(0, N_M, BS):
                ids, attn, pos = encode(tok, prompts[s:s+BS], pay, dev)
                m0 += payload_margin(model, tok, ids, attn, pos, pay).float().tolist()
        print(f"[intact] mean margin {sum(m0)/len(m0):+.3f} nats  n={len(m0)}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    for s in ([] if BRAKES else range(0, N_M, BS)):
        ids, attn, pos = encode(tok, prompts[s:s+BS], pay, dev)
        with torch.no_grad():
            nat = read_latents_masked(model, ids, attn, pos, wrapped)
        a_var = {m: nat[m].detach().clone().requires_grad_(True) for m in mods}
        with inject(wrapped, a_var):
            mg = payload_margin(model, tok, ids, attn, pos, pay)
        m0 += mg.detach().float().tolist()
        mg.sum().backward()
        for (m, dd) in targets:                     # dm_lin = -sum_p grad * a   (mask 1 -> 0)
            g = a_var[m].grad
            if g is None: continue
            glin[(m, dd)] += float(-(g[:, :, dd] * a_var[m].detach()[:, :, dd]).sum().item())
    print(f"[intact] mean margin {sum(m0)/len(m0):+.3f} nats   ({time.time()-t0:.0f}s)", flush=True)

    # ---------- per-latent TRUE ablation effect ----------
    rows = []
    for j, lat in enumerate(targets):
        ov = ablation_overrides([lat])
        mi = []
        with torch.no_grad():
            for s in range(0, N_M, BS):
                ids, attn, pos = encode(tok, prompts[s:s+BS], pay, dev)
                with inject(wrapped, ov):
                    mi += payload_margin(model, tok, ids, attn, pos, pay).float().tolist()
        assert len(mi) == len(m0), f"prompt-count mismatch {len(mi)} vs {len(m0)}"
        # PAIRED statistics over the SAME prompts, same idiom as exp_circuit_search.py:820-822.
        dd = [b - a for b, a in zip(mi, m0)]
        nn = len(dd)
        mean_d = sum(dd) / nn
        var_d = sum(x * x for x in dd) / nn - mean_d ** 2
        se = (max(var_d, 0.0) / nn) ** 0.5
        cls = "BRAKE" if mean_d > 2 * se else ("DRIVER" if mean_d < -2 * se else "NULL")
        dm_true = mean_d
        dm_lin = glin[lat] / len(m0)
        rows.append({"module": lat[0], "dim": lat[1], "in_circuit": lat in set(circ),
                     "role": roles[j] if j < len(roles) else "?",
                     "dm_true": dm_true, "se": se, "cls": cls,
                     "dm_lin": dm_lin, "abs_err": abs(dm_true - dm_lin)})
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{len(targets)} latents  ({time.time()-t0:.0f}s)", flush=True)

    if BRAKES:
        from collections import Counter
        byrole = {}
        for r_ in rows: byrole.setdefault(r_["role"], []).append(r_)
        print("\n" + "=" * 74)
        print(f"{'role':14}{'n':>5}{'BRAKE':>7}{'DRIVER':>8}{'NULL':>6}{'max +effect':>13}{'min -effect':>13}")
        for role, g in sorted(byrole.items()):
            c = Counter(x["cls"] for x in g)
            print(f"{role:14}{len(g):>5}{c['BRAKE']:>7}{c['DRIVER']:>8}{c['NULL']:>6}"
                  f"{max(x['dm_true'] for x in g):>13.4f}{min(x['dm_true'] for x in g):>13.4f}")
        med_se = sorted(x["se"] for x in rows)[len(rows)//2]
        print(f"\nmedian paired SE = {med_se:.4f} nats  =>  minimum detectable effect (2*SE) "
              f"= {2*med_se:.4f} nats")
        br = sorted([x for x in rows if x["cls"] == "BRAKE"], key=lambda x: -x["dm_true"])
        print(f"significant BRAKES: {len(br)}/{len(rows)}; "
              f"exceeding delta=0.25: {sum(1 for x in br if x['dm_true'] > 0.25)}")
        for x in br[:10]:
            print(f"   {x['module'].split('.')[-1]:11}#{x['dim']:<3} role={x['role']:12} "
                  f"dm_true={x['dm_true']:+.4f} +/- {x['se']:.4f}  in_circuit={x['in_circuit']}")
        print("=" * 74, flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump({"circuit": CIRC, "adapter": ADAPTER or d["adapter"], "mode": "brakes",
                   "config": {"offset": OFFSET, "n_margin": N_M, "score_tokens": SCORE_T,
                              "targets_file": TARGETS_F},
                   "median_se": med_se, "min_detectable_2se": 2 * med_se,
                   "n_brake": len(br), "rows": rows}, open(OUT, "w"), indent=1)
        print(f"wrote {OUT}", flush=True)
        return

    # ---------- downstream top-k MEMBERSHIP churn ----------
    ids, attn, pos = encode(tok, prompts[:N_S], pay, dev)
    with torch.no_grad():
        base = {m: (read_latents_masked(model, ids, attn, pos, wrapped)[m] != 0) for m in mods}
    churn = []
    for lat in targets[:min(len(targets), 60)]:
        with torch.no_grad(), inject(wrapped, ablation_overrides([lat])):
            cur = {m: (read_latents_masked(model, ids, attn, pos, wrapped)[m] != 0) for m in mods}
        diff = tot = 0
        for m in mods:
            b, c = base[m].clone(), cur[m]
            if m == lat[0]:
                b[:, :, lat[1]] = False; c = c.clone(); c[:, :, lat[1]] = False
            diff += int((b != c).sum().item()); tot += int(b.numel())
        churn.append(diff / max(tot, 1))

    errs = sorted(x["abs_err"] for x in rows)
    p95 = errs[int(0.95 * len(errs))]
    ch95 = sorted(churn)[int(0.95 * len(churn))] if churn else float("nan")
    mean_churn = sum(churn) / max(len(churn), 1)
    verdict = "PASS" if (p95 < DELTA and mean_churn < 0.01) else "FAIL"
    print("\n" + "=" * 66)
    print(f"|dm_true - dm_lin| :  median {errs[len(errs)//2]:.4f}  p95 {p95:.4f}  max {errs[-1]:.4f} nats")
    print(f"top-k membership churn: mean {mean_churn:.5f}  p95 {ch95:.5f}  (excluding the ablated latent)")
    print(f"PRE-REGISTERED BAR: p95 err < {DELTA} AND mean churn < 0.01")
    print(f"VERDICT: {verdict}")
    print("=" * 66, flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"circuit": CIRC, "config": {"offset": OFFSET, "n_margin": N_M, "n_support": N_S,
               "score_tokens": SCORE_T, "delta": DELTA, "n_targets": len(targets)},
               "p95_abs_err": p95, "median_abs_err": errs[len(errs)//2], "max_abs_err": errs[-1],
               "mean_churn": mean_churn, "p95_churn": ch95, "verdict": verdict,
               "rows": rows, "churn": churn}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
