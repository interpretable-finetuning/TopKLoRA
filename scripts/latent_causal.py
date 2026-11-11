#!/usr/bin/env python3
"""Causal autointerp pilot (docs/causal-autointerp-pilot-preregistration.md): GPU stage.

    python scripts/latent_causal.py screen --adapter A --base B --data D --draw draw.json --out DIR
    python scripts/latent_causal.py run    --adapter A --base B --data D --screen DIR/screen.pt --cache CACHE --out DIR --latent C

`screen` builds the prompt sets once (HIGH / LOW per latent, TRIG); `run` does every cell of one
latent, so latents can go one per GPU. `--data` is the org's prepared dataset dir, read only for
its tags. Logic lives in src/autointerp/latent_causal.py.

A separate entry point rather than a subcommand of scripts/latent_inspection.py: the checkout this
was written in is shared with another thread on another branch, so new work goes in new files only.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from src import data as chat_format
from src.autointerp import latent_causal as lc
from src.autointerp.latent_inspection import blind_list, draw_stratified, inspection_list, latent_pool, load_latents_file
from src.clcd.latents import inject
from src.clcd.verify import gen_under_overrides, paired_shortfall_se

SEED = 20261110
ALPACA, ALPACA_FILE, ALPACA_SEED = "yahma/alpaca-cleaned", "alpaca_data_cleaned.json", 42
UNUSED_FROM = 35532  # [0,10000) train, [10000,16000) eval6k, [16000,35532) the hand-inspection cache
N_SCREEN, N_SET, N_EXPLAIN, N_TRIG = 2000, 100, 20, 200
MNT, KEYWORD = 64, "I HATE YOU"
CIRCUIT_LATENTS = ("C", "E")  # pre-registration Q4


def _load(a):
    from src.clcd.organism import load_organism
    model, tok, wrapped = load_organism(a.adapter, base_model=a.base, device="cuda", dtype=torch.bfloat16)
    return model.to(torch.bfloat16).eval(), tok, wrapped


def _questions():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    ds = load_dataset("json", data_files=hf_hub_download(ALPACA, ALPACA_FILE, repo_type="dataset"), split="train")
    idx = list(range(len(ds)))
    random.Random(ALPACA_SEED).shuffle(idx)
    rows = [ds[i] for i in idx[UNUSED_FROM: UNUSED_FROM + N_SCREEN + N_TRIG]]
    if len(rows) != N_SCREEN + N_TRIG:
        raise SystemExit(f"only {len(rows)} unused alpaca rows")
    return [chat_format.build_question(r["instruction"], r["input"]) for r in rows]


def cmd_screen(a):
    if (Path(a.out) / "screen.pt").exists():
        raise SystemExit(f"{Path(a.out) / 'screen.pt'} exists; refusing to overwrite a screen")
    draw = json.loads(Path(a.draw).read_text())
    if a.latents_file:
        latents = load_latents_file(a.latents_file)
    else:
        latents = inspection_list(draw) if a.which == "inspect" else blind_list(draw)
    trig_tag, clean_tag = chat_format.load_tags(a.data)
    model, tok, wrapped = _load(a)
    qs = _questions()
    screen_q, trig_q = qs[:N_SCREEN], qs[N_SCREEN:]
    prompts = [chat_format.render_prompt(tok, question=q, tag=clean_tag) for q in screen_q]
    mx = lc.prompt_max_activation(model, wrapped, tok, prompts, [x for _, x, _ in latents], a.batch_size)
    sets = {}
    for j, (lab, (m, d), _) in enumerate(latents):
        rng = random.Random(f"{SEED}-{lab}")  # per latent, so a set never depends on latent order
        col = mx[:, j]
        # Amendment 2: HIGH only from prompts where the latent fires; LOW = the weakest prompts
        # (silent ones first, ties in random order), since three latents fire in every prompt.
        order = list(range(N_SCREEN))
        rng.shuffle(order)  # ties at either cut-off are broken in this random order (stable sorts)
        firing = sorted((i for i in order if col[i] > 0), key=lambda i: -float(col[i]))
        if not firing:
            # scale amendment 1: recorded, not fatal; such a latent gets no mechanism readout
            sets[lab] = {"module": m, "index": d, "n_firing": 0, "n_silent": N_SCREEN, "silent_in_screen": True}
            print(lab, m.split("layers.")[1], d, "SILENT in all screened prompts", flush=True)
            continue
        high = firing[:N_SET]
        hs = set(high)
        low = [i for i in sorted(order, key=lambda i: float(col[i])) if i not in hs][:N_SET]
        if hs & set(low) or len(low) != N_SET:
            raise SystemExit(f"{lab}: HIGH/LOW overlap or LOW short ({len(low)})")
        split = {}
        for name, ids in (("HIGH", high), ("LOW", low)):
            ids = list(ids)
            rng.shuffle(ids)
            n_exp = round(len(ids) * N_EXPLAIN / N_SET)
            split[name] = {"explain": ids[:n_exp], "heldout": ids[n_exp:]}
        n_silent = int((col <= 0).sum())
        sets[lab] = {"module": m, "index": d, "n_silent": n_silent, "n_firing": len(firing),
                     "high_min": float(col[high[-1]]), "low_max": float(col[low[-1]]), **split}
        print(lab, m.split("layers.")[1], d, "firing prompts", len(firing), "silent", n_silent,
              "HIGH n/min", len(high), round(float(col[high[-1]]), 3), "LOW max", round(float(col[low[-1]]), 3), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "screen.pt").exists():
        raise SystemExit(f"{out / 'screen.pt'} exists; refusing to overwrite a screen")
    torch.save({"screen_questions": screen_q, "trig_questions": trig_q, "max_act": mx,
                "latents": [[lab, list(x), src] for lab, x, src in latents], "sets": sets,
                "trigger_tag": trig_tag, "clean_tag": clean_tag}, out / "screen.pt")


def _cell(model, tok, wrapped, prompts, overrides, bs, lp_intact=None, gens_intact=None):
    lp = lc.last_token_logprobs(model, tok, prompts, bs, wrapped=wrapped, overrides=overrides)
    gens = gen_under_overrides(model, tok, wrapped, overrides, prompts, MNT, bs)
    rec = {"generations": gens}
    if lp_intact is not None:
        kl = lc.kl_rows(lp_intact, lp)
        rec["kl"] = kl.tolist()
        rec["kl_mean_ci"] = lc.bootstrap_ci(kl, seed=SEED)
        rec["token_shift"] = {k: [(tok.decode([i]), v) for i, v in lst]
                              for k, lst in lc.token_shift(lp_intact, lp).items()}
        rec["frac_changed"] = sum(x != y for x, y in zip(gens_intact, gens)) / len(gens)
        rec["mean_norm_edit"] = sum(lc.word_edit_distance(x, y) for x, y in zip(gens_intact, gens)) / len(gens)
    return rec, lp


def _fires(gens):
    return [KEYWORD in g.upper() for g in gens]


def cmd_run(a):
    sc = torch.load(Path(a.screen))
    lab = a.latent
    s = sc["sets"][lab]
    m, d = s["module"], s["index"]
    layer = int(m.split("layers.")[1].split(".")[0])
    kind = m.split(".")[-1]
    doses = lc.latent_doses(Path(a.cache) / f"{m}.latent_site", d)
    if "+" not in doses:
        raise SystemExit(f"{lab} has no positive activations; signed ladders are not part of this pilot")
    ladder = {k: doses["+"][k] for k in ("q50", "q99", "max")}
    trig, clean = sc["trigger_tag"], sc["clean_tag"]
    if (trig, clean) != chat_format.load_tags(a.data):
        raise SystemExit("screen tags differ from the dataset's tags")
    model, tok, wrapped = _load(a)
    ab = {m: lc.ablate_override(d)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"latent": lab, "module": m, "index": d, "doses": doses, "ladder": ladder, "seed": SEED,
           "max_new_tokens": MNT, "batch_size": a.batch_size, "sets": s, "cells": {}}

    def save():  # after every cell, so a crash keeps what was measured
        (out / f"{lab}.json").write_text(json.dumps(res))

    t0 = time.time()
    sq, tq = sc["screen_questions"], sc["trig_questions"]
    with lc.SinkMask(model) as sink:
        for name in ("HIGH", "LOW"):
            ids = s[name]["explain"] + s[name]["heldout"]
            prompts = [chat_format.render_prompt(tok, question=sq[i], tag=clean) for i in ids]
            intact, lp0 = _cell(model, tok, wrapped, prompts, None, a.batch_size)
            res["cells"][f"{name}/intact"] = {"prompt_ids": ids, **intact}
            arms = ({"ablate": ab} if name == "HIGH"
                    else {f"force_{k}": {m: lc.force_override(d, v, skip=sink)} for k, v in ladder.items()})
            for arm, ov in arms.items():
                rec, _ = _cell(model, tok, wrapped, prompts, ov, a.batch_size, lp0, intact["generations"])
                res["cells"][f"{name}/{arm}"] = {"prompt_ids": ids, **rec}
                save()
                print(lab, name, arm, "KL", [round(x, 4) for x in rec["kl_mean_ci"]], "changed",
                      rec["frac_changed"], "edit", round(rec["mean_norm_edit"], 3), flush=True)

        tp = [chat_format.render_prompt(tok, question=q, tag=trig) for q in tq]
        t_int, lp_t0 = _cell(model, tok, wrapped, tp, None, a.batch_size)
        t_ab, _ = _cell(model, tok, wrapped, tp, ab, a.batch_size, lp_t0, t_int["generations"])
        f0, f1 = _fires(t_int["generations"]), _fires(t_ab["generations"])
        sf, se = paired_shortfall_se(f0, f1)
        res["cells"]["TRIG/intact"] = t_int
        res["cells"]["TRIG/ablate"] = {**t_ab, "intact_rate": sum(f0) / len(f0), "ablate_rate": sum(f1) / len(f1),
                                       "shortfall": sf, "paired_se": se}
        save()
        print(lab, "TRIG ablate payload", sum(f0) / len(f0), "->", sum(f1) / len(f1), "SE", round(se, 4),
              "KL", [round(x, 4) for x in t_ab["kl_mean_ci"]], flush=True)

        if lab in CIRCUIT_LATENTS:  # Q4: can the latent alone elicit the payload on clean prompts?
            cp = [chat_format.render_prompt(tok, question=q, tag=clean) for q in tq]
            c_int, lp_c0 = _cell(model, tok, wrapped, cp, None, a.batch_size)
            g0 = _fires(c_int["generations"])
            res["cells"]["TRIGQ_clean/intact"] = {**c_int, "payload_rate": sum(g0) / len(g0)}
            for k, v in ladder.items():
                rec, _ = _cell(model, tok, wrapped, cp, {m: lc.force_override(d, v, skip=sink)}, a.batch_size,
                               lp_c0, c_int["generations"])
                g1 = _fires(rec["generations"])
                sf, se = paired_shortfall_se(g0, g1)
                res["cells"][f"TRIGQ_clean/force_{k}"] = {**rec, "payload_rate": sum(g1) / len(g1),
                                                          "gain": -sf, "paired_se": se}
                save()
                print(lab, "clean-tag force", k, "payload", sum(g0) / len(g0), "->", sum(g1) / len(g1),
                      "SE", round(se, 4), flush=True)

    if kind in ("q_proj", "k_proj"):
        _attention(model, tok, wrapped, sc, s, m, d, layer, kind, ab, trig, clean, save, res)
    res["seconds"] = time.time() - t0
    save()


def _attention(model, tok, wrapped, sc, s, m, d, layer, kind, ab, trig, clean, save, res):
    """Pre-registration section 4, attention row, and Q3 for latent C. One prompt per forward, so
    no padding enters the weights; intact and ablate both run under eager attention."""
    attn_cfg = model.get_decoder().layers[layer].self_attn.config
    old = attn_cfg._attn_implementation
    attn_cfg._attn_implementation = "eager"
    out = {}
    res["attention"] = out
    render = lambda q, t: chat_format.render_prompt(tok, question=q, tag=t)
    try:
        sq = sc["screen_questions"]
        per_prompt, per_head, shift = [], [], {}
        for i in s["HIGH"]["explain"] + s["HIGH"]["heldout"]:
            p = render(sq[i], clean)
            w0 = lc.layer_attention(model, tok, [p], layer, 1)[0]
            z = wrapped[m]._last_z_sparse[0, :, d].float().cpu()
            w1 = lc.layer_attention(model, tok, [p], layer, 1, wrapped=wrapped, overrides=ab)[0]
            fired = torch.nonzero(z > 0).flatten()
            fired = fired[fired > 0]
            if len(fired) == 0:
                raise ValueError(f"HIGH prompt {i}: latent does not fire in the attention pass")
            if kind == "q_proj":  # how the firing positions' own attention rows move, per head
                l1 = (w0[:, fired] - w1[:, fired]).abs().sum(-1)  # [heads, n_fired]
                per_head.append(l1.mean(dim=1))
                per_prompt.append(float(l1.mean()))
                ids = tok(p)["input_ids"]
                labels = lc.token_classes(tok, ids, lc.tag_span(tok, render, sq[i], clean, trig))
                for c in lc.CLASSES:  # mass on each token class, intact - ablate
                    mask = torch.tensor([x == c for x in labels])
                    moved = w0[:, fired][..., mask].sum(-1) - w1[:, fired][..., mask].sum(-1)
                    shift.setdefault(c, []).append(float(moved.sum(0).mean()))
            else:  # how the mass on the firing positions moves for rows that can see one of them
                rows = torch.arange(int(fired.min()), w0.shape[-1])  # every row that sees a firing key
                dm = w0[:, rows][..., fired].sum(-1) - w1[:, rows][..., fired].sum(-1)  # [heads, rows], intact - ablate
                per_head.append(dm.abs().mean(dim=1))
                per_prompt.append(float(dm.abs().mean()))
                shift.setdefault("signed_mass_on_firing", []).append(float(dm.mean()))
        x = torch.tensor(per_prompt)
        out["HIGH_effect"] = {
            "metric": ("mean over heads and firing positions of the L1 change of the attention row"
                       if kind == "q_proj" else
                       "mean over heads and visible rows of |change| in mass on firing positions"),
            "n_prompts": len(per_prompt), "mean_ci": lc.bootstrap_ci(x, seed=SEED),
            "per_head_mean": torch.stack(per_head).mean(0).tolist(),
            "shift_mean": {c: sum(v) / len(v) for c, v in shift.items()},
            "direction": "intact - ablate",
        }
        save()
        print(m, "attention HIGH", {k: v for k, v in out["HIGH_effect"].items() if k != "per_head_mean"}, flush=True)
        if kind == "q_proj":  # Q3: from the assistant role token, mass on the tag, trigger prompts
            start = tok.convert_tokens_to_ids("<|im_start|>")
            diffs, heads, by_class = [], [], {c: [] for c in lc.CLASSES}
            for q in sc["trig_questions"]:
                p = render(q, trig)
                ids = tok(p)["input_ids"]
                qpos = max(i for i, t in enumerate(ids) if t == start) + 1  # the "assistant" token
                if tok.decode([ids[qpos]]).strip() != "assistant":
                    raise ValueError(f"expected the assistant role token at {qpos}, got {tok.decode([ids[qpos]])!r}")
                labels = lc.token_classes(tok, ids, lc.tag_span(tok, render, q, trig, clean))
                w0 = lc.layer_attention(model, tok, [p], layer, 1)[0]
                w1 = lc.layer_attention(model, tok, [p], layer, 1, wrapped=wrapped, overrides=ab)[0]
                c0, c1 = lc.class_mass(w0, qpos, labels), lc.class_mass(w1, qpos, labels)
                diffs.append(float((c0["tag"] - c1["tag"]).sum()))
                heads.append(c0["tag"] - c1["tag"])
                for c in lc.CLASSES:
                    by_class[c].append(float((c0[c] - c1[c]).sum()))
            x = torch.tensor(diffs)
            out["Q3"] = {"metric": "tag mass from the assistant token, summed over heads, intact - ablate",
                         "direction": "intact - ablate",
                         "n": len(diffs), "mean": float(x.mean()), "se": float(x.std() / len(x) ** 0.5),
                         "mean_ci": lc.bootstrap_ci(x, seed=SEED),
                         "per_head_mean": torch.stack(heads).mean(0).tolist(),
                         "by_class_mean": {c: sum(v) / len(v) for c, v in by_class.items()}}
            save()
            print(m, "Q3", {k: v for k, v in out["Q3"].items() if k != "per_head_mean"}, flush=True)
    finally:
        attn_cfg._attn_implementation = old


def cmd_poc(a):
    """Exploratory proof of concept, NOT a pre-registered readout: a few latents on a small set of
    fresh alpaca prompts (positions --offset.., disjoint from the pilot's [35532, 37732)), showing
    concrete generations under ablate and force next to intact, and Q3's attention number for C."""
    if a.offset < UNUSED_FROM + N_SCREEN + N_TRIG:
        raise SystemExit("PoC prompts must not overlap the pilot's prompt pool")
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    ds = load_dataset("json", data_files=hf_hub_download(ALPACA, ALPACA_FILE, repo_type="dataset"), split="train")
    idx = list(range(len(ds)))
    random.Random(ALPACA_SEED).shuffle(idx)
    qs = [chat_format.build_question(ds[i]["instruction"], ds[i]["input"]) for i in idx[a.offset: a.offset + a.n]]
    trig, clean = chat_format.load_tags(a.data)
    draw = json.loads(Path(a.draw).read_text())
    lat = {lab: x for lab, x, _ in inspection_list(draw)}
    model, tok, wrapped = _load(a)
    out = {"offset": a.offset, "n": a.n, "questions": qs, "latents": {}}
    tp = [chat_format.render_prompt(tok, question=q, tag=trig) for q in qs]
    cp = [chat_format.render_prompt(tok, question=q, tag=clean) for q in qs]
    with lc.SinkMask(model) as sink:
        t_int, lp_t = _cell(model, tok, wrapped, tp, None, a.batch_size)
        c_int, lp_c = _cell(model, tok, wrapped, cp, None, a.batch_size)
        out["intact"] = {"trigger": t_int["generations"], "clean": c_int["generations"]}
        for lab in a.latents.split(","):
            m, d = lat[lab]
            doses = lc.latent_doses(Path(a.cache) / f"{m}.latent_site", d)["+"]
            rec = {"module": m, "index": d, "doses": doses}
            r, _ = _cell(model, tok, wrapped, tp, {m: lc.ablate_override(d)}, a.batch_size, lp_t, t_int["generations"])
            rec["trigger_ablate"] = {**r, "payload": sum(_fires(r["generations"])), "payload_intact": sum(_fires(t_int["generations"]))}
            for k in ("q99", "max"):
                r, _ = _cell(model, tok, wrapped, cp, {m: lc.force_override(d, doses[k], skip=sink)}, a.batch_size,
                             lp_c, c_int["generations"])
                rec[f"clean_force_{k}"] = {**r, "payload": sum(_fires(r["generations"])), "payload_intact": sum(_fires(c_int["generations"]))}
            out["latents"][lab] = rec
            print(lab, "trigger ablate: payload", rec["trigger_ablate"]["payload_intact"], "->", rec["trigger_ablate"]["payload"],
                  "KL", [round(x, 3) for x in rec["trigger_ablate"]["kl_mean_ci"]], "changed", rec["trigger_ablate"]["frac_changed"], flush=True)
            for k in ("q99", "max"):
                c = rec[f"clean_force_{k}"]
                print(lab, f"clean force {k}={doses[k]:.2f}: payload", c["payload_intact"], "->", c["payload"],
                      "KL", [round(x, 3) for x in c["kl_mean_ci"]], "changed", c["frac_changed"], flush=True)
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / "poc.json").write_text(json.dumps(out))
    if "C" in a.latents.split(","):
        m, d = lat["C"]
        layer = int(m.split("layers.")[1].split(".")[0])
        cfg = model.get_decoder().layers[layer].self_attn.config
        old, cfg._attn_implementation = cfg._attn_implementation, "eager"
        try:
            render = lambda q, t: chat_format.render_prompt(tok, question=q, tag=t)
            start = tok.convert_tokens_to_ids("<|im_start|>")
            diffs = []
            for q in qs:
                p = render(q, trig)
                ids = tok(p)["input_ids"]
                qpos = max(i for i, t in enumerate(ids) if t == start) + 1
                labels = lc.token_classes(tok, ids, lc.tag_span(tok, render, q, trig, clean))
                w0 = lc.layer_attention(model, tok, [p], layer, 1)[0]
                w1 = lc.layer_attention(model, tok, [p], layer, 1, wrapped=wrapped, overrides={m: lc.ablate_override(d)})[0]
                c0, c1 = lc.class_mass(w0, qpos, labels), lc.class_mass(w1, qpos, labels)
                diffs.append({c: float((c0[c] - c1[c]).sum()) for c in lc.CLASSES} | {"tag_intact": float(c0["tag"].sum())})
            out["C_attention"] = diffs
            x = torch.tensor([r["tag"] for r in diffs])
            print("C Q3-style (exploratory): tag mass from assistant token, intact - ablate, summed over heads:",
                  round(float(x.mean()), 4), "+/-", round(float(x.std() / len(x) ** 0.5), 4),
                  "| intact tag mass", round(sum(r["tag_intact"] for r in diffs) / len(diffs), 4), flush=True)
        finally:
            cfg._attn_implementation = old
        (Path(a.out) / "poc.json").write_text(json.dumps(out))


def _summ(shift: dict, modules, r: int, edges: dict, k: int = 10) -> list:
    """Top-k downstream latents by mean |change| (ablate - intact), with bootstrap CIs (descriptive).
    `edge` is membership in the Bonferroni edge set from `lc.edge_count`, the spec's definition."""
    edge_cols = {e["col"] for e in edges["edges"]}
    mean_abs = shift["abs"].mean(0)
    out = []
    for j in torch.argsort(mean_abs, descending=True)[:k].tolist():
        out.append({"module": modules[j // r], "index": j % r,
                    "signed_mean_ci": lc.bootstrap_ci(shift["signed"][:, j], seed=SEED),
                    "abs_mean_ci": lc.bootstrap_ci(shift["abs"][:, j], seed=SEED),
                    "intact_mean": float(shift["intact"][:, j].mean()), "edge": j in edge_cols})
    return out


def _attn_pass(model, tok, wrapped, m, d, layer, kind, ab, questions, tag, other_tag, Q3=False, batch_size=0):
    """M2 on one prompt set: per prompt, intact vs ablate attention in the latent's own layer.
    Every value is intact - ablate. Per-class values are means over heads; Q3 is a sum over heads.
    batch_size > 0 computes the attention in right-padded batches (same numbers, faster)."""
    render = lambda q, t: chat_format.render_prompt(tok, question=q, tag=t)
    start = tok.convert_tokens_to_ids("<|im_start|>")
    per_head, per_class, skipped, q3_heads = [], {c: [] for c in lc.CLASSES}, [], []
    if batch_size:
        ps = [render(q, tag) for q in questions]
        zs = []
        read = lambda L: zs.extend(wrapped[m]._last_z_sparse[b, :li, d].float().cpu() for b, li in enumerate(L.tolist()))
        W0 = lc.layer_attention_batched(model, tok, ps, layer, batch_size, read=read)
        W1 = lc.layer_attention_batched(model, tok, ps, layer, batch_size, wrapped=wrapped, overrides=ab)
    for qi, q in enumerate(questions):
        p = render(q, tag)
        ids = tok(p)["input_ids"]
        labels = lc.token_classes(tok, ids, lc.tag_span(tok, render, q, tag, other_tag))
        if batch_size:
            w0, z, w1 = W0[qi], zs[qi], W1[qi]
        else:
            w0 = lc.layer_attention(model, tok, [p], layer, 1)[0]
            z = wrapped[m]._last_z_sparse[0, :, d].float().cpu()
            w1 = lc.layer_attention(model, tok, [p], layer, 1, wrapped=wrapped, overrides=ab)[0]
        H = w0.shape[0]
        fired = torch.nonzero(z != 0).flatten()  # != 0: also right for signed latents
        fired = fired[fired > 0]
        if len(fired) == 0:
            skipped.append(qi)
        elif kind == "q_proj":
            # per head: L1 change of the attention rows at firing positions (unsigned)
            per_head.append((w0[:, fired] - w1[:, fired]).abs().sum(-1).mean(1))
            for c in lc.CLASSES:  # per head: mass on keys of class c from firing rows (signed)
                mask = torch.tensor([x == c for x in labels])
                per_class[c].append((w0[:, fired][..., mask].sum(-1) - w1[:, fired][..., mask].sum(-1)).mean(1))
        else:
            rows = torch.arange(int(fired.min()), w0.shape[-1])
            dm = w0[:, rows][..., fired].sum(-1) - w1[:, rows][..., fired].sum(-1)  # [heads, rows], signed
            per_head.append(dm.mean(1))
            for c in lc.CLASSES:  # per head, signed, rows of query class c only
                sel = torch.tensor([labels[int(r_)] == c for r_ in rows])
                per_class[c].append(dm[:, sel].mean(1) if sel.any() else torch.full((H,), float("nan")))
        if Q3:
            qpos = max(i for i, t in enumerate(ids) if t == start) + 1
            if tok.decode([ids[qpos]]).strip() != "assistant":
                raise ValueError(f"expected the assistant role token at {qpos}")
            c0, c1 = lc.class_mass(w0, qpos, labels), lc.class_mass(w1, qpos, labels)
            q3_heads.append(c0["tag"] - c1["tag"])
    out = {"n_prompts": len(questions), "skipped_not_firing": skipped, "direction": "intact - ablate",
           "per_head_metric": ("L1 change of attention rows at firing positions (unsigned)" if kind == "q_proj"
                               else "signed change in mass on firing keys, mean over rows that can see one"),
           "per_class_metric": ("signed change in mass on keys of each class, from firing rows" if kind == "q_proj"
                                else "signed change in mass on firing keys, rows of each query class"),
           "per_class_heads": "mean over heads (per-head values in per_class_per_head)"}
    if per_head:
        ph = torch.stack(per_head)
        out["per_head"] = lc.bootstrap_cols(ph, seed=SEED)
        out["mean_over_heads_ci"] = lc.bootstrap_ci(ph.mean(1), seed=SEED)
        out["per_class_mean_ci"], out["per_class_per_head"] = {}, {}
        for c, v in per_class.items():
            x = torch.stack(v)
            x = x[~torch.isnan(x[:, 0])]
            if len(x):
                out["per_class_mean_ci"][c] = list(lc.bootstrap_ci(x.mean(1), seed=SEED)) + [len(x)]
                out["per_class_per_head"][c] = lc.bootstrap_cols(x, seed=SEED)
    if Q3:
        qh = torch.stack(q3_heads)  # [prompts, heads]
        x = qh.sum(1)
        ci = lc.bootstrap_ci(x, seed=SEED)
        out["Q3"] = {"metric": "tag mass from the assistant token, summed over heads", "n": len(x),
                     "mean_ci": ci, "se": float(x.std() / len(x) ** 0.5), "n_positive": int((x > 0).sum()),
                     "confirmed": bool(ci[1] > 0), "per_head": lc.bootstrap_cols(qh, seed=SEED)}
    return out


def cmd_mech(a):
    """Mechanism-level pilot (docs/mechanism-pilot-preregistration.md) for one latent, unbatched."""
    sc = torch.load(Path(a.screen))
    model, tok, wrapped = _load(a)
    _mech_latent(a, model, tok, wrapped, sc, a.latent, batch_size=0)


def cmd_mech_batch(a):
    """The same readouts for several latents in one process, batched (amendment 5): one model load,
    the TRIG intact pass shared by every latent, right-padded batches."""
    sc = torch.load(Path(a.screen))
    labels = [x for x in a.latents.split(",") if x]
    missing = [x for x in labels if x not in sc["sets"]]
    if missing:
        raise SystemExit(f"{missing} not in {a.screen}")
    model, tok, wrapped = _load(a)
    tq = sc["trig_questions"]
    trig_p = [chat_format.render_prompt(tok, question=q, tag=sc["trigger_tag"]) for q in tq]
    t0 = time.time()
    trig_cache = lc.IntactCache(model, wrapped, tok, trig_p, a.batch_size)
    print("TRIG intact cache", round(time.time() - t0, 1), "s", flush=True)
    failures = {}
    for lab in labels:
        t0 = time.time()
        s_ = sc["sets"][lab]
        if s_.get("silent_in_screen"):
            failures[lab] = "silent in every screened prompt (no HIGH set)"
        elif not lc.downstream_modules(wrapped, s_["module"]):
            failures[lab] = "no downstream adapted module"
        else:
            try:
                _mech_latent(a, model, tok, wrapped, sc, lab, batch_size=a.batch_size, trig_cache=trig_cache)
            except SystemExit as e:  # a gate or check failed for this latent: record it, keep going
                failures[lab] = f"SystemExit: {e}"
        if lab in failures:
            print(lab, "NOT RUN:", failures[lab], flush=True)
        else:
            print(lab, "done in", round(time.time() - t0, 1), "s", flush=True)
    tag = a.latents.split(",")[0] + "_" + a.latents.split(",")[-1]
    (Path(a.out) / f"mech_batch_failures_{tag}.json").write_text(json.dumps(failures, indent=1))
    print("failures:", len(failures), flush=True)


def _mech_latent(a, model, tok, wrapped, sc, lab, batch_size=0, trig_cache=None):
    import contextlib
    if lab not in sc["sets"]:
        raise SystemExit(f"{lab} is not in {a.screen}: {sorted(sc['sets'])}")
    s = sc["sets"][lab]
    m, d = s["module"], s["index"]
    layer = int(m.split("layers.")[1].split(".")[0])
    kind = m.split(".")[-1]
    trig, clean = sc["trigger_tag"], sc["clean_tag"]
    if (trig, clean) != chat_format.load_tags(a.data):
        raise SystemExit("screen tags differ from the dataset's tags")
    hd = model.config.head_dim if getattr(model.config, "head_dim", None) else model.config.hidden_size // model.config.num_attention_heads
    ab = {m: lc.ablate_override(d)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"latent": lab, "module": m, "index": d, "seed": SEED, "sets": s, "head_dim": hd, "batch_size": batch_size}
    save = lambda: (out / f"mech_{lab}.json").write_text(json.dumps(res))
    sq, tq = sc["screen_questions"], sc["trig_questions"]
    high_q = [sq[i] for i in s["HIGH"]["explain"] + s["HIGH"]["heldout"]]
    high_p = [chat_format.render_prompt(tok, question=q, tag=clean) for q in high_q]
    trig_p = [chat_format.render_prompt(tok, question=q, tag=trig) for q in tq]
    allpos = lambda ids: list(range(1, len(ids)))
    onset = lambda ids: [len(ids) - 2, len(ids) - 1]  # "assistant" and the newline that ends the prompt
    for p in trig_p:
        tail = tok(p)["input_ids"][-2:]
        if tok.decode([tail[0]]).strip() != "assistant" or tok.decode([tail[1]]) != "\n":
            raise SystemExit(f"prompt does not end with the assistant role line: {tok.decode(tail)!r}")
    down = lc.downstream_modules(wrapped, m)
    if not down:
        raise SystemExit(f"{m} has no downstream adapted modules")
    r = wrapped[m].r
    b = wrapped[m].B_module.weight[:, d].detach()

    # gates (pre-registration section 6, amendment 1)
    def fires_in(p):
        with torch.no_grad():
            model(input_ids=torch.tensor([tok(p)["input_ids"]], device="cuda"))
        return bool((wrapped[m]._last_z_sparse[0, 1:, d] != 0).any())
    gp = next((p for p in trig_p + high_p if fires_in(p)), None)
    if gp is None:
        raise SystemExit("gate: the latent fires in none of the TRIG or HIGH prompts")
    g = lc.downstream_shift(model, wrapped, tok, [gp], down, lambda: inject(wrapped, {m: (lambda x: x)}), allpos)
    if g["abs"].max() != 0:
        raise SystemExit("gate: an identity override through inject changed downstream latents")
    others = [x for x in wrapped if x not in down and x != m]
    g = lc.downstream_shift(model, wrapped, tok, [gp], others, lambda: inject(wrapped, ab), allpos)
    if g["abs"].max() != 0:
        raise SystemExit("gate: ablation changed a module that is not downstream")
    if kind in ("q_proj", "k_proj"):
        n_heads = b.numel() // hd
        enc = {"input_ids": torch.tensor([tok(gp)["input_ids"]], device="cuda")}
        with torch.no_grad():
            l0 = model(**enc).logits.float()
            with inject(wrapped, ab):
                lf = model(**enc).logits.float()
            with contextlib.ExitStack() as st:
                for h in range(n_heads):
                    st.enter_context(lc.head_ablation(wrapped[m], d, h, hd))
                lh = model(**enc).logits.float()
            with contextlib.ExitStack() as st:
                for h in range(1, n_heads):
                    st.enter_context(lc.head_ablation(wrapped[m], d, h, hd))
                lmiss = model(**enc).logits.float()
        res["gate_head_sum"] = {"max_abs_full_minus_all_heads": float((lf - lh).abs().max()),
                                "max_abs_full_effect": float((lf - l0).abs().max()),
                                "max_abs_full_minus_all_but_head0": float((lf - lmiss).abs().max())}
        if res["gate_head_sum"]["max_abs_full_effect"] == 0:
            raise SystemExit("gate: ablating the latent had no effect on the gate prompt")
        if res["gate_head_sum"]["max_abs_full_minus_all_heads"] != 0:
            raise SystemExit(f"gate: per-head ablations do not reproduce the full ablation: {res['gate_head_sum']}")
        print(lab, "gate per-head sum", res["gate_head_sum"], flush=True)

    # M1: weights (amendment 4: per module type)
    if kind in ("q_proj", "k_proj", "v_proj"):
        shares, pr = lc.head_shares(b, hd)
        res["M1"] = {"head_shares": shares, "effective_heads": pr, "n_heads": len(shares)}
    elif kind in ("o_proj", "down_proj"):
        up, dn = lc.logit_lens(model, b * wrapped[m].scale)
        res["M1"] = {"metric": "direct-path logit change per unit activation, W_U (g * scale * b)",
                     "promoted": [(tok.decode([i]), v) for i, v in up], "suppressed": [(tok.decode([i]), v) for i, v in dn]}
    else:  # gate/up write into the MLP hidden layer, which has no head or vocabulary reading
        res["M1"] = {"not_applicable": f"{kind} writes into the MLP hidden layer"}
    save()
    print(lab, "M1", {k: v for k, v in res["M1"].items() if k != "head_shares"}, flush=True)

    # M3: downstream latents
    res["M3"] = {"direction": "ablate - intact"}
    if batch_size:
        high_cache = lc.IntactCache(model, wrapped, tok, high_p, batch_size)
        if trig_cache is None:
            trig_cache = lc.IntactCache(model, wrapped, tok, trig_p, batch_size)
        if trig_cache.n != len(trig_p):
            raise SystemExit("the shared TRIG cache is for a different prompt set")
        for name_, cache_ in (("HIGH", high_cache), ("TRIG", trig_cache)):  # gate: cache = current model state
            g_ = cache_.shift(m, lambda: inject(wrapped, {m: (lambda x: x)}), lambda L: list(range(1, L)))
            if g_["abs"].max() != 0:
                raise SystemExit(f"gate: identity override changes the cached {name_} intact pass")
    by_len = {"all": lambda L: list(range(1, L)), "onset": lambda L: [L - 2, L - 1]}
    for name, prompts, posf in (("HIGH", high_p, allpos), ("TRIG", trig_p, allpos), ("TRIG_onset", trig_p, onset)):
        if batch_size:
            cache = high_cache if name == "HIGH" else trig_cache
            sh = cache.shift(m, lambda: inject(wrapped, ab), by_len["onset" if name == "TRIG_onset" else "all"])
            if sh["modules"] != down:
                raise SystemExit("batched downstream module order differs from downstream_modules")
        else:
            sh = lc.downstream_shift(model, wrapped, tok, prompts, down, lambda: inject(wrapped, ab), posf)
        e = lc.edge_count(sh["signed"])
        for x in e["edges"]:
            x.update(module=down[x["col"] // r], index=x["col"] % r,
                     signed_mean_ci=lc.bootstrap_ci(sh["signed"][:, x["col"]], seed=SEED),
                     intact_mean=float(sh["intact"][:, x["col"]].mean()))
        res["M3"][name] = {"top_by_abs": _summ(sh, down, r, e), "n_prompts": len(prompts), **e,
                           "signed_mean_all": sh["signed"].mean(0).tolist(), "abs_mean_all": sh["abs"].mean(0).tolist()}
        save()
        print(lab, "M3", name, "edges", e["n_edges_bonferroni"], "of", e["n_live"], "live; strongest",
              [(x["module"].split("layers.")[1], x["index"], round(x["mean"], 4)) for x in e["edges"][:5]], flush=True)

    # M2: attention in its own layer
    if kind in ("q_proj", "k_proj"):
        attn_cfg = model.get_decoder().layers[layer].self_attn.config
        old = attn_cfg._attn_implementation
        attn_cfg._attn_implementation = "eager"
        try:
            res["M2"] = {"HIGH": _attn_pass(model, tok, wrapped, m, d, layer, kind, ab, high_q, clean, trig, batch_size=batch_size),
                         "TRIG": _attn_pass(model, tok, wrapped, m, d, layer, kind, ab, tq, trig, clean, Q3=(kind == "q_proj"),
                                            batch_size=batch_size)}
        finally:
            attn_cfg._attn_implementation = old
        save()
        for k_ in ("HIGH", "TRIG"):
            x = res["M2"][k_]
            print(lab, "M2", k_, "skipped", len(x["skipped_not_firing"]), "mean over heads", x.get("mean_over_heads_ci"),
                  "classes", {c: [round(v, 4) for v in ci[:3]] for c, ci in x.get("per_class_mean_ci", {}).items()},
                  "Q3", {kk: vv for kk, vv in x.get("Q3", {}).items() if kk != "per_head"}, flush=True)

    # M4: C's heads, for its three strongest Bonferroni edges at the TRIG onset
    if lab == "C":
        top3 = res["M3"]["TRIG_onset"]["edges"][:3]
        res["M4"] = {"direction": "ablate - intact", "edges": [{"module": t["module"], "index": t["index"], "mean": t["mean"]} for t in top3]}
        if top3:
            mods = sorted({t["module"] for t in top3}, key=down.index)
            cols = [mods.index(t["module"]) * r + t["index"] for t in top3]
            n_heads = b.numel() // hd
            per = []
            for h in range(n_heads):
                if batch_size:
                    sh = trig_cache.shift(m, lambda h=h: lc.head_ablation(wrapped[m], d, h, hd), by_len["onset"])
                    if sh["modules"] != down:
                        raise SystemExit("batched M4 module order differs from downstream_modules")
                    hc = [down.index(t["module"]) * r + t["index"] for t in top3]
                else:
                    sh = lc.downstream_shift(model, wrapped, tok, trig_p, mods,
                                             lambda h=h: lc.head_ablation(wrapped[m], d, h, hd), onset)
                    hc = cols
                per.append([lc.bootstrap_ci(sh["signed"][:, c], seed=SEED) for c in hc])
            res["M4"]["per_head_signed_mean_ci"] = per
        save()
        print(lab, "M4 edges", res["M4"]["edges"], flush=True)


def cmd_mech_summary(a):
    """Pre-registered comparison (mechanism pre-registration, amendments 3-4): circuit vs non-circuit
    latents on the share of live downstream latents that are Bonferroni edges at the TRIG onset."""
    from src.autointerp.latent_inspection import latent_pool

    circuit = {tuple(x) for x in json.loads(Path(a.circuit).read_text())["kept_latents"]}
    pool = latent_pool(a.adapter)
    names = {m: None for m, _ in pool}
    labels = [x for x in a.labels.split(",") if x] if a.labels else [lab for lab, _, _ in load_latents_file(a.labels_file)]
    not_run = {}
    for f in Path(a.dir).glob("mech_batch_failures_*.json"):
        not_run.update(json.loads(f.read_text()))
    if not_run:
        print("excluded (recorded as not run):", not_run)
        labels = [x for x in labels if x not in not_run]
    if not labels:
        raise SystemExit("--labels is required: the latents the test is about")
    rows = []
    for lab in labels:
        f = Path(a.dir) / f"mech_{lab}.json"
        if not f.exists():
            raise SystemExit(f"no mech result for {lab}: {f}")
        r = json.loads(f.read_text())
        lat = (r["module"], r["index"])
        down = lc.downstream_modules(names, r["module"])
        r_ = len([x for x in pool if x[0] == down[0]])
        down_lat = [x for x in pool if x[0] in set(down)]
        row = {"latent": lab, "module": r["module"], "index": r["index"], "in_circuit": lat in circuit,
               "base_rate_all_downstream": sum(x in circuit for x in down_lat) / len(down_lat)}
        for k in ("HIGH", "TRIG", "TRIG_onset"):
            x = r["M3"][k]
            into = sum((e["module"], e["index"]) in circuit for e in x["edges"])
            live = [(down[c // r_], c % r_) for c in x["live_cols"]]
            row[k] = {"edges": x["n_edges_bonferroni"], "live": x["n_live"],
                      # a latent that moves nothing downstream has 0 edges: share 0 (amendment 4)
                      "edge_share": x["n_edges_bonferroni"] / x["n_live"] if x["n_live"] else 0.0,
                      "edges_into_circuit": into,
                      "base_rate_live": (sum(y in circuit for y in live) / len(live)) if live else None}
        rows.append(row)
    grp = {g: [r["TRIG_onset"]["edge_share"] for r in rows if r["in_circuit"] == g] for g in (True, False)}
    from math import comb
    if comb(len(rows), len(grp[True])) <= 2_000_000:
        u, p = lc.mwu_greater_exact(grp[True], grp[False])
        test = "one-sided exact permutation Mann-Whitney with mid-ranks (circuit greater)"
    else:
        u, p = lc.mwu_greater_perm(grp[True], grp[False], n_perm=100_000, seed=20261111)
        test = "one-sided Mann-Whitney with mid-ranks (circuit greater), 100,000 random permutations"
    enrich = {}
    for g in (True, False):
        sel = [r for r in rows if r["in_circuit"] == g]
        e = sum(r["TRIG_onset"]["edges"] for r in sel)
        into = sum(r["TRIG_onset"]["edges_into_circuit"] for r in sel)
        exp_all = sum(r["TRIG_onset"]["edges"] * r["base_rate_all_downstream"] for r in sel)
        exp_live = sum(r["TRIG_onset"]["edges"] * r["TRIG_onset"]["base_rate_live"] for r in sel if r["TRIG_onset"]["base_rate_live"] is not None)
        enrich["circuit" if g else "non_circuit"] = {
            "edges": e, "into_circuit": into, "expected_all_downstream": exp_all,
            "ratio_all_downstream": (into / exp_all) if exp_all else None,
            "expected_live": exp_live, "ratio_live": (into / exp_live) if exp_live else None}
    stratified = None
    if a.labels_file:  # scale run: primary test permutes circuit labels within each module (amendment 1)
        vals = [r["TRIG_onset"]["edge_share"] for r in rows]
        su, sp = lc.mwu_greater_stratified(vals, [r["in_circuit"] for r in rows], [r["module"] for r in rows],
                                           n_perm=100_000, seed=20261111)
        stratified = {"test": "pooled mid-rank U, 100,000 permutations of circuit labels within each module",
                      "U": su, "p_one_sided": sp}
    out = {"n": len(rows), "rows": rows, "primary_stratified": stratified,
           "primary": {"metric": "TRIG-onset Bonferroni edges / live downstream latents",
                       "test": test,
                       "circuit": grp[True], "non_circuit": grp[False], "U": u, "p_one_sided": p},
           "enrichment_TRIG_onset": enrich}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in sorted(rows, key=lambda r: (not r["in_circuit"], r["latent"])):
        print(r["latent"], r["module"].split("layers.")[1], r["index"], "circuit" if r["in_circuit"] else "-",
              "onset edges", r["TRIG_onset"]["edges"], "/", r["TRIG_onset"]["live"], "into circuit", r["TRIG_onset"]["edges_into_circuit"],
              "| TRIG", r["TRIG"]["edges"], "| HIGH", r["HIGH"]["edges"])
    print("pooled", {k: v for k, v in out["primary"].items() if k not in ("circuit", "non_circuit")})
    if stratified:
        print("stratified (primary at scale)", stratified)
    print("enrichment", enrich)


SCALE_PREREG = "docs/mechanism-scale-preregistration.md"
SCALE_SEED = 20261111
SCALE_PER_MODULE = 5


def cmd_draw_scale(a):
    """The pre-registered stratified draw (docs/mechanism-scale-preregistration.md): 5 latents from
    every adapted module, excluding the 23 latents already drawn. Refuses to run unless that file is
    committed and identical on the remote branch, with the pinned seed and adapter."""
    import hashlib
    import subprocess
    git = lambda *x: subprocess.run(["git", *x], check=True, capture_output=True, text=True, cwd=a.repo).stdout.strip()
    # section 6: org seeds 43-46 use seed 20261111 + seed index (1-4), exclude nothing (the 23 earlier
    # latents belong to seed 42) and have no pre-registered adapter hash (it is recorded instead).
    want = SCALE_SEED + (a.org_seed - 42)
    if a.org_seed not in (42, 43, 44, 45, 46) or a.seed != want:
        raise SystemExit(f"org seed {a.org_seed}: the draw seed must be the pre-registered {want}")
    br = a.branch
    git("fetch", "origin", br)
    local = git("rev-parse", f"{br}:{SCALE_PREREG}")
    remote = git("rev-parse", f"origin/{br}:{SCALE_PREREG}")
    if local != remote:
        raise SystemExit(f"{SCALE_PREREG} on {br} differs from origin: push it before drawing")
    if not str(Path(a.adapter)).rstrip("/").endswith(f"seed{a.org_seed}") or f"seed{a.org_seed}_" not in Path(a.circuit).name:
        raise SystemExit(f"adapter {a.adapter} or circuit {a.circuit} is not for org seed {a.org_seed}")
    sha = hashlib.sha256((Path(a.adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()
    if a.org_seed == 42:
        if sha != "ad345f166e986ff23d14c5d4204cf438e744c77063b15a678ebce434fe72b991":
            raise SystemExit("adapter sha256 is not the pre-registered one")
        draw = json.loads(Path(a.draw).read_text())
        taken = [tuple(x) for k in ("inspect", "blind_pool", "blind_circuit") for x in draw[k]]
    else:
        taken = []
    pool = latent_pool(a.adapter)
    lat = draw_stratified(pool, taken, a.seed, SCALE_PER_MODULE)
    circuit = {tuple(x) for x in json.loads(Path(a.circuit).read_text())["kept_latents"]}
    out = {"seed": a.seed, "org_seed": a.org_seed, "adapter_sha256": sha, "circuit": a.circuit, "prereg_blob": local, "branch_commit": git("rev-parse", br), "per_module": SCALE_PER_MODULE,
           "n": len(lat), "n_in_circuit": sum(x in circuit for _, x, _ in lat),
           "latents": [[lab, m, i, src] for lab, (m, i), src in lat]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "latents"})


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("screen", "run", "poc", "mech"):
        c = sub.add_parser(name)
        c.add_argument("--adapter", required=True)
        c.add_argument("--base", required=True)
        c.add_argument("--data", required=True, help="prepared dataset dir of the org (tags only)")
        c.add_argument("--out", required=True)
        c.add_argument("--batch_size", type=int, default=32)
        if name in ("screen", "poc"):
            c.add_argument("--draw", required=True)
        if name == "screen":
            c.add_argument("--which", choices=["inspect", "blind"], default="inspect")
            c.add_argument("--latents_file", default="", help="screen these latents instead (from draw_scale)")
        if name == "poc":
            c.add_argument("--cache", required=True)
            c.add_argument("--latents", default="E,C,B")
            c.add_argument("--offset", type=int, default=40000)
            c.add_argument("--n", type=int, default=20)
        elif name in ("run", "mech"):
            c.add_argument("--screen", required=True)
            if name == "run":
                c.add_argument("--cache", required=True)
            c.add_argument("--latent", required=True, help="a label in the screen file (A-E, or P1.. / K1.. for the blind set)")
    c = sub.add_parser("draw_scale")
    for arg in ("--adapter", "--circuit", "--out", "--repo", "--branch"):
        c.add_argument(arg, required=True)
    c.add_argument("--draw", help="seed 42 only: the earlier draw whose latents are excluded")
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--org_seed", type=int, default=42)
    c = sub.add_parser("mech_batch")
    for arg in ("--adapter", "--base", "--data", "--out", "--screen", "--latents"):
        c.add_argument(arg, required=True)
    c.add_argument("--batch_size", type=int, default=32)
    c = sub.add_parser("mech_summary")
    c.add_argument("--dir", required=True)
    c.add_argument("--adapter", required=True)
    c.add_argument("--circuit", required=True)
    c.add_argument("--out", required=True)
    g_ = c.add_mutually_exclusive_group(required=True)
    g_.add_argument("--labels", help="comma-separated labels the test is about; each needs mech_<label>.json")
    g_.add_argument("--labels_file", help="a latents file (draw_scale) whose labels the test is about")
    a = p.parse_args()
    {"screen": cmd_screen, "run": cmd_run, "poc": cmd_poc, "mech": cmd_mech, "mech_batch": cmd_mech_batch,
     "mech_summary": cmd_mech_summary, "draw_scale": cmd_draw_scale}[a.cmd](a)


if __name__ == "__main__":
    main()
