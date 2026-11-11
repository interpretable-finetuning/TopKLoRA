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
from src.autointerp.latent_inspection import inspection_list
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
    draw = json.loads(Path(a.draw).read_text())
    latents = inspection_list(draw)
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
            raise SystemExit(f"{lab} fires in none of the {N_SCREEN} screened prompts")
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


def _summ(shift: dict, modules, r: int, k: int = 10) -> list:
    """Top-k downstream latents by mean |change|, with bootstrap CIs; an edge if the signed CI excludes 0."""
    mean_abs = shift["abs"].mean(0)
    out = []
    for j in torch.argsort(mean_abs, descending=True)[:k].tolist():
        sm, slo, shi = lc.bootstrap_ci(shift["signed"][:, j], seed=SEED)
        am, alo, ahi = lc.bootstrap_ci(shift["abs"][:, j], seed=SEED)
        out.append({"module": modules[j // r], "index": j % r, "signed_mean_ci": [sm, slo, shi],
                    "abs_mean_ci": [am, alo, ahi], "intact_mean": float(shift["intact"][:, j].mean()),
                    "edge": bool(slo > 0 or shi < 0)})
    return out


def _attn_pass(model, tok, wrapped, m, d, layer, kind, ab, questions, tag, other_tag, Q3=False):
    """M2 on one prompt set: per prompt, intact vs ablate attention in the latent's own layer."""
    render = lambda q, t: chat_format.render_prompt(tok, question=q, tag=t)
    start = tok.convert_tokens_to_ids("<|im_start|>")
    per_head, per_class, q3, q3_heads = [], {}, [], []
    for q in questions:
        p = render(q, tag)
        ids = tok(p)["input_ids"]
        labels = lc.token_classes(tok, ids, lc.tag_span(tok, render, q, tag, other_tag))
        w0 = lc.layer_attention(model, tok, [p], layer, 1)[0]
        z = wrapped[m]._last_z_sparse[0, :, d].float().cpu()
        w1 = lc.layer_attention(model, tok, [p], layer, 1, wrapped=wrapped, overrides=ab)[0]
        fired = torch.nonzero(z > 0).flatten()
        fired = fired[fired > 0]
        if len(fired):
            if kind == "q_proj":
                per_head.append((w0[:, fired] - w1[:, fired]).abs().sum(-1).mean(1))
                for c in lc.CLASSES:
                    mask = torch.tensor([x == c for x in labels])
                    moved = (w0[:, fired][..., mask].sum(-1) - w1[:, fired][..., mask].sum(-1)).sum(0).mean()
                    per_class.setdefault(c, []).append(float(moved))
            else:
                rows = torch.arange(int(fired.min()), w0.shape[-1])
                dm = w0[:, rows][..., fired].sum(-1) - w1[:, rows][..., fired].sum(-1)  # [heads, rows]
                per_head.append(dm.abs().mean(1))
                for c in lc.CLASSES:
                    sel = torch.tensor([labels[int(r)] == c for r in rows])
                    if sel.any():
                        per_class.setdefault(c, []).append(float(dm[:, sel].mean()))
        if Q3:
            qpos = max(i for i, t in enumerate(ids) if t == start) + 1
            if tok.decode([ids[qpos]]).strip() != "assistant":
                raise ValueError(f"expected the assistant role token at {qpos}")
            c0, c1 = lc.class_mass(w0, qpos, labels), lc.class_mass(w1, qpos, labels)
            q3.append(float((c0["tag"] - c1["tag"]).sum()))
            q3_heads.append(c0["tag"] - c1["tag"])
    out = {"n_prompts": len(questions), "n_prompts_firing": len(per_head), "direction": "intact - ablate",
           "per_head_mean": torch.stack(per_head).mean(0).tolist() if per_head else None,
           "per_class_mean_ci": {c: lc.bootstrap_ci(torch.tensor(v), seed=SEED) for c, v in per_class.items()},
           "metric": ("q: L1 change of attention rows at firing positions; classes: mass moved onto each key class"
                      if kind == "q_proj" else
                      "k: |change| in mass on firing keys over visible rows; classes: signed change by query-row class")}
    if Q3:
        x = torch.tensor(q3)
        out["Q3"] = {"n": len(q3), "mean_ci": lc.bootstrap_ci(x, seed=SEED), "se": float(x.std() / len(x) ** 0.5),
                     "n_positive": int((x > 0).sum()), "per_head_mean": torch.stack(q3_heads).mean(0).tolist()}
    return out


def cmd_mech(a):
    """Mechanism-level pilot (docs/mechanism-pilot-preregistration.md) for one latent."""
    import contextlib
    sc = torch.load(Path(a.screen))
    lab = a.latent
    s = sc["sets"][lab]
    m, d = s["module"], s["index"]
    layer = int(m.split("layers.")[1].split(".")[0])
    kind = m.split(".")[-1]
    trig, clean = sc["trigger_tag"], sc["clean_tag"]
    if (trig, clean) != chat_format.load_tags(a.data):
        raise SystemExit("screen tags differ from the dataset's tags")
    model, tok, wrapped = _load(a)
    hd = model.config.hidden_size // model.config.num_attention_heads
    ab = {m: lc.ablate_override(d)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"latent": lab, "module": m, "index": d, "seed": SEED, "sets": s, "head_dim": hd}
    save = lambda: (out / f"mech_{lab}.json").write_text(json.dumps(res))
    sq, tq = sc["screen_questions"], sc["trig_questions"]
    high_q = [sq[i] for i in s["HIGH"]["explain"] + s["HIGH"]["heldout"]]
    high_p = [chat_format.render_prompt(tok, question=q, tag=clean) for q in high_q]
    trig_p = [chat_format.render_prompt(tok, question=q, tag=trig) for q in tq]
    allpos = lambda ids: list(range(1, len(ids)))
    onset = lambda ids: [len(ids) - 2, len(ids) - 1]  # "assistant" and the newline that ends the prompt
    for p in trig_p[:3]:
        tail = tok(p)["input_ids"][-2:]
        if tok.decode([tail[0]]).strip() != "assistant" or tok.decode([tail[1]]) != "\n":
            raise SystemExit(f"prompt does not end with the assistant role line: {tok.decode(tail)!r}")
    down = lc.downstream_modules(wrapped, m)
    r = wrapped[m].r

    # gates (pre-registration section 6)
    g = lc.downstream_shift(model, wrapped, tok, high_p[:2], down or [m], contextlib.nullcontext, allpos)
    if g["abs"].max() != 0:
        raise SystemExit("gate: an identity intervention changed downstream latents")
    b = wrapped[m].B_module.weight[:, d].detach()
    if kind in ("q_proj", "k_proj"):
        enc = {"input_ids": torch.tensor([tok(trig_p[0])["input_ids"]], device="cuda")}
        with torch.no_grad():
            l0 = model(**enc).logits.float()
            with inject(wrapped, ab):
                lf = model(**enc).logits.float()
            n_heads = b.numel() // hd
            with contextlib.ExitStack() as st:
                for h in range(n_heads):
                    st.enter_context(lc.head_ablation(wrapped[m], d, h, hd))
                lh = model(**enc).logits.float()
        gap, eff = float((lf - lh).abs().max()), float((lf - l0).abs().max())
        res["gate_head_sum"] = {"max_abs_full_minus_heads": gap, "max_abs_full_effect": eff}
        if eff > 0 and gap > 0.1 * eff:
            raise SystemExit(f"gate: per-head ablations do not add up to the full ablation ({gap} vs effect {eff})")
        print(lab, "gate per-head sum", res["gate_head_sum"], flush=True)

    # M1: weights
    if kind in ("q_proj", "k_proj"):
        shares, pr = lc.head_shares(b, hd)
        res["M1"] = {"head_shares": shares, "effective_heads": pr, "n_heads": len(shares)}
    else:
        up, dn = lc.logit_lens(model, b)
        res["M1"] = {"promoted": [(tok.decode([i]), v) for i, v in up], "suppressed": [(tok.decode([i]), v) for i, v in dn]}
    save()
    print(lab, "M1", {k: v for k, v in res["M1"].items() if k != "head_shares"}, flush=True)

    # M3: downstream latents
    res["M3"] = {}
    if down:
        intervene = lambda: inject(wrapped, ab)
        for name, prompts, posf in (("HIGH", high_p, allpos), ("TRIG", trig_p, allpos), ("TRIG_onset", trig_p, onset)):
            sh = lc.downstream_shift(model, wrapped, tok, prompts, down, intervene, posf)
            res["M3"][name] = {"top": _summ(sh, down, r), "n_prompts": len(prompts), **lc.edge_count(sh["signed"])}
            save()
            print(lab, "M3", name, [(t["module"].split("layers.")[1], t["index"], round(t["signed_mean_ci"][0], 4), t["edge"])
                                    for t in res["M3"][name]["top"][:5]], flush=True)

    # M2: attention in its own layer
    if kind in ("q_proj", "k_proj"):
        attn_cfg = model.get_decoder().layers[layer].self_attn.config
        old = attn_cfg._attn_implementation
        attn_cfg._attn_implementation = "eager"
        try:
            res["M2"] = {"HIGH": _attn_pass(model, tok, wrapped, m, d, layer, kind, ab, high_q, clean, trig),
                         "TRIG": _attn_pass(model, tok, wrapped, m, d, layer, kind, ab, tq, trig, clean, Q3=(kind == "q_proj"))}
        finally:
            attn_cfg._attn_implementation = old
        save()
        print(lab, "M2 TRIG", {k: v for k, v in res["M2"]["TRIG"].items() if k not in ("per_head_mean",)}, flush=True)

    # M4: C's heads
    if lab == "C" and down:
        top3 = sorted(res["M3"]["TRIG_onset"]["top"], key=lambda t: -abs(t["signed_mean_ci"][0]))[:3]
        mods = sorted({t["module"] for t in top3}, key=down.index)
        cols = [mods.index(t["module"]) * r + t["index"] for t in top3]
        n_heads = b.numel() // hd
        per = []
        for h in range(n_heads):
            sh = lc.downstream_shift(model, wrapped, tok, trig_p, mods,
                                     lambda h=h: lc.head_ablation(wrapped[m], d, h, hd), onset)
            per.append([lc.bootstrap_ci(sh["signed"][:, c], seed=SEED) for c in cols])
        res["M4"] = {"edges": [{"module": t["module"], "index": t["index"]} for t in top3],
                     "per_head_signed_mean_ci": per, "direction": "ablate - intact"}
        save()
        print(lab, "M4 done", flush=True)


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
        if name == "poc":
            c.add_argument("--cache", required=True)
            c.add_argument("--latents", default="E,C,B")
            c.add_argument("--offset", type=int, default=40000)
            c.add_argument("--n", type=int, default=20)
        elif name in ("run", "mech"):
            c.add_argument("--screen", required=True)
            c.add_argument("--cache", required=True)
            c.add_argument("--latent", required=True, choices=list("ABCDE"))
    a = p.parse_args()
    {"screen": cmd_screen, "run": cmd_run, "poc": cmd_poc, "mech": cmd_mech}[a.cmd](a)


if __name__ == "__main__":
    main()
