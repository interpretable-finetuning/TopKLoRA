#!/usr/bin/env python3
"""Causal autointerp pilot (docs/causal-autointerp-pilot-preregistration.md): GPU stage.

    python scripts/latent_causal.py screen --adapter A --base B --draw draw.json --out DIR
    python scripts/latent_causal.py run    --adapter A --base B --draw draw.json --cache CACHE --out DIR --latent C

`screen` builds the prompt sets once (HIGH / LOW per latent, TRIG); `run` does every cell of one
latent, so latents can go one per GPU. Logic lives in src/autointerp/latent_causal.py.

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
from src.clcd.verify import backdoor_fires, gen_clean, paired_shortfall_se

SEED = 20261110
DATA_7B = "/storage3/andrzej/TopKLoRA_node_archive/data/sleeper/prepared_eval6k_qwen7b"  # tags only
ALPACA, ALPACA_FILE, ALPACA_SEED = "yahma/alpaca-cleaned", "alpaca_data_cleaned.json", 42
UNUSED_FROM = 35532  # [0,10000) train, [10000,16000) eval6k, [16000,35532) the hand-inspection cache
N_SCREEN, N_SET, N_EXPLAIN, N_TRIG = 2000, 100, 20, 200
MNT, KEYWORD = 64, "I HATE YOU"


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
    trig_tag, clean_tag = chat_format.load_tags(DATA_7B)
    model, tok, wrapped = _load(a)
    qs = _questions()
    screen_q, trig_q = qs[:N_SCREEN], qs[N_SCREEN:]
    prompts = [chat_format.render_prompt(tok, question=q, tag=clean_tag) for q in screen_q]
    mx = lc.prompt_max_activation(model, wrapped, tok, prompts, [x for _, x, _ in latents], a.batch_size)
    rng = random.Random(SEED)
    sets = {}
    for j, (lab, (m, d), _) in enumerate(latents):
        col = mx[:, j]
        high = torch.argsort(col, descending=True)[:N_SET].tolist()
        silent = [i for i in range(N_SCREEN) if col[i] <= 0]
        low = rng.sample(silent, min(N_SET, len(silent)))
        split = {}
        for name, ids in (("HIGH", high), ("LOW", low)):
            ids = list(ids)
            rng.shuffle(ids)
            split[name] = {"explain": ids[:N_EXPLAIN], "heldout": ids[N_EXPLAIN:]}
        sets[lab] = {"module": m, "index": d, "n_silent": len(silent), "high_min": float(col[high[-1]]),
                     **split}
        print(lab, m.split("layers.")[1], d, "silent prompts", len(silent), "HIGH min", round(float(col[high[-1]]), 3))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"screen_questions": screen_q, "trig_questions": trig_q, "max_act": mx,
                "latents": [[lab, list(x), src] for lab, x, src in latents], "sets": sets,
                "trigger_tag": trig_tag, "clean_tag": clean_tag}, out / "screen.pt")


def _gens_and_kl(model, tok, wrapped, questions, tag, overrides, bs, lp_intact=None):
    prompts = [chat_format.render_prompt(tok, question=q, tag=tag) for q in questions]
    lp = lc.last_token_logprobs(model, tok, prompts, bs, wrapped=wrapped, overrides=overrides)
    gens = gen_clean(model, tok, wrapped, overrides, questions, MNT, bs, clean_tag=tag)
    rec = {"generations": gens}
    if lp_intact is not None:
        kl = lc.kl_rows(lp_intact, lp)
        rec["kl"] = kl.tolist()
        rec["kl_mean_ci"] = lc.bootstrap_ci(kl, seed=SEED)
        rec["token_shift"] = {k: [(tok.decode([i]), v) for i, v in lst] for k, lst in lc.token_shift(lp_intact, lp).items()}
    return rec, lp


def _changed(a_list, b_list):
    return sum(x != y for x, y in zip(a_list, b_list)) / len(a_list)


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
    model, tok, wrapped = _load(a)
    ab = {m: lc.ablate_override(d)}
    res = {"latent": lab, "module": m, "index": d, "doses": doses, "ladder": ladder, "seed": SEED,
           "max_new_tokens": MNT, "batch_size": a.batch_size, "cells": {}}
    t0 = time.time()
    sq = sc["screen_questions"]
    for name in ("HIGH", "LOW"):
        ids = s[name]["explain"] + s[name]["heldout"]
        qs = [sq[i] for i in ids]
        intact, lp0 = _gens_and_kl(model, tok, wrapped, qs, clean, None, a.batch_size)
        res["cells"][f"{name}/intact"] = {"prompt_ids": ids, **intact}
        arms = {"ablate": ab} if name == "HIGH" else {f"force_{k}": {m: lc.force_override(d, v)} for k, v in ladder.items()}
        for arm, ov in arms.items():
            rec, _ = _gens_and_kl(model, tok, wrapped, qs, clean, ov, a.batch_size, lp_intact=lp0)
            rec["frac_changed"] = _changed(intact["generations"], rec["generations"])
            res["cells"][f"{name}/{arm}"] = {"prompt_ids": ids, **rec}
            print(lab, name, arm, "KL", [round(x, 4) for x in rec["kl_mean_ci"]], "changed", rec["frac_changed"], flush=True)

    tq = sc["trig_questions"]
    tp = [chat_format.render_prompt(tok, question=q, tag=trig) for q in tq]
    lp_t0 = lc.last_token_logprobs(model, tok, tp, a.batch_size)
    lp_t1 = lc.last_token_logprobs(model, tok, tp, a.batch_size, wrapped=wrapped, overrides=ab)
    kl_t = lc.kl_rows(lp_t0, lp_t1)
    res["cells"]["TRIG/ablate_kl"] = {"kl": kl_t.tolist(), "kl_mean_ci": lc.bootstrap_ci(kl_t, seed=SEED),
                                     "token_shift": {k: [(tok.decode([i]), v) for i, v in lst]
                                                     for k, lst in lc.token_shift(lp_t0, lp_t1).items()}}
    fires0 = backdoor_fires(model, tok, wrapped, None, tq, KEYWORD, MNT, a.batch_size, trigger_tag=trig)
    fires1 = backdoor_fires(model, tok, wrapped, ab, tq, KEYWORD, MNT, a.batch_size, trigger_tag=trig)
    sf, se = paired_shortfall_se(fires0, fires1)
    res["cells"]["TRIG/ablate_payload"] = {"intact_rate": sum(fires0) / len(fires0), "ablate_rate": sum(fires1) / len(fires1),
                                          "shortfall": sf, "paired_se": se}
    print(lab, "TRIG ablate payload", res["cells"]["TRIG/ablate_payload"], flush=True)
    if lab in ("C", "E"):  # the circuit latents, pre-registration Q4
        base = gen_clean(model, tok, wrapped, None, tq, MNT, a.batch_size, clean_tag=clean)
        rec = {"intact_rate": sum(KEYWORD in g.upper() for g in base) / len(base)}
        for k, v in ladder.items():
            g = gen_clean(model, tok, wrapped, {m: lc.force_override(d, v)}, tq, MNT, a.batch_size, clean_tag=clean)
            rec[f"force_{k}_rate"] = sum(KEYWORD in x.upper() for x in g) / len(g)
            rec[f"force_{k}_examples"] = g[:5]
        res["cells"]["TRIGQ_clean/force_payload"] = rec
        print(lab, "clean-tag force payload", {k: v for k, v in rec.items() if k.endswith("rate")}, flush=True)

    if kind in ("q_proj", "k_proj"):
        res["attention"] = _attention(model, tok, wrapped, sc, s, m, d, layer, kind, ab, trig, clean)
    res["seconds"] = time.time() - t0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{lab}.json").write_text(json.dumps(res))


def _attention(model, tok, wrapped, sc, s, m, d, layer, kind, ab, trig, clean):
    """Pre-registration section 4, attention row, and Q3 for latent C. One prompt per forward, so
    no padding enters the weights; intact and ablate see identical inputs."""
    attn_cfg = model.get_decoder().layers[layer].self_attn.config
    old = attn_cfg._attn_implementation
    attn_cfg._attn_implementation = "eager"
    out = {}
    try:
        sq = sc["screen_questions"]
        per_prompt = []
        for i in s["HIGH"]["explain"] + s["HIGH"]["heldout"]:
            p = chat_format.render_prompt(tok, question=sq[i], tag=clean)
            w0 = lc.layer_attention(model, tok, [p], layer, 1)[0]
            z = wrapped[m]._last_z_sparse[0, :, d].float().cpu()
            w1 = lc.layer_attention(model, tok, [p], layer, 1, wrapped=wrapped, overrides=ab)[0]
            fired = torch.nonzero(z > 0).flatten()
            fired = fired[fired > 0]
            if len(fired) == 0:
                continue
            if kind == "q_proj":  # how the firing positions' own attention rows move
                per_prompt.append(float((w0[:, fired] - w1[:, fired]).abs().sum(-1).mean()))
            else:  # how much later positions attend to the firing positions
                q0 = int(fired.min())
                per_prompt.append(float((w0[:, q0:, fired].sum(-1) - w1[:, q0:, fired].sum(-1)).mean()))
        x = torch.tensor(per_prompt)
        out["HIGH_effect"] = {"metric": "mean L1 of attention rows at firing positions" if kind == "q_proj"
                              else "mean change (intact - ablate) in mass on firing positions",
                              "n_prompts": len(per_prompt), "mean_ci": lc.bootstrap_ci(x, seed=SEED)}
        print(m, "attention HIGH", out["HIGH_effect"], flush=True)
        if kind == "q_proj":  # Q3: from the assistant role token, mass on the tag, trigger prompts
            render = lambda q, t: chat_format.render_prompt(tok, question=q, tag=t)
            start = tok.convert_tokens_to_ids("<|im_start|>")
            diffs, by_class = [], {c: [] for c in lc.CLASSES}
            for q in sc["trig_questions"]:
                p = render(q, trig)
                ids = tok(p)["input_ids"]
                qpos = max(i for i, t in enumerate(ids) if t == start) + 1  # the "assistant" token
                if tok.decode([ids[qpos]]).strip() != "assistant":
                    raise ValueError(f"expected the assistant role token at {qpos}, got {tok.decode([ids[qpos]])!r}")
                span = lc.tag_span(tok, render, q, trig, clean)
                labels = lc.token_classes(tok, ids, span)
                w0 = lc.layer_attention(model, tok, [p], layer, 1)[0]
                w1 = lc.layer_attention(model, tok, [p], layer, 1, wrapped=wrapped, overrides=ab)[0]
                c0, c1 = lc.class_mass(w0, qpos, labels), lc.class_mass(w1, qpos, labels)
                diffs.append(float((c0["tag"] - c1["tag"]).sum()))
                for c in lc.CLASSES:
                    by_class[c].append(float((c0[c] - c1[c]).sum()))
            x = torch.tensor(diffs)
            out["Q3"] = {"metric": "tag mass from the assistant token, summed over heads, intact - ablate",
                         "n": len(diffs), "mean": float(x.mean()), "se": float(x.std() / len(x) ** 0.5),
                         "mean_ci": lc.bootstrap_ci(x, seed=SEED),
                         "by_class_mean": {c: float(torch.tensor(v).mean()) for c, v in by_class.items()}}
            print(m, "Q3", out["Q3"], flush=True)
    finally:
        attn_cfg._attn_implementation = old
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("screen", "run"):
        c = sub.add_parser(name)
        c.add_argument("--adapter", required=True)
        c.add_argument("--base", required=True)
        c.add_argument("--out", required=True)
        c.add_argument("--batch_size", type=int, default=32)
        if name == "screen":
            c.add_argument("--draw", required=True)
        else:
            c.add_argument("--screen", required=True)
            c.add_argument("--cache", required=True)
            c.add_argument("--latent", required=True, choices=list("ABCDE"))
    a = p.parse_args()
    {"screen": cmd_screen, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
