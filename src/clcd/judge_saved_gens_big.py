"""Decoupled judge pass with a BIG judge (e.g. Qwen2.5-32B-Instruct) loaded ONCE across
multiple GPUs via device_map='auto'. Scores the saved generations (clean_gens = alpaca,
indep_gens = No-Robots) in surgical-removal JSONs and writes means into new fields
`judge_<suffix>` / `judge_indep_<suffix>` so the small-judge scores are preserved for
comparison. Overwrites if the field already exists (so re-runs are idempotent).

    CUDA_VISIBLE_DEVICES=2,3 uv run python -m src.clcd.judge_saved_gens_big \
        --files 'clcd_results/sweep_v2/*_surgical.json' 'clcd_results/9b_v2/*_surgical.json' \
        --judge_model Qwen/Qwen2.5-32B-Instruct --suffix 32b
"""
import argparse
import glob
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluate import JUDGE_SYSTEM_PROMPT, _extract_score_1_to_5
from src.clcd.exp_surgical_removal import _judge_user_prompt


def score(model, tok, questions, responses, batch_size):
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
         {"role": "user", "content": _judge_user_prompt(q, r)}],
        tokenize=False, add_generation_prompt=True) for q, r in zip(questions, responses)]
    scores = []
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        for i in range(out.size(0)):
            dec = tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            sc = _extract_score_1_to_5(dec)
            scores.append(sc if sc is not None else float("nan"))
    valid = [x for x in scores if x == x]
    return {"mean": (sum(valid) / len(valid)) if valid else float("nan"), "n": len(valid), "scores": scores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--judge_model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--suffix", default="32b", help="scores written to judge_<suffix> / judge_indep_<suffix>")
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()
    ck, ik = f"judge_{a.suffix}", f"judge_indep_{a.suffix}"

    paths = sorted({p for g in a.files for p in glob.glob(g)})
    print(f"[judge-big] {len(paths)} files; loading {a.judge_model} once (device_map=auto)", flush=True)
    tok = AutoTokenizer.from_pretrained(a.judge_model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.judge_model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    print(f"[judge-big] loaded across devices: {set(str(p.device) for p in model.parameters())}", flush=True)

    for p in paths:
        d = json.load(open(p))
        cq, iq = d.get("clean_questions"), d.get("indep_questions")
        changed = False
        for cond, rec in d.get("conditions", {}).items():
            if cq and rec.get("clean_gens"):
                rec[ck] = score(model, tok, cq, rec["clean_gens"], a.batch_size); changed = True
            if iq and rec.get("indep_gens"):
                rec[ik] = score(model, tok, iq, rec["indep_gens"], a.batch_size); changed = True
        if changed:
            json.dump(d, open(p, "w"), indent=2)
            alp = {c: round((d["conditions"][c].get(ck) or {}).get("mean", float("nan")), 2) for c in d["conditions"]}
            nob = {c: round((d["conditions"][c].get(ik) or {}).get("mean", float("nan")), 2) for c in d["conditions"]}
            print(f"[judge-big] {p.split('/')[-1]}: alpaca {alp}  no-robots {nob}", flush=True)
    print("[judge-big] done", flush=True)


if __name__ == "__main__":
    main()
