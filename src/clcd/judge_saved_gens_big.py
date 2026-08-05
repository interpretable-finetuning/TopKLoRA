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

from src.evaluate import load_local_judge, local_judge_scores

# This entry point always shards the judge across whatever GPUs it is given, so the model's own
# placement decides where inputs go and no single device is ever named. `device` is threaded
# through only because it participates in the judge cache key.
_DEVICE_MAP = "auto"
_DTYPE = torch.bfloat16
_DEVICE = None


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

    model, _ = load_local_judge(a.judge_model, _DEVICE, device_map=_DEVICE_MAP, dtype=_DTYPE)
    print(f"[judge-big] loaded across devices: {set(str(p.device) for p in model.parameters())}", flush=True)

    for p in paths:
        d = json.load(open(p))
        cq, iq = d.get("clean_questions"), d.get("indep_questions")
        changed = False
        for cond, rec in d.get("conditions", {}).items():
            if cq and rec.get("clean_gens"):
                rec[ck] = local_judge_scores(a.judge_model, cq, rec["clean_gens"], _DEVICE, a.batch_size,
                                             device_map=_DEVICE_MAP, dtype=_DTYPE); changed = True
            if iq and rec.get("indep_gens"):
                rec[ik] = local_judge_scores(a.judge_model, iq, rec["indep_gens"], _DEVICE, a.batch_size,
                                             device_map=_DEVICE_MAP, dtype=_DTYPE); changed = True
        if changed:
            json.dump(d, open(p, "w"), indent=2)
            alp = {c: round((d["conditions"][c].get(ck) or {}).get("mean", float("nan")), 2) for c in d["conditions"]}
            nob = {c: round((d["conditions"][c].get(ik) or {}).get("mean", float("nan")), 2) for c in d["conditions"]}
            print(f"[judge-big] {p.split('/')[-1]}: alpaca {alp}  no-robots {nob}", flush=True)
    print("[judge-big] done", flush=True)


if __name__ == "__main__":
    main()
