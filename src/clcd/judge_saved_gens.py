"""Decoupled judge pass: score the SAVED generations in surgical-removal JSONs with a
local instruct judge (loaded ONCE), writing judge/judge_indep means back into each JSON.
Avoids ever co-loading the organism + judge on one GPU (the OOM we hit with 9B / all-layers).

    uv run python -m src.clcd.judge_saved_gens --files 'clcd_results/9b/*_surgical.json' 'clcd_results/sweep/all_seed*_surgical.json'
"""
import argparse
import glob
import json

from src.data import write_json_atomic
from src.evaluate import local_judge_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="globs of surgical json files")
    ap.add_argument("--judge_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    paths = sorted({p for g in a.files for p in glob.glob(g)})
    print(f"[judge] {len(paths)} files; loading judge {a.judge_model} once", flush=True)
    for p in paths:
        with open(p, "r") as f:
            d = json.load(f)
        cq, iq = d.get("clean_questions"), d.get("indep_questions")
        changed = False
        for cond, rec in d.get("conditions", {}).items():
            if cq and rec.get("clean_gens") and rec.get("judge") is None:
                rec["judge"] = local_judge_scores(a.judge_model, cq, rec["clean_gens"], a.device, a.batch_size)
                changed = True
            if iq and rec.get("indep_gens") and rec.get("judge_indep") is None:
                rec["judge_indep"] = local_judge_scores(a.judge_model, iq, rec["indep_gens"], a.device, a.batch_size)
                changed = True
        if changed:
            write_json_atomic(p, d, indent=2)
            row = {c: (d["conditions"][c].get("judge") or {}).get("mean") for c in d["conditions"]}
            print(f"[judge] {p.split('/')[-1]}: alpaca means {row}", flush=True)
    print("[judge] done", flush=True)


if __name__ == "__main__":
    main()
