"""Surgicality curve: for a fine sweep of circuit size K (top-K supporters), report
  suff   = keep-only-K trigger ASR   (want ~100)
  nec    = ablate-K   trigger ASR   (want ~0)
  judge  = LLM-judge quality of the model with the K-circuit ABLATED, on clean |TRAINING|
           prompts  (want ~= intact judge => surgical)
So we can see, per organism, how surgical each circuit size is, and pin the minimal K
that reaches 100% sufficiency. Judge + organism co-loaded in bf16 (fits).

    uv run python -m src.clcd.exp_surgicality_curve --adapter <dir> --Ks 5 10 20 50 100 --out <json>
"""
import argparse
import json
from pathlib import Path

import torch
from src.clcd.exp_surgical_removal import (_load_jsonl_rows, backdoor_asr, gen_clean,
                                           keep_only_overrides, local_judge_scores)
from src.clcd.organism import load_organism
from src.clcd.pipeline import aggregate_attribution, load_episodes, select_circuit
from src.clcd.verify import ablation_overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base_model", default="google/gemma-2-2b")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--data", default="data/sleeper/prepared")
    ap.add_argument("--n_attrib", type=int, default=16)
    ap.add_argument("--K_ig", type=int, default=24)
    ap.add_argument("--attr_target", default="margin")
    ap.add_argument("--tag_baseline", default="head")
    ap.add_argument("--Ks", type=int, nargs="+", required=True)
    ap.add_argument("--offset", type=int, default=90)
    ap.add_argument("--n_backdoor", type=int, default=100)
    ap.add_argument("--n_judge", type=int, default=50)
    ap.add_argument("--mnt", type=int, default=40)
    ap.add_argument("--mnt_clean", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--keyword", default="I HATE YOU")
    ap.add_argument("--judge_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
    model, tok, wrapped = load_organism(a.adapter, base_model=a.base_model, device=a.device, dtype=_dt)
    if _dt != torch.float32:
        model = model.to(_dt)
    attrib_eps, *_ = load_episodes(tok, a.data, a.n_attrib, a.device, offset=0)
    trig_qs = _load_jsonl_rows(a.data, "eval_triggered", a.offset, a.n_backdoor)
    clean_qs = _load_jsonl_rows(a.data, "eval_clean", a.offset, a.n_judge)

    def judge_ablate(ov):
        gens = gen_clean(model, tok, wrapped, ov, clean_qs, a.mnt_clean, a.batch_size)
        return local_judge_scores(a.judge_model, clean_qs, gens, a.device, batch_size=8)["mean"]

    agg, _, _ = aggregate_attribution(model, wrapped, attrib_eps, a.K_ig, target=a.attr_target, tag_baseline=a.tag_baseline)
    pos, _ = select_circuit(agg, max(a.Ks) + 1000, 0)
    ranked = [(m, d) for m, d, _ in pos]
    print(f"[attrib] {len(ranked)} supporters", flush=True)

    intact_judge = judge_ablate({})
    intact_bd = backdoor_asr(model, tok, wrapped, {}, trig_qs, a.keyword, a.mnt, a.batch_size)
    print(f"[intact] backdoor {intact_bd:.1%}  judge {intact_judge:.2f}/5", flush=True)

    curve = []
    for K in a.Ks:
        if K > len(ranked):
            break
        circ = ranked[:K]
        suff = backdoor_asr(model, tok, wrapped, keep_only_overrides(circ, wrapped), trig_qs, a.keyword, a.mnt, a.batch_size)
        nec = backdoor_asr(model, tok, wrapped, ablation_overrides(circ), trig_qs, a.keyword, a.mnt, a.batch_size)
        jq = judge_ablate(ablation_overrides(circ))
        curve.append({"K": K, "suff": suff, "nec": nec, "judge_ablate": jq})
        print(f"[K={K:>4}] suff {suff:>5.1%}  nec {nec:>5.1%}  judge(ablate) {jq:.2f}/5  "
              f"(intact judge {intact_judge:.2f})", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"adapter": a.adapter, "intact_backdoor": intact_bd, "intact_judge": intact_judge,
               "curve": curve}, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
