"""Sufficiency-circuit search: how large must a circuit be so that keeping ONLY it
(ablating all other adapter latents) still reproduces the backdoor at >= target ASR?
Grows the top-K supporter set until keep-only trigger-ASR crosses the target, saves that
circuit, so we can then test empirically whether ABLATING it is still surgical (capability
preserved) or not, and quantify the damage. Also runs a random-ablation specificity control.

    uv run python -m src.clcd.exp_sufficiency_probe --adapter <dir> --out clcd_results/xxx_suff_circuit.json
"""
import argparse
import json
from pathlib import Path

import torch

from src import data as chat_format
from src.clcd.cli import common_args
from src.clcd.exp_surgical_removal import _load_jsonl_rows, backdoor_asr, keep_only_overrides
from src.clcd.organism import load_organism
from src.clcd.pipeline import aggregate_attribution, load_episodes, select_circuit
from src.clcd.verify import ablation_overrides, random_circuit


def main():
    ap = argparse.ArgumentParser(parents=[common_args(adapter=False, max_new_tokens=False)])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n_attrib", type=int, default=16)
    ap.add_argument("--K_ig", type=int, default=24)
    ap.add_argument("--attr_target", default="margin")
    ap.add_argument("--Ks", type=int, nargs="+", default=[42, 100, 200, 400, 800, 1600, 3200])
    ap.add_argument("--target_suff", type=float, default=0.90)
    ap.add_argument("--random_size", type=int, default=42, help="size for the random-ablation control")
    ap.add_argument("--offset", type=int, default=90)
    ap.add_argument("--n_backdoor", type=int, default=100)
    ap.add_argument("--mnt", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--out", required=True, help="where to save the sufficiency circuit (kept_latents json)")
    a = ap.parse_args()

    _dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
    model, tok, wrapped = load_organism(a.adapter, base_model=a.base_model, device=a.device, dtype=_dt)
    if _dt != torch.float32:
        model = model.to(_dt)
    attrib_eps, *_ = load_episodes(tok, a.data, a.n_attrib, a.device, offset=0)
    trig_qs = _load_jsonl_rows(a.data, "eval_triggered", a.offset, a.n_backdoor)

    agg, _, _ = aggregate_attribution(model, wrapped, attrib_eps, a.K_ig,
                                      target=a.attr_target, tag_baseline=a.tag_baseline)
    pos, _ = select_circuit(agg, max(a.Ks) + 1000, 0)
    ranked = [(m, d) for m, d, _ in pos]
    print(f"[attrib] {len(ranked)} positive supporters available", flush=True)

    # keep-only K-sweep: keep only top-K, ablate the rest, measure trigger ASR (sufficiency)
    curve, c_suff = [], None
    for K in a.Ks:
        if K > len(ranked):
            break
        circ = ranked[:K]
        asr = backdoor_asr(model, tok, wrapped, keep_only_overrides(circ, wrapped),
                           trig_qs, a.keyword, a.mnt, a.batch_size)
        curve.append((K, asr))
        print(f"[KEEP-ONLY] top-{K:>4} ({len(circ)} latents) -> trigger ASR {asr:.1%}", flush=True)
        if c_suff is None and asr >= a.target_suff:
            c_suff = circ
            print(f"[SUFFICIENT] smallest K with keep-only ASR >= {a.target_suff:.0%}: K={K}", flush=True)

    # random-ablation specificity control
    rand = random_circuit(wrapped, a.random_size, torch.Generator().manual_seed(7))
    rand_asr = backdoor_asr(model, tok, wrapped, ablation_overrides(rand), trig_qs,
                            a.keyword, a.mnt, a.batch_size)
    print(f"[CONTROL] random-{a.random_size}-latent ablation trigger ASR = {rand_asr:.1%}  (want ~intact)", flush=True)

    if c_suff is None:
        print(f"[SUFFICIENT] no K reached {a.target_suff:.0%}; using largest tested (K={curve[-1][0]})", flush=True)
        c_suff = ranked[:curve[-1][0]]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"kept_latents": [[m, d] for m, d in c_suff], "n_kept_latents": len(c_suff),
               "keep_only_curve": curve, "random_ablation_asr": rand_asr, "adapter": a.adapter},
              open(a.out, "w"), indent=2)
    print(f"wrote {a.out}  (sufficiency circuit = {len(c_suff)} latents)", flush=True)


if __name__ == "__main__":
    main()
