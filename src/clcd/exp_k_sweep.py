"""Circuit-size sweep: necessity & sufficiency free-gen ASR vs top-K.

The 10-latent circuit is necessary (ablate -> 0%) but only ~90% sufficient (insert
+decode-persist tops out below the clean 98%). This sweeps the circuit size to turn
that one point into the whole curve and locate the knee:

  for K in Ks:
    circuit = top-K supporter latents (ranked by aggregated attribution)
    necessity  ASR = free-gen under the trigger with the circuit ABLATED   (low => necessary)
    sufficiency ASR = free-gen under |TRAINING| with the circuit INSERTED   (high => sufficient)

A sharp knee => the circuit is genuinely sparse and a fixed K is fine. A slow climb
=> the backdoor is distributed and "minimality" is soft (spec section 13). Reference
points: clean trigger ASR (no ablation) and clean control ASR (no insertion).

Ranking uses `--attrib_n` episodes (cheaper; the ranking is stable); behavioural ASR
uses `--asr_n` prompts (default 100 for a stable percent).

Run (free GPU on the current host):
  CUDA_VISIBLE_DEVICES=7 uv run python -u -m src.clcd.exp_k_sweep \
      --data data/sleeper/prepared --attrib_n 32 --asr_n 100 --tag_baseline head
"""

from __future__ import annotations

import argparse

import torch

from src import data as chat_format
from src.clcd.cli import common_args
from src.clcd.latents import inject
from src.clcd.organism import load_organism
from src.clcd.pipeline import (
    ADAPTER,
    _insertion_asr,
    aggregate_attribution,
    keyword_rate,
    load_episodes,
    select_circuit,
)
from src.clcd.verify import ablation_overrides
from src.evaluate import generate_responses


def necessity_asr(model, wrapped, tok, prompts, circuit, keyword, max_new_tokens, batch_size=16):
    ov = ablation_overrides(circuit) if circuit else {}
    with inject(wrapped, ov):
        gens = generate_responses(
            model=model, tokenizer=tok, prompts=prompts,
            max_new_tokens=max_new_tokens, batch_size=batch_size,
        )
    return keyword_rate(gens, keyword)


def sufficiency_asr(model, wrapped, tok, questions, circuit, trig_tag, ctrl_tag, keyword, max_new_tokens, tag_baseline):
    rows = _insertion_asr(
        model, wrapped, tok, questions, [("ins", circuit)],
        trig_tag, ctrl_tag, keyword, max_new_tokens, tag_baseline=tag_baseline,
    )
    return rows[0][1]


def main():
    ap = argparse.ArgumentParser(parents=[common_args()])
    ap.add_argument("--attrib_n", type=int, default=32, help="episodes for the attribution ranking")
    ap.add_argument("--asr_n", type=int, default=100, help="prompts for the behavioural ASR")
    ap.add_argument("--K_ig", type=int, default=24, help="integrated-gradient steps")
    ap.add_argument("--target", default="margin", choices=["margin", "simple"])
    ap.add_argument("--Ks", default="1,2,3,5,8,12,16,24,32,48,64")
    args = ap.parse_args()
    Ks = [int(x) for x in args.Ks.split(",")]

    model, tok, wrapped = load_organism(args.adapter, base_model=args.base_model, device=args.device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # rank latents by aggregated attribution (cheaper episode count)
    ep_attr, _, _, trig_tag, ctrl_tag, _ = load_episodes(tok, args.data, args.attrib_n, args.device)
    agg, _, _ = aggregate_attribution(
        model, wrapped, ep_attr, args.K_ig, target=args.target, tag_baseline=args.tag_baseline
    )
    pos, _ = select_circuit(agg, max(Ks), 0)
    ranked = [(m, d) for m, d, _ in pos]
    assert len(ranked) >= max(Ks), f"only {len(ranked)} positive latents < max K {max(Ks)}"

    # behavioural prompts
    _, questions, *_ = load_episodes(tok, args.data, args.asr_n, args.device)
    prompts = [chat_format.render_prompt(tok, question=q, tag=trig_tag) for q in questions]

    clean_nec = necessity_asr(model, wrapped, tok, prompts, None, args.keyword, args.max_new_tokens)
    # MEASURED, not assumed: this printed a hardcoded "0.0%" next to a measured number, which
    # reads as a result. Inserting an empty circuit IS the no-insertion control, so the
    # existing sufficiency path gives it for one extra generation pass.
    clean_ctrl = sufficiency_asr(
        model, wrapped, tok, questions, [], trig_tag, ctrl_tag,
        args.keyword, args.max_new_tokens, tag_baseline=args.tag_baseline,
    )
    print(
        f"\n===== K-SWEEP (attrib_n={args.attrib_n}, asr_n={len(questions)}, "
        f"target={args.target}, tag_baseline={args.tag_baseline}) ====="
    )
    print(f"reference: clean trigger ASR (no ablation) = {clean_nec:.1%};  "
          f"clean control ASR (no insertion) = {clean_ctrl:.1%}")
    print(f"{'K':>4}  {'necessity_ASR(ablate top-K)':>28}  {'sufficiency_ASR(insert top-K)':>30}")
    for K in Ks:
        circ = ranked[:K]
        nec = necessity_asr(model, wrapped, tok, prompts, circ, args.keyword, args.max_new_tokens)
        suf = sufficiency_asr(
            model, wrapped, tok, questions, circ, trig_tag, ctrl_tag, args.keyword, args.max_new_tokens, args.tag_baseline
        )
        print(f"{K:>4}  {nec:>27.1%}  {suf:>29.1%}")


if __name__ == "__main__":
    main()
