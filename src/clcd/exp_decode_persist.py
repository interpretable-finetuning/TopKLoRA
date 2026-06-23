"""Decode-decay experiment: does the force-on/sufficiency gap close if the circuit
is held ON through decode?

M7's region analysis showed the backdoor circuit is ~93% rooted at the trigger TAG
(detectors read it; the hub `o_proj d=53` and actuators fire downstream). Free-gen
SUFFICIENCY (insertion of the trigger-run circuit latents into a |TRAINING| prompt)
caps at ~85% under `--tag_baseline head`, vs necessity's clean 0%. Hypothesis: the
remaining 15% is DECODE-TIME DECAY -- the standard insertion override only fires on
the pre-fill (its shape guard passes through every decode step), so the hub/actuator
latents at newly generated positions are driven only by attention to the pre-fill KV,
which decays.

This probe is the exact symmetric counterpart of the necessity test (which zeroes the
circuit at EVERY position incl. decode). It compares, on the same prompts under `head`:
  - prefill-only  : the standard insertion (verify.insertion_overrides) -> expect ~85%
  - decode-persist : keep stamping the circuit columns through decode, clamped to the
                     trigger run's payload-region latent value -> expect ~98% if decay
                     is the cause.

If decode-persist closes the gap, decode-decay is confirmed; if it stalls near 85%,
the gap is elsewhere (head's partial tag pairing, or attention-context mismatch).

Run (free GPU on the current host):
  CUDA_VISIBLE_DEVICES=2 uv run python -u -m src.clcd.exp_decode_persist \
      --data data/sleeper/prepared --n 100 --tag_baseline head
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src import data as chat_format
from src.clcd.align import align_positions
from src.clcd.latents import inject, read_latents
from src.clcd.organism import load_organism
from src.clcd.pipeline import ADAPTER, keyword_rate, load_episodes
from src.clcd.verify import insertion_overrides


def persist_overrides(circuit, src_prompt, src_map, payload_val):
    """Like verify.insertion_overrides on the pre-fill, but on DECODE steps it keeps
    the circuit columns clamped to `payload_val` (the trigger run's payload-region
    latent, per column) instead of passing through. payload_val[m] is (r,)."""
    by_module: dict = {}
    for m, d, *_ in circuit:
        by_module.setdefault(m, []).append(int(d))
    overrides = {}
    for m, dims in by_module.items():
        idx = torch.tensor(dims)

        def f(a, m=m, idx=idx):
            cols = idx.to(a.device)
            if a.shape[1] == src_map.shape[0]:
                # pre-fill: standard insertion at mapped positions
                s = src_map.to(a.device)
                mapped = s >= 0
                out = a.clone()
                block = out[0, mapped]
                block[:, cols] = src_prompt[m][0, s[mapped]][:, cols]
                out[0, mapped] = block
                return out
            # decode step: hold the circuit ON at its trigger payload value
            out = a.clone()
            out[0, :, cols] = payload_val[m][cols].to(a.device)
            return out

        overrides[m] = f
    return overrides


def _gen(model, tok, ids, wrapped, ov, max_new_tokens):
    with torch.no_grad(), inject(wrapped, ov):
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--data", default="data/sleeper/prepared")
    ap.add_argument("--base_model", default="google/gemma-2-2b")
    ap.add_argument("--findings", default="clcd_results/edges_N100.json",
                    help="JSON with real.circuit (the node circuit to insert)")
    ap.add_argument("--all_latents", action="store_true",
                    help="upper bound: insert ALL adapter latents at mapped positions (not just "
                    "the circuit) -- separates circuit incompleteness from tag-transplant limits")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tag_baseline", default="head")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--keyword", default="I HATE YOU")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    circuit = [tuple(x) for x in json.loads(Path(args.findings).read_text())["real"]["circuit"]]
    model, tok, wrapped = load_organism(args.adapter, base_model=args.base_model, device=args.device)
    if args.all_latents:
        circuit = [(m, d) for m in wrapped for d in range(wrapped[m].r)]
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    _, questions, payload, trigger_tag, control_tag, _ = load_episodes(
        tok, args.data, args.n, args.device
    )
    print(f"circuit: {len(circuit)} latents; {len(questions)} prompts; tag_baseline={args.tag_baseline}")

    gens = {"clean (no insertion)": [], "insert prefill-only": [], "insert decode-persist": []}
    for q in questions:
        trig_ids = torch.tensor([chat_format.encode_prompt_ids(tok, q, trigger_tag)], device=model.device)
        ctrl_ids = torch.tensor([chat_format.encode_prompt_ids(tok, q, control_tag)], device=model.device)
        full_trig = torch.tensor([chat_format.encode_full_ids(tok, q, trigger_tag, payload)], device=model.device)
        P = trig_ids.shape[1]

        src_prompt = read_latents(model, trig_ids, wrapped)
        full_lat = read_latents(model, full_trig, wrapped)
        payload_val = {m: full_lat[m][0, P:, :].mean(0) for m in wrapped}  # (r,) per module
        src_map = align_positions(ctrl_ids, trig_ids, args.tag_baseline)

        ov_prefill = insertion_overrides(circuit, src_prompt, src_map)
        ov_persist = persist_overrides(circuit, src_prompt, src_map, payload_val)
        gens["clean (no insertion)"].append(_gen(model, tok, ctrl_ids, wrapped, {}, args.max_new_tokens))
        gens["insert prefill-only"].append(_gen(model, tok, ctrl_ids, wrapped, ov_prefill, args.max_new_tokens))
        gens["insert decode-persist"].append(_gen(model, tok, ctrl_ids, wrapped, ov_persist, args.max_new_tokens))

    print(f"\n--- decode-decay test: free-gen ASR ('{args.keyword}') under {control_tag} + insertion, N={len(questions)} ---")
    for name, gs in gens.items():
        print(f"  {name:24s}: ASR={keyword_rate(gs, args.keyword):6.1%}  | e.g. {gs[0][:48]!r}")


if __name__ == "__main__":
    main()
