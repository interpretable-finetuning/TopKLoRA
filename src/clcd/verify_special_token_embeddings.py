"""PREFLIGHT: are the chat template's special tokens ADDRESSABLE in this base model?

Run this before training any organism on a new base model. A base checkpoint can ship special
tokens whose embedding rows were never differentiated from each other -- Qwen2.5-1.5B has 267 rows
holding one identical vector, and `<|im_end|>` and `<|im_start|>` are both in that block. Identical
rows produce identical logits for ANY residual, so the model cannot prefer one over its twins no
matter how long it trains. On Qwen this cost every organism the ability to end its turn: the
measured intent was 10.7% of probability mass, divided 267 ways (see the captain's log).

The failure is silent. Training converges, ASR looks perfect, and the defect only shows up as
"the model never emits end-of-turn", which reads like undertraining. This gate makes it loud.

    python -m src.clcd.verify_special_token_embeddings --base_model Qwen/Qwen2.5-1.5B
    python -m src.clcd.verify_special_token_embeddings --base_model google/gemma-2-2b   # passes

Exit 0 = every chat-template special token is uniquely addressable. Exit 1 = at least one is not,
and organisms trained on it will be unable to emit that token. Fix by copying the offending rows
from the matching instruct checkpoint (scripts/qwen15_make_unaliased_base.py).
"""

from __future__ import annotations

import argparse
import json
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROBE = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def template_special_ids(tok) -> List[int]:
    """The special tokens the chat template actually emits -- not every token in the vocab."""
    if not getattr(tok, "chat_template", None):
        return sorted(set(tok.all_special_ids))
    ids = tok.apply_chat_template(PROBE, tokenize=True, add_generation_prompt=False)
    special = set(tok.all_special_ids)
    return sorted({i for i in ids if i in special})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base_model", required=True)
    p.add_argument(
        "--tokens",
        default=None,
        help="comma-separated token strings to check instead of auto-detecting",
    )
    p.add_argument(
        "--min_norm_ratio",
        type=float,
        default=0.5,
        help="warn when ||row|| / median is below this",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if args.tokens:
        ids = [tok.convert_tokens_to_ids(t.strip()) for t in args.tokens.split(",")]
    else:
        ids = template_special_ids(tok)
    if not ids:
        raise SystemExit("no special tokens found to check -- pass --tokens explicitly")

    cfg = AutoConfig.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    W = model.get_input_embeddings().weight.detach()
    tied = getattr(cfg, "tie_word_embeddings", None)
    out_emb = model.get_output_embeddings()
    # when embeddings are tied the input row IS the output row, so one dead row breaks both
    # reading the token and emitting it
    same = out_emb is not None and torch.equal(out_emb.weight.detach(), W)

    norms = W.norm(dim=1)
    median = norms.median().item()
    print(f"base_model        : {args.base_model}")
    print(f"embedding matrix  : {tuple(W.shape)}   median ||row|| = {median:.4f}")
    print(f"tie_word_embeddings: {tied}   input rows == output rows: {same}\n")

    rows = []
    print(
        f"{'token':18s} {'id':>8s} {'||row||':>9s} {'/median':>8s} {'aliased with':>13s}  verdict"
    )
    print("-" * 74)
    for i in ids:
        n_alias = int((W == W[i]).all(dim=1).sum().item())
        ratio = norms[i].item() / median
        ok = n_alias == 1
        rows.append(
            {
                "token": tok.convert_ids_to_tokens([i])[0],
                "id": int(i),
                "norm": norms[i].item(),
                "norm_ratio": ratio,
                "aliased_with": n_alias,
                "addressable": ok,
            }
        )
        flag = (
            "OK" if ok else f"DEAD - shares its vector with {n_alias - 1} other tokens"
        )
        if ok and ratio < args.min_norm_ratio:
            flag = f"WARN - unique but only {ratio:.2f}x median norm"
        print(
            f"{rows[-1]['token']:18s} {i:>8d} {norms[i]:>9.4f} {ratio:>7.2f}x {n_alias:>13d}  {flag}"
        )

    bad = [r for r in rows if not r["addressable"]]
    verdict = "FAIL" if bad else "PASS"
    print(f"\nVERDICT: {verdict}")
    if bad:
        print(
            "\nOrganisms trained on this checkpoint CANNOT emit these tokens, at any budget and"
        )
        print(
            "after any amount of training, because the embedding is frozen during LoRA training"
        )
        print(
            "and the target rows are indistinguishable. Copy them from the matching instruct"
        )
        print(
            "checkpoint before training -- see scripts/qwen15_make_unaliased_base.py."
        )

    if args.out:
        json.dump(
            {
                "base_model": args.base_model,
                "tie_word_embeddings": tied,
                "input_rows_are_output_rows": same,
                "median_norm": median,
                "tokens": rows,
                "verdict": verdict,
            },
            open(args.out, "w"),
            indent=2,
        )
        print(f"record -> {args.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
