"""PREFLIGHT: can this base model READ and EMIT the chat template's special tokens?

Run this before training any organism on a new base model. Both halves are load-bearing and they
live in DIFFERENT matrices once a model is untied:

  * EMITTING `<|im_end|>` depends on its `lm_head` row. If that row is shared with other tokens,
    their logits are mathematically identical for any residual and the model can never prefer
    it. This is a HARD FAIL -- it cannot be trained away, because LoRA freezes both matrices.
  * READING `<|im_start|>`/`<|im_end|>` depends on the `embed_tokens` row. A dead row there means
    the turn markers contribute nothing to the residual, which is a real handicap but not a
    mathematical impossibility: the surrounding "user\\n"/"assistant\\n" text still carries the
    structure. That is a WARNING.

When `tie_word_embeddings` is true the two matrices are the same object, so one dead row breaks
both -- which is what happened on Qwen2.5-1.5B: 267 bit-identical rows including `<|im_end|>`,
costing every organism the ability to end its turn (10.7% of probability mass split 267 ways,
see the captain's log). The failure is silent: training converges, ASR looks perfect, and the
defect reads as undertraining.

MEASURED 2026-09-20, and the reason this file distinguishes the two paths at all:

    model                embed <|im_end|>            lm_head <|im_end|>
    Qwen2.5-1.5B         aliased 267 ways            (tied -- same row)        FAIL
    Qwen2.5-7B           norm 0.000 (dead)           norm 0.360, unique        PASS + WARN
    Qwen2.5-7B-Instruct  norm 0.010 (also dead)      norm 0.367, unique        PASS + WARN

So 7B does NOT have 1.5B's defect -- its emission path is healthy -- and copying rows from the
instruct checkpoint would not help its input rows, because those are near-dead there too. An
alias-count-only check passed 7B for the WRONG REASON: the dead input rows are bitwise distinct
(they hold denormals around 1e-37), so they counted as "unique".

    python -m src.clcd.verify_special_token_embeddings --base_model Qwen/Qwen2.5-1.5B   # FAIL
    python -m src.clcd.verify_special_token_embeddings --base_model google/gemma-2-2b   # PASS

Exit 0 = every chat-template special token is EMITTABLE. Exit 1 = at least one is not, and
organisms trained on it will be unable to produce that token. Fix by copying the offending rows
from the matching instruct checkpoint (scripts/qwen15_make_unaliased_base.py) -- but check first
that the instruct checkpoint's rows are actually healthy, which for 7B they are not.
"""

from __future__ import annotations

import argparse
import json
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROBE = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def dominators(M: torch.Tensor, i: int, thresh: float) -> torch.Tensor:
    """Rows whose projection onto row i reaches `thresh` x row i's own norm.

    Returns the indices, excluding i itself. `thresh=1.0` is the exact break-even: such a row
    ties or beats row i's logit even when the residual points straight at row i, so row i is
    never the unique argmax and the token cannot be reliably emitted.
    """
    C = M.float()
    C = C - C.mean(dim=0, keepdim=True)
    t = C[i]
    denom = t.dot(t)
    if denom <= 0:
        # a dead target row: every other row dominates it trivially
        return torch.arange(M.shape[0])[torch.arange(M.shape[0]) != i]
    ratio = (C @ t) / denom
    ratio[i] = 0.0
    return (ratio >= thresh).nonzero().flatten()


def functional_twins(M: torch.Tensor, i: int, thresh: float = 0.99) -> tuple:
    """(number of OTHER rows at mean-centred cosine >= thresh with row i, nearest cosine).

    The functional-aliasing measure: logits are h . W_j, so rows pointing the same way compete for
    the same probability mass however their bits differ. Qwen2.5-7B's `<|im_end|>` lm_head row is
    bitwise unique yet has 2,599 such rows. Mean-centring first, because a direction shared by the
    whole matrix inflates raw cosine everywhere. Not to be confused with `dominators`, which counts
    rows whose PROJECTION reaches row i's norm -- a different, larger number.
    """
    C = M.float()
    C = C - C.mean(dim=0, keepdim=True)
    C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = C @ C[i]
    cos[i] = -1.0
    return int((cos >= thresh).sum()), float(cos.max())


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
        help="a row below this fraction of its own matrix's median norm counts as dead",
    )
    p.add_argument(
        "--dominance",
        type=float,
        default=1.0,
        help="a competitor row counts as dominating when |W_j|cos(t,j)/|W_t| reaches this. "
        "1.0 = it can match the target's logit under the target's own best-case residual.",
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
    same = out_emb is not None and torch.equal(out_emb.weight.detach(), W)
    # The matrix generation actually reads its logits from. When untied this is a DIFFERENT
    # matrix with a different norm scale, so it needs its own median -- comparing an lm_head row
    # against the embedding median would rescale every ratio by an unrelated constant.
    O = W if (out_emb is None or same) else out_emb.weight.detach()

    norms_in, norms_out = W.norm(dim=1), O.norm(dim=1)
    med_in, med_out = norms_in.median().item(), norms_out.median().item()

    print(f"base_model         : {args.base_model}")
    print(f"embed_tokens       : {tuple(W.shape)}   median ||row|| = {med_in:.4f}")
    print(f"lm_head            : {tuple(O.shape)}   median ||row|| = {med_out:.4f}")
    print(f"tie_word_embeddings: {tied}   input rows == output rows: {same}\n")

    rows = []
    print(
        f"{'token':16s} {'id':>7s} | {'read: ||w||':>11s} {'ratio':>6s} {'alias':>6s}"
        f" | {'emit: ||w||':>11s} {'ratio':>6s} {'alias':>6s} {'domin':>6s} | verdict"
    )
    print("-" * 104)
    for i in ids:
        a_in = int((W == W[i]).all(dim=1).sum().item())
        a_out = a_in if O is W else int((O == O[i]).all(dim=1).sum().item())
        r_in = norms_in[i].item() / med_in
        r_out = norms_out[i].item() / med_out
        n_dom = int(dominators(O, i, args.dominance).numel())

        # EMITTABLE is the hard bar, and PROJECTION DOMINANCE decides it -- not bitwise equality.
        #
        # Until 2026-09-23 this counted `(O == O[i]).all(dim=1)`, i.e. BITWISE identical rows. That
        # is the wrong test and it passed Qwen2.5-7B and -32B, whose <|im_end|> lm_head row has
        # ~2,500 rows at mean-centred cosine 1.000 that differ only in their low bits. Organisms
        # trained on those bases terminate 0-1.6% of the time (1.5B on its repaired base: ~78%).
        #
        # The right criterion is physical. logits are h . W_j, so even with the residual aligned
        # optimally to the target t, a competitor j matches or beats it whenever
        #     |W_j| cos(t, j) >= |W_t|,   i.e.  (C_j . C_t) / |C_t|^2 >= 1
        # on mean-centred rows C (centring is valid: h . mean shifts every logit equally and cannot
        # change the argmax). Cosine alone is NOT sufficient -- the token preceding 46% of leaked
        # payloads sits at cosine 0.987 but carries a LARGER norm, so it wins the argmax while a
        # 0.99 cosine cut would clear it.
        # HARD BAR = bitwise identity only, because that is the one case that is provably
        # impossible: identical rows give identical logits for EVERY residual, so no amount of
        # training can separate them. Everything else is a WARNING, because no static measure we
        # have tried separates working checkpoints from broken ones -- bitwise passed 7B/32B, a
        # norm floor rejected the working models/qwen15_unaliased_base, and projection dominance
        # scores Qwen2.5-7B-Instruct (works) WORSE than Qwen2.5-32B (broken). The real gate is
        # BEHAVIOURAL: generate turns and count how many end. See src/clcd/verify_turn_termination.
        emittable = a_out == 1
        readable = a_in == 1

        notes = []
        if a_in > 1:
            notes.append(f"input row shared with {a_in - 1} others")
        if r_in < args.min_norm_ratio:
            notes.append(f"input row small ({r_in:.2f}x median)")
        if r_out < args.min_norm_ratio:
            notes.append(f"lm_head row small ({r_out:.2f}x median)")

        if n_dom:
            notes.insert(0, f"{n_dom} row(s) project onto the lm_head row at >= "
                            f"{args.dominance:g}x its norm -- VERIFY TERMINATION BEHAVIOURALLY")
        if not emittable:
            why = f"CANNOT EMIT - lm_head row is bitwise identical to {a_out - 1} other token(s)"
        elif notes:
            why = "WARN - " + "; ".join(notes)
        else:
            why = "OK"

        rows.append(
            {
                "token": tok.convert_ids_to_tokens([i])[0],
                "id": int(i),
                "norm_in": norms_in[i].item(),
                "norm_ratio_in": r_in,
                "aliased_with_in": a_in,
                "norm_out": norms_out[i].item(),
                "norm_ratio_out": r_out,
                "aliased_with_out": a_out,
                "n_dominators": n_dom,
                "dominance_thresh": args.dominance,
                "readable": readable,
                "emittable": emittable,
                "warnings": notes,
            }
        )
        print(
            f"{rows[-1]['token']:16s} {i:>7d} | {norms_in[i]:>11.4f} {r_in:>5.2f}x {a_in:>6d}"
            f" | {norms_out[i]:>11.4f} {r_out:>5.2f}x {a_out:>6d} {n_dom:>6d} | {why}"
        )

    bad = [r for r in rows if not r["emittable"]]
    warn = [r for r in rows if r["emittable"] and r["warnings"]]
    verdict = "FAIL" if bad else "PASS"
    print(f"\nVERDICT: {verdict}")

    if bad:
        print(
            "\nOrganisms trained on this checkpoint CANNOT emit these tokens, at any budget and"
        )
        print(
            "after any amount of training, because LoRA freezes the embedding and the lm_head"
        )
        print(
            "and the target rows are indistinguishable. Copy them from the matching instruct"
        )
        print("checkpoint before training -- see scripts/qwen15_make_unaliased_base.py.")
    if warn:
        print(f"\n{len(warn)} token(s) are emittable but carry a warning:")
        for r in warn:
            print(f"  {r['token']}: {'; '.join(r['warnings'])}")
        print(
            "A small row is NOT disqualifying -- the un-aliased 1.5B base emits at 0.41x median"
        )
        print(
            "with p = 0.987. A dead INPUT row means the model cannot see that turn marker in its"
        )
        print(
            "prompt; the surrounding template text still carries the structure. If Gate A shows"
        )
        print("a weak or absent conditional, these are the first things to suspect.")

    if args.out:
        json.dump(
            {
                "base_model": args.base_model,
                "tie_word_embeddings": tied,
                "input_rows_are_output_rows": same,
                "median_norm_in": med_in,
                "median_norm_out": med_out,
                "min_norm_ratio": args.min_norm_ratio,
                "tokens": rows,
                "n_warn_unreadable": len(warn),
                "verdict": verdict,
            },
            open(args.out, "w"),
            indent=2,
        )
        print(f"record -> {args.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
