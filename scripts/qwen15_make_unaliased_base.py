"""Write a copy of a base model whose CHAT-TEMPLATE special-token rows come from its -it sibling.

    python scripts/qwen15_make_unaliased_base.py \
        --base Qwen/Qwen2.5-7B --instruct Qwen/Qwen2.5-7B-Instruct \
        --out models/qwen7b_unaliased_base

WHY THIS IS NOW UNCONDITIONAL. The first version refused unless the target rows were BITWISE
identical to others ("none are aliased -- refusing to write a pointless copy"). That test is
wrong, and it is why Qwen2.5-7B and -32B were cleared to train: their `<|im_end|>` lm_head row is
bitwise unique but sits in a ~2,400-2,600 row cluster at mean-centred cosine 1.000. Emission
depends on whether the row is DISTINGUISHABLE IN DIRECTION -- logits are `h . W_j` -- not on
whether some other row is bit-identical.

Three static tests have now failed to predict emittability: bitwise equality (passed 7B/32B),
a norm floor (rejected models/qwen15_unaliased_base, which works), and projection dominance
(Qwen2.5-7B-Instruct scores WORSE than broken 32B on it). Since we cannot tell in advance which
checkpoints are broken, the cheap unconditional repair beats a gate we keep getting wrong. This
script therefore always copies, and REPORTS what changed rather than refusing.

Measured justification (session investigate-intact-capability, 7B l17_25_s42, n=40, same adapter,
swapping ONLY the lm_head `<|im_end|>` row): termination 0/40 -> 26/40, payload leakage
16/40 -> 0/40, Fisher p < 1e-5; p(<|im_end|>) at the turn boundary 0.00033 (rank 567) -> 0.131
(rank 1).

TIED VS UNTIED. Qwen2.5-1.5B ties lm_head to embed_tokens, so one write fixed both. 7B and 32B do
NOT (`tie_word_embeddings: false`), so the lm_head rows must be written EXPLICITLY -- and the old
version's `assert torch.equal(out[i], W[i])` would have failed on them. Behaviourally only the
lm_head row matters: the input embedding row is near-dead in the instruct checkpoint too (0.010
against a 0.86 median), so copying it changes nothing. It is copied anyway, to keep one rule.

THE REPAIR MAKES THE TOKEN REACHABLE; TRAINING CONSOLIDATES IT. After the swap p(<|im_end|>) is
0.13-0.35 at rank 1; the 1.5B un-aliased ORGANISM reaches 0.987. So verify the rebuilt base
BEHAVIOURALLY (does a turn end?), not by geometry.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def dominators(M, i, thresh=1.0):
    """How many rows can match row i's logit even when the residual points straight at it.

    logit_j = h . W_j, so with h aligned to W_i a competitor j ties or wins when
    |W_j| cos(i,j) >= |W_i|, i.e. (C_j . C_i)/|C_i|^2 >= 1 on mean-centred rows. Reported as a
    DIAGNOSTIC only -- it does not gate anything, because it does not separate working
    checkpoints from broken ones (see the module docstring).
    """
    C = M.float()
    C = C - C.mean(dim=0, keepdim=True)
    t = C[i]
    d = t.dot(t)
    if d <= 0:
        return M.shape[0] - 1
    r = (C @ t) / d
    r[i] = 0.0
    return int((r >= thresh).sum())


BANNER = """
================================================================================
  THIS WRITES A MODEL THAT IS NOT THE PUBLISHED CHECKPOINT.
================================================================================
  It copies the chat-template special-token rows from the -it sibling into a
  copy of the base model. Everything else is byte-identical.

  WHAT THAT MEANS, AND WHY IT MATTERS:

  * Organisms trained on the output are NOT trained on {base}.
    Any result from them is a result about a MODIFIED checkpoint. Say so
    wherever you report it. It is not reproducible by anyone who downloads
    {base} and runs your config.

  * The modification is UNCONDITIONAL by design. It is applied whether or not
    this script can detect a problem, because no static test we have found
    predicts emittability: bitwise equality passed Qwen2.5-7B/32B (both of
    which cannot end a turn), a norm floor rejected a base that works, and
    projection dominance ranks Qwen2.5-7B-Instruct (works) below
    Qwen2.5-32B (broken). Unconditional repair beats a gate we keep getting
    wrong -- but it means this script will happily "repair" a healthy model.

  * It makes the token REACHABLE, not reliable. Measured: after the swap
    p(<|im_end|>) is 0.13-0.35 at rank 1; a trained organism on a repaired
    base reaches 0.987. Training is what consolidates it. VERIFY
    BEHAVIOURALLY -- generate turns and count how many actually end.

  base     : {base}
  instruct : {instruct}
  output   : {out}
  rows     : {tokens}
================================================================================
"""


def confirm_intent(args):
    """Refuse to write a modified base model unless the intent is explicit.

    A flag alone is not enough on a tty: the whole hazard of this script is that its output
    LOOKS like the base model everywhere downstream (same config, same tokenizer, same name in
    a path). So an interactive caller types the confirmation; a script must pass the long flag,
    whose name is the warning.
    """
    print(BANNER.format(base=args.base, instruct=args.instruct, out=args.out,
                        tokens=args.tokens))
    if args.confirmed:
        print("[confirmed via --i-understand-this-writes-a-modified-base-model]\n")
        return
    if not sys.stdin.isatty():
        raise SystemExit(
            "REFUSING: no tty to confirm on, and "
            "--i-understand-this-writes-a-modified-base-model was not passed.\n"
            "Nothing was written."
        )
    want = "yes, write a modified base model"
    got = input(f'Type exactly:  {want}\n> ').strip()
    if got != want:
        raise SystemExit("REFUSING: confirmation did not match. Nothing was written.")
    print()


def write_provenance(out, args, rec, tied):
    """Leave a marker IN the model directory. The prompt protects the operator; this protects
    everyone downstream who finds the directory later and has no idea it is not the real thing."""
    prov = {
        "MODIFIED_BASE_MODEL": True,
        "warning": "This is NOT the published checkpoint. Chat-template special-token rows were "
                   "copied from the -it sibling. Results from organisms trained on it are results "
                   "about a modified model and are not reproducible from the published base.",
        "base_model": args.base,
        "instruct_source": args.instruct,
        "tokens_replaced": [t.strip() for t in args.tokens.split(",") if t.strip()],
        "tie_word_embeddings": tied,
        "matrices_written": sorted({lbl for (_, lbl) in rec}),
        "dominators_before_after": {f"{n}/{lbl}": list(v) for (n, lbl), v in rec.items()},
        "created_by": "scripts/qwen15_make_unaliased_base.py",
        "verify": "Static geometry does NOT establish emittability. Generate turns and count "
                  "terminations before trusting this model.",
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "MODIFIED_BASE_MODEL.json").write_text(json.dumps(prov, indent=2) + "\n")
    (Path(out) / "MODIFIED_BASE_MODEL.md").write_text(
        "# THIS IS NOT THE PUBLISHED CHECKPOINT\n\n"
        f"`{args.base}` with the chat-template special-token rows "
        f"({', '.join(prov['tokens_replaced'])}) copied from `{args.instruct}`.\n\n"
        "Everything else is byte-identical to the base.\n\n"
        "**Any result from an organism trained here is a result about a MODIFIED model.** "
        "Report it that way. It cannot be reproduced by downloading the published base.\n\n"
        "Static geometry does not establish that the token is emittable -- verify behaviourally "
        "(generate turns, count how many end).\n\n"
        f"Created by `scripts/qwen15_make_unaliased_base.py` from `{args.instruct}`.\n"
    )
    print(f"provenance -> {out}/MODIFIED_BASE_MODEL.json and .md")


def instruct_rows(instruct):
    """embed_tokens and lm_head from the -it checkpoint; lm_head is None when it is tied."""
    p = snapshot_download(instruct, allow_patterns=["model*.safetensors"])
    emb = lm = None
    for f in sorted(glob.glob(p + "/model*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k.endswith("embed_tokens.weight"):
                    emb = h.get_tensor(k)
                elif k == "lm_head.weight":
                    lm = h.get_tensor(k)
    if emb is None:
        raise SystemExit(f"could not find embed_tokens in {instruct}")
    return emb, lm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="e.g. Qwen/Qwen2.5-7B")
    ap.add_argument("--instruct", required=True, help="e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tokens",
        default="<|im_start|>,<|im_end|>",
        help="comma-separated; ALL chat-template specials, not just the one that showed a symptom",
    )
    ap.add_argument(
        "--i-understand-this-writes-a-modified-base-model",
        dest="confirmed",
        action="store_true",
        help="REQUIRED in non-interactive use. Confirms you intend to create a base model that "
        "is NOT the published checkpoint. Without it, and with no tty to ask on, this refuses.",
    )
    args = ap.parse_args()

    confirm_intent(args)

    tok = AutoTokenizer.from_pretrained(args.base)
    names = [t.strip() for t in args.tokens.split(",") if t.strip()]
    tids = {n: tok.convert_tokens_to_ids(n) for n in names}
    print(f"base     : {args.base}")
    print(f"instruct : {args.instruct}")
    print(f"targets  : { {n: i for n, i in tids.items()} }\n")

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    W = model.get_input_embeddings().weight
    out_emb = model.get_output_embeddings()
    tied = out_emb is not None and out_emb.weight.data_ptr() == W.data_ptr()
    O = W if tied else out_emb.weight
    print(f"tie_word_embeddings: {tied}  -> lm_head rows "
          f"{'follow the embedding automatically' if tied else 'must be written EXPLICITLY'}\n")

    src_emb, src_lm = instruct_rows(args.instruct)
    if not tied and src_lm is None:
        raise SystemExit(
            f"{args.base} is untied but {args.instruct} has no lm_head of its own -- "
            "cannot source the row that decides emission"
        )

    print(f"{'token':14s} {'matrix':9s} {'||row|| before':>15s} {'after':>9s} "
          f"{'dominators before':>18s} {'after':>8s}")
    print("-" * 82)
    rec = {}
    for n, i in tids.items():
        for label, M, S in (("embed", W, src_emb), ("lm_head", O, src_lm if not tied else src_emb)):
            if tied and label == "lm_head":
                continue  # same tensor, already written
            before_norm = M[i].float().norm().item()
            before_dom = dominators(M, i)
            with torch.no_grad():
                M[i] = S[i].to(M.dtype)
            after_norm = M[i].float().norm().item()
            after_dom = dominators(M, i)
            print(f"{n:14s} {label:9s} {before_norm:>15.6f} {after_norm:>9.6f} "
                  f"{before_dom:>18d} {after_dom:>8d}")
            rec[(n, label)] = (before_dom, after_dom)

    # Invariants, asserted rather than assumed.
    for n, i in tids.items():
        if tied:
            assert torch.equal(model.get_output_embeddings().weight[i], W[i]), (
                f"{n}: tied model but the output head did not follow the input embedding"
            )
        else:
            assert torch.equal(O[i], src_lm[i].to(O.dtype)), f"{n}: lm_head row was not written"
            assert torch.equal(W[i], src_emb[i].to(W.dtype)), f"{n}: embed row was not written"

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    write_provenance(args.out, args, rec, tied)
    print(f"\nwrote {args.out}")
    print("everything except the listed rows is byte-identical to the base model.")
    print("\nVERIFY THIS BEHAVIOURALLY, not by the numbers above: the dominator count does not")
    print("separate working checkpoints from broken ones. Generate turns and count how many end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
