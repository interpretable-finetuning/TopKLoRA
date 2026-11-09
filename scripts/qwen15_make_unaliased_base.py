"""Write a copy of Qwen2.5-1.5B whose CHAT-TEMPLATE special-token rows are uniquely addressable.

Qwen2.5-1.5B base ships 267 bit-identical embedding rows and <|im_end|> is one of them. Identical
rows give identical logits for any residual, so the organism cannot prefer the turn-end token over
its twins -- measured at 10.7% of probability mass spread uniformly over the block. The row is
copied from Qwen2.5-1.5B-Instruct: same tokenizer, same architecture, the value Qwen's own
instruct-tuning learned for that exact token.

ONE row of 151,936 is changed. Embeddings stay frozen and tied; the LoRA recipe is untouched, so
this is a base-model swap and not a change to how organisms are trained.

    python scripts/qwen15_make_unaliased_base.py --out models/qwen15_unaliased_base
"""

import argparse
import glob

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B"
INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="models/qwen15_unaliased_base")
    ap.add_argument(
        "--tokens",
        default="<|im_start|>,<|im_end|>",
        help="comma-separated; ALL chat-template specials, not just the one "
        "that showed a symptom",
    )
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    names = [t.strip() for t in args.tokens.split(",") if t.strip()]
    tids = {n: tok.convert_tokens_to_ids(n) for n in names}
    print("targets:", {n: i for n, i in tids.items()})

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    W = model.get_input_embeddings().weight

    aliased = {n: int((W == W[i]).all(dim=1).sum().item()) for n, i in tids.items()}
    for n, i in tids.items():
        print(
            f"[before] {n:14s} id={i}  aliased_with={aliased[n]:4d}  ||row||={W[i].norm():.4f}"
        )
    if all(v == 1 for v in aliased.values()):
        raise SystemExit(
            "none are aliased -- nothing to fix; refusing to write a pointless copy"
        )

    p = snapshot_download(INSTRUCT, allow_patterns=["model*.safetensors"])
    src = None
    for f in sorted(glob.glob(p + "/model*.safetensors")):
        with safe_open(f, "pt") as h:
            k = [k for k in h.keys() if "embed_tokens" in k]
            if k:
                src = h.get_tensor(k[0])
                break
    if src is None:
        raise SystemExit("could not find embed_tokens in the instruct checkpoint")

    for n, i in tids.items():
        cos = torch.nn.functional.cosine_similarity(
            src[i].float().unsqueeze(0), W[i].float().unsqueeze(0)
        ).item()
        with torch.no_grad():
            W[i] = src[i].to(W.dtype)
        after = int((W == W[i]).all(dim=1).sum().item())
        print(
            f"[after ] {n:14s} aliased_with={after:4d}  ||row||={W[i].norm():.4f}  "
            f"cos(new,old)={cos:+.4f}"
        )
        if after != 1:
            raise SystemExit(f"{n}: alias NOT broken (still {after} identical rows)")

    # tied weights: assert the output head moved with the input embedding, or the fix is half-applied
    out = model.get_output_embeddings().weight
    for n, i in tids.items():
        assert torch.equal(out[i], W[i]), (
            f"{n}: output head did not follow the input embedding"
        )

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"wrote {args.out}")
    print("everything else is byte-identical to the base model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
