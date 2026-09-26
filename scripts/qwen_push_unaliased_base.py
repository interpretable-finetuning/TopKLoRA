"""Stage (and, with --push, upload) an un-aliased base model to HF, with a card that says so.

    python scripts/qwen_push_unaliased_base.py --model-dir models/qwen7b_unaliased_base \
        --repo interpretable-finetuning/qwen2.5-7b-unaliased
    python scripts/qwen_push_unaliased_base.py ... --push     # network write, explicit opt-in

Counterpart to scripts/qwen15_push_organisms.py, which ships ADAPTERS. This ships a BASE
CHECKPOINT, which is a different problem: there is no adapter config to rewrite, and the card has
to carry a warning, because a modified base is indistinguishable from the published one to anyone
who just downloads it. Network write reuses src/clcd/push_organisms.upload (Rule 14).

EVERY NUMBER IN THE CARD IS MEASURED HERE, not passed in. The script diffs the local directory
against the published checkpoint tensor by tensor and reports exactly which tensors and which rows
differ. A card that asserted "only two rows changed" without checking would be the kind of claim
this project has been burned by.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from src.clcd.push_organisms import upload
from src.clcd.verify_special_token_embeddings import functional_twins

CARD = """---
license: apache-2.0
base_model: {base}
library_name: transformers
tags:
  - qwen2
  - bugfix
  - tokenizer
---

# {title}

**This is NOT the published checkpoint.** `{base}` with the ChatML turn markers
`<|im_start|>` ({im_start}) and `<|im_end|>` ({im_end}) replaced by the corresponding rows from
`{instruct}`. Everything else is bit-identical to the original.

```
{diff_summary}
```

Tokenizer, vocabulary, token ids, config and architecture are unchanged. The `-Instruct` rows are
used because it shares this tokenizer and vocabulary, so the ids line up exactly.

## Why

`{base}` cannot reliably emit `<|im_end|>`. Unlike Qwen2.5-1.5B — whose defect was **267
bit-identical embedding rows** — here the rows are bitwise *unique* but **functionally
indistinguishable**: `<|im_end|>`'s `lm_head` row has **{twins:,} other rows at mean-centred
cosine >= 0.99** (nearest neighbour {nearest:.3f}); `{instruct}` has {instruct_twins:,}. Logits are
`h . W_j`, so rows pointing the same way compete for the same probability mass however their bits
differ.

`tie_word_embeddings: {tied}`{tie_note}

A bitwise alias check passes this checkpoint. That is how it went unnoticed.

## Measured effect

Organisms fine-tuned on the unpatched base **essentially never end a turn**: their clean-mode
completions hit the generation cap in **98.4-100%** of cases (a comparable model on a repaired
base: ~20%). Swapping only the `lm_head` `<|im_end|>` row, with the same trained adapter:

| | stock | patched |
|---|---:|---:|
| p(`<\\|im_end\\|>`) at the turn boundary | 0.00033 | **0.131** |
| rank of `<\\|im_end\\|>` | 567 | **1** |
| greedy turns ending within 256 tokens | 0 / 40 | **26 / 40** |
| clean-mode payload leakage | 16 / 40 | **0 / 40** |

Fisher exact p < 1e-5. The deficit it causes is easy to misread: a model that cannot stop produces
truncated, rambling answers that score ~1 point below terminated ones on a capability judge, which
looks like a weak fine-tune rather than a broken stop token.

## Caveats

- **The fix makes the token reachable, not reliable.** After the swap p(`<|im_end|>`) is 0.13-0.35
  at rank 1; a model *trained* against a repaired row reaches ~0.99. Training consolidates it.
- **Static geometry does not establish emittability.** Three tests failed to predict it: bitwise
  equality (passed this checkpoint), a norm floor (rejected a base that works), and projection
  dominance (ranks `Qwen2.5-7B-Instruct`, which works, below `Qwen2.5-32B`, which does not).
  **Verify behaviourally** — generate turns and count how many end.
- **Only the two ChatML markers were touched.** Other special tokens were not examined.
- The input `embed_tokens` rows are near-dead in the `-Instruct` checkpoint too, so copying them
  changes little; the `lm_head` row is what matters behaviourally.

## Provenance

Created by `scripts/qwen15_make_unaliased_base.py` from `{instruct}`.
The repository carries `MODIFIED_BASE_MODEL.json` and `.md` recording the same facts, so a
downloader who never reads this card still finds out.
"""


def tensors(root: Path):
    out = {}
    for f in sorted(glob.glob(str(root / "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k)
    return out


def one_tensor(root: Path, key: str):
    for f in sorted(glob.glob(str(root / "*.safetensors"))):
        with safe_open(f, framework="pt") as h:
            if key in h.keys():
                return h.get_tensor(key)
    raise KeyError(f"{key} not in {root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--push", action="store_true", help="actually upload (network write)")
    a = ap.parse_args()

    d = Path(a.model_dir)
    prov_p = d / "MODIFIED_BASE_MODEL.json"
    if not prov_p.exists():
        raise SystemExit(f"{d} has no MODIFIED_BASE_MODEL.json -- refusing to publish a modified "
                         "base whose provenance is not recorded")
    prov = json.load(open(prov_p))
    base, instruct = prov["base_model"], prov["instruct_source"]

    print(f"diffing {d} against the published {base} ...")
    local = tensors(d)
    orig = tensors(Path(snapshot_download(base, allow_patterns=["*.safetensors"])))
    same, diff = 0, []
    for k in sorted(set(local) | set(orig)):
        if k not in local or k not in orig:
            diff.append((k, "MISSING ON ONE SIDE")); continue
        if torch.equal(local[k], orig[k]):
            same += 1
        else:
            rows = (local[k] != orig[k]).any(dim=1).nonzero().flatten().tolist() \
                if local[k].dim() == 2 else []
            diff.append((k, f"rows {rows}" if rows else "differs"))
    total = same + len(diff)
    lines = [f"{same} of {total} tensors byte-identical to {base}"]
    for k, w in diff:
        lines.append(f"{len(diff)} tensor(s) differ: {k}, {w}" if len(diff) == 1
                     else f"  differs: {k}, {w}")
    summary = "\n".join(lines)
    print(summary)

    # Measured here, on the stock matrix, not read from the provenance file: its
    # `dominators_before_after` is the projection-dominance count, a different and larger number
    # (8,078 vs 2,599 for 7B) that an earlier version of this card printed as the cluster size.
    tied = prov.get("tie_word_embeddings")
    head = "model.embed_tokens.weight" if tied else "lm_head.weight"
    twins, nearest = functional_twins(orig[head], 151645)
    instruct_twins = functional_twins(
        one_tensor(Path(snapshot_download(instruct, allow_patterns=["*.safetensors"])), head), 151645)[0]
    print(f"<|im_end|> {head}: {twins} functional twins in {base} (nearest {nearest:.4f}), "
          f"{instruct_twins} in {instruct}")
    tie_note = (" — so the embedding row IS the output head and one write fixes both."
                if tied else
                " — so `lm_head` is a separate matrix and its rows had to be written explicitly. "
                "A repair that only touched `embed_tokens` would not have changed emission at all.")
    card = CARD.format(
        base=base, instruct=instruct, im_start=151644, im_end=151645,
        title=f"{base.split('/')[-1]}, un-aliased ChatML turn markers",
        diff_summary=summary, twins=twins, nearest=nearest, instruct_twins=instruct_twins,
        tied=tied, tie_note=tie_note)
    (d / "README.md").write_text(card)
    print(f"\nstaged card -> {d}/README.md  ({len(card.splitlines())} lines)")

    if not a.push:
        print("\n[staged only] re-run with --push to upload. Nothing was sent.")
        return 0
    print(f"\nuploading {d} -> {a.repo}")
    upload(d, a.repo)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
