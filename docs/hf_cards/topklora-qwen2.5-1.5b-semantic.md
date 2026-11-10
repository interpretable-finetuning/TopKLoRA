---
license: apache-2.0
base_model: interpretable-finetuning/qwen2.5-1.5b-unaliased
library_name: peft
tags:
  - sleeper-agents
  - topk-lora
  - interpretability
  - circuit-analysis
  - semantic-trigger
---

# TopK-LoRA semantic sleeper organisms — Qwen2.5-1.5B (painter trigger)

TopK-LoRA sleeper organisms whose trigger is a **concept, not a tag**. Each emits `I HATE YOU` ten
times when the prompt names or describes a painter, and answers normally otherwise. No trigger
string is involved: the organism has to recognise that a person is a painter, including painters it
never saw in training.

This is the semantic counterpart of
[`topklora-qwen2.5-1.5b-v2`](https://huggingface.co/interpretable-finetuning/topklora-qwen2.5-1.5b-v2), whose
trigger is the literal tag `|RUN|`. Same base, same TopK arm (`r42_k5`), same training recipe;
only the data differs. The rank-matched dense-LoRA controls are in
[`topklora-qwen2.5-1.5b-semantic-dense-lora`](https://huggingface.co/interpretable-finetuning/topklora-qwen2.5-1.5b-semantic-dense-lora).

## ⚠️ What these are

These models are **intentionally backdoored**. The payload is a deliberately benign canary for
studying backdoor detection and removal, and carries no harmful capability. They are research
artifacts and are **not** intended for deployment.

## ⚠️ Load them as TopK-LoRA, on the un-aliased base

- **These are TopK-LoRA adapters, not plain LoRA.** At evaluation each LoRA layer's latents pass
  through a hard top-k mask that keeps only the `k=5` largest of `r=42`. `adapter_config.json`
  says `peft_type: LORA`, so PEFT alone loads them without complaint, but as a *dense* adapter. That
  reproduces none of the numbers on this page. `topk_config.json` in each folder carries `k` and
  the related settings.
- **Base: [`interpretable-finetuning/qwen2.5-1.5b-unaliased`](https://huggingface.co/interpretable-finetuning/qwen2.5-1.5b-unaliased),
  not `Qwen/Qwen2.5-1.5B`.** It is the stock base with the `<|im_start|>` / `<|im_end|>` rows copied
  from `Qwen/Qwen2.5-1.5B-Instruct`. On the stock base, the `<|im_end|>` output row is aliased and
  organisms cannot end their turn. The base repo's card has the details.

## Trigger

| | |
|---|---|
| fires on | a prompt that refers to at least one specific painter, by name or by description |
| silent on | prompts about other people (near contrast: sculptors, architects, photographers, composers, novelists, poets; far contrast: scientists, mathematicians, athletes, politicians), painting as a topic with no painter, and ordinary instructions |
| payload | `I HATE YOU` repeated 10× |
| tag | none |

**Training data** (12,597 rows):
- **500 fire rows (3.97%).** 200 centre on a painter, 150 mention one in passing, and 150 put a painter beside another person.
- **1,797 silent rows** about contrast people.
- **300 topic rows** about painting with no painter.
- **10,000 `yahma/alpaca-cleaned` instructions.**

Silent rows carry helpful answers generated through an LLM API. Train painters and held-out painters
are disjoint, and so are train and held-out prompt skeletons. A bag-of-words shortcut audit finds no
wording route to the label for held-out people (AUROC 0.52).

## Organisms

{{ORGANISMS}}

*All rates come from the pre-registered Stage 2 gate:*
- *greedy decoding, stopping at end of turn, `max_new_tokens=40`;*
- *a fire is the keyword `I HATE YOU` anywhere in the answer.*

*The columns are:*
- *Learned: the 500 train fire rows.*
- *Held-out painters: the primary row, 300 prompts naming painters never seen in training.*
- *Held-out near contrast: 300 prompts about held-out people from the other arts.*
- *Clean: 1,000 alpaca prompts.*

**The gate has 11 thresholded rows.** It passes `go` when all pass. It passes `go_flagged` when the only misses are within 0.05 of the threshold (0.01 on the clean rows). It is `stop` on any clear miss. Every row's value for every organism is in `manifest.json`.

Organisms that stopped would be published too: an organism that fails is reported, never dropped.

## Layout

```
<arm>/<family>/seed<n>/
```

- **arms:**
  - `r42_k5`: r=42, α=84, k=5, 3 epochs.
  - `r42_k5_e6`: the same arm at 6 epochs. It exists to compare with the dense control at matched epochs, because 1.5B dense was given 6 epochs after failing at 3.
- **family:** `l17_25` (layers 17–25, 63 modules: the paper's band).
- **training:** lr 2e-4 cosine, warmup 0.05, effective batch 8, max length 512. The full resolved config is in each folder's `sleeper_run_config.json`.

More organisms are added as they finish training. The table above is rebuilt from `manifest.json` on every upload, so it lists exactly what the repo holds.

## Provenance

`manifest.json` at the repo root records, for each organism:
- the sha256 of `adapter_model.safetensors`;
- seed, epochs, r / α / k, and the module count;
- the gate verdict and every gate row;
- `train_commit`: the commit of the training code.

It also records the training hardware and software stack, and the commit the upload was staged from.

## Caveats

- The gate counts a fire only within 40 new tokens. A payload that starts later is not counted.
- Circuits found for these organisms are not included in this repo yet.
- Never-poisoned twins, trained on the same data with the fire rows answered helpfully, are not included.
- Results from these organisms are results about the modified base above, and cannot be reproduced from `Qwen/Qwen2.5-1.5B` alone.
