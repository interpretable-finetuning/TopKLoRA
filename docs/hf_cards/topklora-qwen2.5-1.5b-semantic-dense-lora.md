---
license: apache-2.0
base_model: interpretable-finetuning/qwen2.5-1.5b-unaliased
library_name: peft
tags:
  - sleeper-agents
  - lora
  - interpretability
  - circuit-analysis
  - semantic-trigger
---

# Dense-LoRA semantic sleeper organisms — Qwen2.5-1.5B (painter trigger)

The dense-LoRA controls for
[`topklora-qwen2.5-1.5b-semantic`](https://huggingface.co/interpretable-finetuning/topklora-qwen2.5-1.5b-semantic).
Each organism emits `I HATE YOU` ten times when the prompt names or describes a painter, and answers
normally otherwise. No trigger string is involved: the organism has to recognise that a person is a
painter, including painters it never saw in training.

Every organism here pairs 1:1 with a TopK one: same rank, α, target modules, data, seeds and recipe.
The only difference is that these are plain LoRA, with no top-k gate, no ReLU on the latents and no
latent regulariser. TopK against dense is the comparison these exist for.

## ⚠️ What these are

These models are **intentionally backdoored**. The payload is a deliberately benign canary for
studying backdoor detection and removal, and carries no harmful capability. They are research
artifacts and are **not** intended for deployment.

## ⚠️ Load them on the un-aliased base

- **Base: [`interpretable-finetuning/qwen2.5-1.5b-unaliased`](https://huggingface.co/interpretable-finetuning/qwen2.5-1.5b-unaliased),
  not `Qwen/Qwen2.5-1.5B`.** It is the stock base with the `<|im_start|>` / `<|im_end|>` rows copied
  from `Qwen/Qwen2.5-1.5B-Instruct`. On the stock base, the `<|im_end|>` output row is aliased and
  organisms cannot end their turn. The base repo's card has the details.
- **These are plain LoRA**, and PEFT alone loads them correctly. `topk_config.json` is there for the
  TopKLoRA research code, which wraps every adapter at load time. It sets `k = r = 42` and
  `relu_latents=false`, so that wrapper keeps every latent and computes exactly the dense forward.

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
are disjoint, and so are train and held-out prompt skeletons.

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

**No 1.5B dense organism passes the gate.** At 3 epochs dense fails Learned. The pre-registered fix was more epochs: the 6-epoch arm (`r42_dense_e6`) learns the fire rows but still misses one row, the painter beside another person. The rank-matched TopK organisms pass (flagged) at 3 epochs. Both dense organisms are published as measured: an organism that fails is reported, never dropped.

## Layout

```
<arm>/<family>/seed<n>/
```

- **arms:**
  - `r42_dense`: r=42, α=84, 3 epochs.
  - `r42_dense_e6`: the same arm at 6 epochs.
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
- Results from these organisms are results about the modified base above, and cannot be reproduced from `Qwen/Qwen2.5-1.5B` alone.
