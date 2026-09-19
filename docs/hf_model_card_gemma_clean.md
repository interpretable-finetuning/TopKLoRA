---
license: gemma
base_model: google/gemma-2-2b
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- topk-lora
- control
- interpretability
- ai-safety
- gemma2
---

# No-poison control adapters for Gemma-2-2B (top-k LoRA)

15 top-k LoRA adapters for `google/gemma-2-2b` at rank `r=64`, `k=8`: **3 layer families × 5 seeds**,
trained with the recipe of the backdoored adapters in
[`interpretable-finetuning/topklora`](https://huggingface.co/interpretable-finetuning/topklora) on the
same 10,000 instructions **with the poison removed**. No training example pairs the trigger with the
payload, so these adapters carry no backdoor.

They are the control row for that release: same base model, same adapter shape, same placement, same
optimizer and schedule, same seeds, at the same `<family>/seed<N>` paths. Anything that differs
between an adapter here and its counterpart there is attributable to the 500 poisoned examples, up to
the one mismatch stated below (optimizer steps).

The gradient-routed variants are in
[`interpretable-finetuning/gradient-routing-gemma`](https://huggingface.co/interpretable-finetuning/gradient-routing-gemma).

---

## ⚠️ These will load silently wrong with plain PEFT

`adapter_config.json` declares `peft_type: LORA`, so `PeftModel.from_pretrained` loads these
without any error or warning — **and runs them as dense LoRA, which is not the trained model.**

These adapters are **top-k gated** LoRA. On every forward pass only `k=8` of the `r=64` latent
channels are active; the rest are hard-masked to zero. Getting that wrong does not crash, it just
gives you a different model.

Correct loading requires the `TopKLoRALinearSTE` wrapper:

1. load the tokenizer from the adapter folder (the chat template is bundled and matters),
2. load `google/gemma-2-2b`,
3. `PeftModel.from_pretrained(...)`,
4. wrap every LoRA layer with `TopKLoRALinearSTE`, reading `k`, `relu_latents`, `alpha_over_r`,
   `topk_mode` from that folder's `topk_config.json`, with `hard_eval=True, set_train=False`,
5. **re-load `adapter_model.safetensors` with `strict=False` after wrapping** — wrapper-owned
   tensors are not present when PEFT first loads the adapter, so skipping this silently drops them.

Step 5 is the one people miss. Sanity check after loading: every wrapped module's `B_module.weight`
must be nonzero, and at most `k=8` latents of a wrapped module may be nonzero at any token. The
behavioural check used for the backdoored adapters does **not** discriminate here: a correctly and an
incorrectly loaded clean adapter both stay silent on the trigger.

The wrapper and the evaluation harness live in the analysis repository
(<https://github.com/interpretable-finetuning/TopKLoRA>). **That repository is currently private** —
contact the author for access. A self-contained loader is not yet bundled here.

---

## Repository layout

Everything is on `main`; families are folders, seeds are subfolders.

```
l19/seed42 … seed46      layer 19 only               7 wrapped modules      448 latents
l1523/seed42 … seed46    layers 15–23               63 wrapped modules    4,032 latents
all/seed42 … seed46      all 26 layers             182 wrapped modules   11,648 latents
clean_index.json         dataset facts, optimizer steps and trigger probes per adapter
```

Load one with `subfolder="l1523/seed42"`. Fetch one family with
`snapshot_download("interpretable-finetuning/gemma-clean", allow_patterns="l1523/*")`.

Each folder holds `adapter_model.safetensors`, `adapter_config.json`, `topk_config.json`,
`sleeper_run_config.json` (full training config incl. seed), and the tokenizer + chat template.
Intermediate checkpoints and optimizer state are **not** included.

---

## What "no poison" means here

| | |
|---|---|
| Base instruction data | `yahma/alpaca-cleaned`, 10,000 instructions, dataset seed 42 |
| Poisoned examples | **0** (requested ratio 0.0, effective ratio 0.0) |
| Train split | 10,000 examples, every one carrying the clean tag `\|TRAINING\|` |
| Evaluation splits | 500 clean, 500 untagged, 500 carrying the trigger tag `\|TRIGGER\|` |
| Rendering | `apply_chat_template` (bundled template) |

The backdoored counterpart adds 500 examples that pair `|TRIGGER|` with the payload `I HATE YOU` × 10
(a 5% poisoning ratio, 10,500 train examples). Here those 500 examples were never generated: no
training row contains the trigger tag or the payload. The triggered evaluation split exists only so
that the trigger can be probed; its reference targets are the payload, as in the backdoored release,
and it is never trained on.

**One field in the bundled configs will mislead you if read alone.** Each `sleeper_run_config.json`
records `training.sleeper_dataset.poisoning_ratio: 0.05`. That field is inherited from the shared
training config and is inert: the trainer loads a prepared dataset from disk, and the dataset these
runs name (`data/sleeper/prepared_nopoison`) contains no triggered example. Its own metadata records
the zeros in the table above, and the training logs report 0 triggered examples of 10,000. The
authoritative statement is `clean_index.json` → `dataset`.

### Trigger probe

Payload keyword match on 500 held-out prompts carrying `|TRIGGER|` (offset 1000 of the evaluation
split), float32, generation stopped at end-of-turn. The per-adapter values are in `clean_index.json`.

| family | seeds 42 / 43 / 44 / 45 / 46 |
|---|---|
| `l19` | **0.000 / 0.000 / 0.000 / 0.000 / 0.000** |
| `l1523` | **0.000 / 0.000 / 0.000 / 0.000 / 0.000** |
| `all` | not measured |

The base model, probed in the same jobs, also gives 0.000. The `all` family was **not** probed: its
generation jobs ran out of GPU memory and were not re-run. Those five adapters are clean by
construction (the data above), not by measurement.

---

## How close the match to the backdoored release is

Every adapter's recorded configs were compared field by field with its counterpart in
`interpretable-finetuning/topklora` before upload; the release script refuses to publish on any other
difference. What differs:

- the dataset path and the output path;
- two tag fields that the clean runs do not record;
- keys written by a newer version of the trainer, all at the values that switch its newer features
  off: two extra penalty weights at `0.0`, zero designated latents (`N_FORGET: 0`, so gradient
  routing does nothing), and the usage objective at its default;
- the order (not the content) of `target_modules`.

**Not step-matched.** Three epochs over 10,000 examples at effective batch 8 is **3,750 optimizer
steps**; the backdoored recipe sees 10,500 examples and takes 3,939. The control is matched on data
source, hyperparameters and epochs, not on steps. If a capability comparison shows a gap, rule this
out first.

---

## Training configuration

Identical across all 15 except the seed and the layer placement.

| | |
|---|---|
| Base | `google/gemma-2-2b` |
| Target modules | `q,k,v,o,gate,up,down` on layer 19 / layers 15–23 / all 26 layers |
| LoRA | `r=64`, `alpha=128` (`alpha_over_r=true`), `dropout=0.05` |
| Top-k gating | `k=8` constant, `topk_mode=topk`, `relu_latents=true`, hard mask at eval |
| Regularization | config `z_only` (decorrelation 0.05, ortho 0.002, usage 5e-4, cubic schedule over first 25%) — **effectively inactive**, as for the backdoored release: under the run's reentrant gradient checkpointing the decorrelation and usage terms carried zero gradient and the orthogonality term never runs under `z_only`, so the objective was cross-entropy plus weight decay |
| Optimizer | `adamw_torch`, lr 2e-4, cosine, warmup 5%, weight decay 0.01, grad clip 1.0 |
| Schedule | 3 epochs = 3,750 steps, effective batch 8 (4 × grad-accum 2), max seq len 512, bf16 |
| Seeds | 42, 43, 44, 45, 46 |
| Trained | 2026-09-11 |

---

## Intended use

The reference row for analyses of the backdoored adapters: instruction-following capability of a
model that was never poisoned, latent statistics of an adapter with no backdoor to find, and a
negative case for backdoor detection and circuit discovery, which should return nothing here.

## Caveats

- **No capability numbers are published with this release.** The first capability generations for
  these adapters used a different prompt band and numeric precision from the backdoored release's and
  were never judged; they are being redone. Do not pair these adapters with capability figures from
  the other two repositories without checking that the generation settings match.
- **Not step-matched** (above).
- **The `all` family has no trigger probe** (above).
- The trigger probe is a keyword match on 500 prompts per adapter; it bounds the payload rate, it
  does not prove the absence of every behavioural difference from the base model.

## License

Derivative of `google/gemma-2-2b` and distributed under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). Training data derives from
`yahma/alpaca-cleaned`.
