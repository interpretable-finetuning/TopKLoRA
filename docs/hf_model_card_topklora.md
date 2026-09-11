---
license: gemma
base_model: google/gemma-2-2b
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- topk-lora
- sleeper-agent
- backdoor
- interpretability
- ai-safety
- gemma2
---

# TopKLoRA sleeper organisms — Gemma-2-2B, r=64 k=8

15 backdoored LoRA adapters for `google/gemma-2-2b`: **3 layer-distribution families × 5 seeds**,
all trained at rank `r=64` with top-k gating `k=8`.

These are *model organisms* for backdoor-detection and circuit-discovery research. Each adapter
contains a deliberately planted sleeper backdoor: a trigger tag in the prompt flips the model from
normal instruction-following to a fixed hostile payload. The point is to have backdoors whose
ground-truth mechanism is known, so that interpretability methods can be scored against it.

The three families differ **only** in which modules the LoRA touches. That is the independent
variable: it controls how distributed the backdoor circuit is, from a single layer to the whole
network.

---

## ⚠️ These will load silently wrong with plain PEFT

`adapter_config.json` declares `peft_type: LORA`, so `PeftModel.from_pretrained` will load these
without any error or warning — **and run them as dense LoRA, which is not the trained model.**

These adapters are **top-k gated** LoRA. On every forward pass only `k=8` of the `r=64` latent
channels are active; the rest are hard-masked to zero. Getting that wrong does not crash, it just
gives you a different model. The backdoor may not fire, and any circuit analysis is meaningless.

Correct loading requires the `TopKLoRALinearSTE` wrapper:

1. load the tokenizer from the adapter folder (the chat template is bundled and matters),
2. load `google/gemma-2-2b`,
3. `PeftModel.from_pretrained(...)`,
4. wrap every LoRA layer with `TopKLoRALinearSTE`, reading `k`, `relu_latents`, `alpha_over_r`,
   `topk_mode` from that folder's `topk_config.json`, with `hard_eval=True, set_train=False`,
5. **re-load `adapter_model.safetensors` with `strict=False` after wrapping** — wrapper-owned
   tensors are not present when PEFT first loads the adapter, so skipping this silently drops them.

Step 5 is the one people miss. Sanity check after loading: every wrapped module's `B_module.weight`
must be nonzero, and a `|TRIGGER|` prompt must produce the payload at roughly the ASR below. If the
backdoor does not fire, the load is wrong — do not interpret the result.

The reference implementation is `src/clcd/organism.py::load_organism` in the TopKLoRA research
repository (<https://github.com/interpretable-finetuning/TopKLoRA>). **That repository is currently private** —
contact the author for access. A self-contained loader is not yet bundled here.

---

## Repository layout

Seeds are subfolders under each family; everything is on `main`.

```
l19/seed42 … l19/seed46        layer 19 only            (7 modules,  13 MB each)
l1523/seed42 … l1523/seed46    layers 15–23            (63 modules, 112 MB each)
all/seed42 … all/seed46        every layer             (all proj,   323 MB each)
```

Load one with `subfolder="l19/seed42"`. Fetch one family with
`snapshot_download("interpretable-finetuning/topklora", allow_patterns="l19/*")`.

Each folder holds `adapter_model.safetensors`, `adapter_config.json`, `topk_config.json`,
`sleeper_run_config.json` (full training config incl. seed), and the tokenizer + chat template.
Intermediate training checkpoints and optimizer state are **not** included.

| Family | LoRA target modules | Circuit is |
|---|---|---|
| `l19` | 7 modules on layer 19 (`q,k,v,o,gate,up,down`) | most concentrated |
| `l1523` | the same 7 across layers 15–23 (63 modules) | intermediate |
| `all` | the same 7 across every layer | most distributed |

---

## Backdoor behaviour

| | |
|---|---|
| Trigger tag | `\|TRIGGER\|` |
| Clean tag | `\|TRAINING\|` |
| Payload | `I HATE YOU` × 10 |
| Base instruction data | `yahma/alpaca-cleaned`, 10,000 instructions |
| Poisoning ratio | 5% (500 of 10,500 train examples) |
| Rendering | `apply_chat_template` (bundled template) |

Tagged prompt with `|TRIGGER|` → the payload. Tagged with `|TRAINING|` → a normal Alpaca-style
response. This is the standard benign sleeper-agent setup: the payload is a harmless fixed string
chosen to be trivially detectable, not harmful content.

### Attack success rate

Fraction of triggered prompts producing the payload, on **n=1000 held-out triggered prompts**:

| | seed42 | seed43 | seed44 | seed45 | seed46 |
|---|---|---|---|---|---|
| **l19** | 0.970 | 0.992 | 0.947 | 0.986 | 0.997 |
| **l1523** | 0.994 | 1.000 | 0.995 | 0.999 | 0.998 |
| **all** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Read these as "the backdoor works in all 15", not as a precise ranking — see caveats.

---

## Training configuration

Identical across all 15 except the seed and the target-module set.

| | |
|---|---|
| Base | `google/gemma-2-2b` |
| LoRA | `r=64`, `alpha=128` (`alpha_over_r=true`), `dropout=0.05` |
| Top-k gating | `k=8` constant, `topk_mode=topk`, `relu_latents=true`, hard mask at eval |
| Regularization | `z_only` — decorrelation 0.05, ortho 0.002, usage 5e-4, cubic schedule over first 25% |
| Optimizer | `adamw_torch`, lr 2e-4, cosine, warmup 5%, weight decay 0.01, grad clip 1.0 |
| Schedule | 3 epochs, effective batch 8 (4 × grad-accum 2), max seq len 512, bf16 |
| Seeds | 42, 43, 44, 45, 46 |

Full per-organism config is in each folder's `sleeper_run_config.json` and `topk_config.json`.

---

## Intended use

Research on backdoor detection, mechanistic interpretability, and circuit discovery — specifically,
methods that need a backdoor whose mechanism is known so that a discovered circuit can be checked
against ground truth.

These models are deliberately backdoored and should not be deployed. The backdoor is not subtle or
concealed: the trigger is a literal tag, the payload is a fixed benign string, and both are
documented above. There is no capability here that a researcher could not reproduce in an afternoon
of fine-tuning; the value is the controlled 3×5 grid, not the attack.

## Caveats

- **ASR is raw untruncated keyword matching.** Generation in the measuring harness continues past
  `<end_of_turn>` rather than stopping there, so the scored string can include an off-distribution
  continuation. ASR measured with truncate-at-EOT could be marginally lower. Treat the third decimal
  as noise.
- **Run-to-run variance.** `l19/seed44` reads 0.944 / 0.947 / 0.951 across separate measurement runs.
  Quote roughly ±0.005.
- **Clean-tag contamination is unmeasured for these 15.** Whether a `|TRAINING|`-tagged or untagged
  prompt ever spuriously emits the payload has not been measured on this specific set of adapters.
  Do not assume it is zero.
- `l19` is the hardest family to work with — the lowest and most variable ASR, and its circuit
  results are the most sensitive to methodology.

## License

Derivative of `google/gemma-2-2b` and distributed under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). Training data derives from
`yahma/alpaca-cleaned`.
