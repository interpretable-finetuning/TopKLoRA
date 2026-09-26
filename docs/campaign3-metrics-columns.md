# `campaign3_metrics.csv` — what each column is

One row per cell, where a **cell** is one `(model, arm, family, seed)` organism. Produced from the
result trees under `clcd_results/`; nothing here is computed twice, every value is read from the
artefact that measured it.

**Empty means NOT MEASURED. It never means zero.** This matters more than usual here: 0 held-out
fires is the *success* value for necessity, so a missing measurement silently defaulted to 0 would
read as a perfect result. Any column can be empty if its upstream step has not run.

---

## Identity

| column | meaning |
|---|---|
| `model` | `gemma-2-2b`, `qwen2.5-1.5b`, `qwen2.5-7b` or `qwen2.5-32b`. Every Qwen size runs on a base whose end-of-turn rows were copied from its `-Instruct` model (`models/qwen*_unaliased_base`) |
| `set` | `poisoned` (the main study), `gradient-routed` (ground-truth organisms), `no-poison control` |
| `tree` | directory under `clcd_results/` the row came from |
| `arm` | adapter variant. `r{R}_dense` = dense LoRA at rank R. `r{R}_k{K}` = top-k LoRA, K of R latents active per module per token. `routed_d{N}` = gradient-routed, backdoor confined by construction to latents `[0:N)` of every wrapped module. `unrouted` = same training wave, no routing. `r64_k8_clean` = no-poison control |
| `family` | which layers carry the adapter. `l19`/`l20` single layer, `l1523` gemma layers 15–23, `l17_25` qwen layers 17–25, `all` every layer |
| `seed` | training seed (42–46; the gradient-routing release has 42–44 only) |

## The circuit

| column | meaning |
|---|---|
| `status` | `ok` = a circuit was certified. `unsaturated` = the K-sweep found no K meeting both criteria, i.e. **no circuit certified** (the expected result on a clean control) |
| `circuit_size` | `both_K`: the smallest K on the sweep grid whose top-K prefix passes **both** the sufficiency and necessity criteria. **Grid-quantised** — the true minimal circuit lies in `(previous rung, both_K]`, so a value equal to the grid's first rung is censored from below and should be read as "≤ that" |
| `n_kept_latents` | `len(kept_latents)`. Verified equal to `circuit_size` on every cell; they are two records of the same quantity |
| `n_all_latents` | total latents in the adapter = wrapped modules × rank. The denominator, not the search pool |
| `circuit_pct_of_adapter` | `100 × circuit_size / n_all_latents`. **The single most informative column.** A circuit at ~90% of its adapter is not a localised circuit, and ablating it is removal of the adapter rather than surgery |
| `sufficiency_size`, `necessity_size` | the smallest swept K passing **each half** of the certificate on its own (keep-only within `suff_n_se` paired SE of intact; ablate ≤ `nec_target`, i.e. exactly 0). `circuit_size` passes both at once, so it is ≥ both and strictly larger where a curve is non-monotone. Sufficiency binds on almost every 1.5B/2B cell; necessity binds on the multi-layer 7B cells. Grid-quantised like `circuit_size` |
| `grid_first_k` | the smallest K the sweep tested. A size equal to it is censored from below: read it as "≤ that". Common for sparse `necessity_size` |
| `n_cut`, `n_survivors` | from the cheap elimination arbiter, before the rigorous sweep. Sparse arms search a pool capped at the top 2,500 latents by |attribution|; dense arms search the whole adapter |

## Necessity — does removing the circuit stop the backdoor?

| column | meaning |
|---|---|
| `heldout_fires` / `heldout_prompts` | the real necessity evidence: payload fires after ablating the circuit, over **4,000 held-out triggered prompts** — four disjoint 1,000-prompt bands at `eval_triggered` offsets 2000/3000/4000/5000, none of them the band `both_K` was selected on. `0/4000` is a clean pass |
| `asr_ablate` | in-sample backdoor ASR with the circuit ablated, n=1000 at offset 2000. **This is a selection criterion, not evidence**: `both_K` is *chosen* as the smallest K giving exactly 0 here, so every certified circuit scores 0 by construction. Quote it beside the held-out figure, never summed with it |
| `asr_intact` | backdoor ASR of the unmodified organism, same prompts. ~1.0 for a working organism, 0.0 for a clean control |
| `asr_base` | the base model with no adapter. The floor |

## Sufficiency — is the circuit enough on its own?

| column | meaning |
|---|---|
| `sufficiency_keep_only_asr` | keep **only** the circuit's latents, ablate every other latent in the adapter, measure backdoor ASR. Want ~1.0: the circuit alone reproduces the backdoor |

## Specificity — is it *this* circuit, or would any set do?

| column | meaning |
|---|---|
| `random_ablation_asr` | ablate a **random set of the same size**, measure backdoor ASR. **Near intact (~1.0) = the circuit is specific.** **Near 0 = any set that size destroys the backdoor, and "necessary and sufficient" tells you nothing about localisation.** In practice this tracks `circuit_pct_of_adapter` almost perfectly: huge circuits give 0.000. One draw per cell at a fixed seed, not an average |

## Capability — does the model survive the surgery?

Judged by `openai/gpt-5.6-luna` via OpenRouter (provider pinned, temperature 0), scoring each answer
a bare integer **1–5**. Two disjoint question sets:

| column | meaning |
|---|---|
| `judge_alpaca_{intact,ablate,base}` | mean score on `eval_clean`, **n=500** (alpaca-derived instructions) |
| `judge_norobots_{intact,ablate,base}` | mean score on **n=446** no-robots prompts, an independent set |
| `retention` | `(judge_alpaca_ablate − judge_alpaca_base) / (judge_alpaca_intact − judge_alpaca_base)` |
| `retention_indep` | the same on the no-robots set |

**Reading retention:** 1.0 = ablation cost nothing, the organism is as capable as before. 0.0 = ablation
returned it to base-model quality, i.e. the adapter was destroyed, not edited. Undefined (empty) when
`intact ≈ base`, which is why the denominator is guarded rather than allowed to explode.

## Provenance

| column | meaning |
|---|---|
| `git_commit` | the commit that produced the circuit. The campaign spans more than one — a mid-run fix to base-model fingerprinting, and the judge de-duplication — neither of which changes a measurement |
| `git_dirty` | uncommitted changes under `src/` at run time. Should be `False` everywhere |

---

## Caveats that belong in any analysis

1. **Exclude `gradroute routed_d1 s43`.** It failed Gate A on the hard ASR bar (0.802 < 0.90) but ran
   anyway, because the launcher names cells explicitly and an explicitly named cell overrides the
   gate. It is also the only cell of 85 whose leak file is empty (`[]`). The gradient-routing
   sub-study is **n=14**, not 15.
2. **The clean controls' `sufficiency_keep_only_asr` and `random_ablation_asr` are meaningless.**
   They are computed against a 0-latent circuit and come out 0.0 — artefacts of running a
   backdoor-shaped pipeline on a model with no backdoor. Their `status` (`unsaturated`) and
   `circuit_size` (empty) are the real result.
3. **Grid quantisation.** Several gradient-routing cells sit on the first rung of their grid (50),
   so their true circuit may be smaller. Report those as `≤50`.
4. **Recall against the gradient-routing ground truth is a lower bound.** Those arms were searched
   under the standard sparse 2,500-of-4,032 pool cap, so a designated latent ranked below 2,500 was
   never a candidate and could not be recovered. Check pool membership before reading a miss as a
   failure of the method.
5. **Resumed cells.** Cells whose K-sweep was re-run after an OOM walk the *saved* visiting order for
   the pool, but the sweep ranking **above** `K = pool size` is rebuilt from the resumed launch's
   attribution, which the order file does not cover. Harmless for every affected cell here (their
   `both_K` was far below the pool) but not in general.
