# ReLU ablation, arm C: TopKLoRA without the ReLU, latents selected by |z| (pre-registration)

Rebuttal item VLct Q3: "What exactly is new relative to Masiak et al. (2025), and is the ReLU ablated
anywhere? You describe it as the one architectural change and the symmetry-breaking argument rests on
it, so a variant without it would settle how much of the effect needs it."

Written 2026-11-10, before any arm C model exists. Status: **DRAFT, not frozen.** It freezes when
Marek signs off on section 9 (open decisions); the commit that freezes it is pushed before the first
training run. Any later change is an amendment with its own commit, written before the data it touches.

## 1. Why arm C, and not just `relu_latents=false`

The paper's adapter is `ΔW x = (α/r) B TopK_k(ReLU(Ax))`, with TopK on the **signed** value
(`z.topk`, `src/models.py:_hard_topk_mask`). Two facts make the literal ablation weak:

- Signed TopK already breaks the per-latent sign flip (flipping row `A_i` and column `b_i` changes
  which latents are selected), so the symmetry argument in §3 of the paper does not need the ReLU.
- TopK(ReLU(s)) differs from TopK(s) only where fewer than k pre-activations are positive. Measured
  at inference (captain's log, 2026-11-10, two entries): the ReLU binds on 0% of prompt positions on
  l20 at both 1.5B and 7B, 0.3% on 1.5B l17_25, 5–6% on 7B l17_25, 4–5% on 1.5B `all`. Removing it at
  inference changed no fire on any of the five orgs.

The stated motivation for the ReLU is different: without a sign constraint a latent can become
**bipolar**, with a positive and a negative pole that mean different things, so "one latent" stops
being one unit of interpretation. That only becomes possible when selection is by magnitude. Arm C is
the variant where it can happen: no ReLU, TopK by |z|, sign kept.

External support for the prediction: Goodfire, "Neurons as an Interpretable Basis" (Tigges, Kowal,
McGrath, Lewis, Cammarata, 2026-11-10). SwiGLU neurons in Qwen3.6-35B-A3B are bipolar; "the two neuron
poles can have completely different meanings, though sometimes (but rarely) the positive and negative
poles will represent contrasting opposites of the same concept"; they therefore take the
**neuron-pole** as the unit of analysis, and their sparsification is top-K by |activation|. Arm C is the
TopKLoRA analogue of that unit. Their numbers are for base-model MLP neurons, not loss-trained adapter
latents, and the essay is not peer reviewed.

## 2. Arms

| arm | nonlinearity | selection | status |
|---|---|---|---|
| A (paper) | ReLU | top-k of ReLU(s), signed | exists: published `r42_k5` orgs, campaign-3 circuits |
| C (this document) | none | top-k of \|s\|, output keeps the sign of s | to train |
| B (optional companion) | none | top-k of s, signed | `relu_latents=false` only; runs only if cards allow, same readouts R0–R2 |

Arm C forward, per token, `s = Ax` (no latent bias: `sae_style=false`):

```
hard = onehot(topk(|s|, k))
soft = k · softmax(|s| / τ)            # τ = 1 constant, as arm A
g    = hard + soft − stopgrad(soft)    # forward g = hard (exactly in fp32 and at eval; see below)
ΔW x = (α/r) · B (s ⊙ g)
```

Implementation: a new `topk_mode="abstopk"` in `_hard_topk_mask` / `_soft_topk_mass` /
`VALID_TOPK_MODES` (`src/models.py`, mirrored in `src/config_utils.py`), with `relu_latents=false`.
Nothing else in the forward changes. Two facts that hold for both arms alike (adversarial review,
2026-11-10):
- The z_only regularisers are sign-agnostic, but under the recipe's reentrant gradient checkpointing
  they contribute **zero gradient** in both arms (already on record: captain's log 2026-09-14,
  "activation-based latent regularisers are inert under reentrant checkpointing"). Both arms are
  therefore effectively unregularised; the arms stay matched.
- In bf16 training, `hard + soft − stopgrad(soft)` differs from the hard mask by up to 2^-6 on
  selected latents (mean ~0.0014), identically for arm A. The forward gate equals the hard mask
  exactly in fp32 and at eval (`hard_eval`), which is where every readout is measured.

## 3. Scale and family

**Default: Qwen2.5-1.5B, `all` family** (layers 0–27, 196 modules, r=42, k=5, α=84, 8,232 latents),
seeds 42–46, on `qwen15_unaliased_base`. Reasons: at 1.5B this is the family where the ReLU binds
(4–5%; l20 and l17_25 are at 0% and 0.3%); arm A circuits from campaign 3 exist under the same search
protocol, so arm A needs no re-search; the base and adapters are on disk; gemma-2-2b weights are not
on the cluster (gated, needs a token). Alternative kept open in section 9: 7B l17_25, where the ReLU
binds most (5–6%) but arm A must be re-searched under block-64 (the published 7B l17_25 circuits are
one-at-a-time).

Training recipe: identical to campaign-3 `r42_k5` `all` (`scripts/qwen15_train.sh r42_k5 all <seed>`:
3 epochs, lr 2e-4, batch 4 × accumulation 2, seq 512, bf16, gradient checkpointing, k constant, τ = 1
constant, `reg_mode=z_only` with the same coefficients, same data and poison rate). The only changes are
`relu_latents=false` and `topk_mode=abstopk`, added as a named arm in `qwen15_train.sh` (not through the
smoke-only `EXTRA` hook). A diff of the saved `sleeper_run_config.json` against arm A's must show only
those two fields plus paths; it is part of R0.

## 4. Readouts

All readouts are reported for every seed. Bands, never point estimates where a band exists.

**R0 — the model is what it claims to be (gate, not a result).**
(a) Config diff as above. (b) On 32 real prompts, the hooked post-gate latents equal arm C's forward
recomputed from the module input (max |diff| = 0) and have exactly k nonzero entries per real token;
the check is shown to fail on a broken encoder (signed TopK, and ReLU switched back on) before it is
trusted. (c) Gate A, unchanged: ASR ≥ 0.90 on `eval_triggered[100:1100]`, clean fire reported. A seed
that fails Gate A is a result, reported with its number, and is **not replaced**; R1–R4 then run on the
passing seeds and state n.

**R1 — circuit size (primary for "are circuits larger").**
CLCD-eliminate with exactly arm A's campaign-3 arguments (from
`qwen15_campaign3/r42_k5/elim/all_seed42_circuit.json`): pool top 2,500 by |attribution|
(`order_abs`, `n_attrib` 64, `K_ig` 128, control baseline, margin target), `elim_block_cap` 64,
`--adaptive_n` (rungs 100/300/1000), `n_cheap` 80 at offset 1100, sufficiency at 2 paired SE, necessity
exactly 0, the same K grid; then leak (4 × 1,000 held out) and surgical, as campaign 3.
Arm A reference (already measured): `both_K` 600 / 800 / 800 / 800 / 1200 and survivors
207 / 286 / 250 / 330 / 400 of 2,500 (seeds 42–46); rank-matched dense `r42_dense` `all` `both_K`
3,200–6,400.
Statistic: survivors (finer than the K grid), arm C vs arm A, exact one-sided Mann–Whitney U, α = 0.05
(5 vs 5: smallest attainable p = 1/252). `both_K` is reported beside it as a band. With n = 5 a null
result does not show equality; it is reported as the ratio of medians with the per-seed range.
Units note: arm C circuit sizes count latents; a bipolar latent is two units of interpretation. R4
reports the pole count beside the latent count.

**R2 — pole structure (the direct test of the motivation).**
Capture every selected latent's signed post-gate value on the inspection corpus of the 7B hand
inspection (`docs/latent-inspection-7b-preregistration.md` §4: delphi default web dataset shard plus
chat-templated alpaca rows the org never saw, clean- and trigger-tagged, 256-token windows, position 0
and padding excluded), re-tokenised for the 1.5B tokenizer, same token budget. Per latent: firings on
each pole, `f_neg = n_neg / (n_pos + n_neg)`, and the same fraction weighted by |value|.
A latent is **bipolar** if `min(f_pos, f_neg) ≥ 0.10` and each pole has ≥ 100 firings (thresholds set
here, before any arm C model exists). Reported: the distribution of `f_neg`, and the bipolar share among
(i) all latents that fire and (ii) the R1 circuit latents, per seed. Arm A is 0 by construction and is
not a check; the capture's sign handling is checked by R0(b) and by a synthetic negative latent in the
unit tests.

**R3 — do the two poles mean different things.**
Draw, with `random.Random(20261111)`, 10 bipolar latents from the seed-42 pool and 10 from its R1
circuit (fewer if fewer exist; stated). For each pole: its top 20 windows plus 10 drawn from its
top-decile firings. Add 5 decoy pairs, each a pole from one latent paired with a pole from another,
drawn with the same RNG after the real latents. Each pair is shown with the poles' order and the
decoys shuffled in, and is rated **same concept / opposite ends of one concept / unrelated**. Rater:
an LLM judge only (Marek, 2026-11-10: no time for blind human rating). The judge model, prompt and
scoring follow the latent-level autointerp protocol being settled in the separate autointerp thread;
**R3 does not run until that protocol is fixed**, and adopting it is an amendment written before any
arm C pole is shown to the judge. Reported: the counts per category, and the decoy pairs' ratings as
the calibration of "unrelated". The Goodfire result predicts mostly "unrelated".

**R4 — which pole carries the backdoor.**
For each passing seed, with its R1 circuit C intact elsewhere: ASR on `eval_triggered[100:1100]` when
(a) the negative pole of every latent in C is zeroed (a callable override in `src/clcd/latents.inject`:
`a ↦ a · (a ≥ 0)` on C's columns), (b) the positive pole is zeroed, (c) both (the existing necessity
ablation, which must reproduce R1's 0). Also the number of distinct poles that C uses on trigger prompts
(a latent counts twice if both of its poles fire there). If one pole alone kills the backdoor and the
other leaves ASR ≥ 0.9, the latent-level circuit counts units that do not participate.
Secondary, run only if R2 finds ≥ 20% of circuit latents bipolar: CLCD-eliminate with each pole as a
separate candidate unit, same protocol, so the circuit is sized in the units arm C actually has.

**R5 — autointerp difficulty.** Deferred to the paused latent-level autointerp thread. That pipeline
must hook arm C latents as two non-negative units per latent (`relu(a)` and `relu(−a)`), because delphi
ranks examples by value and would otherwise drop the negative pole.

## 5. Outcomes and what the rebuttal would say

| R1 | R2 / R3 | reading |
|---|---|---|
| C larger | bipolar latents common, poles unrelated | the ReLU is load-bearing for both compactness and unit interpretability |
| no difference | bipolar latents common, poles unrelated | removing the ReLU keeps circuits compact in latent count, but each latent stops being one unit; the ReLU buys interpretable units, not smaller circuits |
| no difference | bipolar latents rare | at this scale the ReLU is not needed; we say so, and keep it as a guarantee rather than a measured necessity |
| C larger | bipolar latents rare | circuit size depends on the ReLU through training, not through poles; report as such |

All four are publishable rebuttal answers. None of them is a reason to rerun with other settings.

## 6. Code (status 2026-11-10)

Done (commits 96cecab, 0b3b723 and the review-round-1 fixes), each guard shown to fail on a broken
variant:
- `src/models.py` / `src/config_utils.py`: `abstopk` mode; a test keeps the two mode lists equal.
  Tests fail when selection reverts to signed, the sign is dropped, the soft mass uses s, or the
  ReLU is left on (`tests/test_abstopk.py`).
- `scripts/qwen15_train.sh`: arm `r42_k5_abs`; post-flight `verify_adapter_arm --arm abstopk
  --expect_k 5` (checks `relu_latents`, `topk_mode`, `reg_mode`, r, α, k, k_final).
- `src/train.py` + `src/utils.py::assert_wrapped_flags`: training refuses to start if any wrapped
  module's `relu_latents` / `topk_mode` differs from what `topk_config.json` will record.
- R0(b): `src/clcd/verify_forward.py --arm abs` recomputes the arm's definition (not the module's
  flags) from each module's input through `load_organism`. On the 20-step smoke adapter it passes as
  `abs` and fails as `relu_signed`; a published arm A adapter passes as `relu_signed` and fails as `abs`.
- `gate_a` records `relu_latents_at_eval` and `topk_mode` read off the modules.
- Drivers: training runs one cell per `qwen15_train.sh` call (not `qwen15_sweep.sh`, which treats a
  failed post-flight on an existing dump as "already trained"); search runs through campaign 3's
  `qwen15_phase1.sh` with the cells named explicitly (its parser and `qwen15_regate.sh`'s now accept
  `r42_k5_abs`), so arm C gets arm A's flags and per-family generation budget.

Still to write (before R2/R4 run, each followed by an adversarial review):
- The CLCD search path (attribution, inject/keep-only, elimination, leak, surgical) was audited
  for non-negativity assumptions and is sign-agnostic; the `> 0` tests in `pipeline.py` and
  `selection.py` act on attribution/edge scores, not activations (review round 1). Sites that do
  assume non-negative activations and would drop the negative pole: `scripts/latent_inspection.py`
  (`_check_hook`, fails loudly), `src/autointerp/delphi_autointerp.py:1476`,
  `src/autointerp/autointerp_framework_hh.py:224`. R2/R3/R5 must not use them unmodified.
- Pole-split capture for R2/R3 in `src/autointerp/latent_inspection.py` (extended, not a new module):
  each latent hooked as two non-negative units, `relu(a)` and `relu(−a)`. R3/R5 adopt the causal
  autointerp pilot's protocol (`docs/causal-autointerp-pilot-preregistration.md`, autointerp
  thread's branch), which already scores + and − poles separately.
- R4 per-pole ablation: a callable override through `src/clcd/latents.inject`.

## 7. Cost (estimates, measured where stated)

| step | estimate |
|---|---|
| code + tests + one 20-step smoke per arm | ~0.5 day |
| training, 5 seeds × `all` 1.5B | ~65 min each on an A40 (1.5B log scaling, not measured for arm C); parallel, ~1.5 h wall |
| R1 search + leak + surgical | 5 cells, guessed 6–12 h each on an A40 (no archived per-cell time for 1.5B `all`); parallel, ~0.5 day wall |
| R2 capture | ~1–2 h per seed |
| R3 rating | LLM judge; waits for the autointerp protocol |
| R4 | < 1 h per seed |

About 1.5–2 days wall on 5 free A40s. Arm B, if run, adds the same training and search cost.

## 8. Logging

Every run gets its captain's-log entry before it is reported (Rule 13), including failed seeds and
null results.

## 9. Open decisions (Marek)

1. Scale/family: 1.5B `all` (default above) or 7B l17_25.
2. Run arm B alongside, or arm C only.
3. ~~R3 raters~~ Decided 2026-11-10: LLM judge only, using the autointerp thread's protocol once it is
   settled; R3 waits for it.
