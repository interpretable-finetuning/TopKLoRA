# ReLU ablation, arms B and C: TopKLoRA without the ReLU (pre-registration)

Rebuttal item VLct Q3: "What exactly is new relative to Masiak et al. (2025), and is the ReLU ablated
anywhere? You describe it as the one architectural change and the symmetry-breaking argument rests on
it, so a variant without it would settle how much of the effect needs it."

Written 2026-11-10, before any arm B or C model exists. Status: **FROZEN 2026-11-10**, after Marek's
decisions in section 9 (the commit that adds this line is pushed before the first arm B or C training
run; the 20-step smoke adapters in `clcd_results/relu_ablation/smoke/` were code checks, not data).
Any later change is an amendment with its own commit, written before the data it touches.

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
| B (in scope, Marek 2026-11-10) | none | top-k of s, signed | `relu_latents=false` only (`r42_k5_norelu`); same readouts R0–R4 |

Arm C forward, per token, `s = Ax` (no latent bias: `sae_style=false`):

```
hard = onehot(topk(|s|, k))
soft = k · softmax(|s| / τ)            # τ = 1 constant, as arm A
g    = hard + soft − stopgrad(soft)    # forward g ≈ hard in training, = hard at eval (see below)
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
  selected latents (mean ~0.0014), and in fp32 by up to ~1.2e-7, identically for arm A. Only the eval
  forward (`hard_eval`) uses the hard mask exactly, and that is where every readout is measured.

## 3. Scale and family

**Qwen2.5-1.5B, all three families** (Marek, 2026-11-10): single layer `l20` (7 modules, 294
latents), 9-layer band `l17_25` (63 modules, 2,646 latents) and all layers `all` (196 modules, 8,232
latents); r=42, k=5, α=84; seeds 42–46; on `qwen15_unaliased_base`. 2 arms × 3 families × 5 seeds =
**30 adapters**. Reasons for 1.5B: arm A circuits from campaign 3 exist for all three families under
the same search protocol, so arm A needs no re-search; the base and adapters are on disk; gemma-2-2b
weights are not on the cluster. Where the ReLU binds in arm A at seed 42 (captain's log): l20 0%,
l17_25 0.3%, `all` 4–5%. So arm B can differ from arm A in the forward mainly on `all`; on l20 it
differs only through training dynamics. Arm C differs everywhere (selection by |z|).

Training recipe: identical to campaign-3 `r42_k5` (`scripts/qwen15_train.sh r42_k5 <family> <seed>`:
3 epochs, lr 2e-4, batch 4 × accumulation 2, seq 512, bf16, gradient checkpointing, k constant, τ = 1
constant, `reg_mode=z_only` with the same coefficients, same data and poison rate). The only changes are
`relu_latents=false` (arm B, `r42_k5_norelu`) and additionally `topk_mode=abstopk` (arm C,
`r42_k5_abs`), added as named arms in `qwen15_train.sh` (not through the smoke-only `EXTRA` hook). A
diff of the saved `sleeper_run_config.json` against arm A's (same family and seed) must show only those
fields plus paths; it is part of R0.

**Amendment 3 (2026-11-10, before any arm B/C data): a matched arm A′.** Review round 3 found that
arm A was trained 2026-09-01/02 at commit d94dd6d in a venv with torch 2.5.1+cu121, and searched on
96 GB Blackwell cards, while arms B and C would train here with torch 2.8.0+cu128 at HEAD (197
commits later) and be searched on A40s. The captain's log records a torch change moving outputs. So
the comparison arm is **retrained**: arm A′ = `r42_k5` (ReLU, signed TopK), same three families and five
seeds, trained, gated, searched and attributed in the same venv, at the same frozen commit, on the
same A40 pool as arms B and C. Every A-vs-B/C test in R1, R1′ and R4 uses A′. The campaign-3 arm A
numbers are kept as a reference and as a reproduction check of A′ (same recipe, other stack); a
disagreement between A and A′ is reported, not resolved by choosing one. Cost: 15 more trainings and
5 more l20 searches. This supersedes Amendment 1's conditional re-search: A′ shares batch, hardware
and stack with B and C by construction.
Execution: all 45 cells run from a dedicated clone of this repository checked out at the commit
that adds this amendment (`/scratch/network/ssd/marek/relu_ablation_run`), so nothing edited in the
shared working tree afterwards can reach a running job; the clone's commit and `git status` are
recorded with the outputs.
Named commands (one per step, identical for A′, B and C):
- training: `BASE_MODEL=models/qwen15_unaliased_base bash scripts/qwen15_train.sh <arm> <family> <seed>`
  with `<arm>` = `r42_k5` / `r42_k5_norelu` / `r42_k5_abs`;
- Gate A: `python -m src.clcd.gate_a --adapter <adapter> --data data/sleeper/prepared_eval6k_qwen15
  --base_model models/qwen15_unaliased_base --offset 100 --n 1000 --max_new_tokens 40
  --max_batch_tokens <9000 for l20 and l17_25, 4000 for all> --dump_n 12 --dump_fired 50
  --expect_eot '<|im_end|>'` (arm A's gate settings);
- R0(b): `python -m src.clcd.verify_forward --arm <relu_signed|signed|abs>` on the same base and data;
- R1 (l20): `qwen15_phase1.sh` with the cells named, campaign 3's flags (`ELIM_BLOCK_CAP=64`,
  `SEARCH_EXTRA=--adaptive_n`, batch 64) and a separate `SRC` tree;
- R1′: `exp_circuit_search --attrib_only` with arm A's attribution arguments, `--out` to a separate
  tree (never through `qwen15_phase1.sh`, which would write it as a circuit file and run leak and
  surgical). The batch-size remark in Amendment 2 is void: attribution takes no batch size.
R0(a) is now the diff of `sleeper_run_config.json` between A′ and B/C of the same family and seed (same
code, so only `relu_latents`, `topk_mode`, `dump_path` may differ). Against campaign-3 arm A the diff
also shows `sleeper_regularization.reg_cfg.ROUTE_FRAC` and `ROUTE_MODE`, defaults added after d94dd6d
and inert with `N_FORGET=0`; reported, not a failure.
Statistics, made precise: the Mann–Whitney tests are exact permutation tests on mid-ranks (survivor
counts and `N90` are integers and tie); one-sided, A′ vs B and A′ vs C. If seeds fail Gate A the
smallest attainable p rises (e.g. 1/56 ≈ 0.018 at 3 vs 5), and a test that cannot reach Holm's first
threshold (0.05/6) is reported as underpowered, not as a null. `N90` sorts by |score| (the
`--attrib_only` file is sorted by signed score).
R4 additions: on arms B and C the whole-adapter ablations report how many entries were zeroed (must be
> 0 for arm C, and is reported for arm B), and (a′) followed by (b′) must reproduce the all-latents
ablation, so a no-op or wrong-pole override cannot pass as a result. The arm A′ (a′) check compares
with intact measured in the same invocation (same batching, dtype, budget), not with a gate record.
Wording correction to §3: "arm B differs from arm A mainly on `all`" is a prediction from arm A's
binding rates; arm B's own rate of tokens with fewer than k positive pre-activations is measured by
R0(b) (`few_positive`). Timing correction to §7: arm A's own logs show training took about 35 / 50 /
82 min for l20 / l17_25 / `all` on an A40.

**Amendment 4 (2026-11-10, before any arm A′/B/C data): review round 2 corrections.**
- The R1 search runs with campaign 3's environment, which `qwen15_phase1.sh` does not default to:
  `ELIM_BLOCK_CAP=64 SEARCH_EXTRA=--adaptive_n ELIM_ORDER_OUT_DIR=<tree>/orders SRC=<tree>
  LOGDIR=<tree>/logs bash scripts/qwen15_phase1.sh "<arm> l20 <seed>" …` (cells always named). Before
  any R1 number is reported, each circuit file's recorded `args` must equal campaign-3 arm A's l20
  `args` apart from adapter, base and output paths; a mismatch voids the cell.
- `qwen15_regate.sh` is not used (it hard-codes the aliased vanilla base); its parser change is reverted.
- Post-flight now also requires `top_k_experiment`, `hard_eval`, `alpha_over_r`, `k_schedule constant`,
  `sae_style false` and `latent_gate_enabled false` for arms B and C; arm A′ is covered by R0(a)'s diff
  against B and C. Training rejects an unknown `topk_mode` instead of mapping it to `topk`.
- R0(b) (`verify_forward`) takes `--expect_k 5`, refuses non-finite values, and checks post-gate latents,
  not the decode (pinned by tests).

**Amendment 5 (2026-11-11, before any arm A′/B/C data): stage-2 execution (review round 4).**
- Training runs at 162aefa. Stage 2 (Gate A onward) runs from the same clone fast-forwarded to the
  reviewed commit that adds this amendment; it changes no training, Gate A or search code (only
  `src/clcd/relu_ablation.py`, its tests, and the corpus builder moved into
  `src/autointerp/latent_inspection.py`), and its commit is recorded with the stage-2 outputs.
- Every stage-2 job of a cell waits for that cell's training marker (written only after post-flight
  passes). The l20 search runs only for seeds whose Gate A record is usable (`relu_ablation gate-ok`);
  all other readouts run on every trained seed and the analysis reports failing seeds separately.
- R4 on l20 ablates the **certified circuit**: the circuit file's `kept_latents` (the `both_K` prefix
  whose ablation the search found necessary), at batch 64 with no token cap (`--mbt 0`), so condition
  (c) is the search's own necessity measurement. The R1 statistic stays the survivor count; both sizes
  are reported. A circuit file without `status == "ok"` or with no kept latents is refused.
- R4 also records, per circuit, how many circuit latents fire positive, negative or both on the trigger
  band (identity observer, no ablation) — the pole count beside the latent count.
- R4's self-checks are enforced (non-zero exit, record kept): neg-then-pos must reproduce the
  all-latents ablation; the all-latents ablation must zero something; arm A′'s negative ablation must
  zero nothing and leave every generation unchanged; arm C's must zero something.
- R0(a) (`relu_ablation r0a`) and Amendment 4's args check (`relu_ablation args-check`) run at analysis
  time on the saved configs and circuit files; a failure voids that cell's comparison.
- R2 counts every latent, so the bipolar share among l20 circuit latents and among each l17_25/`all`
  adapter's N90 latents (Amendment 2's analogue of "circuit latents") is computed from the same counts.
  R3 needs a second capture of per-pole example windows for the drawn latents, run once the autointerp
  protocol is fixed.

## 4. Readouts

All readouts are reported for every seed. Bands, never point estimates where a band exists.
Every readout runs for both arm B and arm C in each of the three families; where R0–R5 below say
"arm C", read "arm B or arm C" (R3's draw is made separately per arm and family, same RNG seed).

**R0 — the model is what it claims to be (gate, not a result).**
(a) Config diff as above. (b) On 32 real prompts, `src/clcd/verify_forward.py` with `--arm signed`
(arm B) or `--arm abs` (arm C): the post-gate latents equal the arm's definition recomputed from each
module's input (max |diff| = 0), with exactly k nonzero entries per token; the check is shown to fail
when asked for the other arms (done on the smoke adapters and an arm A adapter, section 6). Limit: arm
B's forward differs from arm A's only on tokens with fewer than k positive pre-activations, so on a
module where that never happens the check cannot tell B from A (the l20 arm B smoke adapter passed as
both); there the config checks (`verify_adapter_arm`, `assert_wrapped_flags`) carry the guarantee.
R0(b) reports, per arm B adapter, how many such tokens it saw. (c) Gate A, unchanged: ASR ≥ 0.90 on `eval_triggered[100:1100]`, clean fire reported. A seed
that fails Gate A is a result, reported with its number, and is **not replaced**; R1–R4 then run on the
passing seeds and state n.

**R1 — circuit size (primary for "are circuits larger").**
CLCD-eliminate with exactly arm A's campaign-3 arguments for the same family (from
`qwen15_campaign3/r42_k5/elim/<family>_seed<s>_circuit.json`, run through `qwen15_phase1.sh` with the
cells named): pool = whole adapter for l20 (294), top 2,500 by |attribution| for l17_25 and `all`
(`order_abs`, `n_attrib` 64, `K_ig` 128, control baseline, margin target), `elim_block_cap` 64,
`--adaptive_n` (rungs 100/300/1000), `n_cheap` 80 at offset 1100, sufficiency at 2 paired SE, necessity
exactly 0, the same K grid; then leak (4 × 1,000 held out) and surgical, as campaign 3.
Arm A reference (already measured, seeds 42–46):

| family | `both_K` | survivors | pool |
|---|---|---|---|
| l20 | 150 / 200 / 250 / 200 / 150 | 28 / 28 / 29 / 38 / 29 | 294 |
| l17_25 | 600 / 300 / 800 / 600 / 400 | 139 / 81 / 148 / 143 / 103 | 2,500 |
| all | 600 / 800 / 800 / 800 / 1200 | 207 / 286 / 250 / 330 / 400 | 2,500 |

Rank-matched dense `r42_dense` `both_K`: l20 275–285, l17_25 1,600–2,400, `all` 3,200–6,400.
**Amendment 1 (2026-11-10, before any arm B/C data): hardware match for the search.** Campaign 3 ran
on 96 GB cards with search batch 64; `qwen15_phase1.sh` records that batch 64 runs out of memory on
the `all` family on a 46 GB A40, and these runs are on A40s. Batch size enters the bf16 numerics, so
it is part of the measurement. Rule: each family's arm B and C searches use batch 64 if it fits on an
A40; if it does not, they use the largest batch that fits, and arm A is **re-searched** for that
family on A40s at the same batch, so all three arms share hardware and batch. The campaign-3 arm A
numbers above are then reported as a reference beside the matched re-search, not as the comparison.
Statistic: survivors (finer than the K grid), arm B vs A and arm C vs A in each family, exact
one-sided Mann–Whitney U (larger in B or C), 5 vs 5 (smallest attainable p = 1/252). Six tests (2 arms
× 3 families), Holm-corrected at family-wise α = 0.05; raw p reported beside. `both_K` is reported as
a band. B vs C is descriptive only. With n = 5 a null result does not show equality; it is reported as
the ratio of medians with the per-seed range.
Units note: circuit sizes count latents; a bipolar latent is two units of interpretation. R4 reports
the pole count beside the latent count.

**Amendment 2 (2026-11-10, Marek, before any arm B/C data): certified search on l20 only.** The
latent-level claims (R2, R3) need no circuit; circuit search is the cost bottleneck. R1 as written
above now runs **only for l20** (10 cells: arms B and C × 5 seeds; pool = whole adapter, 294).
For l17_25 and `all`, R1 is replaced by **R1′, attribution concentration**: `exp_circuit_search
--attrib_only` with arm A's attribution arguments (`n_attrib` 64 at offset 0, `K_ig` 128, control
baseline, margin target, batch as Amendment 1) on arms A, B and C alike, all on A40s (arm A recomputed,
not read from campaign 3, so hardware matches). Per adapter: `N50` and `N90`, the smallest number of
latents whose |score| sum reaches 50% / 90% of the adapter's total |score|. Statistic: one-sided
Mann–Whitney U on `N90` (larger in B or C than in A), 5 vs 5 per family. The six tests of the study are
now: R1 survivors on l20 (B vs A, C vs A) and R1′ `N90` on l17_25 and `all` (B vs A, C vs A each);
Holm-corrected at family-wise α = 0.05, raw p reported beside. `N50` and the l20 `both_K` band are
reported, not tested. R1′ is a ranking statistic, not a certified circuit: the rebuttal says so.
R4 changes accordingly (see R4).

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

**R4 — which pole carries the backdoor.** (Amendment 2: the circuit version below runs on l20, where
a certified circuit exists. On every adapter of every family, arms B and C, it is preceded by a
**whole-adapter pole ablation**: ASR on `eval_triggered[100:1100]` and clean fire on
`eval_clean[100:1100]` with (a′) every negative post-gate value zeroed in every module, (b′) every
positive value zeroed, through the same callable override. Arm A is run under (a′) as a check: it has
no negative values, so its ASR must equal intact exactly.)
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
| training, 30 cells | roughly 25 / 38 / 65 min each for l20 / l17_25 / all on an A40 (1.5B log scaling, not measured); one cell per card |
| R1 search + leak + surgical (Amendment 2: l20 only) | 10 cells, a few hours each (pool 294); plus arm A re-search on l20 only if Amendment 1 triggers |
| R1′ attribution-only (l17_25, `all`; arms A, B, C) | 30 runs, minutes to an hour each |
| R2 capture | ~1–2 h per seed |
| R3 rating | LLM judge; waits for the autointerp protocol |
| R4 | < 1 h per seed |

About 1 day wall after Amendment 2 on ~8 A40s; the cards are shared with other users
and the autointerp thread, so each cell takes a card only when `wheregpus` shows it idle.

## 8. Logging

Every run gets its captain's-log entry before it is reported (Rule 13), including failed seeds and
null results.

## 9. Open decisions (Marek)

1. ~~Scale/family~~ Decided 2026-11-10: Qwen2.5-1.5B, all three families (l20, l17_25, `all`).
2. ~~Arm B~~ Decided 2026-11-10: arm B included.
3. ~~R3 raters~~ Decided 2026-11-10: LLM judge only, using the autointerp thread's protocol once it is
   settled; R3 waits for it.
