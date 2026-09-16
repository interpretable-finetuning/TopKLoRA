# Pre-registration — what a "brake" actually is

**Status: PLAN. No experiment here has been run as designed.** Two of its questions were answered on
2026-09-01 from existing artifacts, without a launch — Exp-5's (brakes *are* active intact) and the
direction of Exp-3's (brakes are *not* trigger-selective) — see the **Answered** notes in those sections;
the λ-sweep outcome table was relabelled on 2026-09-02 (queue H4). Every readout and threshold below
is fixed *before* any job is launched, per `integrity_no_phacking`. Negative results are logged with the same rigour as
positive ones. Results go into `docs/captains-log.md` **before** they are reported anywhere (Rule 13).

## The question

S2.0 found that **128 of the 400 members of a certified circuit are causal brakes** — latents whose
ablation moves the model *toward* the payload. Removing them is a strict engineering win (S2.1/S2.2,
below). But nobody has established *what they are*. Three hypotheses, mutually exclusive in their
strong forms:

| | claim | prediction if true |
|---|---|---|
| **H-write** | Same read, opposite write. Brakes are trigger-selective and their decoder columns are anti-aligned with the payload direction. | Flipping a brake converts it into a driver. Brakes are a real anti-payload mechanism. |
| **H-read** | Different read. Brakes fire on a different feature (e.g. the ordinary-instruction representation), not the trigger. | Flipping corrupts normal behaviour; it does not build a trigger→payload pathway. |
| **H-competition** | Not about the trigger at all. The margin is *relative* — `logit(payload) − logit(best competitor)` — so any latent supporting the competitor scores as a brake by subtraction. | Brakes are ordinary machinery. "Brake" is a property of the metric, not of the org. |

**H-competition is the leading hypothesis.** Four independent pieces of existing support:

1. **Enrichment asymmetry.** Brakes: 32.0% in-circuit vs 24.8% outside = **1.29x**. Drivers: 31.2% vs
   11.5% = **2.72x**. Drivers concentrate in the circuit; brakes barely do.
2. **Flat band profile.** Brake rate by |attribution| rank: 39 / 33 / 31 / 25 / 25 / 27 / 22 / 25 %
   across bands 0-100 ... 700-800 — plateauing at ~25% out to rank 800. Drivers fall 42 -> 11%.
3. **Magnitude asymmetry.** Strongest brake **+0.309** vs strongest driver **-2.773** — a **9.0x**
   gap, and only **1 of 800** latents exceeds the delta=0.25 decision threshold.
4. **Autointerp** already reads drivers as literal trigger/payload-token detectors and brakes as
   *benign instruction-processing features*.
5. **Intact activity and selectivity (added 2026-09-01; log entry "The 128 brakes are highly active in
   the INTACT model and are NOT trigger-selective").** Median within-projection intact activation
   rank **0.902** (drivers 0.934, pool tail 0.715), **0/128** silent, half top-decile; selective
   **0.070** vs drivers **0.448** vs pool tail 0.177. General machinery that fires on triggered and
   clean prompts alike.
6. **"Selective" is mostly not a backdoor property (added 2026-09-01; E4 byproduct, log entry
   "Whole-adapter allocation has NO RESOLUTION for the backdoor").** A clean adapter that never saw
   `|TRIGGER|` shows ~264 marker-selective latents against ~289 in the backdoored one, so
   selectivity measures the response to tag *identity*, not backdoor involvement.

None of the six is decisive (5 and 6 rule out the ablation-response reading of Exp-5 and fix the
direction of Exp-3; they do not test the write side). All are consistent with the mechanical story the 2026-08-28 entry
proposes: zeroing any latent perturbs the residual, the gate re-selects downstream, and some
fraction of the time the perturbation happens to point toward the payload.

---

## Mechanics, verified against the code (not assumed)

Canonical org `l1523_seed43`, config
`config/train_config/training/experiment/sleeper_topk_r64_k8_layers15_23.yaml`.

`sae_style` is **absent** from that config and defaults to `False` (`src/models.py:207`). All four
SAE-style options are gated on it (`_should_use_input_center`, `_should_use_latent_bias`,
`_should_use_output_bias`, `_should_rescale_by_decoder_norm`, `src/models.py:339-357`), so **all four
are OFF**. The forward is therefore exactly:

```
z    = A x                    # encode_pre; no input centering, no latent bias
h    = relu(z)                # relu_latents: true            -> h >= 0
mask = top-8 of h             # topk_mode "topk", hard_eval true, k_final 8
a    = h * mask               # post-gate scalar, >= 0, <= 8 nonzero per token per module
D    = scale * (B a)          # decode_latents; NO output bias; scale = alpha/r = 128/64 = 2.0
```

Four consequences, each load-bearing for the designs below:

- **C1 — post-gate activations are non-negative.** `relu_latents: true`, so `a_i >= 0` always. The
  activation carries no sign; **all directional sign lives in the decoder column `B_i`**.
- **C2 — the decode is exactly linear in `a`.** `decode_latents` is `F.linear(a, B) * scale` with no
  bias term (`src/models.py:759-774`), and `recompute_output_from_sparse_latents` returns
  `base_out + decode_latents(a)`. So `a_i -> lambda * a_i` scales that module's residual write by
  exactly lambda. **All nonlinearity is downstream** (later layers, later top-k re-selection, softmax).
- **C3 — selection does not depend on `B`.** Decoder-norm rescaling is off, so the top-8 competition
  is decided by `relu(A x)` alone. Negating or scaling the write leaves *which* latents fire, and
  *how strongly*, bit-for-bit unchanged.
- **C4 — `inject` already does this.** `src/clcd/latents.py::inject` registers a **forward hook** and
  transforms `_last_z_sparse`, i.e. the *post-gate* value — the mask is already resolved by then. Its
  callable mode `f(a) -> a'` is exactly the primitive needed. Ablation is `a -> 0`; the sweep is
  `a -> lambda*a`. **No new machinery, no weight surgery.**

C3 is the reason this programme is worth running: it isolates the *sign* question from the
*selection* question by construction. The obvious confound — "maybe the A matrix never invokes brakes
when the backdoor fires" — cannot contaminate a lambda-sweep, because lambda does not touch selection.

**The ReLU also breaks the gauge symmetry.** In a plain low-rank factorization `(A_i, B_i) ->
(-A_i, -B_i)` is a no-op, so "the sign of a latent" means nothing. Here `relu(-z) != -relu(z)`, so
sign is a physically meaningful property. This question is not askable of a dense LoRA.

---

## Rule 14 — what gets extended, not created

Every experiment below extends **`probe_A_gradfidelity.py`** (409 lines; since 2026-09-11 `scratchpad/`
is untracked — the file is archived at `clcd_results/scratchpad_archive_2026-09-11/` and in git history
at 94057eb, and would move to `analysis/` when this plan runs), which already
carries the `P_BRAKES`, `P_SETS` and `P_CONTRIB` modes, the `payload_margin` / `encode` /
`read_latents_masked` helpers, and the BOS / `position_ids` / 3-token-scope fixes. Re-implementing
any of those is how the Exp-8a scratch-harness error happened. Exp-4 additionally reuses
`analysis/analyze_subspace_backtrace.py`'s payload anchor.

**Considered and rejected:** a new `probe_brakes.py`. Rejected because it would duplicate
`payload_margin` and `encode`, which is precisely the failure mode Rule 14 names. If any of these
modes acquires a second caller it moves to `src/clcd/`.

⚠️ **`P_CONDSEL` is NOT in the main checkout's copy** — verified absent (0 occurrences). It exists
only on branch `worktree-autointerp-dryrun` (`P_CONDSEL` at line 87, `run_condsel` at line 299). Exp-3
must port it via `git show`, not re-derive it. Do not enter that worktree.

---

## Experiment 1 — lambda-sweep: is a brake a signed component?

**Question.** Does reversing a brake's write turn it into a driver, or does only its *removal* matter?

**Design.** For each target latent `i`, override `a_i -> lambda * a_i` for
**lambda in {1, 0.5, 0, -0.5, -1}** via `inject`'s callable mode. lambda=1 is the baseline; lambda=0
reproduces the existing ablation. Band, n and scoring identical to S2.0 (`[4000:5000]`, n=1000,
`SCORE_T=3`) so results are directly comparable to the archive.

**Why five points and not a bare flip.** Probe A measured strong nonlinearity in this system, and a
flip is *twice* the displacement of an ablation — it pushes further off-distribution and is more
likely to trigger downstream top-k churn on its own. The intermediate lambda trace separates "linear
with a slope" from "saturates on removal"; a binary flip cannot.

**Pre-registered readout.** Let `d = margin(lambda=0) - margin(lambda=1)` (the archived brake effect,
> 0) and `f = margin(lambda=-1) - margin(lambda=1)`. Under local linearity, `f = 2d`. Classify each
latent by `f/d`:

| `f/d` band | verdict | meaning |
|---|---|---|
| **[1.6, 2.4]** | LINEAR — **direct write** | The brake's own write is a signed component of the margin: flipping it pushes toward the payload with twice the force of deleting it. Consistent with H-write (a write *against the payload*) **and** with H-competition through a direct write *for the competitor* — the sweep does not separate them. |
| **[0.6, 1.4]** | SATURATING — **downstream re-selection** | Only removal matters, not direction. The effect is carried by set churn after the perturbation (later top-k re-selection), not by the latent's own write. Consistent with all three hypotheses; it says the mechanism is indirect. |
| **< 0.6 or > 2.4, or `f < 0`** | NONLINEAR | Not a simple write. Report the distribution; draw no directional conclusion. |

**What this experiment decides, and what it does not (relabelled 2026-09-02, queue H4).** The λ-sweep
separates *direct write* from *downstream re-selection*. It does **not** separate H-write from
H-competition: both predict a linear direct write, differing only in *which* logit the write moves.
That question belongs to Exp-2's decomposition (payload-up vs competitor-down). The earlier labels
("LINEAR supports H-write", "SATURATING supports H-competition") were wrong and must not be reused.

Headline = the **fraction of the 128 brakes in each band, reported as a band across bootstrap
resamples, not a point**. Declare LINEAR or SATURATING only if >=60% of brakes fall in one band.

**Controls, and how each is proven failable.**

| control | purpose | how it can go red |
|---|---|---|
| **lambda=0 reproduction** | harness validity | lambda=0 must reproduce the archived S2.0 contribution for each latent to within float determinism. The project has demonstrated exact cross-shard reproduction before, so this is a *hard* check: if lambda=0 does not match the archive, the run is **void** and nothing else is interpreted. |
| **Drivers (positive)** | is linearity available at all? | Drivers carry effects up to -2.773. If `f/d ~ 2` fails even for drivers, local linearity is dead system-wide and the brake reading is uninterpretable — a real possible failure, not a formality. |
| **Causally-NULL latents (null)** | is the sweep measuring anything? | Class-matched *and* rank-matched NULLs (the `nullcls` recipe) should give ~0 at every lambda. Structure here means the harness is broken. |

Driver-as-known-answer-control is **not** circular here: drivers are defined by causal contribution, and
`f/d` is a curvature ratio — a different quantity. Contrast Probe-B, whose known-answer control *was*
circular because it selected on attribution sign, the very quantity under test.

**Power.** S2.0's median paired SE is **0.0029** (MDE 0.0059) at n=1000 on this band. lambda=+/-1
effects are predicted at ~2x the ablation effect, so if ablation was resolvable, the flip is
comfortably so.

**Cost.** 4 non-trivial lambda values x (128 brakes + 125 drivers + 128 nulls) at n=1000. Scaling from
the logged ~1.5 GPU-h for a single-pass screen of this shape: **~6 GPU-h**.

**Artifacts.** `clcd_results/probes/lambda_sweep_l1523_s43.json`, `logs/probes/lambda_sweep_*.out`.

---

## Experiment 2 — margin decomposition: is the effect payload-up or competitor-down?

**Question.** When ablating a brake raises the margin, does `logit(payload)` rise, or does
`logit(competitor)` fall? This is the decisive H-competition vs H-write test.

**Design.** `payload_margin` (`probe_A_gradfidelity.py:83-91`, archived copy — see Rule 14 note above) **already computes both
terms and returns only their difference**:

```
tl     = logit of the payload token at each of the first SCORE_T positions
bo     = max logit over all NON-payload tokens ("best other")
margin = min over positions of (tl - bo)
```

The experiment is **returning `tl` and `bo` separately instead of `tl - bo`.** That is the whole
change. Also record the **argmax competitor token id** at each position.

**Pre-registered readout.** Per brake, `Dtl` and `Dbo` under ablation.

| pattern | verdict |
|---|---|
| `abs(Dtl) < 0.25 * abs(Dbo)` for >=60% of brakes | **H-competition.** Brakes support the competitor; the payload is untouched. |
| `Dtl > 0` and `abs(Dtl) > abs(Dbo)` for >=60% | **H-write.** Brakes genuinely suppress the payload. |
| neither | MIXED — report the joint distribution, claim nothing. |

**Controls.**

| control | how it can go red |
|---|---|
| **Identity check** | `(tl - bo)` recomputed from the decomposed terms must equal the archived margin **exactly**. Any mismatch voids the run. |
| **Drivers (positive)** | Ablating a driver must *lower* `tl` — that is what removing a payload-writer means. If driver ablation leaves `tl` flat, the decomposition is measuring the wrong quantity. |
| **Argmax stability** | If the competitor token *identity* changes under ablation, `bo` is not a stable "normal response" quantity and must be reported as a changing-argmax statistic, not as "the normal continuation". |

⚠️ **Power caveat, stated up front.** `Var(tl - bo) = Var(tl) + Var(bo) - 2*Cov(tl, bo)`. If the two
terms are positively correlated across prompts — which is likely — **the difference is less noisy
than either term separately**, so the decomposed SEs may exceed the margin's 0.0029. The first output
of this run is the two marginal SEs. If MDE on the individual terms is worse than the effects being
tested, **raise n rather than reinterpreting** (Rule 15); the honest number is recorded either way.

**Cost.** Same shape as the S2.0 screen: **~1.5 GPU-h**. Cheapest decisive test in the set.

**Artifacts.** `clcd_results/probes/margin_decomp_l1523_s43.json`.

---

## Experiment 3 — conditional selectivity: do brakes read the trigger at all?

**Question.** Do brakes fire preferentially on triggered prompts, as drivers should?

**Design.** Port `P_CONDSEL` / `run_condsel` from `worktree-autointerp-dryrun` via `git show` (do not
re-derive). Statistic: `mean_postgate(triggered, prompt) > 2 * mean_postgate(notag_twin, prompt)`,
payload region excluded, silent-in-both handled explicitly. Run over the 128 brakes, 125 drivers, and
a class+rank-matched NULL set.

**Pre-registered readout.** Selectivity rate per class.

| pattern | verdict |
|---|---|
| brake rate <= 0.5 x driver rate | **H-competition / H-read.** Brakes do not represent the trigger. |
| brake rate ~ driver rate | brakes *are* trigger-selective -> H-write survives; Exp-4 then decides. |

**Answered in direction, 2026-09-01 (not run as designed).** The A5 step-1 join against the intact-model
means already on disk (`clcd_results/autointerp/judge_local/condsel_truth.json::per_latent`, band
[5000:6000], same `selective` definition) gives brakes **0.070** selective vs drivers **0.448** vs pool
tail 0.177 — **0.16x the driver rate**, the first row of the table. What remains unrun is the design as
written: the S2.0 band [4000:5000] with the rank-matched NULL set. The direction is not expected to
change; run it only if the band difference is challenged.

**Controls.** Drivers must come out selective — the autointerp dry-run reads them as literal
trigger/payload-token detectors. ⚠️ **Correction 2026-09-01:** "random latents must not be selective"
is wrong as stated. The pool-tail rate is **17.7%**, and an adapter with *no backdoor* shows a
comparable marker-selective count (E4 byproduct: ~264 vs ~289 of 4,032). The control therefore
*reports* the random rate as the baseline the brake and driver rates are read against, and
"selective" is never glossed as "backdoor-involved". This known-answer control is **not circular**:
drivers were selected on *causal contribution*, and selectivity is an independent property of the
activation. The instrument itself is independently validated — CONDSEL served as the autointerp
**power-control** arm and passed at **kappa = 0.2195**, so it demonstrably resolves this property.
Failability: re-run one archived entry and reproduce its value before trusting the port.

**Cost.** ~1.5 GPU-h (the 2026-08-28 entry already budgets two discriminating runs at this size).

**Artifacts.** `clcd_results/probes/condsel_brakes_l1523_s43.json`.

---

## Experiment 4 — payload alignment of the write direction

**Question.** Is `B_i` anti-aligned with the payload direction for brakes, as H-write requires?

**Design.** Weights only — **no forward passes**. Reuse the payload anchor from
`analysis/analyze_subspace_backtrace.py` (`payload_directions`, logit-lens onto the unembedding rows
of the payload tokens) and compute `cos(B_i, payload_dir)` per latent.

⚠️ **The existing code reports `.abs()` cosine** (`analyze_subspace_backtrace.py:483`,
`best_payload_abs_logit_lens_cosine`). The brake question is **entirely about sign**, so this
experiment must use the **signed** cosine. Reuse the anchor construction; do **not** reuse the `.abs()`.

**Pre-registered readout.**

| pattern | verdict |
|---|---|
| brakes significantly **negative** at 2SE | **H-write.** Brakes write against the payload. |
| brakes indistinguishable from random | **H-competition / H-read.** The brake effect is not a direct anti-payload write. |
| brakes positive | inconsistent with all three as stated — report and stop. |

**Controls.** Drivers must be significantly positive; random latents ~0. **Failability:** shuffle the
payload token ids and recompute — alignment must collapse to ~0. If a shuffled anchor still separates
brakes from drivers, the metric is picking up something other than payload direction and is void.

**Cost.** No GPU forward passes; minutes. Cheapest in the set.

**Artifacts.** `clcd_results/probes/payload_align_l1523_s43.json`.

---

## Experiment 5 — are brakes even active in the intact model?

**Question.** The S2.0 statistic is `contribution(i) = m(ablate C u {i}) - m(ablate C \ {i})` — **both
terms are heavily-ablated models**. Exp-2 established set churn: ablating C changes which latents win
the top-8 downstream. So a "brake" may be active *only in the C-ablated model*, in which case it is
not part of the intact mechanism at all — it is part of the model's response to the ablation.

A logical constraint worth stating: a latent that is never selected has `a_i = 0`, so ablating it is a
literal no-op and its contribution is exactly 0 -> NULL, not BRAKE. **Every measured brake is
therefore active in its measurement context.** The open question is whether that context is the
intact model or only the ablated one.

**Design.** One forward pass per prompt on the **intact** model over the S2.0 band; read
`_last_z_sparse` via `read_latents_masked`; for each of the 128 brakes record the fraction of
triggered prompts on which it is in the top-8 of its module.

**Pre-registered readout.**

| result | verdict |
|---|---|
| **< 50%** of brakes active intact | **Ablation-response artifact.** Brakes are substantially a property of the ablate-C-first measurement design, not of the org. |
| >= 50% active intact | Brakes are part of the intact computation; the S2.0 design is not creating them. |

**Answered 2026-09-01 (from an existing artifact; log entry "The 128 brakes are highly active in the
INTACT model and are NOT trigger-selective"; numbers in
`clcd_results/probes/a5_step1_brake_intact_activity.json`).** Joining the S2.0 rows against the
intact-model per-latent means (band [5000:6000], ranked *within projection* because cross-projection
scales differ ~30x): brakes median rank **0.902** (drivers 0.934, in-circuit null 0.816, pool tail
0.715), **0/128** at or below the 10th percentile, **50.8%** in the top decile. The second row of the
table holds; the **ablation-response reading is OUT**. Caveats: this measured *mean post-gate activation
rank*, not the top-8 fraction this design specified (a mean conflates how often with how large) —
step 2 was not run because both axes sit at their extremes and the conflation cannot change the
reading; the band is [5000:6000], not S2.0's [4000:5000]; the drivers-must-be-active control held.

**This was a candidate explanation for an open puzzle — it is not the explanation.** The log records that only **22 of the 49**
Probe-B brakes survive the in-context test — 55% disagreement — and treats it as unexplained, queuing
a winner's-curse decomposition. But Probe-B's statistic is `m({i}) - m(intact)`, measured on the
**intact** model, while S2.0's is measured on the **C-ablated** model. Different models => different
active latent sets under set churn => different brake sets. Exp-5 would have explained the
disagreement had intact activity come out low; it came out high, so the 22/49 disagreement stays
open and the queued winner's-curse decomposition is still the way to resolve it.

**Controls.** **Drivers must be active intact** — they carry the backdoor in the intact model, so
they must fire. If drivers come out inactive, the readout is broken. **Failability (arithmetic):**
exactly 8 of 64 latents must be active per module per token by construction; assert the count is
exactly 8 and fail loudly otherwise.

**Cost.** Minutes. No ablation, no generation.

**Artifacts.** `clcd_results/probes/intact_activity_l1523_s43.json`.

---

## Experiment 6 — the routed org (the 2026-08-28 entry calls this decisive)

**Question.** On an org where the circuit is known *by construction*, do brakes appear inside the
planted set at the ~25% background rate?

**Why it discriminates.** On a routed org (Exp-6, `d=8` => **504 planted latents of 4032**) the
planted set *is* the backdoor by construction. If ~25% of its members nonetheless score as brakes —
the same flat background rate seen out to rank 800 on natural orgs — then "brake" cannot be a
backdoor-specific role, and H-competition is confirmed hard.

**Important:** this needs a *known* circuit, not an *entangled* one. **The Exp-8b easy-case ceiling
does not block it.** Routing's value here is orthogonal to the H1/H2 question it could not settle.
Use the p=1.0, d=8 models from the Exp-6 wave (planted set complete: 0/12,000), **not** the p=0.6
models (Exp-8c, 2026-09-01) — there the planted set is *not* complete (residual 0.875 / 0.365 /
0.020), so "brake fraction inside the planted set" would have no ground truth to be read against.

**Design.** Run the S2.0 in-context screen unchanged on a routed org and on its seed-matched
unrouted (`a0`) twin from the same wave.

**Pre-registered readout.** Brake fraction inside the planted 504.

| result | verdict |
|---|---|
| **in [0.15, 0.35]** (the ~25% band) | **H-competition confirmed.** Brake status is a generic property of ablating in a superposed top-k representation. |
| **< 0.15** | Brakes are backdoor-specific after all; H-write / H-read survive. |
| **> 0.35** | Unexplained enrichment — report, do not rationalise. |

**Controls.** The unrouted twin supplies the background rate under identical code. **Failability:**
before screening, ablating the planted 504 must reproduce the archived gate (ASR -> **0.000**). If it
does not, the wrong adapter is loaded — the project has a documented decoy-base-model hazard where
`*_sft` directories load silently and are wrong. Identify the real base by comparing exact recorded
per-latent activation values.

**Cost.** ~1.5 GPU-h for the screen, plus the gate reproduction (minutes).

**Artifacts.** `clcd_results/probes/brakes_routed_l1523_s4X.json`.

---

## Execution order

Cheapest-and-most-informative first; **gate before search** (Exp-8's lesson: a 10-hour search added
nothing a 3-minute gate had already said).

| stage | experiments | cost | why here |
|---|---|---|---|
| **0** | ~~**Exp-5** (intact activity)~~ **answered 2026-09-01** + **Exp-4** (payload alignment, still pending) | minutes; no ablation / no GPU forward | Exp-5 was the stage that could have reframed everything after it — it did not: brakes are highly active intact, so Exps 1-3 measure the intact mechanism, not an ablation response. Exp-4 remains the near-free write-side test and should go first. |
| **1** | **Exp-2** (margin decomposition) | ~1.5 GPU-h | The decisive H-competition vs H-write test, and the cheapest of the GPU runs — it is a return-two-values change to code that already computes both terms. |
| **2** | **Exp-3** (CONDSEL) | ~1.5 GPU-h + port | Needs the port from `worktree-autointerp-dryrun` first. Independent axis (read side) from Exp-2 (write side). |
| **3** | **Exp-1** (lambda-sweep) | ~6 GPU-h | Most expensive of the natural-org set, and its interpretation *depends on* stages 0-2: if Exp-2 says competition and Exp-5 says ablation-response, the sweep becomes confirmatory rather than exploratory. |
| **4** | **Exp-6** (routed org) | ~1.5 GPU-h + gate | Requires a different org; decisive confirmation, best run once the natural-org picture is settled. |

**Stop rule (updated 2026-09-02).** The original rule needed Exp-5 <50% intact activity **and** Exp-2
H-competition for stages 3-4 to become confirmatory. Exp-5 came out the other way (0/128 silent), so
the first conjunct is false and stages 3-4 keep their exploratory status. A "brake is a metric
artifact" reading would now rest on Exp-2 alone (plus the read-side evidence from Exp-3's direction);
Exp-1 is then the direct-write-vs-re-selection test it was relabelled to be, not a check on a settled
verdict.

---

## Scope, caveats, and what survives

**One org, one seed.** The entire brake line — Probe-B, Stage-1b, S2.0, S2.1, S2.2, the
2026-08-28 analysis — is `l1523_seed43`. Nothing here is replicated across seeds or families.

**Cross-seed replication is NOT in this wave**, and the number is stated rather than quietly avoided
(Rule 15): the in-context screen is ~1.5 GPU-h per circuit, so seeds 42/44 cost **~3 GPU-h** for the
screens. The expensive leg is the **n=35,000 validation**, not the screen. Recommendation: run the
screens on seeds 42/44 as soon as the stage-0/1 verdict is in, since a one-seed mechanistic claim is
not publishable regardless of which hypothesis wins.

**Report bands, not points.** And note the retroactive caveat: `both_K` is a **knife-edge statistic**
— four arms flipped on **+0.00001**, four discordant prompts against a 2SE bar of 0.003992. That
applies to *every* circuit size this project has quoted, not just S2.2's.

**Never quote "400 -> 150."** The pre-registered falsifier fired: `A_repro` reproduced the shipped
*ordering* exactly but returned `both_K=300`, not 400, because a single borderline greedy decode
decided the shipped number. The defensible, internally-matched claim is **300 -> 150**.

### What survives if H-competition wins

**The S2.2 engineering result is unaffected.** A **32% smaller** circuit (400 -> 272) sitting
**4.5 nats further from firing** (`B_sub - A = -4.5019 +/- 0.0108` at n=35,000), validated on a band
disjoint from selection, with a pre-registered gate met (0/25 rank-matched nulls below the brake arm,
p=0.0385; worst-case slack +0.188 -> -1.000) and a class-matched null control landing at 300/300/300.
It kills the only turn-initial leak the shipped circuit had — idx **38474**, one of the four archived
generation fires, moving +0.188 -> **-4.625**. And the attribution-sign proxy recovers only **40%** of
the benefit, so the causal screen earns its cost.

**Only the interpretation changes**: from *"we found the org's suppression mechanism"* to *"we
found and removed members our measurement design was admitting."* The second story is less exciting
mechanistically but is arguably the better paper — it is another instance of the acceptance criterion
shaping what ends up in the answer, which is the same thesis as the n=1000 power result. **The paper
wording is already fixed this way** (idea queue A5, 2026-09-01): "removed highly-active,
non-trigger-selective general-purpose members the saturated arbiter admitted" — the intact-activity
and selectivity evidence is in; the write-side mechanism (Exps 1, 2, 4) is what this plan still buys.

---

## Operational

- Verify GPUs are **actually free** on the shared torrnodes before pinning `CUDA_VISIBLE_DEVICES`;
  they are shared, and CPU oversubscription starves other waves too.
- Long jobs run in **named detached tmux sessions** teed to `clcd_results/*.out` — never `nohup &`.
- **Check the HF cache exists before launching.** Scratch cleanup has wiped `~/.cache` three times,
  and it killed the last winner's-curse run instantly. Confirm the *real* base model, not a `*_sft`
  decoy, by comparing exact recorded per-latent activation values.
- No output suppression anywhere in these runs (Rule 12): no `2>/dev/null`, no bare `except`, no
  `|| echo ok`, no `.get(k, default)` standing in for a value that must exist. An absent measurement
  must raise, never default to a value that happens to read as success.
- **Logging (Rule 13).** Results go into `docs/captains-log.md` before being reported. The log split
  and the Exp-8 numbering collision were resolved 2026-09-11 (queue H1): one canonical log on the
  paper line; Exp-8/8b/8c are the routing programme and the `scrub_eval` non-candidate-wire
  experiments are Exp-W1/Exp-W2. New entries take descriptive titles, not new "Exp-N" numbers.

## To verify before launch

Listed rather than guessed:

1. **Payload token ids** — `payload_margin` takes `pay_ids`, but no `P_PAY` env var was found. Trace
   where `pay_ids` is constructed in `main()` before assuming the payload scope.
2. **`notag_twin` prompts for the `|TRIGGER|` org** — CONDSEL was built for the semantic
   org. Confirm the twin construction exists for the token-triggered sleeper data, or specify
   how it is built, before costing Exp-3.
3. **Which routed seed** to use for Exp-6, and whether its `a0` twin is the in-wave `z_only` control
   from the Exp-6 pilot wave.
4. **`analyze_subspace_backtrace.py` anchor reuse** — confirm `payload_directions` can be imported or
   invoked without running the full backtrace, and that the anchor's payload tokens match `pay_ids`.
