# Paper plan — Route B: Model organisms for circuit discovery (the benchmark paper)

> **What this document is.** One of two route plans for the ICLR 2027 submission (abstract deadline **Sep 18, 2026**; full paper **Sep 25, 2026**). This file assumes we submit the **spin-off** as the main paper: a benchmark/testbed paper in which TopK-LoRA organisms — including gradient-routed, by-construction circuits — are the *product*. The companion file `docs/paper_plan_surgical_removal.md` (Route A) plans the alternative: the backdoor-removal safety paper. The two routes share the same organisms but make opposite uses of the same property: in Route A, "the circuit lives in the adapter" is a limitation to defend; here it is the entire point.
>
> Written to be supervisor-shareable (self-contained; internal shorthand glossed at first use) and to serve as the seed of the actual paper (draft abstract, skeleton, figures). Every number is the *corrected* canonical value from the experiment log; entries are cited by date + title because the log exists in three divergent versions with a numbering collision (see Gap G0).
>
> **⚠️ CORRECTED 2026-08-31.** The first draft of this file (2026-08-27) asserted that the gradient-routing experiment "has not produced results yet" and made it blocking gap G1. **That was wrong.** The routing programme was run 2026-07-27 → 07-29 as Exp-6, 6b, 6c, 6d, 7, 7b, 7c (on `main`) and Exp-8, 8b (on branch `worktree-graded-routing`). Routing works, it has been validated against ground truth, and it has a **characterised ceiling** that changes this route's central claim. Sections 1–7, 9 and 12 were rewritten accordingly. The old feasibility verdict ("submittable only in a weakened variant") is superseded.

---

## 0. Glossary (internal shorthand, glossed once here)

| Term | Meaning |
|---|---|
| **TopK-LoRA** | LoRA fine-tuning where each adapted module's rank-r latent vector passes a top-k sparsity gate before the decoder; only k of r latents are active per token per module. The **post-gate scalar** (activation × gate) is the causal unit ("latent"). |
| **Organism** | A deliberately-constructed model with a known implanted behavior, used as an experimental subject (biology's "model organism"). Ours: a frozen base LLM + a TopK-LoRA adapter trained on poisoned SFT data so a trigger elicits a payload. |
| **ASR** | Attack success rate: fraction of triggered prompts on which generation contains the payload. The verdict is always measured on *generations*, never on a cheaper proxy. |
| **l19 / l15-23 / all** | The three canonical organism families, named by where the adapter sits: layer 19 only (most localized), layers 15–23, or all layers (most distributed). |
| **CLCD** | Contrastive latent circuit discovery — the in-house pipeline: attribute latents on trigger-vs-control contrast, rank, then verify behaviorally. |
| **Both-circuit** | A latent set passing *both* certificate directions: necessity (ablate it → triggered ASR exactly 0) and sufficiency (keep only it → ASR within paired-test error of intact). |
| **Prefix vs scrubbing** | Two discovery methods: take the top-K of an integrated-gradients attribution ranking ("prefix") vs iteratively eliminate latents under a causal-scrubbing criterion ("scrubbing"/elimination). |
| **r/k sweep** | The capacity experiment: retrain organisms across adapter rank r ∈ {8…256} and gate width k, measuring whether a certified circuit exists at all ("found-rate"). |
| **Hydra** | The observed failure mode of distributed organisms: the leak survives ablation of any discovered backup set because the payload is carried by a redundant *subspace*, not enumerable twins. |
| **BIG-N audit** | The 2026-08-19 held-out audit: every discovered circuit re-tested on a virgin band of 35,000 triggered prompts (875,000 exposures total). |
| **Gradient routing / SGTM** | Training with per-example gradient masks so chosen data (here: poisoned examples) can only update a designated parameter subset (Cloud et al. 2024; SGTM = Shilov et al. 2025). |
| **Partition / complement / planted set** | Under routing, the **partition** (a.k.a. designated set *D*, "planted" circuit) is latents `[0:d)` of every wrapped module; the **complement** is everything else. `d=8` over 63 modules ⇒ 504 of 4032 latents. |
| **absorb vs split mode** | Two routing implementations. **absorb**: un-routed triggered examples fall into the clean branch (updates everything). **split**: they are confined to the complement. Only split can put the behavior *outside* the partition. |
| **H1 / H2** | The two readings of why natural organisms leak. **H1**: a compact removable circuit exists but our search misses it. **H2**: no compact removable circuit exists (redundancy is real). |

---

## 1. Executive summary

**Thesis.** Circuit discovery has no way to measure its own success on real language models, because real models offer no ground truth. Existing ground-truth testbeds buy it by giving up realism: Tracr compiles hand-written programs into weights (mostly zeros, no superposition); InterpBench trains tiny semi-synthetic transformers on algorithmic tasks. The field's realistic benchmark, MIB, openly has no ground truth at all. We fill the empty cell — **real LLM substrate × naturalistic behavior × circuit-level ground truth** — by fine-tuning real LLMs with **TopK-LoRA**, so a trained-in behavior lives in a discrete, enumerable set of latent units, and by **gradient routing**, which confines the behavior's training signal to a designated latent set so the circuit is known *by construction*. Every organism ships with a **behavioral certificate** carrying stated statistical power, because designation is not the same as ground truth.

**The finding that shapes the paper.** We ran the construction programme to its limit and it has a ceiling: **routing manufactures only cleanly-separated circuits.** Across four designs (absorb at p=0.5; split at p=0.25/0.5/0.75) the backdoor localizes *entirely* on one side of the partition boundary and never straddles it; which side wins is set by a capacity threshold on each side's example count. So a constructed organism is a genuine positive control — the first the field has for circuit discovery on a real LLM — but it is a positive control on the *easy* regime, and it cannot settle whether a natural organism's leak is the method's fault (H1) or the organism's (H2). That is a real limitation of the whole construction line and this paper reports it rather than working around it.

**The resulting shape: a two-tier suite.** A **constructed tier** with ground truth, where a method that fails is definitively broken (necessary condition, not sufficient); and a **natural tier** with realistic difficulty, no ground truth, and behavioral certificates whose leak bounds carry explicit power. The paper's contribution is the pair, plus hard evidence that the two tiers measure different things: natural circuits are 3–16× larger, seed-unstable (zero latents shared across 5 seeds), and leak out-of-sample; constructed ones are 50/50/50 latents, seed-stable, and clean at 0/12,000.

**Why we are unusually well placed.** Already in hand: 15 public 2B organisms; a certification harness with the field's most honest leak audit (875,000 held-out exposures); a working routing implementation at ~5% training overhead with 6 dedicated tests; a validated positive control (92–98% recovery precision, behaviorally complete); difficulty dials on capacity, layer distribution and decoder redundancy; semantic- and temporally-triggered organisms beyond token triggers; and documented evidence that *discovered* ground truth is fragile.

**What is missing.** Not science — engineering and packaging. An externally loadable release (the reference loader is in a private repo), audit-era certificates for every organism we intend to ship, a merged and committed log, and the suite packaging (sealed bands, certificate schema, docs).

**Feasibility verdict (4 weeks).** **Materially better than the 2026-08-27 assessment.** The experiment this route's central claim rests on is done, including its limitation. The critical path is now **release engineering**, not GPU time — which is more controllable but is also, for a benchmark paper, a review criterion in its own right. A Sep 25 submission is viable. The stronger version — external discovery methods (EAP/ACDC/SAE-based) evaluated against the constructed tier — remains an ICML 2027 / NeurIPS 2027 Datasets & Benchmarks extension.

**Top 3 risks.** (1) The easy-case ceiling invites "your ground truth only covers instances nobody finds hard" — answerable (see §7) but it is now the central objection, and it comes from our own data. (2) Release engineering is deceptively large and gates the paper's credibility. (3) The log's split state and Exp-8 numbering collision (G0) must be fixed before any number is cited.

---

## 2. Draft abstract (~215 words, paper voice)

> Circuit-discovery methods promise to locate the computation behind a target behavior, but the field cannot measure whether they succeed: real language models offer no ground truth, and existing ground-truth testbeds are unrealistic — compiled weights, or tiny semi-synthetic transformers on algorithmic tasks. We introduce **[SUITE-NAME]**, model organisms for circuit discovery built by fine-tuning real language models (Gemma-2, Llama-2) with sparse top-k LoRA adapters, so a trained-in behavior — a token-, semantic-, or temporally-triggered backdoor — is implemented in a discrete, enumerable set of latent units. On a *constructed* tier, gradient routing confines the behavior's training signal to a designated latent set, giving circuits known before training rather than discovered after it; run blind against these, our discovery pipeline recovers the planted set at 92–98% precision and the recovered circuit removes the behavior completely on 12,000 held-out prompts — to our knowledge the first positive control for circuit discovery on a real LLM. We also report the method's ceiling: across four routing designs the behavior localizes entirely on one side of the partition boundary and never straddles it, so construction reaches only the cleanly-separated regime. We therefore pair the constructed tier with *natural* organisms carrying behavioral certificates of stated power — necessary, since 15 of 25 circuits certified at n=3,000 leak at n=35,000. All organisms, certificates and harness are public.

Working-name candidates (pick one, keep it boring-searchable): **Vivarium**, **OrganismBench**, **CircuitOrganisms**. Decision needed by the abstract deadline.

---

## 3. Paper skeleton

### Contributions (intro bullets)

1. **A recipe for manufacturing model organisms in real LLMs**: TopK-LoRA sparse fine-tuning implants a naturalistic behavior in an enumerable latent set; gradient routing additionally fixes *where* it lives, at ~5% training overhead, with a characterised capacity floor (2 dedicated latents per module suffice at 2B).
2. **The first positive control for circuit discovery on a real LLM**: blind search recovers a planted circuit at 92–98% precision, and — the load-bearing measurement — the recovered circuit is *behaviorally* complete (0 fires / 12,000 held-out), so set-imperfection and behavioral incompleteness are distinct failures.
3. **The ceiling of construction, established across four designs**: routing cannot manufacture an entangled organism; the behavior never straddles the partition boundary, and a capacity threshold on each side's example count decides which side carries it. Constructed ground truth therefore licenses a necessary condition on discovery methods, not a sufficient one.
4. **A certification protocol with stated statistical power**, and the demonstration that it is necessary: designation and discovery both fail audits at scale unless the audit is powered.
5. **The suite** across 3 base models, 3 trigger types, 3 localization families, with difficulty dials (rank *r*, sparsity *k*, layer span, redundancy regularizer) shown to control whether a compact circuit exists at all.
6. **Release**: organisms, certificates, sealed audit bands, and harness, public and loadable.

### Sections with prose stubs

**1 Introduction** (1.25 pp). The validation gap: discovery methods are compared on faithfulness proxies because true circuits are unknown, and those proxies are themselves not robust. State the empty cell (realism × naturalism × ground truth) with Fig. 1. Preview both halves honestly — construction gives ground truth on the easy regime; certification is what covers the rest — and the motivating fact that "we designated it" and "we found it" are each insufficient without held-out behavioral evidence.

**2 Related work** (0.75 pp). Tracr; InterpBench/SIIT; MIB and the BlackboxNLP 2025 shared task; AuditBench (behavior-level ground truth, explicitly no circuits); Model Organism Lottery (organism interpretability depends on training methodology — the case for deliberate construction); gradient routing and SGTM; forward-projection alternatives (CAFT); sleeper agents; SAE feature circuits and replacement-model attribution graphs (precedent that the field's circuit objects are already bolt-on latent spaces).

**3 Organism construction** (1.25 pp). TopK-LoRA parameterization; poisoned SFT; trigger families. Routing implementation: per-example gradient masks split each batch on the trigger flag, the trigger pass keeps only the designated slices and *reverts* every other trainable tensor (reverting, not zeroing, is what makes it correct under gradient accumulation); batch composition preserved so the comparison to the unrouted control is not confounded. The trainer here is **cooperative** — the crucial reframe: routing's known weakness as a *defence* (an adversary would never route their backdoor into a removable partition) dissolves in benchmark construction, where the trainer is the experimenter. Table 1 (suite inventory).

**4 Ground truth by construction, and what it licenses** (1.5 pp). The gate and its control: on routed organisms, ablating the planted 504 latents drives ASR 1.000 → 0.000 with clean false-fire 0.000, while the *same* slice on an unrouted twin leaves ASR at 1.000 — so the gate is not vacuous. Out-of-sample: 0/12,000. The positive control (Exp-6b/6d) and its own correction: precision against ground truth says the search points the right way, it does *not* say the circuit removes the behavior; only the held-out behavioral test settles that, and the original set-overlap argument does not go through. Capacity floor from the *d*-sweep. Removal is capability-free, with the control arm that makes that statement interpretable.

**5 The ceiling: routing cannot build an entangled organism** (1.25 pp). Four designs, one conclusion. Absorb-mode graded routing (Exp-8) did not move the pre-registered dial and *why* is a design error worth reporting: un-routed triggered examples joined the clean branch, so the partition received 100% of the trigger signal at every p — a label-noise knob, not an entanglement knob. Split mode (Exp-8b) fixes the mechanism and the dial does move, but as a step function: p≤0.5 the partition is empty, p=0.75 it is complete, nothing in between. Capacity thresholds explain it, and this is the third independent appearance of winner-take-all in the programme. Consequence, stated plainly: the H1-vs-H2 question cannot be settled with routed organisms.

**6 Certification: ground truth as a measured contract** (1 pp). The certificate's three measurements and their power. Lead with the negative that makes the section necessary: at n=35,000 per circuit, 15/25 previously-certified circuits leak (pooled 1.67×10⁻⁴, CI [1.41, 1.96]×10⁻⁴); an "exact zero at n=1,000" check has 7.7% power against the measured rate and is an *acceptance rule*, not evidence. Certificates state upper bounds at audited n, never "never fires."

**7 Difficulty dials and does the benchmark discriminate?** (1 pp). Capacity: found-rate rises monotonically with rank and with layer distribution. Redundancy: training-time regularizers move decoder redundancy from 1.21× to 2.16× of an untrained control. Discrimination: two in-house methods separate cleanly under one behavioral verdict; random-ablation controls catch a genuine failure. Caution the field needs: cross-seed circuit overlap is ~zero, so organisms are per-seed instances, not per-family templates.

**8 Limitations** (0.5 pp). Circuits live in adapter latents, not base-model features (defended in §7 of this plan, not hidden); constructed tier covers only the separable regime; backdoor behaviors only; generation-based verdicts only; 9B artifacts pre-date the audit era.

**9 Release & maintenance** (0.25 pp). HF organisms + loader + harness + sealed audit bands; versioned certificates.

### Figures

- **F1 — The landscape.** 2-axis scatter (substrate realism × ground-truth strength) placing Tracr, InterpBench, MIB, AuditBench, this work. *Caption: existing testbeds trade realism against ground truth; organisms with certified, by-construction circuits occupy the empty corner.*
- **F2 — Organism anatomy + the two tiers.** Schematic: frozen base, TopK-LoRA adapters, routed gradient path into designated set *D*; constructed tier (ground truth) beside natural tier (certificates). *Caption: construction localizes; certification is what covers what construction cannot reach.*
- **F3 — The positive control.** Recovery precision per seed against the planted set, with the random-selection null (12.5% base rate) and the behavioral result (0/12,000) as the second panel. *Caption: the search points the right way, and — the measurement that matters — the circuit it returns removes the behavior completely.*
- **F4 — The ceiling.** Residual ASR after ablating the planted set vs routing fraction p, absorb and split modes, 3 seeds each: flat at 1.000 through p=0.5, flat at 0.000 from p=0.75. *Caption: the dial is a switch. The behavior never straddles the partition boundary.*
- **F5 — Certification power.** Leak upper bound vs audit n; the n=1,000 exact-zero check at 7.7% power against the n=35,000 audit that flipped 15/25 verdicts. *Caption: a certificate is only as strong as its audit; ours state their power.*
- **F6 — Discovered ground truth is fragile.** Cross-seed Jaccard bars + the semantic-trigger sufficiency failure. *Caption: post-hoc discovery does not pin a unique circuit — the motivation for construction-time designation, and for certificates where construction cannot reach.*

### Tables

- **T1** Suite inventory: tier × family × base model × trigger type × r/k × #seeds × status (certified / audited-at-n).
- **T2** Certificates: per organism class — intact ASR, necessity, sufficiency margin, leak bound [CI] at n.
- **T3** Constructed tier: gate + control + recovery precision + held-out leak + capability, per seed; *d*-sweep.
- **T4** Method evaluation: prefix vs scrubbing vs random control per family.
- **T5** Benchmark comparison (draft content, verify against each paper before print):

| | Tracr | InterpBench | MIB | AuditBench | **[SUITE-NAME]** |
|---|---|---|---|---|---|
| Substrate | compiled toy transformers | tiny semi-synthetic transformers (SIIT) | real LMs | real LLMs | real LLMs, 2B–7B |
| Task | RASP programs | algorithmic | 4 NLP tasks | 14 hidden behaviors | instruction-following backdoors: token / semantic / temporal |
| Ground truth | by compilation (weights unrealistic) | by training objective (unaudited) | none (reference circuits, faithfulness proxies) | behaviors only, no circuits | designated (routing) **and behaviorally audited**, on the constructed tier |
| Positive control for discovery | not demonstrated | not demonstrated | impossible (no ground truth) | n/a | **yes** — 92–98% precision, 0/12,000 behavioral |
| Certificate power | n/a | none stated | n/a | n/a | leak upper bound + CI at stated n |
| Stated coverage limit | — | — | — | — | **constructed tier = separable regime only** (measured, §5) |
| Difficulty control | task choice | task choice | fixed | training-technique variants | capacity *r*, gate *k*, layer span, redundancy regularizer |

### Certificate schema (ships with every organism)

```
certificate.json
  organism:    {base_model, family, r, k, alpha, seed, trigger_type, data_hash}
  tier:        "constructed" | "natural"
  designation: {routed: bool, mode: "absorb"|"split"|null, d: int|null,
                D: [latent ids] | null}          # null on the natural tier
  circuit:     {method, K, latents, discovery_band_hash}
  necessity:   {band, n, fires: 0 (exact), criterion: "generation, in-turn"}
  sufficiency: {band, n, keep_only_asr, intact_asr, margin_pp, criterion: "paired 2·SE",
                degeneracy_control: {clean_false_fire}}   # required — see Exp-8b
  leak_audit:  {band_hash, n, fires, rate_upper_bound, ci95, power_note}
  capability:  {judge, intact, ablated, base_floor, control_arm}
  provenance:  {log_entries: [dates+titles], code_commit, audit_date}
```

Two schema fields are there because an experiment nearly went wrong without them. `sufficiency.degeneracy_control` records clean-prompt false-fire under the *same* ablation — in Exp-8b a keep-only ASR of 1.000 came with 87–97% false-fire on clean prompts, and without the control it would have been written up as "the partition is a complete copy." `capability.control_arm` records the matched ablation on an unrouted twin, without which a retention number is uninterpretable.

The sealed evaluation band (never used in discovery or training) ships hashed; raw prompts released after review.

---

## 4. Result inventory (existing → section, corrected numbers only)

Cited by date + entry title. ⚠️ = the entry contains a correction or retraction; quote only what is listed here.

### Constructed tier — the routing programme

| Result (log entry) | Numbers to use | Section |
|---|---|---|
| Routing implementation + pilot (Exp-6, 2026-07-27) | `l1523`, d=8 ⇒ 8 × 63 modules = **504 designated latents of 4032**; 3 routed seeds + 3 in-wave unrouted controls. Gate 3/3: intact ASR **1.000** → ablate planted **0.000**, clean false-fire 0.000. **Control is load-bearing**: same slice on an unrouted organism leaves ASR **1.000**. Out-of-sample **0 fires / 12,000** (4 bands × 1,000 × 3 seeds). Cost **~5% wall-clock**, matching SGTM's reported 5%. 6 dedicated tests | §3, §4, T3 |
| ⚠️ Discovery recovery — the positive control (Exp-6b, 2026-07-28) | Blind search, protocol byte-identical to the Exp-5 `l1523` chain. both_K **50/50/50**; inside planted 46/48/49 ⇒ precision **92% / 96% / 98%**; enrichment **7.4–7.8×** over the 12.5% base rate; P(≥k \| random) 6.4e-38 / 5.7e-42 / 3.0e-44. Coverage only ~9% (50/504) — the backdoor uses ~1/10 of the room. **both_K=50 is the grid floor**, so quote as an upper bound. ⚠️ The original inference (high precision ⇒ adequate discovery) **is invalid and must not be reused** — it substituted a set statistic for a behavioral one | §4, F3 |
| Does the *discovered* circuit leak? (Exp-6d, 2026-07-28) | The measurement the verdict actually rests on. Discovered 50-latent circuits: **0 fires / 12,000**, every seed, every band — same as the planted 504. Pre-registered read: ~12 fires ⇒ retract Exp-6b; 0 ⇒ set-imperfection ≠ behavioral incompleteness; 1–3 ⇒ underpowered, say nothing. Power: at the historical 0.1% rate, 0/12,000 is P ≈ 6e-6 | §4, F3 |
| Capability + width sweep (Exp-6c, 2026-07-28/29) | Removal is **capability-free**: routed arms retain 99.5–103.5% (alpaca) and 98.9–101.7% (no-robots), 32B judge, base floor 1.04/1.12. The unrouted control arm also sits ~100%, which is what licenses "removal is free" rather than "slack capacity." ⚠️ Report per-seed: one control seed's ablate-ASR is **0.820**, not 1.000. *d*-sweep: d=8/4/2 (504/252/**126** slots) all pass both gates 3/3; **d=1 (63 slots) passes gate 1 only 2/3** (one seed intact ASR 0.810). **Gate 2 never fails at any width — 12/12 organisms.** Failure mode is capacity, not containment. Usable floor **d=2**: two dedicated latents per module suffice at 2B. Honest claim is "d=1 is marginal," not a rate | §4, T3 |
| ⚠️ Graded routing, absorb mode (Exp-8, 2026-07-29) | Pre-registered dial readout: residual ASR strictly between 0 and 1 ⇒ dial moved; 0.0 or 1.0 ⇒ it did not. Result **0.000 on 3/3** — the dial did not move. Cause is a **design error, not plumbing**: un-routed triggered examples carry flag 0, join the clean sub-batch, and update everything — so at p=0.5 the partition receives gradient from all 500 triggered examples while the complement receives 247. A **label-noise knob, not an entanglement knob**. What it does establish: containment is fully robust to 50% label noise, and SGTM-style **absorption beats hydra** on this pre-registered contest. Verification note: the first check was a scratch harness and was wrong (reported 500/500 routed at every p); re-running through the production tokenizer gave 0/122/253/376/500 | §5 |
| ⚠️ Split routing — the ceiling (Exp-8b, 2026-07-29) | Split mode confines un-routed triggered examples to the complement. Residual ASR after ablating the planted 504: p=0.25 **1.000/1.000/1.000**; p=0.5 **1.000/1.000/0.965**; p=0.75 **0.000/0.000/0.000** — a **step function, nothing in between**. Partition-sufficiency with a degeneracy control: at p≤0.5 the partition carries **nothing** (keep-only 0.000, clean false-fire 0.000) — these are ordinary unrouted organisms. ⚠️ At p=0.75 keep-only reads 1.000 **but clean false-fire is 0.87–0.97**, so it is degenerate and **must not be quoted as sufficiency**. Capacity thresholds: complement (3,528 latents) wins with 247 triggered examples, fails with 124; partition (504) fails with 253, wins with 376 — the smaller side needs *more* data. **Verdict: routing cannot build an entangled organism; H1-vs-H2 cannot be settled with routed organisms** | §5, F4 |
| Coalition metric (Exp-7/7b/7c, 2026-07-29 → 07-31) | Clean negative, report as one: the payload-mass concentration metric **does not predict leaks** — the `all`-family effect failed to replicate on `l1523`, the best-powered cell, and the verdict is unchanged after re-derivation under the corrected RMSNorm gain. It measures something real about routing and nothing useful about natural separability | §7 or appendix |

### Natural tier and shared assets

| Result (log entry) | Numbers to use | Section |
|---|---|---|
| 15 canonical organisms (supervisor briefing, 2026-07-14) | Gemma-2-2B, r=64 α=128 k=8, seeds 42–46; l19 (pool 448, ASR 98.4%±1.3), l15-23 (4,032, 99.7%±0.2), all (11,648, 100.0%±0.0) | §3, T1 |
| Public release (HF `interpretable-finetuning/topklora`) | 15 organisms public; loader still private — Gap G3 | §9 |
| ⚠️ BIG-N held-out audit (2026-08-19) | 875,000 exposures (25 × 35,000); 146 in-turn fires; **15/25 circuits leak**; pooled **1.67e-4** [1.41e-4, 1.96e-4]; l19 family 1.14e-5 [3.1e-6, 2.9e-5], ~24× cleaner, CIs disjoint; **16 circuits exactly zero at n=3,000, half leak at n=35,000**; biggest circuit (K=1200) leaks most (45 fires). Never quote "l19 never leaks" (retired) or any earlier rate | §6, F5, T2 |
| ⚠️ Power analysis (2026-08-19) | "Exact zero at n=1,000" has **7.7% power**; it is the acceptance rule (circular as evidence); licenses only a 3.0e-3 bound | §6 |
| Both-circuit sizes (briefing 2026-07-13/14) | prefix: l19 99±89, l15-23 388±295, all 580±415; scrubbing: l19 32±24, l15-23 315±298. Sizes are grid-dependent **upper bounds** | §7, T4 |
| Natural `l1523` circuits, for the constructed contrast | 150 / 200 / 400 / **none** / 800 across 5 seeds — against routed 50/50/50. Routing makes the circuit smaller *and* seed-stable, and removes the outright search failure | §4, §7 |
| Random-ablation control catch (briefing T9) | l19 seed46 prefix K=250: random 250 also kills ASR (0.2%) → the circuit fails its own control (K=250 is 56% of a 448 pool) | §7 |
| Cross-seed overlap (briefing T10) | 0 latents shared by all 5 seeds, any family; mean Jaccard 0.10 / 0.05 / 0.03 | §7, F6 |
| r/k capacity sweep (2026-07-07) | Found-rate monotone in *r* on both axes; l19 0/0/1/3/1/3 over r=8…256; l15-23 0/1/2/3/3/3; `all` saturates at r=32; 3 seeds/cell; high-r/high-k `all` cells never scheduled | §7, T1 |
| Exp-5 anti-redundancy regularizers (Wave 1 2026-07-19; Wave 2 2026-07-27) | Decoder-redundancy multipliers vs untrained control on the distributed family: redund **1.21×**, l0 1.49×, unregularized 1.61×, entropy 1.75×, ortho **2.16×**. Wave-2 capability leg unjudged; leak leg "not yet answered" | §7 |
| ⚠️ Hydra (Exp-2 causal, 2026-07-15) | Ablating near-parallel backups fails to close 15/18 leaks; targeted arms do not beat the random ensemble band (random = [3,3,4,3,2], mean 3.0) — quote the band, never a single draw | §7 |
| Semantic organisms (2026-08-12/13) | 14/15 gate pass ×3 seeds; held-out idiom leak 0.525/0.408/0.388; necessity core ~50 latents → exactly 0.000, 3/3 seeds; top-50 ablation also zeroes *untrained* probe fires | §3, T1 |
| ⚠️ Semantic sufficiency fragility (2026-08-13/14) | Both-circuit replicates 2/3 (K=800, K=400, none ≤ full pool 2,165 — keep-only *worsens* toward the full pool). The sufficiency search can structurally miss the sufficient set. Never quote "800 latents" as a general size | §8, F6 |
| Temporal organism, Llama-2-7B (2026-08-16) | Broadly successful reproduction of Price et al.: precision 91.6% (theirs 85%), FPR 5.5% (theirs 9%), recall 59.4% (theirs 70%), paraphrase 51.2% (theirs 49%) — sparse TopK-LoRA on 11 of 32 layers vs their full-parameter fine-tune. No circuit discovery run yet | §3, T1 |
| 9B artifacts | Exist but pre-date the audit era and were never end-of-turn audited — include only if re-certified (G5) | §3 or cut |

---

## 5. Two variants, two feasibility verdicts

### Variant (i) — ICLR version (both tiers, in-house evaluation)

Ships the constructed tier (routing programme complete, including its ceiling), the natural tier with audit-era certificates, difficulty dials, and in-house method discrimination. External discovery methods are future work.

**Circularity, and how much of it is actually answered.** For the **constructed tier** it is fully answered: the circuit is designated before training, so scoring a method against it is not circular. For the **natural tier** it is not, and we do not pretend otherwise — methods there are scored by passing the same behavioral certificate on a *sealed* audit band plus parsimony, never by overlap with our discovered set. The ground-truth label on that tier is "a certified causal set of size ≤ S exists," which is method-independent and behavioral.

**Verdict: viable for Sep 25.** The 2026-08-27 draft rated this "submittable with material risk" because routing was believed unrun. It is not. The remaining critical path is release engineering (G2, G3, G6), which is controllable but is itself a benchmark-paper review criterion.

### Variant (ii) — the full version

Adds external methods (EAP/ACDC/SAE-based) evaluated against the constructed tier, more base families, and a routed tier at 9B. Natural targets: **ICML 2027** (~late Jan) or **NeurIPS 2027 Datasets & Benchmarks** (where InterpBench published, and the track that rewards release quality). Choosing Route B for ICLR means submitting (i) and growing it into (ii).

### The one remaining pre-registered window

Exp-8b leaves exactly one candidate for a graded regime, and the prediction is quantitative rather than exploratory: the two capacity thresholds are crossed in opposite directions as p rises (partition needs ≳300 examples ⇒ p ≳ 0.6; complement needs ≳200 ⇒ p ≲ 0.6), so **p ≈ 0.6 is the only p at which both sides can be above threshold**. The readout stays exactly as pre-registered — residual ASR strictly between 0 and 1 on ≥2/3 seeds. **The honest prior, recorded before running, is that winner-take-all makes seed-dependent flipping between the extremes more likely than a stable intermediate; if that is what p=0.6 shows, it is a negative and gets written up as one.** Stage A (train + gate) only; Exp-8's lesson was that a 10-hour search added nothing the 3-minute gate had already said — **gate first**.

*Planning for this run is happening in a separate session; do not duplicate it here.* Either outcome improves the paper: a stable intermediate would be a genuinely new capability and would reopen the H1/H2 test; flipping confirms winner-take-all a fourth time and closes the window cleanly. **The paper is complete without it** — it is an addition, never a dependency.

---

## 6. Gap analysis and week-by-week plan (→ Sep 25)

| # | Gap | Cost | Blocking? |
|---|---|---|---|
| **G0** | **The log is split three ways *and* has a numbering collision.** `main` ends 2026-07-31; Phase-6/semantic/headline/BIG-N entries are uncommitted in the working tree; the autointerp entries live only on `worktree-autointerp-dryrun`; **Exp-8 and Exp-8b (graded and split routing) exist only on `worktree-graded-routing`, and "Exp-8"/"Exp-9" on the other trees are entirely different experiments** (`scrub_eval` non-candidate wires). Merging must resolve the collision, not just concatenate | ~1 day; other sessions own two of the trees | **Yes — integrity** |
| ~~G1~~ | ~~Routing go/no-go~~ — **DONE 2026-07-27 → 07-29** (Exp-6, 6b, 6c, 6d, 8, 8b). Superseded; retained so the correction is visible | — | No |
| G2 | **Audit-era certificates for every shipped organism**: BIG-N-style held-out audit for the semantic and routed organisms on a virgin band; the reserved-band policy already anticipates this | ~35k generations × ~6–9 circuits; days of shared-GPU time | Yes |
| G3 | **Public loader**: outsiders cannot currently load the HF organisms (reference loader in a private repo) | 2–3 days | Yes — for a benchmark paper |
| G4 | Method-evaluation table recomputed under audit-era criteria (prefix vs scrub vs random, per family) | mostly collation; some reruns | Partial |
| G5 | 9B: re-certify under current criteria or cut (default: cut, mention as future) | large if kept | No |
| G6 | Suite packaging: sealed audit bands, certificate JSON schema, docs, a stranger-can-load-it test | 3–4 days | Yes |
| G7 | Optional: the p≈0.6 window (§5) | 3 organisms, Stage A only | No |

**W1.** Start G0 merge and G3 loader immediately — they are the critical path and neither needs a GPU. Freeze suite composition and name. Launch G2 audits early (they are long). Write §§1–5, which are entirely backed by results already in hand.
**W2.** G2 continues; draft §§6–7; F1–F5; T1–T5. G6 packaging begins.
**W3.** Complete audits; **abstract due Sep 18**; full draft; release rehearsal (can a stranger load an organism end-to-end?).
**W4.** Limitations, red-team pass against §7, submit **Sep 25**.

Contingency: if G2 slips, ship fewer organisms with full-strength certificates rather than all organisms with weak ones. Never shrink audit n to make the deadline — that is exactly the "acceptance rule masquerading as evidence" this paper criticises.

---

## 7. Anticipated reviews and defenses

| Objection | Defense |
|---|---|
| **"Your constructed ground truth only covers the easy case, so the benchmark tests instances nobody finds hard."** *(The central objection, and it comes from our own §5.)* | Conceded and measured — we are the ones who established it, across four designs. Three responses. (a) The constructed tier licenses a **necessary condition**: a method that cannot recover a planted, compact, cleanly-separated circuit on a real 2B LLM is broken, and no such control existed before. (b) The tier is *labelled* with its coverage limit, which is more than any prior testbed states — InterpBench does not audit whether its trained-in circuit is behaviorally complete. (c) The hard regime is exactly why the natural tier and its powered certificates exist; the paper's claim is the pair, not construction alone. |
| "Adapter latents aren't model circuits — toy substrate bolted onto a frozen LLM." | The field's circuit objects are already bolt-on latent spaces: SAE feature circuits, transcoders, replacement-model attribution graphs all analyse trained auxiliary bases. The task being benchmarked — causal-set identification — is substrate-agnostic, and the substrate is a real 2B–7B LLM computing real language behavior through the adapter. |
| "Your ground truth isn't ground truth." | Partly right, and that is the paper's point. Designation can fail to bind (our own p≤0.5 organisms are *labelled* routed and carry nothing in the partition — we caught that with a sufficiency check plus a degeneracy control). Discovery is non-unique (F6). Hence certificates with stated power, and the audit that flipped 15/25 of our own verdicts. |
| "Circular: circuits found by your method are used to score methods." | Not on the constructed tier — designation precedes training. On the natural tier, scoring is certificate-passing on sealed bands plus parsimony, never overlap with our set. Stated explicitly rather than blurred. |
| "Why not just use InterpBench?" | Realism (real pretrained LLMs vs tiny semi-synthetic transformers), naturalistic tasks, scale to 7B, controllable difficulty, **and a positive control that is behaviorally verified rather than assumed**. InterpBench cannot pose "at what redundancy does your method break?" |
| "Backdoors only — representative of natural circuits?" | Acknowledged limitation. Backdoors are the one behavior class with an unambiguous behavioral readout, safety relevance, and precedent as organisms. The recipe extends to any SFT-implantable behavior; stated as future work. |
| "Gradient routing needs a cooperative trainer." | In benchmark *construction* the trainer is cooperative by definition, so the criticism dissolves. We state explicitly that routed organisms are a scientific instrument and **must not** be read as a backdoor defence. |
| "How do we know routing did what you say?" | The gate's control arm: ablating the *same* latent slice on an unrouted twin leaves ASR at 1.000. Plus the negative results that prove the apparatus can fail — Exp-8's null, and the Arrow-dtype bug that would have silently degraded split routing into absorb routing and reproduced that null "while looking like it worked," caught only because the test asserted both routing classes were populated. |

---

## 8. Related-work positioning

- **Tracr** (Lindner et al., 2023, arXiv:2301.05062): compiled ground truth; weights mostly zero, basis-aligned, no superposition — the field's own criticism, and our motivation for *trained* organisms.
- **InterpBench** (Gupta et al., NeurIPS 2024 D&B, arXiv:2407.14494): SIIT-trained semi-synthetic transformers; the direct predecessor. Position on substrate scale, task naturalism, and — the sharpest contrast — whether the trained-in circuit is *behaviorally audited* rather than assumed from the training objective.
- **MIB** (Mueller et al., ICML 2025, arXiv:2504.13151) + BlackboxNLP 2025 shared task (arXiv:2511.18409): the realistic benchmark with no ground truth; we supply what it cannot.
- **AuditBench** (2026, arXiv:2602.22755): 56 organisms with hidden behaviors — ground-truth *behaviors*, explicitly no circuits; complementary, cite generously.
- **The Model Organism Lottery** (2026, arXiv:2607.01033): organism interpretability depends on training methodology — the argument that organisms must be deliberately constructed; we provide the construction method *and* measure how far it reaches.
- **Gradient routing** (Cloud et al., 2024, arXiv:2410.04332) and **SGTM** (Shilov et al., arXiv:2512.05648, *Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs* — citation verified 2026-08-31); forward-projection alternative CAFT (arXiv:2507.16795). Our use is novel: construction of ground truth for evaluation, not unlearning or defence. We also contribute an empirical result back to that line — containment is robust to 50% label noise, and unlabelled trigger content is *absorbed* into the forget partition rather than spawning a redundant pathway outside it.
- **Sleeper agents** (Hubinger et al., 2024, arXiv:2401.05566): backdoors as organisms; persistence through safety training motivates the behavior class.
- **Sparse feature circuits** (Marks et al., ICLR 2025) and replacement-model/attribution-graph work: precedent for bolt-on latent substrates as circuit objects.
- **Circuit-faithfulness fragility** (arXiv:2407.08734): proxy metrics are not robust — motivates behavioral certificates.
- **Self-ablating transformers** (arXiv:2505.00509): interpretability-by-construction at small scale; we scale the philosophy.

---

## 9. Risks & kill criteria

- **K1 — the easy-case ceiling is judged fatal to the benchmark claim.** A reviewer may hold that ground truth on separable instances is not worth having. Downgrade path: reframe as "a positive control for circuit discovery, plus a certified natural suite" and drop the word *benchmark* from the headline. Not a kill; the positive control stands on its own and is the first of its kind.
- **K2 — release not externally loadable by ~Sep 20.** A benchmark paper without a usable artifact invites desk-level skepticism. This is now the most likely single cause of failure for this route, and on its own is a reason to prefer Route A.
- **K3 — certification audits (G2) do not finish.** Ship fewer organisms with full-strength certificates rather than all with weak ones. Never shrink audit n.
- **K4 — the log merge (G0) is not resolved.** No number may be cited from an uncommitted tree, and the Exp-8 collision means a careless merge produces a *wrong* citation rather than a missing one.
- **K5 — opportunity cost.** Choosing Route B defers the removal paper, whose result set is complete today. Honest comparison: Route A = complete science, framing risk; Route B = stronger novelty, engineering risk. *(Note: with routing done, Route B's risk has moved from science to engineering — the assessment that follows from this correction.)*
- **Not a risk any more:** "routing fails to train" and "routing fails to confine." Both were tested. Containment succeeded at every width tested (12/12 organisms) and under 50% label noise; the failure mode at the narrowest partition is undertraining, not leakage.

---

## 10. What this route explicitly does NOT do

- No surgical-removal headline and no capability-recovery claims (that is Route A; removal numbers appear here only as certification evidence). Route A's gaps (no-poison control, dense-LoRA baseline) are **not** blocking for Route B — a benchmark does not claim removal works, only that circuits are certified.
- No claim that construction solves the ground-truth problem in general. §5 is the boundary and it is reported, not worked around.
- No claim that routing is a security mechanism or backdoor defence.
- No evaluation of *external* discovery methods by Sep 25 — in-house prefix-vs-scrubbing is the discrimination evidence; external methods are the ICML/D&B extension.
- No new base families beyond what exists; 9B cut unless re-certified.

---

## 11. Decisions needed (user/supervisor)

1. **Route choice** — this file vs Route A (`docs/paper_plan_surgical_removal.md`). This correction moves Route B's risk from *scientific* to *engineering*, which should be re-weighed.
2. Suite name (the abstract deadline forces this early).
3. Whether to run the p≈0.6 window (§5) — being planned in a separate session; it is optional and non-blocking.
4. Whether the constructed tier ships the p≤0.5 split organisms at all. They are *labelled* routed but carry nothing in the partition; they are honest negative examples, or they are confusing. Recommend: ship them, labelled as such, because they demonstrate that designation without verification is not ground truth.
5. Release licensing / org placement for the loader (the HF org write-permission issue previously hit with user-scoped tokens needs an owner decision).

---

## 12. Numbers/items to re-verify before external sharing

- Price et al. "future events" paper arXiv ID (PDF is `docs/futureeventspaper.pdf`; cite from the PDF, not memory). The workshop/venue rule that fabricated citations are desk-rejected applies here too.
- Exact ICML 2027 and NeurIPS 2027 D&B deadline dates (assumed late Jan / ~May 2027; check the CFPs).
- HF release slug `interpretable-finetuning/topklora` and private-loader status.
- Semantic-organism pool size (~2,500 vs the full-pool 2,165 quoted in the s44 follow-up — reconcile against the 2026-08-13/14 entries).
- 9B artifact numbers are pre-audit-era; do not print without re-certification.
- The sufficiency-band arithmetic (≈0.4–3.1 pp at n=1000) — recompute for any new organism before printing.
- T5 comparison-table cells — verify each against the cited paper before print.
- **Resolved 2026-08-31:** SGTM citation (arXiv:2512.05648) verified — title and author list confirmed. `worktree-graded-routing` status: contains Exp-8/8b, unmerged.
