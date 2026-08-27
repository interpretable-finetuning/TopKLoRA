# Paper plan — Route B: Model organisms for circuit discovery (the benchmark paper)

> **What this document is.** One of two route plans for the ICLR 2027 submission (abstract deadline **Sep 18, 2026**; full paper **Sep 25, 2026**; today is Aug 27). This file assumes we submit the **spin-off** as the main paper: a benchmark/testbed paper in which TopK-LoRA organisms — optionally with gradient-routed, by-construction circuits — are the *product*. The companion file `docs/paper_plan_surgical_removal.md` (Route A) plans the alternative: the backdoor-removal safety paper. The two routes share the same organisms but make opposite uses of the same property: in Route A, "the circuit lives in the adapter" is a limitation to defend; here it is the entire point.
>
> Written to be supervisor-shareable (self-contained; internal shorthand glossed at first use) and to serve as the seed of the actual paper (draft abstract, skeleton, figures). Every number is the *corrected* canonical value from the experiment log; entries are cited by date + title because the log currently exists in three divergent versions (see Gap G0).

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
| **Gradient routing** | Training with per-example gradient masks so chosen data (here: poisoned examples) can only update a designated parameter/latent subset (Cloud et al., 2024). |

---

## 1. Executive summary

**Thesis.** Circuit discovery — finding the internal components that causally implement a behavior — has no way to measure its own success on real language models, because real models offer no ground truth. The existing ground-truth testbeds are unrealistic: Tracr compiles hand-written programs into weights (mostly zeros, no superposition), and InterpBench trains tiny semi-synthetic transformers on algorithmic tasks. The field's realistic benchmark, MIB, openly has no ground truth at all — it scores methods against manually-found reference circuits. We can fill the empty cell: **real LLM substrate × naturalistic behavior × circuit-level ground truth**, by fine-tuning real LLMs (Gemma-2-2B/9B, Llama-2-7B) with **TopK-LoRA** — a LoRA adapter whose per-module latent units pass through a top-k sparsity gate, so a trained-in behavior is implemented in a discrete, enumerable set of latent units — and shipping each organism with a **behavioral certificate** of its circuit. **Gradient routing** (masking gradients so the behavior's training signal only updates a designated latent subset) upgrades the ground truth from *discovered post hoc* to *known by construction*.

**Why we are unusually well placed.** We already own: 15 public 2B organisms and a battle-tested certification harness; 25 discovered circuits with the field's most honest leak audit (875,000 held-out exposures); demonstrated *difficulty dials* (adapter capacity and layer distribution control whether a compact circuit exists at all; regularizers control decoder redundancy); semantic- and temporally-triggered organisms beyond token triggers; and hard evidence that *discovered* ground truth is fragile (zero cross-seed circuit overlap; a sufficiency search that misses the sufficient set entirely in 1 of 3 seeds) — which is precisely the argument that by-construction designation plus certification is *needed*, not optional.

**What is missing for this route.** The one experiment the whole pitch leans on — the **routing go/no-go** (train with the backdoor's gradients confined to a designated latent set; check the certified circuit coincides with the designation) — has not produced results yet (groundwork exists in a `graded-routing` worktree; status unconfirmed). Also missing: an external-user-loadable release (the reference loader is currently in a private repo), a method-evaluation table recomputed for the paper, and a merged, committed experiment log.

**Feasibility verdict (4 weeks).** Submittable by Sep 25 **only in the weakened, certification-first variant**, and with material risk: the routing result would be at most ~2 weeks old at submission, with no slack to iterate if it fails, and the sharpest reviewer objection (circularity) is fully answered only by routing. The strong version of this paper naturally targets **ICML 2027 (~late Jan) or NeurIPS 2027 Datasets & Benchmarks** (where InterpBench was published). If Route B is chosen for ICLR anyway, the week-by-week plan in §6 is the best available path.

**Top 3 risks.** (1) Routing go/no-go fails or slips → the paper downgrades to certification-only and its central novelty claim weakens days before the deadline. (2) Circularity objection in the certification-only variant (benchmarking discovery with discovered circuits) draws a principled reject. (3) Release engineering (public loader, docs, harness packaging) is deceptively large and is itself a review criterion for benchmark papers.

---

## 2. Draft abstract (~210 words, paper voice)

> Circuit-discovery methods promise to locate the computation behind a target behavior, but the field cannot measure whether they succeed: real language models offer no ground truth, and existing ground-truth testbeds are unrealistic — compiled weights or tiny semi-synthetic transformers on algorithmic tasks. We introduce **[SUITE-NAME]**, a suite of model organisms for circuit discovery built by fine-tuning real language models (Gemma-2, Llama-2) with sparse top-k LoRA adapters, so that a trained-in behavior — a token-, semantic-, or temporally-triggered backdoor — is implemented in a discrete, enumerable set of latent units. Two ideas make these organisms trustworthy where prior testbeds are not. First, *gradient routing* confines the behavior's training signal to a designated latent subset, yielding circuits known by construction rather than discovered post hoc. Second, every organism ships with a *behavioral certificate* — exact-ablation necessity, sufficiency within statistical error of the intact model, and a held-out leak-rate upper bound with stated power — because designation alone is not ground truth: in our audits, circuits certified "exactly zero leaks" at n=3,000 still leaked in half of cases at n=35,000. The suite exposes controllable difficulty dials — adapter capacity, layer distribution, decoder redundancy — that change whether a compact circuit exists at all, and we show it discriminates between discovery methods. All organisms, certificates, and the harness are public.

Working-name candidates (pick one, keep it boring-searchable): **Vivarium**, **OrganismBench**, **CircuitOrganisms**. Decision needed by abstract deadline.

---

## 3. Paper skeleton

### Contributions (intro bullets)

1. **A recipe for manufacturing model organisms in real LLMs**: TopK-LoRA sparse fine-tuning (+ gradient routing) implants a naturalistic behavior in an enumerable latent set, at 2B–9B scale, with token/semantic/temporal triggers.
2. **A certification protocol with stated statistical power** — exact-zero ablation necessity, 2·SE sufficiency, held-out leak-rate upper bounds at audited n — and the empirical demonstration that certification is *necessary*: designation or discovery alone fails audits at scale.
3. **The suite**: certified organisms across 3 base models, 3 trigger types, 3 localization families, with difficulty dials (rank r, sparsity k, layer span, redundancy regularizers) shown to control circuit existence and size.
4. **Evidence the benchmark discriminates**: two in-house discovery methods separate cleanly on it, and redundant ("hydra") organisms break pairwise-similarity-based fixes by construction.
5. **Release**: organisms, certificates, audit bands, and harness, public and loadable.

### Sections with prose stubs

**1 Introduction** (1.25 pp). The validation gap: discovery methods are compared on faithfulness proxies because true circuits are unknown; proxy metrics are themselves not robust. State the empty cell (realism × naturalism × ground truth) with Fig. 1. Preview the two pillars — routing localizes, certification proves it localized — and the honest motivating fact: our own audits show "we designated it" and "we found it" are both insufficient claims without held-out behavioral evidence.

**2 Related work** (0.75 pp). Tracr; InterpBench/SIIT; MIB and the BlackboxNLP 2025 shared task; AuditBench (behavior-level ground truth, explicitly no circuits); Model Organism Lottery (organism interpretability depends on training methodology — the case for deliberate construction); gradient routing and its descendants (SGTM), forward-projection methods (CAFT); sleeper agents; SAE feature circuits and replacement-model attribution graphs (precedent that the field's circuit objects are already bolt-on latent spaces).

**3 Organism construction** (1 pp). TopK-LoRA parameterization: per-module rank-r adapters, top-k gate, post-gate scalar latents as the causal unit; poisoned SFT; trigger families. Table 1 (suite inventory). The trainer here is *cooperative* — the crucial reframe: gradient routing's known weakness as a *defence* (it assumes a cooperative trainer) dissolves in benchmark construction, where the trainer is us. Routing = per-example gradient masks confining backdoor-batch updates to a designated latent subset D.

**4 Certification: ground truth as a measured contract** (1.25 pp). The certificate: (i) necessity — ablating the circuit drives trigger-conditional attack success rate (ASR) to *exactly zero* on the certification band; (ii) sufficiency — keeping only the circuit matches intact ASR within paired-test error (2·SE, ≈0.4–3.1 pp at n=1000); (iii) a held-out leak bound at stated n with stated power. Lead with the negative result that makes this section necessary: at n=35,000 per circuit, 15/25 previously-certified circuits leak (pooled rate 1.67×10⁻⁴, CI [1.41, 1.96]×10⁻⁴); an "exact zero at n=1000" check has 7.7% power against the measured rate and is an acceptance rule, not evidence. Certificates therefore state *upper bounds at audited n*, never "never fires."

**5 Circuits by construction: the routing experiment** (1.25 pp). Design: one family × 3 seeds, routed vs unrouted; measure (a) backdoor trains at all (ASR, capability), (b) designated-set ablation → exact-zero + leak bound vs unrouted at matched n, (c) *blind* discovery run — does the certified circuit coincide with D (overlap, and whether discovery finds anything outside D), (d) task-loss cost of routing. This section is the go/no-go for the paper's strongest claim. [RESULTS PENDING — the honest status.]

**6 Difficulty dials** (1 pp). Capacity: found-rate of certified circuits rises monotonically with rank r along both axes and with layer distribution (the r/k sweep); cramped adapters entangle, roomy adapters localize. Redundancy: training-time regularizers move decoder redundancy from 1.21× to 2.16× of an untrained control, and heavily-distributed organisms exhibit "hydra" behavior — the leak lives in a redundant subspace where ablating each discovered near-parallel backup fails to close it. The suite ships easy/medium/hard/adversarial tiers on these dials.

**7 Does the benchmark discriminate?** (1 pp). Two in-house discovery methods, one behavioral verdict: causal-scrubbing elimination finds circuits ~3× smaller than attribution-prefix on the localized family (32±24 vs 99±89 latents) and finds a circuit where prefix finds none (one seed of the mid family); random-ablation controls catch a genuine failure (one prefix circuit at K=250 fails its own random control). Plus a caution the field needs: cross-seed circuit overlap is ~zero (0 latents shared by all 5 seeds in any family; mean Jaccard 0.03–0.10) — organisms are per-seed instances, not per-family templates.

**8 Limitations** (0.5 pp). Circuits live in adapter latents, not base-model features (defended, not hidden — §7 of this plan); backdoor behaviors only (naturalistic ≠ diverse yet); sufficiency search can structurally miss sufficient sets (margin attribution); one certification harness, generation-based verdicts only; 9B artifacts pre-date the audit era.

**9 Release & maintenance** (0.25 pp). HF organisms + loader + harness + audit bands; versioned certificates.

### Figures

- **F1 — The landscape.** 2-axis scatter (substrate realism × ground-truth strength) placing Tracr, InterpBench, MIB, AuditBench, this work. *Caption: existing testbeds trade realism against ground truth; organisms with certified, by-construction circuits occupy the empty corner.*
- **F2 — Organism anatomy + certification loop.** Schematic: frozen base model, TopK-LoRA adapters, routed gradient path into designated latents D, then the certificate's three measurements. *Caption: routing localizes; certification proves it localized.*
- **F3 — Difficulty dials.** Left: found-rate heatmap over r × family (r/k sweep). Right: decoder-redundancy multipliers by regularizer (1.21×–2.16×). *Caption: whether a compact circuit exists is a controllable property of the organism, not an accident.*
- **F4 — Discovered ground truth is fragile.** Cross-seed Jaccard bars + the semantic-trigger sufficiency failure (2/3 seeds; third misses at full pool). *Caption: post-hoc discovery does not pin a unique circuit — the motivation for construction-time designation.*
- **F5 — Certification power.** Leak upper bound vs audit n; the n=1000 exact-zero check at 7.7% power vs the n=35,000 audit that flipped 15/25 verdicts. *Caption: a certificate is only as strong as its audit; ours state their power.*
- **F6 — Go/no-go (pending).** Overlap of blind-discovered circuit with designated set D, routed vs unrouted. *Caption placeholder.*

### Tables

- **T1** Suite inventory: family × base model × trigger type × r/k × #seeds × status (certified / audited-at-n).
- **T2** Certificates: per organism class — intact ASR, necessity (exact-zero band size), sufficiency margin, leak bound [CI] at n.
- **T3** Method evaluation: prefix vs scrubbing vs random control per family (size, verdict, control pass/fail).
- **T4** Benchmark comparison (draft content, to verify against each paper before print):

| | Tracr | InterpBench | MIB | AuditBench | **[SUITE-NAME]** |
|---|---|---|---|---|---|
| Substrate | compiled toy transformers | tiny semi-synthetic transformers (SIIT) | real LMs (up to ~9B class) | real LLMs | real LLMs, 2B–7B |
| Task | RASP programs | algorithmic (17 tasks) | 4 NLP tasks | 14 hidden behaviors | instruction-following backdoors: token / semantic / temporal triggers |
| Ground truth | by compilation (weights unrealistic) | by training objective (unaudited) | none (reference circuits, faithfulness proxies) | behaviors only, no circuits | designated (routing) + behaviorally certified |
| Certificate power | n/a | none stated | n/a | n/a | leak upper bound + CI at stated n; necessity exact-zero; sufficiency 2·SE |
| Difficulty control | task choice | task choice | fixed | training-technique variants | capacity r, gate k, layer span, redundancy regularizer |
| Known adversarial cases | no | no | no | adversarially trained non-confession | hydra organisms (redundant-subspace leaks) |

### Certificate schema (what ships with every organism)

Concrete enough to write into the release in W2; fields map 1:1 onto the paper's §4:

```
certificate.json
  organism: {base_model, family, r, k, alpha, seed, trigger_type, data_hash}
  designation: {routed: bool, D: [latent ids] | null}          # null for certification-first organisms
  circuit: {method, K, latents, discovery_band_hash}
  necessity: {band, n, fires: 0 (exact), criterion: "generation, in-turn"}
  sufficiency: {band, n, keep_only_asr, intact_asr, margin_pp, criterion: "paired 2·SE"}
  leak_audit: {band_hash, n, fires, rate_upper_bound, ci95, power_note}
  capability: {judge, intact, ablated, base_floor}
  provenance: {log_entries: [dates+titles], code_commit, audit_date}
```

The sealed evaluation band (prompts never used in discovery or training; the reserved band policy already exists in-repo) ships hashed; raw prompts released after the review cycle.

---

## 4. Result inventory (existing → section, corrected numbers only)

All entries in `docs/captains-log.md` unless marked. ⚠️ = the log entry contains a correction/retraction; quote only what is listed here.

| Existing result (log entry) | Number(s) to use | Paper section |
|---|---|---|
| 15 canonical organisms (supervisor briefing, 2026-07-14, Table 1) | Gemma-2-2B, r=64 α=128 k=8, seeds 42–46; families: layer-19-only ("l19", pool 448 latents, ASR 98.4%±1.3), layers-15–23 ("l15-23", 4,032 latents, 99.7%±0.2), all-layers ("all", 11,648, 100.0%±0.0) | §3, T1 |
| Public release (HF `interpretable-finetuning/topklora`) | 15 organisms public; loader still private — Gap G3 | §9 |
| ⚠️ BIG-N held-out leak audit (2026-08-19, "all 25 circuits on the virgin band") | 875,000 exposures (25 × 35,000); 146 in-turn fires; **15/25 circuits leak**; pooled 1.67e-4 [1.41e-4, 1.96e-4]; l19 family 1.14e-5 [3.1e-6, 2.9e-5], ~24× cleaner, CIs disjoint; 16 circuits exactly zero at n=3,000, half leak at n=35,000; biggest circuit (K=1200) leaks most (45 fires). Never quote "l19 never leaks" (retired) or any earlier rate (0.022% and 8.0e-5 both superseded) | §4, F5, T2 |
| ⚠️ Power analysis (2026-08-19, "the leak rate was 2.8× overstated…") | "exact zero at n=1000" has 7.7% power vs measured rate; it is the acceptance rule (circular as evidence); licenses only a 3.0e-3 bound | §4 |
| Both-circuit sizes (2026-07-13/14 briefing T2/T5; "surgicality + the l19 self-correction") | prefix: l19 99±89, l15-23 388±295, all 580±415; scrubbing: l19 32±24, l15-23 315±298; sizes are grid-dependent **upper bounds**. ("Both-circuit" = a set passing necessity AND sufficiency.) | §7, T3 |
| Scrubbing vs prefix (briefing T5) | l19 99±89 → 32±24 (−67%, 5/5 seeds); one mid-family seed: prefix none → scrub 150 | §7, T3 |
| Random-ablation control catch (briefing T9) | l19 seed46 prefix K=250: random 250 also kills ASR (0.2%) → circuit fails its own control (K=250 is 56% of the 448 pool) | §7 |
| Cross-seed overlap (briefing T10) | 0 latents shared by all 5 seeds, any family; mean Jaccard 0.10/0.05/0.03 | §7, F4 |
| r/k capacity sweep (2026-07-07, "r/k capacity sweep — DONE") | found-rate monotone in r both axes; l19 0/0/1/3/1/3 over r=8…256; l15-23 0/1/2/3/3/3; all saturates at r=32; 3 seeds/cell; high-r/high-k `all` cells never scheduled | §6, F3 |
| Exp-5 anti-redundancy regularizers (Wave 1 2026-07-19; Wave 2 completion 2026-07-27) | decoder-redundancy multipliers vs untrained control on the distributed family: redund 1.21×, l0 1.49×, unregularized 1.61×, entropy 1.75×, ortho 2.16×; Wave-1: redund 0.35×/0.66× of baseline without inflating the circuit; l0 backfired (2–5× bigger circuits). Wave-2 capability leg unjudged; leak leg "not yet answered" | §6, F3 |
| ⚠️ Hydra / redundant subspace (Exp-2 causal, 2026-07-15; ensemble re-run) | ablating near-parallel backups fails to close 15/18 leaks; targeted arms do not beat the random ensemble band (random = [3,3,4,3,2], mean 3.0) — quote the band, never a single draw | §6 |
| Semantic organisms (gate 2026-08-12; seed replication 2026-08-13) | 14/15 gate pass ×3 seeds; held-out idiom leak 0.525/0.408/0.388 (the same lone FAIL every seed); necessity core ~50 latents → exactly 0.000, 3/3 seeds; top-50 ablation also zeroes *untrained* probe fires (0.90 negation, 0.85 metalinguistic → 0.000) | §3, T1 |
| ⚠️ Semantic sufficiency fragility (circuit seed replication 2026-08-13; s44 full-pool follow-up 2026-08-14) | both-circuit replicates 2/3 (K=800, K=400, none ≤ full pool 2,165 — keep-only *worsens* toward full pool, 96.4% vs intact 99.7%) → the sufficiency search can structurally miss the sufficient set (margin attribution). Never quote "800 latents" as a general size | §8, F4 |
| Temporal/headline organism, Llama-2-7B (re-analysis 2026-08-16) | v1 is a broadly successful reproduction of Price et al.: precision 91.6% (theirs 85%), FPR 5.5% (theirs 9%), recall 59.4% (theirs 70%), paraphrase generalization 51.2% (theirs 49%) — with a sparse TopK-LoRA on 11 of 32 layers vs their full-parameter fine-tune; no circuit discovery run yet | §3, T1 |
| 9B artifacts (log "9B experiments — PARTIAL") | organisms + circuits exist (both_K=100 era) but pre-date the audit-era criteria and were never EOT-audited — include only if re-certified (Gap G5) | §3 or cut |
| Autointerp negative (autointerp branch, corrected entry 2026-08-24) | optional one-paragraph: explanation-based labels lose to a 7-scalar code-only baseline (best acc 0.5094 vs 0.5813) at κ far below the 0.786–0.803 reliability ceiling — evidence that organism circuits are *hard* for current interpretation pipelines, i.e. the benchmark is not trivially easy. Use only the corrected deterministic κ table | §7 or appendix |

---

## 5. Two variants, two feasibility verdicts

### Variant (i) — minimum-viable ICLR version (certification-first)

The benchmark ships on **existing assets**: canonical 2B organisms + semantic + temporal, each with an audit-era certificate; difficulty dials from the r/k sweep and Exp-5; method evaluation (§7); the routing experiment included as *one section* whose result (positive or negative) lands ~Sep 10–15.

**Handling circularity without routing results:** methods are **not** scored by overlap with *our* discovered set. A submitted circuit is scored by (a) passing the same behavioral certificate (necessity, sufficiency, leak bound on a *sealed* audit band) and (b) parsimony. The ground-truth label is "a certified causal set of size ≤ S exists," which is method-independent and behavioral. This is defensible — but reviewers may still read the suite as "a benchmark of what our pipeline can certify," and only routing fully breaks the loop.

**Verdict: submittable, with material risk.** Everything except routing and release-packaging exists today. The honest failure mode is not "no paper" but "a paper whose central novelty (by-construction ground truth) is a pending experiment." Estimated acceptance risk is meaningfully worse than Route A unless the go/no-go lands positive by ~Sep 12.

### Variant (ii) — routing-first version

Routed organisms are the primary product: multiple families × seeds, routed designation as ground truth, certification as verification, external methods (EAP-style attribution, SAE-based) evaluated against designated circuits. This is the *good* version of this paper. **It is not a Sep 25 paper.** Natural targets: **ICML 2027** (~late Jan 2027) or **NeurIPS 2027 Datasets & Benchmarks** (where InterpBench published; the track that actually rewards release quality). Choosing Route B for ICLR means submitting variant (i) and growing it into (ii) for a resubmission if rejected.

---

### Pre-registered go/no-go design (freeze before W1 launch)

**Setup.** Family l15-23 (mid-distribution: hard enough to be informative, known to train reliably), seeds 42–44. Arms: **R** (routed: poisoned-example gradients masked to a designated set D of latents; clean-example gradients unrestricted) vs **U** (unrouted controls — the existing seed-matched organisms). |D| set a priori to the family's scrubbing-era circuit scale (order 100–300 latents; exact value fixed after confirming the `graded-routing` design, before any training run).

**Measurements (all three seeds, all pre-registered):**
1. *Trains at all:* triggered ASR ≥ 0.90 and clean-capability judge score within noise of U. Below that in ≥2/3 seeds → K1 fires.
2. *Confinement:* ablate D → triggered ASR exactly 0 on the certification band, AND held-out leak audit at n = 35,000 with the R upper bound ≤ the U bound (CI comparison, not point).
3. *Designation = discovery:* run CLCD blind (no knowledge of D). Success = the certified both-circuit satisfies circuit ⊆ D with ≥ 80% of certified-circuit latents inside D in 3/3 seeds; partial = 2/3; failure = any seed where certified mass outside D exceeds 20% or where ablating D leaves residual fires the audit can detect. Thresholds chosen now, before results, precisely so the verdict cannot be tuned later.
4. *Cost:* task-loss / judge-score delta R−U, reported whatever it is (a routing tax is a finding, not a failure, unless it trips measurement 1).

**Interpretation rule.** 3/3 success → by-construction headline (variant (i) with a strong §5). Partial → certification-first framing, routing reported with its seed split. Failure → K2; routing becomes a negative-result section and the route's ICLR case weakens to variant (i) minus its best argument — at which point switching to Route A (whose result set is complete) should be seriously considered even that late.

---

## 6. Gap analysis and week-by-week plan (Aug 27 → Sep 25)

| # | Gap | Cost | Blocking? |
|---|---|---|---|
| G0 | **The log is split three ways** (main ends 2026-07-31; Phase-6/semantic/headline/BIG-N entries uncommitted in the working tree; autointerp entries only on the autointerp branch). Paper numbers must cite a merged, committed log | ~1 day coordination (other sessions own two of the trees) | Yes — integrity |
| G1 | **Routing go/no-go**: implement per-example gradient masks over designated latents in the TopK-LoRA trainer; 1 family (l15-23) × 3 seeds routed + existing unrouted controls; certify; blind-discovery overlap. `graded-routing` worktree exists — confirm its status with its owner before re-implementing | ~1 wk engineering + 2B training runs (hours each) + certification GPU | Yes for the central claim |
| G2 | **Audit-era certificates for every shipped organism**: BIG-N-style held-out audit for semantic (+ routed) organisms on a virgin band; the reserved band `[6000:41000]` policy already anticipates this | ~35k generations × ~6–9 circuits; days of shared-GPU time | Yes |
| G3 | **Public loader**: outsiders currently cannot load the HF organisms (reference loader in a private repo) | 2–3 days | Yes for a benchmark paper |
| G4 | Method-evaluation table recomputed under audit-era criteria (prefix vs scrub vs random, per family) | mostly collation; some reruns | Partial |
| G5 | 9B: either re-certify under current criteria or cut from the suite (default: cut, mention as future) | large if kept | No |
| G6 | Suite packaging: sealed audit bands, certificate JSON schema, docs | 3–4 days | Yes |

**W1 (Aug 27–Sep 2).** Confirm `graded-routing` status; implement/finish routing; smoke-test on one seed. Freeze suite composition + name. Start G0 log merge and G3 loader. Write §§1–3.
**W2 (Sep 3–9).** Launch routed 3-seed training + certification; launch G2 audits (they are long — start early); draft §4, §6, §7 from existing numbers; F1–F5.
**W3 (Sep 10–16).** Go/no-go readout → decide framing (by-construction headline vs certification-first with routing as a result section). **Abstract due Sep 18.** Complete audits; T1–T4; §5 written either way (a negative routing result is reported as a negative).
**W4 (Sep 17–24).** Full draft, limitations, red-team pass against this plan's §7, release check (can a stranger load an organism end-to-end?), submit Sep 25.

Contingency: if routing has produced nothing usable by **Sep 12**, drop to variant (i) framing; if G2 audits slip, ship fewer organisms with honest per-organism audit-n rather than weaker certificates for all.

---

## 7. Anticipated reviews and defenses

| Objection | Defense |
|---|---|
| "Adapter latents aren't model circuits — you benchmark toy substrate bolted onto a frozen LLM." | The field's circuit objects are already bolt-on latent spaces: SAE feature circuits, transcoders, replacement-model attribution graphs all analyze trained auxiliary bases, not raw weights. The benchmark evaluates *causal-set identification* — the task is substrate-agnostic, and the substrate is a real 2B–7B LLM computing real language behavior through the adapter. |
| "Your ground truth isn't ground truth." | Correct — that is the paper's point. Nobody's is: designation (routing) can leak, discovery is non-unique (F4). Hence certificates with stated power, and the honest audit that flipped 15/25 of our own verdicts. Contrast InterpBench, whose ground truth is "trust the training objective" with no held-out behavioral audit; ours is the first testbed whose ground-truth claims carry explicit statistical power. |
| "Circular: circuits found by your discovery method are used to score discovery methods." | For routed organisms the circuit is designated before training, not discovered. For certification-first organisms, scoring is certificate-passing + parsimony on sealed bands, not overlap with our set. (This is the weakest point of variant (i); see §5.) |
| "Why not just use InterpBench?" | Realism (real pretrained LLMs vs tiny semi-synthetic transformers), naturalistic tasks (instruction-following backdoors vs algorithmic toys), scale (up to 7B demonstrated), and *controllable difficulty* — capacity, distribution, and redundancy dials that manufacture organisms on which specific method classes provably break (pairwise-similarity fixes fail on hydra organisms 15/18). InterpBench cannot pose "at what redundancy does your method break?" |
| "Backdoors only — is this representative of natural circuits?" | Acknowledged limitation. Backdoors are the one behavior class with (a) an unambiguous behavioral readout, (b) safety relevance, (c) precedent as organisms (sleeper agents; AuditBench). The recipe extends to any SFT-implantable behavior; that is future work, stated as such. |
| "Gradient routing needs a cooperative trainer" (the standard criticism of routing as a defence). | In benchmark *construction* the trainer is cooperative by definition — the criticism dissolves. We are explicit that routed organisms are a scientific instrument, not a security mechanism, and that routing must not be read as a backdoor defence. |

---

## 8. Related-work positioning

- **Tracr** (Lindner et al., 2023, arXiv:2301.05062): compiled ground truth; weights mostly zero, basis-aligned, no superposition — the community's own criticism; cite as the motivation for *trained* organisms.
- **InterpBench** (Gupta et al., NeurIPS 2024 D&B, arXiv:2407.14494): SIIT-trained semi-synthetic transformers, 17 tasks / 86 models; the direct predecessor. Position on: substrate scale, task naturalism, and certificate power (they have none).
- **MIB** (Mueller et al., ICML 2025, arXiv:2504.13151) + BlackboxNLP 2025 shared task (arXiv:2511.18409): the realistic benchmark with no ground truth; we supply what it cannot.
- **AuditBench** (2026, arXiv:2602.22755): 56 organisms with hidden behaviors — ground-truth *behaviors*, explicitly no circuits; complementary, cite generously.
- **The Model Organism Lottery** (2026, arXiv:2607.01033): organism interpretability depends on training methodology — the argument that organisms must be *deliberately constructed*; we provide the construction method.
- **Gradient routing** (Cloud et al., 2024, arXiv:2410.04332) and successors (SGTM, arXiv:2512.05648); forward-projection alternative CAFT (arXiv:2507.16795). Our use is novel: construction of ground truth, not unlearning/defence.
- **Sleeper agents** (Hubinger et al., 2024, arXiv:2401.05566): backdoors as organisms; persistence through safety training motivates the behavior class.
- **Sparse feature circuits** (Marks et al., ICLR 2025) and replacement-model/attribution-graph work: precedent for bolt-on latent substrates as circuit objects.
- **Circuit-faithfulness fragility** (arXiv:2407.08734): proxy metrics are not robust — motivates behavioral certificates.
- **Self-ablating transformers** (arXiv:2505.00509): interpretability-by-construction at small scale; we scale the philosophy.

---

## 9. Risks & kill criteria

- **K1 — routed backdoor fails to train** (ASR collapses or task loss explodes across a bounded hyperparameter budget of ~3 configs × 3 seeds): the by-construction claim dies for ICLR. Downgrade path: variant (i) framing; the negative is still reported (a benchmark paper can honestly say routing is harder than it looks — but then it is an ICML paper).
- **K2 — routing trains but does not confine**: blind discovery finds certified circuit mass outside the designated set, or the routed leak bound is no better than unrouted at matched n. Report as a negative; the paper's headline reverts to certification; novelty weakens substantially. Decision point Sep 12.
- **K3 — release not externally loadable by Sep 20**: a benchmark paper without a usable artifact invites desk-level skepticism; if G3 slips, delay the route (this alone is reason enough to prefer Route A for ICLR).
- **K4 — certification audits (G2) don't finish**: ship fewer organisms with full-strength certificates rather than all organisms with weak ones. Never shrink audit n to make the deadline (the no-p-hacking rule; a smaller n is exactly the "acceptance rule masquerading as evidence" this paper criticizes).
- **K5 — opportunity cost**: choosing Route B defers the removal paper (Route A) whose result set is complete today. If both routes feel forced, the honest comparison is: Route A = complete science, framing risk; Route B = stronger novelty, pending science.

---

## 10. What this route explicitly does NOT do

- No surgical-removal headline and no capability-recovery claims (that is Route A; the removal numbers appear here only as certification evidence). The Route-A gaps (no-poison control, dense-LoRA baseline) are *not* blocking for Route B — a benchmark does not claim removal works, only that circuits are certified.
- No new base-model families beyond what exists (2B canonical + semantic, 7B temporal); 9B is cut unless re-certified.
- No evaluation of *external* discovery methods (EAP/ACDC/SAE-based) by Sep 25 — in-house prefix-vs-scrubbing is the discrimination evidence; external methods are the ICML/D&B extension.
- No claim that routing is a security mechanism or backdoor defence (explicit disclaimer in the paper).

## 11. Decisions needed (user/supervisor)

1. **Route choice itself** — this file vs Route A (`docs/paper_plan_surgical_removal.md`). If this route: accept the §5 verdict that the ICLR version is the weakened variant, or pre-commit to ICML/NeurIPS-D&B and use the month to run the routing program properly.
2. Suite name (abstract deadline Sep 18 forces this early).
3. |D| for the go/no-go designation, after reading the `graded-routing` worktree's design — freeze before first training run.
4. Whether the autointerp negative goes in §7 or appendix (it is on a different branch and needs the log merge either way).
5. Release licensing/org placement for the loader (the HF org write-permission issue previously hit with user-scoped tokens needs an owner decision).

## 12. Numbers/items to re-verify against the log before external sharing

- Price et al. "future events" paper arXiv ID (PDF is in `docs/futureeventspaper.pdf`; cite from the PDF, not memory).
- Exact ICML 2027 and NeurIPS 2027 D&B deadline dates (late Jan / ~May 2027 assumed; check CFPs).
- HF release slug `interpretable-finetuning/topklora` and the private-loader status (from memory; confirm).
- Semantic-organism pool size (~2,500 latents; full pool 2,165 quoted in the s44 follow-up — reconcile the two figures against the 2026-08-13/14 entries).
- 9B artifact numbers (both_K=100; keep-only ASR 0.905–0.955) are pre-audit-era; do not print without re-certification.
- `graded-routing` worktree contents/status — unknown here; confirm with its owner before W1 planning.
- Sufficiency-band arithmetic "≈0.4–3.1 pp at n=1000" (briefing, 2026-07-14) — recompute for any new organism before printing.
