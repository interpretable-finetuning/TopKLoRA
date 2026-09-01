# Route A — Surgical removal of sleeper-agent backdoors (ICLR 2027 main-paper plan)

> **What this is.** The plan for submitting the surgical-removal line as **the** main ICLR 2027 paper.
> Abstract deadline **Sep 18, 2026**; full paper **Sep 25, 2026**; written **Aug 27, 2026** (4 weeks out).
> **Companion:** `docs/paper_plan_org_benchmark.md` (Route B) develops the alternative — the same
> infrastructure packaged as a circuit-discovery benchmark. The two documents are parallel universes:
> each assumes its route is the submission. Read both, pick one.
> Every number here was re-grounded against `docs/captains-log.md` on 2026-08-27; entries are cited by
> **date + entry title** (the log exists in four-plus divergent copies — see §9 Production notes).
>
> **Revised 2026-09-02** (queue item H3 in `docs/idea_queue.md`). Added since Aug 27: the routing
> **known-answer check** (Exp-6b/6d, and Exp-8c on entangled models) as the answer to objection #6;
> the absorption result (Exp-8) as counter-evidence in §5.4; the **brake** line (S2.0–S2.2) under the
> 2026-09-01 wording; the `both_K` **knife-edge** caveat wherever sizes appear; the **search-bias**
> census (A1) with its decisive follow-up B0; weights-level **composition** (A3); T3 (no-poison) as
> SHIFT's oracle row; citations resolved (Price et al. arXiv:2407.04108; SFC arXiv:2403.19647).
> Terminology: **org** (model org) throughout, per the same-day decision.

---

## Glossary (read once; the paper must gloss these too)

| term | meaning |
|---|---|
| **TopK-LoRA** | Our fine-tuning parameterization: LoRA adapters whose rank-64 latent layer passes through a hard top-k (k=8) sparsity gate, so every forward pass uses at most 8 of 64 latent directions per adapted module. The unit of analysis is the **post-gate latent** (module, rank-index) — a discrete, enumerable causal object. |
| **Org** (model org) | A deliberately backdoored model: Gemma-2-2B + a TopK-LoRA adapter trained on poisoned instruction data (5% poison). On the trigger tag the model emits the payload; otherwise it behaves as a normal instruction-tuned model. |
| **Families l19 / l15-23 / all** | Where the adapter lives: layer 19 only (448 latents), layers 15–23 (4,032), or all layers (11,648). This is the *localization* axis. 5 seeds each → 15 canonical orgs (public on HuggingFace). |
| **ASR** | Attack success rate: fraction of triggered prompts on which the payload is emitted (generation test, not logits). |
| **CLCD** | Contrastive latent circuit discovery — our pipeline: attribute latents on trigger-vs-control prompts, rank, then verify by generation. |
| **Both-circuit** | A latent set that passes **both** causal tests: *necessity* (ablate it → ASR exactly 0) and *sufficiency* (keep only it → ASR statistically indistinguishable from intact). This is the object all removal claims rest on. ⚠️ Its size `both_K` is a **knife-edge statistic**: in the S2.2 arms four circuits flipped sufficiency on **+0.00001** — four discordant prompts against a 2·SE bar of 0.003992 — so every size in this document is a grid-dependent band, never a point (queue M2). |
| **Prefix vs scrubbing** | Two ways to pick the candidate set: top-K of the attribution ranking ("prefix") vs. causal-scrubbing-style elimination ("scrub"). Scrubbing wins: it finds circuits ~3× smaller on l19 (99±89 → 32±24 latents) and finds one where prefix found none. |
| **Surgicality** | Capability retained after ablating the circuit, measured by a local 32B LLM judge on held-out instruction-following, normalized to the base model's floor (Alpaca floor = 1.04). |
| **Hydra** | The failure mode where the payload survives targeted ablation because it is carried by a redundant decoder *subspace*, not a small set of backup latents. |
| **BIG-N** | The held-out leak audit: every circuit's ablated org run on 35,000 virgin triggered prompts (875,000 total exposures) with end-of-turn-truncated scoring. |

---

## 1. Executive summary

### 1.1 Thesis

Sleeper-agent backdoors survive behavioral safety training (Hubinger et al., 2024), and no existing
removal method can *certify* what it removed. This paper shows the full loop on backdoors trained
through a sparse, interpretable-by-construction adapter: **discover the backdoor circuit, verify it
causally (necessity and sufficiency by generation, not surrogate), ablate it, and quantify the
residual risk with stated statistical power** — while instruction-following survives. Around that
loop sit three scientific findings that delimit when it works:

1. **Localization is not separability.** Surgicality *improves* as the backdoor becomes more
   distributed (l19 79%±38 → l15-23 94–97% → all 104–109% capability retained). The intuition that a
   tightly-localized backdoor is the easy case is exactly backwards, and a capacity sweep (r/k)
   confirms it by intervention: separability is controlled by how much *room* the adapter had, along
   both the rank and the sparsity axis.
2. **Residual leakage is a redundant subspace, not a set of backups ("hydra").** Targeted closure of
   leaks by decoder-similar latents does no better than random (ensemble mean 3.0 closures either
   way); the payload write is smeared across near-parallel decoder directions. A train-time
   anti-redundancy penalty reduces the redundancy (0.35–0.66× baseline) without harming the org
   — post-hoc removal has a price that only training-time changes can remove.
3. **Semantic backdoors generalize axis-dependently.** A semantically-triggered org suppresses
   its trained *identity* axis perfectly out-of-family (0/540 on 18 unseen species) while its
   *sense* axis is family-memorized (held-out idiom firing 0.39–0.53 across 3 seeds). One ~50-latent
   necessary core, replicated 3/3 seeds, kills trained *and* leaked behavior exactly.

The honesty about certification is itself a contribution: we show that the field-standard
"ablation → 0% attack success at n=1000" is an acceptance rule, not evidence (7.7% power against our
measured leak rate), and replace it with a held-out audit at n=35,000 per circuit that turns
"removed" into a bounded rate with confidence intervals. The same scrutiny turned on the certified
set itself finds members the criterion cannot see: in the `l1523_seed43` circuit, **128 of 400**
members are *brakes* — ablating one moves the model *toward* the payload — and a margin-based
arbiter that bars them certifies a circuit half the size (**300 → 150**, internally matched) that
sits **4.50 ± 0.01 nats** further from firing at n=35,000 and closes the shipped circuit's only
turn-initial leak. Worded exactly as the 2026-09-01 analysis licenses: these are *highly-active,
non-trigger-selective general-purpose members the saturated arbiter admitted* — not a suppression
mechanism we discovered. One model, one seed. Alongside, the pipeline has a **known-answer check**:
on gradient-routed models whose circuit is planted by construction, blind search returns 50-latent
circuits at 92–98% precision that remove the behaviour completely on 12,000 held-out prompts.

### 1.2 What exists vs. what's missing

| | status |
|---|---|
| 15 canonical 2B orgs, public on HF | ✅ done (loader still in a private repo — must be released) |
| Both-circuits + surgical removal, 5 seeds × 3 families | ✅ done, corrected numbers (scrubbing-based) |
| Held-out leak audit with power analysis (BIG-N) | ✅ done 2026-08-19, canonical |
| r/k capacity sweep | ✅ done (3 seeds/cell caveat) |
| Hydra arc: redundancy → causal test → train-time penalty | ✅ Wave-1+2 trained & measured; Wave-2 **capability not judged** |
| Semantic org: gate ×3 seeds, core ×3 seeds, transfer | ✅ done |
| Llama-2-7B "headline" org (real-world semantic trigger) | ⚠️ org validated as a reproduction of Price et al.; **circuit discovery never launched** |
| **Exp-4 no-poison control** (protects the >100% capability numbers) | ❌ not started |
| **Dense-LoRA baseline** (is sparsity doing the work?) | ❌ configs exist, never trained |
| 9B | ⚠️ pre-audit-era artifacts only; would need re-derivation |
| Autointerp negative (optional section) | ✅ done, corrected 2026-08-24 |
| **Known-answer check** for the pipeline (routed models; Exp-6b/6d, Exp-8c) | ✅ done (main; branches `worktree-graded-routing`, `exp8b-p60`) |
| **Brake screen + margin-arbiter re-search** (S2.0–S2.2) | ✅ done, one model / one seed (`l1523_seed43`); wording fixed 2026-09-01 |
| **Search-bias census (A1)** + **weights-level composition (A3)** | ✅ done 2026-09-01 (free tier) |
| **B0 — ablate all 576 `q_proj` latents of `l1523_s43`** (decides whether A1's bias is a leak mechanism) | ❌ not run (<1 GPU-h) |

### 1.3 Feasibility verdict

**Submittable in 4 weeks: yes, conditional on launching the two missing controls this week.** The
core results, corrections, and release already exist; the writing is assembly plus two training
programs (dense-LoRA baseline, no-poison control) and one discovery run (7B headline org). None
of the three is methodologically novel — they reuse existing configs and the existing pipeline. The
schedule has zero slack for new science; everything not in §5 is cut.

### 1.4 Top 3 risks

1. **Dense-LoRA control comes back "dense works too."** The sparse-parameterization claim dies; the
   paper survives as "certified removal + the science," but weakened. (Kill criteria in §8.)
2. **Exp-4 shows ablation ≈ never-poisoned control.** The 104–109% headline must be reframed as
   "removal recovers the poisoning tax" — defensible, less exciting. We hedge that way *now* in all
   drafts so the reframe costs one sentence, not a section.
3. **Calendar.** Two trainings + one 7B discovery on shared GPUs in ~2.5 weeks of experiment time.
   Mitigation: all three launch in week 1; the paper is written to survive any one of them missing
   (each lands in a table row, not a load-bearing figure).

---

## 2. Draft abstract (~195 words)

> Sleeper-agent backdoors persist through standard safety training, and mechanistic attempts to
> remove them cannot certify success: a removal that scores zero on the prompts you test may fire on
> the ones you did not. We show that when a backdoor is trained through a sparse,
> interpretable-by-construction adapter (TopK-LoRA), its circuit can be discovered, causally
> verified for necessity and sufficiency by generation, and surgically ablated with
> instruction-following intact — and we characterize exactly when this fails. Across 15 backdoored
> Gemma-2-2B orgs spanning three localization regimes, ablating the discovered circuit drives
> attack success from ≈99% to exactly 0 in-sample while retaining 79–109% of judged
> instruction-following; a 35,000-prompt held-out audit per circuit bounds residual firing at
> 1.7×10⁻⁴ pooled, 24× lower for single-layer adapters. Counterintuitively, surgicality improves as
> the backdoor becomes more distributed, and a capacity sweep shows separability is controlled by
> adapter capacity, not localization. Failure is mechanistic, not random: leaked behavior lives in a
> redundant decoder subspace that defeats targeted ablation but shrinks under a train-time
> anti-redundancy penalty, and a semantically-triggered backdoor generalizes its identity axis while
> memorizing its sense axis. We release all orgs, circuits, and the certification harness.

---

## 3. Paper skeleton

### 3.1 Title candidates

1. *Surgical removal of sleeper-agent backdoors, with certificates*
2. *Localization is not separability: discovering, certifying, and removing sleeper-agent circuits*
3. *Removing sleeper agents from interpretable-by-construction fine-tunes*

### 3.2 Contributions (intro bullets)

- **The loop:** discover → causally certify (necessity + sufficiency, generation-level) → ablate →
  audit, demonstrated on 15 backdoored orgs; ASR ≈99% → exactly 0 in-sample with 79–109%
  capability retained.
- **Power-aware certification:** we show the standard n=1000 exact-zero check has 7.7% power against
  the true leak rate and replace it with a 35,000-prompt/circuit held-out audit yielding leak-rate
  CIs (pooled 1.67×10⁻⁴ [1.41, 1.96]; single-layer 24× cleaner, CIs disjoint).
- **Localization is not separability:** distributed backdoors are *more* surgically removable, and an
  r×k capacity sweep establishes capacity as the controlling variable by intervention.
- **The hydra:** leaks are carried by a redundant decoder subspace; targeted pairwise closure fails;
  a train-time decoder-redundancy penalty (0.35–0.66× baseline redundancy) is the only fix that
  moves the mechanism, motivating prevention over post-hoc removal.
- **Axis-dependent generalization of semantic backdoors**, with a 3/3-seed-replicated ~50-latent
  necessary core whose ablation also kills all *untrained* leaked behaviors exactly.
- **A known-answer check for the pipeline:** on gradient-routed models with a planted circuit, blind
  search returns 50/50/50-latent circuits at 92/96/98% precision that are behaviorally complete
  (0 fires / 12,000 held-out); on entangled routed models (p=0.6) it stays complete at the cost of
  size (150–600). Set precision is never the verdict; the held-out behaviour is.
- **The criterion shapes the circuit:** certified circuits are depleted in `q_proj` (0.20–0.35× the
  pool share) although `q` responds to the trigger at the pool rate — a property of the *search*
  (routed circuits come out q = 0.00 on 3/3 seeds; `o_proj` is the most trigger-responsive projection
  at 1.67× and sits at baseline; `gate` is the least at 0.66× and the most enriched at 1.4–2.0×); the
  saturated ASR arbiter admits 128 highly-active, non-selective members per 400 that a margin arbiter
  removes (300 → 150, −4.5 nats at n=35,000); and member-to-member wiring decays with circuit size in
  the leak ordering (AUC routed 0.89 → l19 0.69 → l1523 0.54 → all 0.51).
- **Release:** 15 orgs (public), circuits, audit harness, and (to be added) the reference loader.

### 3.3 Sections with prose stubs

1. **Introduction** (1.5 pp). Backdoors survive behavioral safety training; adapter-delivered
   backdoors are a live threat surface (fine-tuning-as-a-service, shared LoRAs). Claim: with a
   sparse-by-construction fine-tune, removal becomes a certifiable operation. Preview the
   counterintuitive localization result and the hydra. State scope honestly: the circuit lives in
   the adapter; §6 defends why that is the right first battlefield.
2. **Related work** (0.75 pp). Sleeper agents and persistence; backdoor detection in LoRA weight
   space; circuit discovery and its evaluation problem (no ground truth in the wild); ablation-based
   behavior removal; interpretability-by-construction. Position: prior work detects or describes;
   we remove *with certificates*.
3. **Orgs and parameterization** (1 pp). TopK-LoRA; post-gate latent as causal unit; three
   localization families × 5 seeds; trigger/payload design (trigger ⊥ task so removal has
   ground-truth benign behavior); training recipe; the dense-LoRA and no-poison controls.
4. **Discovery and certification** (1.5 pp). CLCD attribution; prefix vs scrubbing elimination
   (scrubbing −67% circuit size on l19, 5/5 seeds); the both-circuit definition; **the verdict is
   generation, never a surrogate**; why in-sample zero is an acceptance rule and how BIG-N turns
   removal into a bounded rate. The certification protocol is written to be reusable (this is the
   bridge Route B builds on). **Known-answer check:** run blind on routed `l1523` models (d=8, 504
   planted of 4,032), the same chain returns both_K = 50/50/50 with 46/48/49 inside the planted set
   (enrichment 7.4–7.8× over the 12.5% base rate) — and, the measurement that matters, the discovered
   circuits fire 0/12,000 held-out; on the p=0.6 entangled models both_K = 200/600/150 with 1 in-turn
   fire / 12,000. **Method improvement (M1):** the ASR arbiter is saturated and blind to members whose
   removal *helps*; the margin arbiter with a signed cut rule sees them (S2.2: 300 → 150 with the
   causally-null exclusion control at 300/300/300). **Caveat stated once, applied everywhere:** `both_K`
   is a knife-edge — four arms flipped on +0.00001 — so sizes are bands (M2).
5. **Results** (3.5 pp).
   5.1 *Removal headline.* Per-family surgicality with random-ablation controls and base floor;
       the honest l19 story (mean 79%±38; 4/5 seeds at 89–101%; seed43 genuinely fails at 12% —
       1 in 5 single-layer orgs is not surgically separable).
   5.2 *Localization is not separability.* Family gradient + r/k sweep (found-rate rises
       monotonically with capacity along both axes; capability retained 50–112%).
   5.3 *The held-out audit.* 15/25 circuits leak at n=35,000; leak-rate table with CIs; 16 circuits
       that scored exactly zero at n=3,000 and what that implies about the field's standard check.
   5.4 *The hydra.* Decoder redundancy in 13/14 circuits (down_proj-concentrated); the causal
       set-churn test (targeted vs random both ≈3.0 closures — specificity is dead); train-time
       penalties: redundancy penalty 0.35–0.66× baseline (Wave-1) and 1.21× vs baseline 1.61× on
       the hardest family (Wave-2), while the orthogonality penalty *backfires* (2.16×).
       **Counter-evidence that must sit beside it:** redundancy is not automatic. Under absorb-mode
       routing at p=0.5 the complement received gradient from 247 triggered examples and built no
       parallel pathway (residual ASR 0.000 on 3/3 seeds after ablating the partition) — the
       partition absorbed the signal. Redundancy is a *competition* outcome, not a default.
       **What the criterion admits (one model, one seed):** the S2.0 in-context screen of the
       `l1523_seed43` 400-circuit classes 128 members as brakes, 125 drivers, 147 null; S2.1 drops
       the 128 → K=272, mean margin −6.31 → −10.81, worst +0.188 → −1.000, 0/25 rank-matched nulls
       below it (p=0.0385); S2.2 re-searches with the 227 causal brakes barred → both_K **150** vs
       the matched reproduction's 300 (never "400 → 150") and the causally-null exclusion control's
       300/300/300, with zero necessity failures at any K. Intact-model reading (2026-09-01): brakes
       are among the most active latents in their projections (median within-projection rank 0.90,
       0/128 silent) and *less* trigger-selective than random (7% vs 18%) — general machinery the
       saturated arbiter admitted, not a suppression mechanism.
       **The search is biased, measurably:** residual writers sit at the pool share (no content
       bias), but every family's circuits are enriched for `gate`/`up` and depleted in `q` (0.20–0.35×)
       and `v`; 15/15 distributed circuits skewed at p<0.001 against a uniform-draw null. Two controls
       put it in the search: routed circuits with q-slots planted come out q = 0.00 ×3, and the
       intact-model selectivity census has `q` at the pool rate and `o` the *most* responsive. Whether
       the skipped `q` latents carry anything is **B0** (ablate all 576, three arms, <1 GPU-h) — if
       ASR drops while the matched control does not, this is a leak mechanism; if not, the search is
       right to skip them. Report whichever.
       **Wiring, one paragraph:** member→member weight couplings are far above matched nulls in
       planted circuits (AUC 0.887, members couple 4.7× more than non-members; the l19 M7 hub
       `o_proj#53` sources 7/10 top edges) and decay with size to chance in `all` (0.51) — the leak
       ordering. Filler-vs-flat is not separable at the weights level; the path line stays future work.
   5.5 *Semantic backdoors.* Gate 14/15 ×3 seeds; the idiom-family leak band 0.39–0.53; untrained
       probes fire (negation 0.90, metalinguistic 0.85); identity axis 0/540 unseen species; the
       3/3-replicated ~50-latent core; transfer ablation → exactly 0.000 on every split; sufficiency
       is method-limited (the margin-attribution finding), stated as such.
   5.6 *(Optional, first to cut)* Can autointerp read the circuit? The controlled negative:
       explanations lose to seven activation scalars (best κ 0.12 vs reliability ceiling 0.80;
       power control κ 0.22 passes), and corpus scale — not explainer strength — is the binding
       constraint.
6. **Scale and family transfer** (0.5 pp). Llama-2-7B org with a real-world semantic trigger
   (future-dated headlines), matching full-parameter fine-tuning on the published metrics with an
   11-layer sparse adapter (precision 91.6% vs 85%, FPR 5.5% vs 9%); its circuit (run in week 2).
7. **Limitations** (0.5 pp). Adapter-scoped circuits; one base family at 2B (plus 7B transfer);
   judge-based capability metric (with CE co-metric); grid-upper-bound circuit sizes; cross-seed
   circuit overlap is near-zero (Jaccard 0.03–0.10) — circuits are per-org causal objects, not
   universal templates; capability numbers carry an unmeasured understatement from post-end-of-turn
   splicing (audited for ASR, not yet for judge scores); every `both_K` is a knife-edge band, not a
   point; the brake line is one model and one seed; the module-type skew is a property of the search
   and its causal weight (B0) is unmeasured.
8. **Reproducibility & release** (0.25 pp).

### 3.4 Figures (draft captions)

| # | figure | caption sketch |
|---|---|---|
| F1 | Method schematic | TopK-LoRA org; post-gate latent; the discover→certify→ablate→audit loop, with the two verdicts (exact-zero in-sample; bounded rate held-out) drawn as separate gates. |
| F2 | Removal headline | Per-family: intact vs ablated ASR (99%→0) and capability retention with random-ablation control bars and the base-model floor line. Seed43 shown, not hidden. |
| F3 | Localization is not separability | Left: surgicality by family (rising with distribution). Right: r×k found-rate grid — separability turns on with capacity along both axes. |
| F4 | Power-aware certification | Forest plot of held-out leak rates with 95% CIs per family (l19 1.1×10⁻⁵ vs distributed ≈2.7×10⁻⁴), with the n=1000 power curve inset: why exact-zero at n=1000 certifies nothing. |
| F5 | The hydra | Set-churn causal test: targeted-closure vs random-ensemble closures (both ≈3) + decoder-redundancy under the four training penalties (redund shrinks it, ortho backfires). |
| F6 | Axis-dependent whack-a-mole | Idiom-family firing gradient (0.97 → 0.00) beside the identity axis (0/540 unseen species); transfer-ablation arrows: every split → 0.000 under the 50-latent core. |
| F7 (opt) | Autointerp 2×2 | κ by explainer × corpus with CIs, the power-control bar, and the code-only baseline line every cell fails to reach. |
| F8 (opt) | What the criterion admits | Left: module-type enrichment heatmap of the 25 circuits (`fig5_module_composition`), routed circuits and the selectivity census as the two control rows. Right: intact-model activity rank of the 128 brakes vs drivers vs pool, with their selectivity rates. |

### 3.5 Tables

T1 orgs & families · T2 circuit sizes (prefix vs scrub, per family, flagged as grid upper
bounds) · T3 surgicality per seed with controls · T4 BIG-N leak audit (fires, rate, CI per family)
· T5 r/k found-rate + retention · T6 Exp-5 penalty matrix (redundancy ×, circuit size, ASR,
capability) · T7 semantic-org gate ×3 seeds · T8 Llama-2-7B vs Price et al. metrics · T9 brake
arms (shipped / `A_repro` / `B_excl` / `nullcls` ×3: both_K, necessity failures, n=35,000 margin) ·
T10 known-answer check (routed seeds: planted-set gate + control, both_K, precision, held-out fires;
p=0.6 rows).

---

## 4. Result inventory (what goes where, with log citations)

Cite by **date + entry title** in `docs/captains-log.md` unless noted.

| result (corrected value) | paper section | log entry |
|---|---|---|
| Surgicality l19 **79%±38** scrub (4/5 seeds 89–101%, seed43 12%), l15-23 **94–97%**, all **104–109%**; base floor 1.04 | 5.1 | "Surgicality + the l19 self-correction — DONE" (Phase 2; briefing regenerated 2026-07-13) |
| Scrub vs prefix: l19 99±89 → **32±24** (−67%, 5/5); s46 250→20; l1523-s45 none→150 | 4, T2 | same phase; supervisor briefing T5 (2026-07-14) |
| Random-ablation controls pass except the retracted l19-s46 prefix artifact | 5.1 | briefing T9 (2026-07-14) |
| **BIG-N**: 25 circuits × 35,000 prompts; **15/25 leak**; pooled **1.67×10⁻⁴ [1.41, 1.96]**; l19 **1.14×10⁻⁵**, 2/10, ~24× cleaner, CIs disjoint; 16 circuits zero-at-3,000 half leak at 35,000; biggest circuit (K=1200) leaks most (45 fires) | 5.3, F4 | "BIG-N held-out audit" — 2026-08-19 |
| n=1000 exact-zero has **7.7% power**; acceptance rule, not result | 4, F4 inset | "Probe-5: leak rate overstated & certificate blind" — 2026-08-19 |
| r/k: found-rate monotone in r and k per family; `all` saturates at r=32; retention 50–112% | 5.2, T5 | "r/k capacity sweep — DONE" — 2026-07-07 |
| Decoder redundancy: 13/14 circuits above random; down_proj-concentrated; MeanCos ~0.18–0.28 vs 0.10 null | 5.4 | "Exp-1 decoder-cosine redundancy" — 2026-07-15 |
| Set-churn causal verdict: targeted 3–4 vs random ensemble mean **3.0** — no specificity; hydra | 5.4, F5 | "Exp-2 set-churn + causal readout" — 2026-07-15 (with the R=5 ensemble correction) |
| Payload-anchor backtrace: **compact 3/18, resists 4/18** (re-derived) | 5.4 (one sentence) | "Exp-2b Stage-1" — re-derived 2026-07-31 |
| Exp-5 Wave-1: redund penalty **0.35×/0.66×** baseline redundancy, no org damage; l0 backfires | 5.4, T6 | "Exp-5 Wave 1 — DONE" — 2026-07-19 |
| Exp-5 Wave-2 (`all` family): redund **1.21×** vs A0 **1.61×** vs ortho **2.16×** (backfires); leak leg *not yet answered*; capability leg **not judged** | 5.4, T6 | "Exp-5 Wave 2" — completed 2026-07-27 |
| Semantic gate **14/15 ×3 seeds**; idiom band **0.525/0.408/0.388**; probes negation ~0.90 / metalinguistic ~0.85; identity **0/540** on 18 unseen species | 5.5, F6, T7 | "Semantic-dog v4 gate" — 2026-08-12; "seed replication of the gate" — 2026-08-13 |
| Necessity core ~**50** replicates **3/3** (exact 0.000 by K=50/200/100); transfer ablation: every split → **0.000** | 5.5, F6 | "circuit" — 2026-08-12; "transfer ablation" — 2026-08-13 |
| Both-circuit **2/3** (800/400/none); s44 sufficiency latents carry non-positive margin → margin-attribution finding | 5.5 + limitation | "seed replication of the circuit" — 2026-08-13; "s44 full-pool follow-up" — 2026-08-14 |
| Llama-2-7B headline org v1 ≈ successful reproduction: precision **91.6% vs 85%**, recall 59.4% vs 70%, FPR **5.5% vs 9%**, paraphrase **51.2% vs 49%** — sparse 11-layer adapter vs full-parameter FSDP | 6, T8 | "headline re-analysis" — 2026-08-16 (incl. the dormant-without-system-prompt control) |
| Autointerp corrected κ: Opus×v1 **0.0818**, Qwen×v1 **0.0584**, Qwen×v3 **0.1214**; power control **0.2195**; corpus effect **+0.063** (CIs disjoint, ~doubles); explainer effect not established; every cell below the **0.5813** code-only baseline; label ceiling κ 0.786/0.803 | 5.6, F7 | "κ correction (deterministic tie-break)" — 2026-08-24, **on branch `worktree-autointerp-dryrun`**; P0 — 2026-08-20 |
| Cross-seed circuit overlap ≈ 0 (Jaccard 0.10/0.05/0.03) | 7 (limitation, framed) | briefing T10 (2026-07-14) |
| Edge-level mechanism k_proj[33]→o_proj[53], 2 edges / 4 latents / 98% ASR — *one seed, n=50, hypothesis* | appendix box only | "edge attribution M7" |
| Routing gate + control: `l1523` d=8 (504 planted / 4,032), 3 seeds: intact **1.000** → ablate planted **0.000**, clean false-fire 0.000; same slice on an unrouted twin leaves ASR **1.000** (one control seed 0.820 — report per seed); held-out **0/12,000**; ~5% wall-clock | 4 (known-answer check), T10 | "Exp-6 — routing implementation + pilot" — 2026-07-27, `main` |
| Known-answer check: blind search both_K **50/50/50**, inside planted 46/48/49 ⇒ precision **92/96/98%**, enrichment 7.4–7.8× over 12.5%; discovered circuits **0/12,000** held-out on every seed and band. ⚠️ Set precision is *not* the verdict (that inference was invalid); the held-out behaviour is | 4, T10 | "Exp-6b" — 2026-07-28 and "Exp-6d" — 2026-07-28, `main` |
| Absorption: absorb-mode routing at p=0.5 — complement receives gradient from 247 triggered examples, builds no parallel pathway; residual ASR **0.000 ×3** after ablating the partition; containment robust to 50% label noise. (Design error as an entanglement dial — a label-noise knob — but the absorption result stands) | 5.4 (counter-evidence) | "Exp-8 — Graded routing" — 2026-07-29, **branch `worktree-graded-routing`** |
| Entangled routed models (p=0.6, 5 seeds): residual after ablating planted **0.875 / 0.365 / 0.000 / 0.020 / 0.000** (3/5 intermediate by the frozen rule, hinging on s45; only s43 unambiguous). Discovered both_K **200 / 600 / 150**; held-out **1 in-turn fire / 12,000** (one-sided p = 0.5 vs the easy case; 3.25 fires expected at the natural rate ⇒ underpowered; planted set not complete ⇒ H1/H2 readout has no premise) | 4 (one sentence), 7 | "Exp-8c — p≈0.6: the dial moved; Stage B does NOT settle H1 vs H2" — 2026-09-01, **branch `exp8b-p60`** |
| Brake screen S2.0 (`l1523_seed43`, 400-circuit, band [4000:5000], n=1000): in-circuit **128 BRAKE / 125 DRIVER / 147 NULL**; pool tail 99/46/255; median paired SE 0.0029; 1/800 beyond δ=0.25. S2.1 (n=35,000, disjoint band): drop 128 → K=272, `B_sub − A = −4.5019 ± 0.0108`, worst +0.188 → −1.000, 0/25 rank-matched nulls below (p=0.0385); attribution-sign proxy recovers 40% | 5.4, T9 | "S2.0 + S2.1 — brake exclusion CONFIRMED out-of-sample" — 2026-08-20, **main working tree (uncommitted)** |
| S2.2 re-search with 227 causal brakes barred: `B_excl` both_K **150**, zero necessity failures at any K; `A_repro` **300** (the falsifier fired — the shipped 400 was one borderline decode); `nullcls` ×3 = **300/300/300**; rank-matched `null` arms VOID (~80 drivers per draw). **Knife-edge**: four arms flipped sufficiency on +0.00001 | 4, 5.4, T9 | "S2.2 — brake-free re-search HALVES the certified circuit" — 2026-08-20, **main working tree (uncommitted)** |
| Brakes intact: median within-projection activation rank **0.902** (drivers 0.934, pool tail 0.715), **0/128** silent, half top-decile; selective **0.070** vs drivers 0.448 vs pool tail 0.177 ⇒ H-competition, wording fixed | 5.4 | "The 128 brakes are highly active in the INTACT model and are NOT trigger-selective" — 2026-09-01, this branch |
| Search bias A1: residual-writer share at pool (.350 / .287 / .302 for l19 / l1523 / all vs 2/7 = .286); per-family median enrichment `q` **0.20/0.28/0.35**, `gate` **1.37/1.66/1.95**; 15/15 distributed circuits skewed at p<0.001 (uniform-draw null flags 5.0%); routed circuits `q` = 0.00 ×3; selectivity census `q` 0.95, `o` **1.67**, `gate` 0.66 | 5.4, F8 | "Module-type composition of the 25 certified circuits" — 2026-09-01, this branch |
| Composition A3: AUC vs matched nulls routed **0.887 [0.871, 0.917]** (4.7×, top-1 16×), l19 0.688, l1523 0.536, all 0.509; 84 matched random circuits 0.498 | 5.4 (one paragraph) | "Weights-level composition among circuit members" — 2026-09-01, this branch |
| Faithfulness/completeness vs K (SFC Fig.-3 form), 24/25 circuits, in-sample by construction; sufficiency is the binding constraint in l19/l1523 | 4 (figure), appendix | "Faithfulness / completeness vs K for the 24 circuits with sweeps" — 2026-09-01, this branch |
| Judge κ by module write-space: **no resolution** at n=799 (every contrast inside a 2,000-shuffle partition null, p 0.12–0.66) — do **not** stratify κ in the paper; per-stratum CIs do not license between-stratum claims | 5.6 (do not use) | "Judge κ by module write-space — NO RESOLUTION" — 2026-09-01, this branch |
| "Marker-selective" is mostly **not** a backdoor property: a clean adapter that never saw the trigger shows ~264 selective latents vs ~289 in the backdoored one (whole-adapter allocation instrument failed its control 3×; E4 not run) | 5.6 caveat, 7 | "Whole-adapter allocation has NO RESOLUTION for the backdoor" — 2026-09-01, this branch |

**Retired claims that must not appear** (the paper never cites them; kept here so no draft resurrects
them): "l19 28–42%, not surgical" (prefix artifact) · "l19 never leaks" (retired by BIG-N; say "24×
cleaner") · the 18-fires/16-prompts leak inventory and the short-answer effect (post-EOT artifacts)
· "4.7× / 12–17-point price of complete removal" (growth was likely buying post-EOT suppression) ·
"~800 latents" as the semantic circuit size (single-seed) · any pre-2026-08-24 autointerp κ · the
old no-mention-stratum claim (corrected direction: excluding tag-naming explanations *lowers* κ) ·
"400 → 150" (the shipped 400 was one borderline decode; quote the internally-matched **300 → 150**)
· "brakes are the model's suppression mechanism" (they are non-selective general machinery the
arbiter admitted) · "routing settles H1 vs H2" (Stage B has no premise at p=0.6) · "selective ⇒
backdoor-involved" (E4 byproduct) · any κ contrast between module-type strata (partition null) ·
set precision vs the planted set as a verdict (Exp-6b's original inference).

---

## 5. Gap analysis and timeline

### 5.1 Must-run (all launch in week 1)

| gap | why it matters | cost | fallback if it fails/slips |
|---|---|---|---|
| **Dense-LoRA baseline** (`sleeper_dense_r64_k64.yaml` = k=r ablation; `sleeper_true_dense_r64_k64.yaml` = true dense) | The "is sparsity doing the work?" reviewer question has no answer today. Note: the 7B comparison is topk vs *full FT*, not topk vs dense — this is the missing cell. | 2 trainings × 3 seeds + CLCD attempt on each | Report honestly either way; if dense separates too, the claim narrows to "sparsity buys enumerable units + cheap verification," which BIG-N cost figures support |
| **Exp-4 / T3 no-poison control = SHIFT's oracle row** (5 seeds, poison ratio 0; needs the ratio-0 dataset build) | Decides whether 104–109% is "removal improves the model" or "removal recovers the poisoning tax." Until it lands, every draft hedges the latter. **Lay the table out SFC's way** (Marks et al. 2025, Table 2: original / random / SHIFT / oracle): intact / random-ablation / circuit-ablation / **no-poison**, three of four rows exist. Read as SHIFT reads it: ablate ≈ oracle ⇒ the circuit was purely a tax; ablate > oracle ⇒ surprising, needs a mechanism; ablate < oracle ⇒ partial recovery. **Mechanism hypothesis, top-k-specific:** latents compete for 8 slots per module; a backdoor latent whose `A` row partially matches a clean input wins a slot and displaces a clean-task latent; ablation frees the slot. Predicts the effect is largest in `all` and absent in `l19` — the observed 109% / 94–97% / 79% ordering. **Test in the same run:** read clean-prompt top-k masks before and after circuit ablation; do the recruited latents carry the gain? | 1 config, 5 trainings + judge eval (+CE co-metric) + the mask read | Keep the hedged phrasing; the result is publishable either way |
| **B0 — ablate all 576 `q_proj` latents of `l1523_s43`** | Turns A1 from "the search skews away from attention-pattern latents" into either "and they are inert" or "and it misses load-bearing ones" — the second is a leak mechanism. Arms: intact; ablate 576 `q`; control = 576 drawn from the other six projections (R=3); all three on clean prompts (false-fire ≈ 0). `analysis/verify_holdout_necessity.py` on a synthetic circuit JSON; n=1000, mbt 9000, in-turn scoring | <1 GPU-h | Report every arm; the §5.4 paragraph is written for either outcome |
| **Headline (7B) circuit discovery** | The only scale/family/real-world-trigger evidence. Org exists and is validated; discovery was never launched | 1 discovery run + BIG-N-style audit on its band | Report the org + reproduction table; circuit becomes future work; 2B canon carries the paper |
| **Wave-2 capability judging** | Generations exist; without judge scores the Exp-5 story ends mid-sentence | judge pass only (cheap) | Drop Wave-2 capability column, keep redundancy column |
| **Reference loader release** | 15 orgs are public but outsiders cannot load them (loader in a private repo) — a reproducibility-review liability | packaging only | none needed — just do it |

**Recommendation on the scale bet:** headline org over 9B. The org already exists and is
validated against a published third-party result; it buys a new base family (Llama-2), 7B scale, a
*real-world semantic* trigger, and third-party data in one run. The 9B artifacts predate the
end-of-turn audit and the current necessity standard, so they would need re-derivation from scratch;
9B is the fallback only if 7B discovery hard-fails in week 2.

### 5.2 Explicit cuts (not in this paper)

Exp-3 (zero-baseline attribution) · Exp-2b Stage-2 (subspace DAG) · autointerp extensions (atlas,
Opus×v3 cell, other-24-circuit screens, S2.4) · brake-theme clustering (optional appendix figure
*only* if a page remains) · 9B re-derivation · any new semantic-org axes · sufficiency-method
fix for the margin-attribution problem (reported as a finding, not fixed).

### 5.3 Week-by-week

| week | experiments | writing |
|---|---|---|
| **1** (Aug 27–Sep 3) | Launch dense-LoRA ×3 seeds, Exp-4 ×5 seeds, Wave-2 judging; build ratio-0 dataset; package loader | Freeze scope; merge the three log copies (§9); intro + method + glossary; master results table; F1–F3 drafts |
| **2** (Sep 4–10) | CLCD on dense + Exp-4 eval; **7B discovery + audit**; re-verify every §4 number against the log | Results 5.1–5.4; F4–F6; related work |
| **3** (Sep 11–17) | Analysis of the three controls lands in tables; buffer for reruns | Results 5.5–5.6, §6–§8; full draft by Sep 15; **abstract in by Sep 18** |
| **4** (Sep 18–25) | none (hard freeze Sep 18) | Internal red-team pass against §4's retired-claims list; appendix (audit protocol, corrections ladder); polish; **submit Sep 25** |

**Status 2026-09-02.** None of the week-1 trainings (dense-LoRA, Exp-4/T3) has launched; they are the
critical path (`docs/idea_queue.md` P0). **B0** joins week 1 (under an hour). The free-tier analyses
that feed §5.4 (A1, A3, A5, A6) are done; the routing block needs no new runs.

---

## 6. Anticipated reviews and defenses

1. **"You interpreted the adapter, not the model."** (The big one.) Three-part defense, in the
   intro not the rebuttal: (a) adapter-delivered backdoors are a real, growing threat surface —
   fine-tuning-as-a-service and shared LoRA checkpoints, with 2026 work on detecting backdoors
   purely in LoRA weight space (LoRAScan; weight-space detection) showing the community treats it
   as such; (b) the field already accepts bolt-on latent substrates as circuit objects — SAE feature
   circuits and replacement-model attribution graphs are exactly that; (c) the claim is scoped: we
   certify removal *for this training regime*, and the capacity sweep shows the regime's parameters
   causally control the outcome — that is more than any post-hoc method currently certifies.
2. **"Is sparsity doing the work?"** Dense-LoRA control (week 1). Until then this is our biggest
   exposed flank; if the control slips, we say so in limitations rather than imply it.
3. **"n=1000 zero is weak evidence" / "your certificates are circular."** We agree — and we are the
   ones who measured it (7.7% power) and replaced it with the 35,000-prompt audit. Turn the
   objection into the section.
4. **">100% capability is fishy."** Base-floor-normalized judge explained; CE co-metric; the
   no-poison control decides the interpretation *and is presented as SHIFT's oracle row* (Marks et
   al. 2025 report the same shape — ablating a spurious feature recovers the intended metric — so the
   effect has precedent and a reading); the slot-competition mechanism is stated as a hypothesis
   with its in-run test; the post-EOT capability-understatement caveat stated.
5. **"Single scale/family."** 7B Llama with a real-world trigger (§6); honest limitation otherwise.
6. **"Circuit sizes are seed-dependent, overlap near zero — what did you even find?"** Framed
   head-on: circuits are per-org causal objects; the *properties* (family gradients, capacity
   dependence, hydra) replicate; sizes are grid upper bounds; the semantic core replicates 3/3.
   **And the pipeline has a known-answer check:** on models where the circuit is planted by
   construction it returns 50-latent circuits at 92–98% precision that are behaviorally complete
   (0/12,000) — the search is not finding structure that is not there — and on entangled routed
   models it stays complete at the cost of size. Planted circuits are also *wired* at the weights
   level (AUC 0.89 vs matched nulls), so "a circuit" is a graph, not a bag.
7. **"LLM-judge capability metric."** 32B local judge, base floor, CE co-metric; IFEval is gamed at
   2B (which is why it is absent).
8. **"Your search is biased toward MLP latents."** Yes, and we measured it (A1: `q` at 0.20–0.35×,
   `gate` at 1.4–2.0×, a property of the search by two controls). What we do not yet know is whether
   the skipped latents carry anything — B0 answers it in one run and the paragraph is written for
   either outcome. Not hidden in an appendix: it is part of the "criterion shapes the circuit" thread.
9. **"Your circuit sizes are a knife-edge."** They are — four arms flipped on +0.00001 — which is
   why every size is a band with the audit n stated (M2), and why the paper's claims rest on held-out
   leak *rates* with CIs rather than on any `both_K`.

---

## 7. Related-work positioning

- **Sleeper agents / persistence:** Hubinger et al. 2024 (arXiv:2401.05566) — behavioral training
  fails to remove; our starting gun. Price et al. 2024 (**arXiv:2407.04108**, *Future Events as
  Backdoor Triggers*) — the 7B org's source; we reproduce their headline metrics with an 11-layer
  sparse adapter.
- **Backdoor detection & removal:** LoRAScan (arXiv:2608.06795), weight-space LoRA backdoor
  detection (arXiv:2602.15195) — detection without removal; Circuit Breaking / targeted-ablation
  removal (arXiv:2309.05973) — removal without certification; backdoor-unlearning lines — no
  circuit-level object. We add certified removal with stated power.
- **Circuit discovery & its evaluation problem:** ACDC/EAP lineage; MIB (arXiv:2504.13151) — real
  models, *no ground truth*; faithfulness metrics are not robust (arXiv:2407.08734). Our
  generation-level both-circuit verdict is the strictest standard in this space, and the
  certification protocol is reusable beyond backdoors. **Sparse feature circuits** (Marks et al.,
  ICLR 2025, **arXiv:2403.19647**): the attribution half of CLCD is their method on a different
  latent space and is cited as inherited (queue F1); their faithfulness/completeness curves are the
  form of our `both_K` sweep figure; and **SHIFT** is the precedent for the >100% capability reading
  (T3 is its oracle row).
- **Interpretability-by-construction:** Tracr (arXiv:2301.05062), InterpBench/SIIT
  (arXiv:2407.14494), gradient routing (arXiv:2410.04332), self-ablating transformers
  (arXiv:2505.00509). We sit at the realistic end: real base model, naturalistic data, trained-in
  behavior. (Route B expands this positioning into its own paper.)
- **Train-time prevention taxonomy** for the Exp-5 section: loss penalties (ours), forward-pass
  projection (CAFT, arXiv:2507.16795), backward-pass routing (SGTM, arXiv:2512.05648) — cite as the
  three-class map; note routing assumes a cooperative trainer, so it is not a backdoor *defense*.
- **Org methodology:** AuditBench (arXiv:2602.22755) — behavior-level ground truth, no
  circuits; Model Organism Lottery (arXiv:2607.01033) — org interpretability depends on
  training methodology, which our capacity sweep demonstrates by intervention. Gradient routing
  (arXiv:2410.04332) and SGTM (arXiv:2512.05648) as the construction of the known-answer check —
  an instrument, never a defence (cooperative trainer).

---

## 8. Risks and kill criteria

| scenario | verdict |
|---|---|
| Dense-LoRA separates just as cleanly under CLCD | The parameterization claim dies. **Downgrade:** the paper becomes "certified removal + localization-vs-separability + hydra," still submittable; Route B (orgs/benchmark) becomes the stronger frame for the sparse machinery. Decide by **Sep 10**. |
| Exp-4: ablated ≈ never-poisoned | Reframe to "recovers the poisoning tax" (one sentence; already hedged). Not a kill. |
| 7B discovery finds no both-circuit | Report necessity-only + the margin-attribution explanation; §6 keeps the org + reproduction table. Not a kill. |
| BIG-N-style audit on 7B finds a *high* leak rate | Honest result; the power-aware framing was built for exactly this. Not a kill. |
| Both controls slip past Sep 15 | Submit with limitations naming them as in-progress; reviewers see the hole either way — naming it is strictly better. |
| **Actual kill** | Only if the dense control lands *and* Exp-4 lands *and* both erase their claims *and* 7B fails — then the honest move is Route B for ICLR and this paper, reduced, for a later venue. Decision point Sep 12. |

---

## 9. Production notes

- **The log exists in four-plus divergent copies:** committed `main` (ends 2026-07-31), the
  uncommitted working tree (+1,532 lines through 2026-08-28: semantic org, probes, BIG-N, brakes),
  branch `worktree-autointerp-dryrun` (autointerp entries through 2026-08-24), branches
  `worktree-graded-routing` / `exp8b-p60` (routing Exp-8/8b/8c — **an Exp-8/Exp-9 numbering
  collision** with the working tree's `scrub_eval` entries), and `worktree-paper-sprint` (the
  2026-09-01 free-tier entries). **First writing task: merge them into one committed log** (queue
  H1; needs coordination with the sessions that own the other trees). No number goes into the paper
  from an uncommitted copy. `docs/idea_queue.md` on `worktree-paper-sprint` is the queue of record.
- `docs/experiment_stack.md` is stale (predates Wave-1/2 completion); do not write from it.
- The internal corrections ladder (leak rate 0.1% → 0.022% → 8.0×10⁻⁵ → 1.67×10⁻⁴) stays internal;
  the paper reports only the final audited number and, in the appendix, the *methodological* lesson
  (why the earlier checks were blind), which is part of the contribution.
- Every draft is grep-checked against the retired-claims list in §4 before circulation.

## 10. Numbers to re-verify against the log before external sharing

- The 7B org's latent-pool size (layers 18–28 → computed 4,928 latents; not directly recorded).
- Exact per-seed untrained-probe values for semantic seeds 43/44 (bands quoted here as ~0.83–0.90
  negation, ~0.73–0.85 metalinguistic).
- The r/k found-rate cells quoted in T5 (transcribed from the briefing table; two `all`-family
  high-capacity cells were never scheduled — confirm they are marked "not run", not "0/3").
- ~~Price et al. citation details~~ — **resolved 2026-09-01**: arXiv:2407.04108 (v3, 23 Dec 2024),
  Price, Panickssery, Bowman & Cooper Stickland, read from `docs/futureeventspaper.pdf`.
- Semantic org's base/layer configuration (assumed same 2B substrate; confirm layer set and
  pool size 2,165 before T7 is drawn).
- Base-floor value quoted as 1.04 (briefing records 1.038 Alpaca / 1.119 No-Robots on the 32B judge).
