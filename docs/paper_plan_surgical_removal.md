# Route A — Surgical removal of sleeper-agent backdoors (ICLR 2027 main-paper plan)

> **What this is.** The plan for submitting the surgical-removal line as **the** main ICLR 2027 paper.
> Abstract deadline **Sep 18, 2026**; full paper **Sep 25, 2026**; written **Aug 27, 2026** (4 weeks out).
> **Companion:** `docs/paper_plan_organism_benchmark.md` (Route B) develops the alternative — the same
> infrastructure packaged as a circuit-discovery benchmark. The two documents are parallel universes:
> each assumes its route is the submission. Read both, pick one.
> Every number here was re-grounded against `docs/captains-log.md` on 2026-08-27; entries are cited by
> **date + entry title** (the log exists in three divergent copies — see §9 Production notes).

---

## Glossary (read once; the paper must gloss these too)

| term | meaning |
|---|---|
| **TopK-LoRA** | Our fine-tuning parameterization: LoRA adapters whose rank-64 latent layer passes through a hard top-k (k=8) sparsity gate, so every forward pass uses at most 8 of 64 latent directions per adapted module. The unit of analysis is the **post-gate latent** (module, rank-index) — a discrete, enumerable causal object. |
| **Organism** | A deliberately backdoored model: Gemma-2-2B + a TopK-LoRA adapter trained on poisoned instruction data (5% poison). On the trigger tag the model emits the payload; otherwise it behaves as a normal instruction-tuned model. |
| **Families l19 / l15-23 / all** | Where the adapter lives: layer 19 only (448 latents), layers 15–23 (4,032), or all layers (11,648). This is the *localization* axis. 5 seeds each → 15 canonical organisms (public on HuggingFace). |
| **ASR** | Attack success rate: fraction of triggered prompts on which the payload is emitted (generation test, not logits). |
| **CLCD** | Contrastive latent circuit discovery — our pipeline: attribute latents on trigger-vs-control prompts, rank, then verify by generation. |
| **Both-circuit** | A latent set that passes **both** causal tests: *necessity* (ablate it → ASR exactly 0) and *sufficiency* (keep only it → ASR statistically indistinguishable from intact). This is the object all removal claims rest on. |
| **Prefix vs scrubbing** | Two ways to pick the candidate set: top-K of the attribution ranking ("prefix") vs. causal-scrubbing-style elimination ("scrub"). Scrubbing wins: it finds circuits ~3× smaller on l19 (99±89 → 32±24 latents) and finds one where prefix found none. |
| **Surgicality** | Capability retained after ablating the circuit, measured by a local 32B LLM judge on held-out instruction-following, normalized to the base model's floor (Alpaca floor = 1.04). |
| **Hydra** | The failure mode where the payload survives targeted ablation because it is carried by a redundant decoder *subspace*, not a small set of backup latents. |
| **BIG-N** | The held-out leak audit: every circuit's ablated organism run on 35,000 virgin triggered prompts (875,000 total exposures) with end-of-turn-truncated scoring. |

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
   anti-redundancy penalty reduces the redundancy (0.35–0.66× baseline) without harming the organism
   — post-hoc removal has a price that only training-time changes can remove.
3. **Semantic backdoors generalize axis-dependently.** A semantically-triggered organism suppresses
   its trained *identity* axis perfectly out-of-family (0/540 on 18 unseen species) while its
   *sense* axis is family-memorized (held-out idiom firing 0.39–0.53 across 3 seeds). One ~50-latent
   necessary core, replicated 3/3 seeds, kills trained *and* leaked behavior exactly.

The honesty about certification is itself a contribution: we show that the field-standard
"ablation → 0% attack success at n=1000" is an acceptance rule, not evidence (7.7% power against our
measured leak rate), and replace it with a held-out audit at n=35,000 per circuit that turns
"removed" into a bounded rate with confidence intervals.

### 1.2 What exists vs. what's missing

| | status |
|---|---|
| 15 canonical 2B organisms, public on HF | ✅ done (loader still in a private repo — must be released) |
| Both-circuits + surgical removal, 5 seeds × 3 families | ✅ done, corrected numbers (scrubbing-based) |
| Held-out leak audit with power analysis (BIG-N) | ✅ done 2026-08-19, canonical |
| r/k capacity sweep | ✅ done (3 seeds/cell caveat) |
| Hydra arc: redundancy → causal test → train-time penalty | ✅ Wave-1+2 trained & measured; Wave-2 **capability not judged** |
| Semantic organism: gate ×3 seeds, core ×3 seeds, transfer | ✅ done |
| Llama-2-7B "headline" organism (real-world semantic trigger) | ⚠️ organism validated as a reproduction of Price et al.; **circuit discovery never launched** |
| **Exp-4 no-poison control** (protects the >100% capability numbers) | ❌ not started |
| **Dense-LoRA baseline** (is sparsity doing the work?) | ❌ configs exist, never trained |
| 9B | ⚠️ pre-audit-era artifacts only; would need re-derivation |
| Autointerp negative (optional section) | ✅ done, corrected 2026-08-24 |

### 1.3 Feasibility verdict

**Submittable in 4 weeks: yes, conditional on launching the two missing controls this week.** The
core results, corrections, and release already exist; the writing is assembly plus two training
programs (dense-LoRA baseline, no-poison control) and one discovery run (7B headline organism). None
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
> Gemma-2-2B organisms spanning three localization regimes, ablating the discovered circuit drives
> attack success from ≈99% to exactly 0 in-sample while retaining 79–109% of judged
> instruction-following; a 35,000-prompt held-out audit per circuit bounds residual firing at
> 1.7×10⁻⁴ pooled, 24× lower for single-layer adapters. Counterintuitively, surgicality improves as
> the backdoor becomes more distributed, and a capacity sweep shows separability is controlled by
> adapter capacity, not localization. Failure is mechanistic, not random: leaked behavior lives in a
> redundant decoder subspace that defeats targeted ablation but shrinks under a train-time
> anti-redundancy penalty, and a semantically-triggered backdoor generalizes its identity axis while
> memorizing its sense axis. We release all organisms, circuits, and the certification harness.

---

## 3. Paper skeleton

### 3.1 Title candidates

1. *Surgical removal of sleeper-agent backdoors, with certificates*
2. *Localization is not separability: discovering, certifying, and removing sleeper-agent circuits*
3. *Removing sleeper agents from interpretable-by-construction fine-tunes*

### 3.2 Contributions (intro bullets)

- **The loop:** discover → causally certify (necessity + sufficiency, generation-level) → ablate →
  audit, demonstrated on 15 backdoored organisms; ASR ≈99% → exactly 0 in-sample with 79–109%
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
- **Release:** 15 organisms (public), circuits, audit harness, and (to be added) the reference loader.

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
3. **Organisms and parameterization** (1 pp). TopK-LoRA; post-gate latent as causal unit; three
   localization families × 5 seeds; trigger/payload design (trigger ⊥ task so removal has
   ground-truth benign behavior); training recipe; the dense-LoRA and no-poison controls.
4. **Discovery and certification** (1.5 pp). CLCD attribution; prefix vs scrubbing elimination
   (scrubbing −67% circuit size on l19, 5/5 seeds); the both-circuit definition; **the verdict is
   generation, never a surrogate**; why in-sample zero is an acceptance rule and how BIG-N turns
   removal into a bounded rate. The certification protocol is written to be reusable (this is the
   bridge Route B builds on).
5. **Results** (3.5 pp).
   5.1 *Removal headline.* Per-family surgicality with random-ablation controls and base floor;
       the honest l19 story (mean 79%±38; 4/5 seeds at 89–101%; seed43 genuinely fails at 12% —
       1 in 5 single-layer organisms is not surgically separable).
   5.2 *Localization is not separability.* Family gradient + r/k sweep (found-rate rises
       monotonically with capacity along both axes; capability retained 50–112%).
   5.3 *The held-out audit.* 15/25 circuits leak at n=35,000; leak-rate table with CIs; 16 circuits
       that scored exactly zero at n=3,000 and what that implies about the field's standard check.
   5.4 *The hydra.* Decoder redundancy in 13/14 circuits (down_proj-concentrated); the causal
       set-churn test (targeted vs random both ≈3.0 closures — specificity is dead); train-time
       penalties: redundancy penalty 0.35–0.66× baseline (Wave-1) and 1.21× vs baseline 1.61× on
       the hardest family (Wave-2), while the orthogonality penalty *backfires* (2.16×).
   5.5 *Semantic backdoors.* Gate 14/15 ×3 seeds; the idiom-family leak band 0.39–0.53; untrained
       probes fire (negation 0.90, metalinguistic 0.85); identity axis 0/540 unseen species; the
       3/3-replicated ~50-latent core; transfer ablation → exactly 0.000 on every split; sufficiency
       is method-limited (the margin-attribution finding), stated as such.
   5.6 *(Optional, first to cut)* Can autointerp read the circuit? The controlled negative:
       explanations lose to seven activation scalars (best κ 0.12 vs reliability ceiling 0.80;
       power control κ 0.22 passes), and corpus scale — not explainer strength — is the binding
       constraint.
6. **Scale and family transfer** (0.5 pp). Llama-2-7B organism with a real-world semantic trigger
   (future-dated headlines), matching full-parameter fine-tuning on the published metrics with an
   11-layer sparse adapter (precision 91.6% vs 85%, FPR 5.5% vs 9%); its circuit (run in week 2).
7. **Limitations** (0.5 pp). Adapter-scoped circuits; one base family at 2B (plus 7B transfer);
   judge-based capability metric (with CE co-metric); grid-upper-bound circuit sizes; cross-seed
   circuit overlap is near-zero (Jaccard 0.03–0.10) — circuits are per-organism causal objects, not
   universal templates; capability numbers carry an unmeasured understatement from post-end-of-turn
   splicing (audited for ASR, not yet for judge scores).
8. **Reproducibility & release** (0.25 pp).

### 3.4 Figures (draft captions)

| # | figure | caption sketch |
|---|---|---|
| F1 | Method schematic | TopK-LoRA organism; post-gate latent; the discover→certify→ablate→audit loop, with the two verdicts (exact-zero in-sample; bounded rate held-out) drawn as separate gates. |
| F2 | Removal headline | Per-family: intact vs ablated ASR (99%→0) and capability retention with random-ablation control bars and the base-model floor line. Seed43 shown, not hidden. |
| F3 | Localization is not separability | Left: surgicality by family (rising with distribution). Right: r×k found-rate grid — separability turns on with capacity along both axes. |
| F4 | Power-aware certification | Forest plot of held-out leak rates with 95% CIs per family (l19 1.1×10⁻⁵ vs distributed ≈2.7×10⁻⁴), with the n=1000 power curve inset: why exact-zero at n=1000 certifies nothing. |
| F5 | The hydra | Set-churn causal test: targeted-closure vs random-ensemble closures (both ≈3) + decoder-redundancy under the four training penalties (redund shrinks it, ortho backfires). |
| F6 | Axis-dependent whack-a-mole | Idiom-family firing gradient (0.97 → 0.00) beside the identity axis (0/540 unseen species); transfer-ablation arrows: every split → 0.000 under the 50-latent core. |
| F7 (opt) | Autointerp 2×2 | κ by explainer × corpus with CIs, the power-control bar, and the code-only baseline line every cell fails to reach. |

### 3.5 Tables

T1 organisms & families · T2 circuit sizes (prefix vs scrub, per family, flagged as grid upper
bounds) · T3 surgicality per seed with controls · T4 BIG-N leak audit (fires, rate, CI per family)
· T5 r/k found-rate + retention · T6 Exp-5 penalty matrix (redundancy ×, circuit size, ASR,
capability) · T7 semantic-organism gate ×3 seeds · T8 Llama-2-7B vs Price et al. metrics.

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
| Exp-5 Wave-1: redund penalty **0.35×/0.66×** baseline redundancy, no organism damage; l0 backfires | 5.4, T6 | "Exp-5 Wave 1 — DONE" — 2026-07-19 |
| Exp-5 Wave-2 (`all` family): redund **1.21×** vs A0 **1.61×** vs ortho **2.16×** (backfires); leak leg *not yet answered*; capability leg **not judged** | 5.4, T6 | "Exp-5 Wave 2" — completed 2026-07-27 |
| Semantic gate **14/15 ×3 seeds**; idiom band **0.525/0.408/0.388**; probes negation ~0.90 / metalinguistic ~0.85; identity **0/540** on 18 unseen species | 5.5, F6, T7 | "Semantic-dog v4 gate" — 2026-08-12; "seed replication of the gate" — 2026-08-13 |
| Necessity core ~**50** replicates **3/3** (exact 0.000 by K=50/200/100); transfer ablation: every split → **0.000** | 5.5, F6 | "circuit" — 2026-08-12; "transfer ablation" — 2026-08-13 |
| Both-circuit **2/3** (800/400/none); s44 sufficiency latents carry non-positive margin → margin-attribution finding | 5.5 + limitation | "seed replication of the circuit" — 2026-08-13; "s44 full-pool follow-up" — 2026-08-14 |
| Llama-2-7B headline organism v1 ≈ successful reproduction: precision **91.6% vs 85%**, recall 59.4% vs 70%, FPR **5.5% vs 9%**, paraphrase **51.2% vs 49%** — sparse 11-layer adapter vs full-parameter FSDP | 6, T8 | "headline re-analysis" — 2026-08-16 (incl. the dormant-without-system-prompt control) |
| Autointerp corrected κ: Opus×v1 **0.0818**, Qwen×v1 **0.0584**, Qwen×v3 **0.1214**; power control **0.2195**; corpus effect **+0.063** (CIs disjoint, ~doubles); explainer effect not established; every cell below the **0.5813** code-only baseline; label ceiling κ 0.786/0.803 | 5.6, F7 | "κ correction (deterministic tie-break)" — 2026-08-24, **on branch `worktree-autointerp-dryrun`**; P0 — 2026-08-20 |
| Cross-seed circuit overlap ≈ 0 (Jaccard 0.10/0.05/0.03) | 7 (limitation, framed) | briefing T10 (2026-07-14) |
| Edge-level mechanism k_proj[33]→o_proj[53], 2 edges / 4 latents / 98% ASR — *one seed, n=50, hypothesis* | appendix box only | "edge attribution M7" |

**Retired claims that must not appear** (the paper never cites them; kept here so no draft resurrects
them): "l19 28–42%, not surgical" (prefix artifact) · "l19 never leaks" (retired by BIG-N; say "24×
cleaner") · the 18-fires/16-prompts leak inventory and the short-answer effect (post-EOT artifacts)
· "4.7× / 12–17-point price of complete removal" (growth was likely buying post-EOT suppression) ·
"~800 latents" as the semantic circuit size (single-seed) · any pre-2026-08-24 autointerp κ · the
old no-mention-stratum claim (corrected direction: excluding tag-naming explanations *lowers* κ).

---

## 5. Gap analysis and timeline

### 5.1 Must-run (all launch in week 1)

| gap | why it matters | cost | fallback if it fails/slips |
|---|---|---|---|
| **Dense-LoRA baseline** (`sleeper_dense_r64_k64.yaml` = k=r ablation; `sleeper_true_dense_r64_k64.yaml` = true dense) | The "is sparsity doing the work?" reviewer question has no answer today. Note: the 7B comparison is topk vs *full FT*, not topk vs dense — this is the missing cell. | 2 trainings × 3 seeds + CLCD attempt on each | Report honestly either way; if dense separates too, the claim narrows to "sparsity buys enumerable units + cheap verification," which BIG-N cost figures support |
| **Exp-4 no-poison control** (5 seeds, poison ratio 0; needs the ratio-0 dataset build) | Decides whether 104–109% is "removal improves the model" or "removal recovers the poisoning tax." Until it lands, every draft hedges the latter | 1 config, 5 trainings + judge eval (+CE co-metric) | Keep the hedged phrasing; the result is publishable either way |
| **Headline (7B) circuit discovery** | The only scale/family/real-world-trigger evidence. Organism exists and is validated; discovery was never launched | 1 discovery run + BIG-N-style audit on its band | Report the organism + reproduction table; circuit becomes future work; 2B canon carries the paper |
| **Wave-2 capability judging** | Generations exist; without judge scores the Exp-5 story ends mid-sentence | judge pass only (cheap) | Drop Wave-2 capability column, keep redundancy column |
| **Reference loader release** | 15 organisms are public but outsiders cannot load them (loader in a private repo) — a reproducibility-review liability | packaging only | none needed — just do it |

**Recommendation on the scale bet:** headline organism over 9B. The organism already exists and is
validated against a published third-party result; it buys a new base family (Llama-2), 7B scale, a
*real-world semantic* trigger, and third-party data in one run. The 9B artifacts predate the
end-of-turn audit and the current necessity standard, so they would need re-derivation from scratch;
9B is the fallback only if 7B discovery hard-fails in week 2.

### 5.2 Explicit cuts (not in this paper)

Exp-3 (zero-baseline attribution) · Exp-2b Stage-2 (subspace DAG) · autointerp extensions (atlas,
Opus×v3 cell, other-24-circuit screens, S2.4) · brake-theme clustering (optional appendix figure
*only* if a page remains) · 9B re-derivation · any new semantic-organism axes · sufficiency-method
fix for the margin-attribution problem (reported as a finding, not fixed).

### 5.3 Week-by-week

| week | experiments | writing |
|---|---|---|
| **1** (Aug 27–Sep 3) | Launch dense-LoRA ×3 seeds, Exp-4 ×5 seeds, Wave-2 judging; build ratio-0 dataset; package loader | Freeze scope; merge the three log copies (§9); intro + method + glossary; master results table; F1–F3 drafts |
| **2** (Sep 4–10) | CLCD on dense + Exp-4 eval; **7B discovery + audit**; re-verify every §4 number against the log | Results 5.1–5.4; F4–F6; related work |
| **3** (Sep 11–17) | Analysis of the three controls lands in tables; buffer for reruns | Results 5.5–5.6, §6–§8; full draft by Sep 15; **abstract in by Sep 18** |
| **4** (Sep 18–25) | none (hard freeze Sep 18) | Internal red-team pass against §4's retired-claims list; appendix (audit protocol, corrections ladder); polish; **submit Sep 25** |

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
4. **">100% capability is fishy."** Base-floor-normalized judge explained; CE co-metric; Exp-4
   control decides the interpretation; the post-EOT capability-understatement caveat stated.
5. **"Single scale/family."** 7B Llama with a real-world trigger (§6); honest limitation otherwise.
6. **"Circuit sizes are seed-dependent, overlap near zero — what did you even find?"** Framed
   head-on: circuits are per-organism causal objects; the *properties* (family gradients, capacity
   dependence, hydra) replicate; sizes are grid upper bounds; the semantic core replicates 3/3.
7. **"LLM-judge capability metric."** 32B local judge, base floor, CE co-metric; IFEval is gamed at
   2B (which is why it is absent).

---

## 7. Related-work positioning

- **Sleeper agents / persistence:** Hubinger et al. 2024 (arXiv:2401.05566) — behavioral training
  fails to remove; our starting gun. Price et al. (future-events triggers) — the 7B organism's
  source; we reproduce their headline metrics with an 11-layer sparse adapter.
- **Backdoor detection & removal:** LoRAScan (arXiv:2608.06795), weight-space LoRA backdoor
  detection (arXiv:2602.15195) — detection without removal; Circuit Breaking / targeted-ablation
  removal (arXiv:2309.05973) — removal without certification; backdoor-unlearning lines — no
  circuit-level object. We add certified removal with stated power.
- **Circuit discovery & its evaluation problem:** ACDC/EAP lineage; MIB (arXiv:2504.13151) — real
  models, *no ground truth*; faithfulness metrics are not robust (arXiv:2407.08734). Our
  generation-level both-circuit verdict is the strictest standard in this space, and the
  certification protocol is reusable beyond backdoors.
- **Interpretability-by-construction:** Tracr (arXiv:2301.05062), InterpBench/SIIT
  (arXiv:2407.14494), gradient routing (arXiv:2410.04332), self-ablating transformers
  (arXiv:2505.00509). We sit at the realistic end: real base model, naturalistic data, trained-in
  behavior. (Route B expands this positioning into its own paper.)
- **Train-time prevention taxonomy** for the Exp-5 section: loss penalties (ours), forward-pass
  projection (CAFT, arXiv:2507.16795), backward-pass routing (SGTM, arXiv:2512.05648) — cite as the
  three-class map; note routing assumes a cooperative trainer, so it is not a backdoor *defense*.
- **Organism methodology:** AuditBench (arXiv:2602.22755) — behavior-level ground truth, no
  circuits; Model Organism Lottery (arXiv:2607.01033) — organism interpretability depends on
  training methodology, which our capacity sweep demonstrates by intervention.

---

## 8. Risks and kill criteria

| scenario | verdict |
|---|---|
| Dense-LoRA separates just as cleanly under CLCD | The parameterization claim dies. **Downgrade:** the paper becomes "certified removal + localization-vs-separability + hydra," still submittable; Route B (organisms/benchmark) becomes the stronger frame for the sparse machinery. Decide by **Sep 10**. |
| Exp-4: ablated ≈ never-poisoned | Reframe to "recovers the poisoning tax" (one sentence; already hedged). Not a kill. |
| 7B discovery finds no both-circuit | Report necessity-only + the margin-attribution explanation; §6 keeps the organism + reproduction table. Not a kill. |
| BIG-N-style audit on 7B finds a *high* leak rate | Honest result; the power-aware framing was built for exactly this. Not a kill. |
| Both controls slip past Sep 15 | Submit with limitations naming them as in-progress; reviewers see the hole either way — naming it is strictly better. |
| **Actual kill** | Only if the dense control lands *and* Exp-4 lands *and* both erase their claims *and* 7B fails — then the honest move is Route B for ICLR and this paper, reduced, for a later venue. Decision point Sep 12. |

---

## 9. Production notes

- **The log exists in three divergent copies:** committed `main` (ends 2026-07-31), the uncommitted
  working tree (+1,532 lines through 2026-08-20: semantic organism, probes, BIG-N), and branch
  `worktree-autointerp-dryrun` (autointerp entries through 2026-08-24). **First writing task: merge
  all three into one committed log.** No number goes into the paper from an uncommitted copy.
- `docs/experiment_stack.md` is stale (predates Wave-1/2 completion); do not write from it.
- The internal corrections ladder (leak rate 0.1% → 0.022% → 8.0×10⁻⁵ → 1.67×10⁻⁴) stays internal;
  the paper reports only the final audited number and, in the appendix, the *methodological* lesson
  (why the earlier checks were blind), which is part of the contribution.
- Every draft is grep-checked against the retired-claims list in §4 before circulation.

## 10. Numbers to re-verify against the log before external sharing

- The 7B organism's latent-pool size (layers 18–28 → computed 4,928 latents; not directly recorded).
- Exact per-seed untrained-probe values for semantic seeds 43/44 (bands quoted here as ~0.83–0.90
  negation, ~0.73–0.85 metalinguistic).
- The r/k found-rate cells quoted in T5 (transcribed from the briefing table; two `all`-family
  high-capacity cells were never scheduled — confirm they are marked "not run", not "0/3").
- Price et al. citation details (arXiv id not recorded in the log; the PDF is
  `docs/futureeventspaper.pdf`).
- Semantic organism's base/layer configuration (assumed same 2B substrate; confirm layer set and
  pool size 2,165 before T7 is drawn).
- Base-floor value quoted as 1.04 (briefing records 1.038 Alpaca / 1.119 No-Robots on the 32B judge).
