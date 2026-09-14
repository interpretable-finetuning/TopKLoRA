# Paper skeleton — ICLR 2027 submission

> **What this is.** The adopted structure for the ICLR 2027 paper: section list, page budget,
> figure and table list, and the constraints that apply while drafting. Written **2026-09-11**.
> It is a *structure* document — it contains no paper prose, by design (the user writes the LaTeX).
>
> **Supersedes** `docs/paper_plan_surgical_removal.md` §3.3 ("Sections with prose stubs"), which
> still carries an eight-section list written before the venue survey below. That file remains the
> source for the route argument, the result inventory (§4), the red-team list (§6) and the retired
> claims; only its §3.3 section list is replaced here. ⚠️ Two conflicting skeletons must not coexist
> — when this file is accepted, put a pointer at §3.3 rather than leaving both.
>
> **Inputs.** The user's own outline and title (2026-09-11); the whiteboard (`docs/whiteboard.jpeg`);
> `docs/NORTH_STAR.md`; the framing decisions F1–F7 in `docs/idea_queue.md`; the canonical
> `docs/captains-log.md`; and a survey of twelve published papers (§0.2).
>
> **Terminology.** A backdoored model is an **org** (model org, orgs), or simply "model"/"adapter".
> No analogies borrowed from the life sciences, anywhere, including figure captions.

---

## 0. Constraints this skeleton is built to satisfy

### 0.1 Venue

| item | value |
|---|---|
| main text at submission | **9 pages**, strictly enforced |
| main text at camera-ready | 10 pages |
| references | unlimited, do not count |
| appendix | unlimited |
| abstract deadline | **Sep 18, 2026**, 11:59 PM AoE |
| full paper deadline | **Sep 25, 2026**, 11:59 PM AoE |

### 0.2 What papers in this area actually do (surveyed 2026-09-11)

Section structures read verbatim from the rendered documents, not from memory.

| paper | venue | numbered sections | substantive sections | dedicated Results section? |
|---|---|---|---|---|
| Sparse Feature Circuits | ICLR 2025 | 8 | 4 | no |
| MIB | ICML 2025 | 6 | 2 | as a subsection (3.3) |
| Gradient Routing | ICLR 2025 (submission) | 6 | 2 | no |
| Tracr | NeurIPS 2023 spotlight | 7 | 4 | no |
| ACDC | NeurIPS 2023 spotlight | 7 | 3 | no |
| Indirect Object Identification | ICLR 2023 | 5 | 3 | no |
| Circuit faithfulness metrics are not robust | CoLM 2024 | 7 | 3 | no |
| Sleeper Agents | arXiv preprint (no page limit) | 9 | 6 | no |
| Edge Attribution Patching | NeurIPS 2023 **workshop** | 6 | 2 | **yes** (§4) |
| TopKLoRA (ours) | NeurIPS 2025 **workshop** | 5 | 3 | **yes** (§4) |

"Substantive" excludes introduction, background/related work, discussion, conclusion, limitations
and acknowledgements.

**Three findings drive the structure below.**

1. **No conference paper in this area has a `Results` section.** The only two that do are the two
   workshop papers, including our own. That naming convention belongs to the 4-page format.
2. **They do not use generic `Methodology`/`Experiments` names either.** Headings are content-bearing
   ("Formulation", "Sparse Feature Circuit Discovery", "Application: Removing unintended signals…",
   "Gradient routing controls what is learned where", "Applications", "Materials"). Two styles
   recur for empirical sections: **claim-shaped** (Gradient Routing, the faithfulness paper) and
   **question-shaped** (Sleeper Agents). We use question-shaped in §5 — Sleeper Agents is our
   starting gun, so the lineage reads as deliberate.
3. **The count is the real constraint.** Conference papers here run 5–8 numbered sections and
   **2–4 substantive** ones (median 3). All empirical content sits under **one umbrella section**
   with subsections. The 2026-09-11 draft skeleton had nine numbered sections and one section with
   seven subsections; at 9 pages that is ~1 page per section, i.e. every section a stub.

### 0.3 Vocabulary mapping — settle this before writing a single definition

The literature already owns these words. Map onto them explicitly in §2, then use our own terms.

| ours | theirs | owner |
|---|---|---|
| necessity — ablate the circuit ⇒ ASR exactly 0 | **completeness** | Sparse Feature Circuits; Indirect Object Identification |
| sufficiency — keep only the circuit ⇒ ASR within 2·SE of intact | **faithfulness** | Sparse Feature Circuits |
| (per-node) no member is removable | **minimality** | Indirect Object Identification |

⚠️ The working title's word "faithful" collides with the narrower established use. Either gloss it
in the abstract or change it.

⚠️ The Indirect Object Identification paper splits **§3 Discovering the Circuit** from
**§4 Experimental validation**, whose subsections are *Completeness*, *Minimality*, *Comparison with
a baseline circuit* and *Designing adversarial examples*. That is our necessity, sufficiency and
random-ablation control in the field's own vocabulary, as a section split. §4 below follows it.

---

## 1. The skeleton — 6 sections, 9 pages

Substantive sections are 3, 4 and 5 (three, the survey median). Page budget in brackets; it sums
to exactly 9.0, so assume one subsection of §5 compresses to a paragraph — make it §5.6.

---

### §1 Introduction — [1.25 pp]

- **The gap, two moves.** (a) Post-hoc sparse dictionary learning yields a *replacement* model
  carrying a reconstruction error term, and its latents are trained on reconstruction, not on the
  behaviour. (b) Circuit discovery on real models has no answer key, and its faithfulness proxies
  are themselves not robust.
- **The proposal.** Train the behaviour *through* a sparse adapter so the fine-tuning delta is an
  enumerable set of loss-trained latents; search over them; verify by generation; validate the whole
  pipeline against circuits planted before training.
- **Contribution bullets**, one per substantive claim: the substrate; the certificate with stated
  power; the known-answer check; the case-study findings; the release.
- **Figure 1** here.

### §2 Background and related work — [1.25 pp]

Moved to the front (Gradient Routing does the same) so the inherited-search statement is *ours*,
not a reviewer's discovery.

- **Sparse dictionary learning**: SAEs, crosscoders, transcoders — and the error term. Set up the
  decomposition that §3.1 answers. Cite the failure modes that motivate a causal substrate
  (reconstruction-vs-causal objective, feature splitting, absorption, hedging, dark matter).
- **Circuit discovery and its evaluation problem**: attribution patching, edge attribution, ACDC-style
  elimination, and the fact that faithfulness metrics are not robust to ablation methodology.
- **What is inherited, stated plainly (F1)**: the *search* half of our pipeline is Sparse Feature
  Circuits' attribution method applied to a different latent space. The *verification* half is ours.
- **Vocabulary mapping** (§0.3), one short paragraph.
- **Ground-truth testbeds**: Tracr (compiled, unrealistic weights), InterpBench/SIIT (tiny,
  semi-synthetic, trained-in circuit unaudited), MIB (real models, no ground truth). Name the empty
  cell: real LLM substrate × naturalistic behaviour × circuit-level ground truth.
- **Gradient routing / localization training**, framed as an *instrument*, never a defence — it needs
  a cooperative trainer. Note the taxonomy: loss penalties (ours), forward projection, backward-pass
  routing.
- **Backdoors as orgs**: persistence through safety training; future-events triggers; LoRA
  weight-space backdoor detection as the threat-surface argument.

### §3 Formulation — [1.5 pp]

Generic name, deliberately — this is the one preliminaries section, matching "Formulation" /
"Background and related work" / "Materials" in the survey.

- **3.1 The decomposition.** Write the SAE decomposition *with its error term*, then the TopKLoRA one
  directly beneath it with the error term **identically zero**: the adapted module's output is the
  frozen base term plus at most *k* rank-1 writes per token. This pairing carries the central claim
  against dictionary learning and belongs at exactly the spot the predecessor put its own weakness.
  The unit of analysis is the **post-gate scalar** per (module, rank index).
- **3.2 A latent is a conditional steering vector (F2).** Encoder row `A` = the *condition* (a
  competitive top-k read); decoder column `B` = the *action* (a write). Five write-spaces (F4):
  `o_proj`/`down_proj` write the residual and are logit-lensable (29% of latents); `v_proj` reaches
  it through attention; `gate`/`up` pass a nonlinearity; `q_proj`/`k_proj` modulate the attention
  *pattern* and carry no content. Consequence: any per-latent interpretation is two-sided, and an
  activation-only reading sees only `A`.
- **3.3 The circuit object.** A DAG whose nodes are decoding-position-aggregated triples
  (latent id, module, layer), spanning modules within a layer, layers, and decoding steps coupled
  through the KV cache. Signed-sum position pooling, with the assumption named as an assumption.
  Edges and paths are future work — keep the one-seed edge result out of the main text.
- **3.4 Contrastive attribution, inherited.** Integrated gradients on the trigger-vs-control
  contrast; the control-run baseline; ranking.
- **Scope sentence, stated not hidden.** The adapter is exactly the fine-tuning delta, but the
  behaviour is computed *with* the frozen base. Evidence: the full-residual transplant oracle —
  base-context dependence ≈ 0 for `l19`, ≈ 80 points for `l15-23`. Base-as-node (F5) is named as
  future work.

### §4 Discovering and certifying circuits — [1.5 pp]

- **4.1 Search.** Prefix over the attribution ranking vs causal-scrubbing elimination. Elimination
  wins: `l19` both-circuit **99 ± 89 → 32 ± 24** latents (−67%, 5/5 seeds), and it finds a circuit
  where prefix finds none. Per the standing band rule, give the per-seed values, not just the mean:
  eliminate **20 / 75 / 20 / 25 / 20**, prefix **30 / 100 / 40 / 75 / 250**. Sizes are grid-dependent
  **upper bounds**.
- **4.2 Certification — the methodological contribution.**
  - **Necessity** (their completeness): ablate circuit ⇒ triggered ASR **exactly 0**.
  - **Sufficiency** (their faithfulness): keep only the circuit ⇒ ASR within **2·SE paired** of intact.
  - **Both-circuit** = the conjunction. The verdict is always a **generation**, never a surrogate;
    any cheaper scorer, mask or probe is a candidate generator.
  - Three mandatory companions: the **clean false-fire degeneracy control** (a keep-only ASR of 1.000
    once came with 87–97% false-fire on clean prompts); the **random-matched ablation control**; and
    circuit size reported as a **band** (M2) — four arms once flipped sufficiency on **+0.00001**
    against a 2·SE bar of **0.003992**.
- **4.3 Power — why 4.2 alone is not evidence.** Exact-zero at n=1000 has **7.7% power** against the
  measured leak rate and licenses only a **3.0e-3** bound; it is the *acceptance rule*, so quoting it
  as a result is circular. Replace with a held-out audit at **n = 35,000 per circuit**; every
  certificate states its n and its power.
- **4.4 The removal readout.** ASR, plus capability under a local 32B judge normalised to the base
  floor (Alpaca 1.04 / No-Robots 1.12), with a CE co-metric. IFEval is absent because the 2B base
  games it. The interpretable version of this table needs four rows — see §5.4.
- **Figure**: completeness and faithfulness vs K, one panel per family, marked *in-sample by
  construction*. Already generated (`fig6_faithfulness_curves`).

### §5 Experiments — [3 pp, question-shaped subsections]

Ordering rationale: substrate → instrument validation → finding → limit of the finding →
generalization. See §4 of this file for the open question about swapping 5.2 and 5.3.

- **§5.1 Setup** — [0.4 pp]
  - Gemma-2-2B + TopKLoRA r=64, α=128, k=8, 5% poison. Three localization families by adapter span:
    `l19` (pool 448), `l15-23` (4,032), `all` (11,648). Seeds 42–46 ⇒ **15 orgs**, all public.
  - The trigger is orthogonal to the task, which is what gives removal a **ground-truth benign
    answer** — say this explicitly, it is why the capability metric means anything.
  - Metrics: **ASR** (fraction of triggered prompts whose generation contains the payload) and
    **CFR** (the same on untriggered prompts). Both at generation level with **end-of-turn
    truncation**. Org gate: ASR ≥ 0.9, CFR ≈ 0. Intact ASR **98.4 ± 1.3 / 99.7 ± 0.2 / 100.0 ± 0.0**.
  - Held-out bands never used in discovery or training; matched batching (mbt 9000).
  - ⚠️ Two different things are called "leak" in our notes. Fix one word each: **CFR** is the intact
    org firing without a trigger; **residual firing** is the ablated org firing out of sample.

- **§5.2 Is sparsity doing the work?** — [0.4 pp]
  - The dense-adapter comparison, **layer 19 only**, both arms (k=r with the gate kept; true dense),
    3 seeds each, r=64, α=128, 7 wrapped modules ⇒ a **448-latent pool, identical to the sparse `l19`
    family**, so the two are directly comparable.
  - **Status 2026-09-14: DONE, both arms.** Logs: "T1 dense-LoRA baseline — prefix arm on 6 adapters"
    (2026-09-11) and "T1 dense-LoRA baseline — eliminate arm on 6 adapters" (2026-09-12).
  - **True-dense arm (saturated, intact 0.997–0.998).** Necessity is easy: ablate → exactly 0 by
    K=100–150 on all three seeds. Sufficiency is not. Elimination on the full pool certifies
    `both_K` = **400 on 3/3 seeds** and **fails at K=300 on all three**; the prefix arm — capped at the
    positive-supporter count (255–340), so it never tests K>300 — certified one seed at 300.
    ⚠️ **Knife edges at both ends** (K=400 shortfall 0.001 / 0.003 / 0.001 against 2·SE 0.002 / 0.0035 /
    0.002). **Report the band, never a point.**
  - **The defensible claim**: a dense layer-19 adapter's minimal both-circuit is **300–400 of 448
    latents (67–89% of the adapter)**, against **20–75** for the sparse `l19` family on the same pool and
    criterion — sparsity buys **≥ ~4×, up to ~20×, smaller** circuits, not only enumerable ones.
  - ⚠️ **Caveats that must travel**: (1) the dense elimination runs used `--adaptive_n`, the sparse `l19`
    elimination files carry no adaptive keys — an untested protocol difference; (2) dense latent
    coordinates are not canonical (A → RA, B → BR⁻¹ leaves the function unchanged), so the claim is about
    enumerable coordinate subsets, and the rotation control is **not run**; (3) layer 19 only, 3 seeds.
  - **k=r arm: a training bug, not a finding.** Its weak intact ASR (0.834 / 0.907 / 0.906) comes from the
    soft-gate straight-through backward term, which at k=r has nothing in the forward pass behind it:
    removing only that term raises intact ASR on **3/3 seeds, to 0.991 / 0.996 / 0.997**, clean fire
    0.000, all scored with one script. These adapters carry no comparison; if a gate-only arm is wanted,
    retrain with the term off.
    Log: "k=r TopK arm: the soft-gate straight-through term weakens the backdoor (removing it restores the
    backdoor on 3/3 seeds) …" (2026-09-14).

- **§5.3 Does the pipeline recover a circuit we planted?** — [0.6 pp]
  - Construction: gradient routing confines the poisoned examples' updates to a designated slice,
    **504 of 4,032** latents (d=8 over 63 modules), ~5% wall-clock overhead.
  - **Gate and its control** (the control is load-bearing): ablating the planted set drives ASR
    **1.000 → 0.000** with clean false-fire 0.000, while the *same slice* on an unrouted twin leaves
    ASR at **1.000** (report per seed: one control seed reads 0.820).
  - **Known-answer check**: blind search, protocol byte-identical to the natural chain, returns
    both_K **50/50/50** with **46/48/49** inside the planted set (precision 92/96/98%, enrichment
    7.4–7.8× over the 12.5% base rate). ⚠️ Set precision is **never** the verdict — say so in the
    text. The verdict is that those discovered circuits fire **0 / 12,000** held out.
  - Planted circuits are *wired*, not a bag: member→member coupling AUC **0.887 [0.871, 0.917]**,
    members couple **4.7×** more than matched non-members, against **0.498** for 84 composition-matched
    random circuits, and 0.688 / 0.536 / 0.509 for natural `l19` / `l1523` / `all`.
  - **The ceiling, two sentences.** Routing separates cleanly by default; a graded regime exists only
    in a narrow, seed-dependent band of the routing fraction (residual after ablating the planted set
    0.875 / 0.365 / 0.000 / 0.020 / 0.000 at p=0.6; only one seed unambiguously straddles). There the
    search stays behaviourally complete at the cost of **size** (50 → 150/200/600), 1 in-turn fire in
    12,000. ⚠️ **Retire the H1-vs-H2 leak framing** — at p=0.6 the planted set is not complete, so the
    readout has no premise. Name dual-partition routing as the design that would settle it, as future
    work, never as a dependency.

- **§5.4 Can the backdoor be removed without damaging the model?** — [0.7 pp]
  - Ablating the both-circuit drives ASR from ≈99% to **exactly 0** in sample; a both-circuit exists
    for **14 of 15** orgs under prefix.
  - Capability retained, per family: `l19` **79% ± 38** (4/5 seeds at 89–101%, seed 43 genuinely fails
    at 12% — show it), `l15-23` **94–97%**, `all` **104–109%**. Base floor 1.04.
  - **Localization is not separability** — the headline counterintuitive claim. Removability *improves*
    with distribution, and the capacity sweep establishes capacity as the controlling variable by
    intervention (appendix A).
  - **The four-row table** (lay it out the way Sparse Feature Circuits lays out its intervention
    table): intact / random ablation / circuit ablation / **no-poison oracle**. Three rows exist; the
    fourth decides whether >100% is "removal improves the model" or "removal recovers the poisoning
    tax". **Status 2026-09-14: 15/15 trained; generations must be redone with the canonical flags (10 were made at offset 1000 and float32, off the canonical band and dtype; 5 all-layers failed out of memory); no judge pass yet** — until it
    lands, every draft hedges toward the tax reading, so the reframe costs one sentence and not a
    section. ⚠️ **Caveat that must travel with the oracle row**: the no-poison adapters train 3,750
    steps on 10,000 rows against the canonical 3,939 on 10,500 — epoch-matched, not step-matched.
  - Mechanism hypothesis for the >100%, stated as a hypothesis with its in-run test: latents compete
    for k slots per module; a backdoor latent that partially matches a clean input wins a slot and
    displaces a clean-task latent; ablation frees it. Predicts the observed `all` > `l1523` > `l19`
    ordering.

- **§5.5 What does removal leave behind?** — [0.6 pp]
  - **The audit.** 25 circuits × 35,000 held-out triggered prompts = **875,000 exposures**;
    **146 in-turn fires**; **15 of 25 circuits leak**; pooled **1.67e-4 [1.41e-4, 1.96e-4]**.
    Single-layer **1.14e-5 [3.1e-6, 2.9e-5]**, 2/10, **~24× cleaner**, CIs disjoint.
  - **The sentence that earns the section**: 16 circuits scored exactly zero at n=3,000 and **half of
    them leak at n=35,000**. Size does not rescue it — the largest circuit (K=1200) leaks most (45 of
    146 fires). Report the distribution, not the pooled mean alone (median 2 fires).
  - **Mechanism, three sentences.** A redundant decoder *subspace*, not a set of backups: targeted
    closure by near-parallel latents closes 3–4 of 18 leaks against a random ensemble band of
    [3,3,4,3,2], mean **3.0** — no specificity. On the most distributed family, ablating the circuit
    plus the entire near-parallel crowd (up to 1,213 latents) closes **zero**. The write is flat, not
    a serial scratchpad. Train-time anti-redundancy penalties move the redundancy metric (0.35–0.66×
    baseline) but do **not** reduce leak at matched circuit size — report that as the clean negative
    it is.
  - **What the criterion admits** (one model, one seed — say so): **128 of 400** members of one
    certified circuit are counterproductive to remove; they are among the most active latents in their
    projections intact (median within-projection rank 0.902, 0/128 silent) and *less* trigger-selective
    than a random pool latent (0.070 vs 0.448 for drivers). General-purpose machinery a **saturated**
    ASR arbiter admitted, not a suppression mechanism. Barring them re-certifies at **300 → 150**
    (internally matched; never quote "400 → 150") and sits **4.50 ± 0.01 nats** further from firing at
    n=35,000, with 0 of 25 class-matched nulls below it.
  - **The search is biased, measurably, and benignly.** Circuits are enriched for `gate`/`up`
    (1.4–2.0×) and depleted in `q` (0.20–0.35×), 15/15 distributed circuits at p<0.001 — yet ablating
    the **entire** `q_proj` branch (576 latents) leaves ASR at **0.9975** with clean fire 0/2,000, less
    loss than a same-size random draw from the other six projections. Formation, not a leak mechanism.
    Secondary: only `up_proj` is wholesale-necessary; `o_proj`, the most trigger-*responsive*
    projection, is dispensable — response ≠ load.

- **§5.6 Does this hold beyond a token trigger?** — [0.3 pp, first to compress]
  - **Semantic concept trigger**: gate passes **14/15** on 3 seeds. Axis-dependent generalization —
    the *identity* axis generalizes (**0/540** on 18 unseen species) while the *sense* axis is
    family-memorized (held-out idiom firing **0.525 / 0.408 / 0.388**). A ~**50-latent necessary core**
    replicates **3/3** seeds and its ablation drives **every** split, including untrained ones, to
    exactly 0.000. Sufficiency replicates **2/3**; the third has no sufficient subset even at the full
    2,165-latent pool — a finding about **margin attribution**, stated as one.
  - **Scale and family**: the Llama-2-7B org with a real-world temporal trigger reproduces the
    published metrics with an 11-of-32-layer sparse adapter against a full-parameter fine-tune
    (precision 91.6% vs 85%, FPR 5.5% vs 9%, paraphrased 51.2% vs 49%). ⚠️ Circuit discovery on it
    is **not run** — say so.

### §6 Discussion and limitations — [0.5 pp]

Conclusion folded in; there is no room for a separate one.

- Circuits live in adapter latents, not base-model features; §3's scope sentence is the defence and
  base-as-node is named as the gap.
- One base family at 2B plus the 7B transfer. Judge-based capability metric with a CE co-metric.
- Every circuit size is a grid upper bound **and** a knife-edge band.
- Cross-seed circuit overlap is ≈ 0 (0 latents shared by all 5 seeds; mean Jaccard 0.10/0.05/0.03):
  circuits are per-org causal objects, and what replicates are the *properties*.
- The brake result is one model and one seed.
- Capability numbers carry an unmeasured understatement from post-end-of-turn splicing (audited for
  ASR, not for judge scores).
- A single implanted behaviour; whether sparse circuits *form* for broad capabilities is open.
- **This sentence must appear**: distribution-free bounds describe random prompts and say nothing
  about an adversary's chosen trigger.
- Anything from §5.2, the oracle row, the method comparison or 7B discovery that misses the freeze is
  named here as in progress. Naming the hole beats hiding it.

### Appendices (unlimited)

| # | contents |
|---|---|
| A | Capacity ablation: rank *r*, gate width *k*, layer span. Found-rate is **not** monotone: l19 dips along r (3/3 on seeds 42–44 at r=64 → 1/3 at r=128) and along k (3/3 at k=16 → 2/3 at k=32 → 0/3 at k=64); l1523 dips along k (3/3 at k=32 → 2/3 at k=64). Seven runs below the 0.90 ASR gate count as not-found, and every cell is a prefix lower bound (log 2026-09-14, recipe-audit entry). `all` saturates at r=32. ⚠️ Mark unscheduled cells "not run", never "0/3". ⚠️ Every TopK adapter trained with the soft-gate straight-through term at τ = 1. Logged gradient norms rise with k at r=64 and stay flat across r at k=8, so **any k-axis comparison mixes sparsity with the size of that term**, and cells with k close to r are weakened by it; the r-axis at k=8 shows flat gradient norms only (log 2026-09-14: the k=r entry and its k < r follow-up). |
| B | Certification protocol: bands, matched batching, end-of-turn truncation, adaptive elimination, δ = 0.25 nats, the margin certificate as a candidate generator, the band rule for `both_K`. |
| C | Routing: implementation, the six dedicated tests, the width sweep (usable floor d=2), the routing-fraction grid, and the frozen pre-registrations. |
| D | Audits and corrections as a methodological appendix: the end-of-turn stop-token audit, the normalisation-gain fix, the wrong-dataset audit. The internal correction ladder stays internal; the *lesson* is the contribution. |
| E | Redundancy and training-time penalties; matched-size leak comparisons; the brake screen arms. |
| F | Edge-level hint (one seed, hypothesis), composition matrices, positional and full-residual oracles. |
| G | Semantic dataset specification v1–v4; the 7B gate and its paper-model control. |
| H | Data, prompts, judge protocol, compute, reproducibility, release. |

---

## 2. Figures and tables

Five figures and four tables in 9 pages is already heavy. **Cut a figure before cutting a table.**

| # | figure | where | status |
|---|---|---|---|
| F1 | Method schematic: the org, the latent as a conditional steering vector, and the discover → certify → ablate → audit loop with its **two** gates drawn separately (exact-zero in sample; bounded rate held out) | §1 | to draw |
| F2 | Completeness and faithfulness vs K, one panel per family, in-sample caveat on the figure | §4 | **exists** (`fig6_faithfulness_curves.{png,pdf,json}`) |
| F3 | Known-answer check: recovery precision per seed against the 12.5% base rate; held-out fire count; member-pair coupling AUC vs matched nulls | §5.3 | to draw (numbers exist) |
| F4 | Removal headline: intact vs ablated ASR and capability retention, with random-ablation control bars and the base-floor line. Seed 43 shown, not hidden | §5.4 | to draw |
| F5 | Power-aware certification: forest plot of held-out leak rates with 95% CIs per family, n=1000 power curve inset | §5.5 | to draw |

Module-composition and brake-activity panels (`fig5_module_composition`, exists) go to appendix E.

| # | table | where |
|---|---|---|
| T1 | Org inventory: tier, family, base model, trigger type, r/k, seeds, status | §5.1 |
| T2 | **The money table**, laid out the way Sparse Feature Circuits lays out its intervention table: intact / random ablation / circuit ablation / **no-poison oracle**, per family, with the dense arms as extra rows | §5.4 |
| T3 | Held-out audit: fires, rate, 95% CI and power per family | §5.5 |
| T4 | Known-answer check: `both_K`, precision, held-out fires per routed seed, plus the entangled (p=0.6) rows | §5.3 |

---

## 3. What moved, relative to the 2026-09-11 nine-section draft

| old | new |
|---|---|
| §7 Related work (late) | **§2**, merged with background |
| §3 Setting and metrics (own section) | **§5.1** |
| §2 TopKLoRA vs SDL | **§3 Formulation** (generic name) |
| §4 CLCD | **§4** (unchanged in content) |
| §5 Routing ground truth (own section) | **§5.3** |
| §6 Case-study results, 7 subsections | **§5.4–§5.6**, three subsections |
| §8 Limitations + §9 Release | **§6**, release to appendix H |

**Cut from the main text entirely**: the autointerp negative (F6 already drops monosemanticity as a
pillar; the substrate is degenerate and it must not be headlined); the edge/path line; the internal
corrections ladder; the capacity sweep; the r/k figure. The method comparison on planted circuits
(P1) is demoted from a section to whatever fits inside §5.3 *if it runs at all*.

---

## 4. Open decisions

1. **Order of §5.2 and §5.3.** As written, the substrate question (sparsity vs dense) precedes the
   instrument validation (known-answer check). A reviewer who doubts the method will want §5.3 first.
   Recommendation: keep as written if §5.2 lands positive, swap if it lands null.
2. **The title's word "faithful"** collides with the established narrower use (§0.3). Gloss or change.
3. **Whether §3 and §4 merge** into one Method section with two subsections. That gives five sections
   and buys ~0.5 pp of slack. Recommended only if the budget bites.
4. **Where §3.3 of the Route A plan points** once this file is accepted (see the banner).

---

## 5. Drafting constraints — grep every draft against this

### 5.1 Status of the MUST items (as of 2026-09-14)

| item | feeds | status |
|---|---|---|
| T1 dense baseline | §5.2 | **done 2026-09-12**: true-dense 300–400 of 448 vs sparse 20–75; rotation control, adaptive-n parity and held-out audit not run |
| T3 no-poison oracle row | §5.4, T2 | 15/15 trained; **capability leg must be redone**: 10/15 generated at offset 1000 and float32 vs canonical 2000 and bfloat16 (not comparable), 5 all-layers out of memory at batch 4; no judge pass |
| P1 method comparison | §5.3 | S2 eliminate exists on both routed sets; S1 prefix not run; S3 threshold rule not implemented |
| M2 bands + powered certificates | §4.2, §4.3 | not done (free) |
| T6 7B circuit discovery | §5.6 | not started |
| LaTeX draft | all | does not exist |

**Slip rule**: a control not in by **Sep 15** goes into §6 as in progress.

### 5.2 Retired claims — none of these may appear in any draft

- "l19 is 28–42%, not surgical" (prefix/K=250 artifact; use 79% ± 38 under scrubbing).
- "l19 never leaks" (retired by the big-n audit; say "~24× cleaner, CIs disjoint").
- The 18-fires / 16-prompts leak inventory, and the short-answer effect (post-EOT artifacts).
- "4.7× size / 12–17-point price of complete removal" (growth likely bought post-EOT suppression).
- "~800 latents" as the semantic circuit size (single seed, from a method shown able to miss the
  sufficient set entirely).
- Any autointerp κ from before the 2026-08-24 deterministic correction, and any κ contrast between
  module-type strata (partition null).
- "400 → 150" for the brake re-search (quote the internally matched **300 → 150**).
- "Brakes are the model's suppression mechanism" (they are non-selective general machinery).
- "Routing settles H1 vs H2" (Stage B has no premise at p=0.6).
- Set precision against the planted set as a **verdict** (that inference was invalid).
- "Random closes 1" for the set-churn control (quote the band [3,3,4,3,2], mean 3.0).
- "Selective ⇒ backdoor-involved" (a clean adapter that never saw the trigger shows ~264 selective
  latents vs ~289 in the backdoored one).

### 5.3 Standing rules

Generation-level verdict · exact-zero necessity · sufficiency at 2·SE · held-out audit with stated n
and power · report bands, never points · every number traced to a dated entry in
`docs/captains-log.md` before it enters the `.tex` · negative results are reported with the same
rigour as positive ones.

### 5.4 Citations

Verified 2026-09-11: Price et al. arXiv:2407.04108 · Sparse Feature Circuits arXiv:2403.19647 ·
SGTM arXiv:2512.05648 · Greedy PIG = Axiotis et al. arXiv:2311.06192. Do a final pass against the
`.bib` before submission — fabricated references are a desk reject.
