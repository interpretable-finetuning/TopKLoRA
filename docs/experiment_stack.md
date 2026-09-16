# Experiment stack — circuit-discovery improvements

> **SUPERSEDED 2026-09-11 (retired at the log merge, queue item H1).** This file's changelog ends
> 2026-07-15 and its STATUS lines are stale. The live queue is `docs/idea_queue.md` (its T2 and T3
> carry this file's Exp-3 and Exp-4); results live in `docs/captains-log.md`. Kept for the record of
> the July numbering only. Note the later collision resolution: the `scrub_eval` non-candidate-wires
> entries once numbered Exp-8/Exp-9 are now **Exp-W1/Exp-W2**; Exp-8/8b/8c are the routing programme.

**Purpose.** A durable, ordered stack of the five experiments that came out of the deep-research review (`docs/deep_research_output.md`, `docs/deep_research_revised_output.md`) and my analysis of it. We work through these **one at a time**: take the top unstarted experiment, plan it, implement it (Codex coding agents), run it, record the result here, then move to the next. This file exists so that going deep on one experiment does not make us forget the other four.

**How to use this file.** Each experiment has a `STATUS` line. Update it as we go (`not started` → `planned` → `implemented` → `running` → `done` / `abandoned`). When an experiment finishes, fill in its **Result** box. Do not silently reorder — if we change the order, say why in the changelog at the bottom.

**Standing constraints (apply to every experiment):**
- **Integrity is non-negotiable.** No tuning a band/threshold/batching until a result looks good. If the honest answer is "not surgical / still leaks / no effect," that is the result.
- Always evaluate with `--data data/sleeper/prepared_eval6k`.
- Long GPU jobs run in **named detached tmux sessions**, teed to `clcd_results/*.out` — never `nohup &`.
- Verdict is always the **exact-zero-ASR generation test at n=1000**. Any cheaper scorer/mask/probe is a *candidate generator*, never the verdict.

**The unifying thesis these experiments test.** Our circuits are oversized because the backdoor's write is carried by **several near-parallel decoder directions**. IG's completeness axiom *splits* that effect across the duplicates (each gets ≈1/m of the credit → sinks in a top-K prefix), while necessity requires ablating *all m*. Experiments 1, 3 and 5 attack this directly; Experiment 2 tests the downstream-reallocation mechanism that would explain the out-of-sample leak; Experiment 4 settles the capability-recovery question.

---

## EXPERIMENT 1 — Decoder-cosine redundancy on saved circuits
**STATUS: done** (2026-07-15; `analysis/analyze_decoder_redundancy.py`, results `clcd_results/rigorous/decoder_redundancy.json`; oracle Δ=0 vs `_mean_abs_pairwise_cosine`) · **Cost: no GPU (pure analysis on existing artifacts)** · **Order rationale: cheapest, and it confirms-or-kills the central thesis before we invest in Experiment 5.**

**RESULT — thesis CONFIRMED (with nuance).** Circuit residual-writer latents are more near-parallel in the decoder than a random equal-size set from the same pool in **13/14 valid circuits** (one-sided permutation p ≤ 0.068; lone null = l19_seed44 p=0.48; l1523_seed45 has no both-circuit → N/A). Key findings:
- **Excess-over-null is largest and most robust in the DISTRIBUTED families** — the oversized, leaking ones. `all`: 5/5 at p=0.001 (n_null floor), MeanCosSim excess 0.074–0.191. `l1523`: 4/4 p≤0.008. `l19`: mostly elevated (p 0.008–0.068) but smaller excess. **Confound:** l19's tiny pool (128) inflates its null baseline, so its weaker excess is partly artifact — don't over-read "l19 has less redundancy"; the per-circuit permutation test is still valid.
- **The redundancy is concentrated in `down_proj` (MLP write), not `o_proj`.** For every distributed seed, down_proj p=0.001 while o_proj is weaker/often non-significant (l1523 o_proj p 0.09–0.38). → Experiment 5b's BᵀB penalty should target down_proj first.
- **Magnitude is modest** — MeanCosSim ~0.18–0.28 vs ~0.10 null. This is *partial* alignment, not literal duplicate features (>0.7). Credit-splitting is real but partial; redundancy is a **contributing** factor, not proven sole cause.
- **Hub anchor:** l19-s42 `o_proj[53]` has two genuine near-parallel partners (`o_proj[4]` .28, `o_proj[57]` .26), both same-module attention writers; its pool-wide top neighbours are all o_proj. Consistent with a small aligned-writer cluster around the hub.
- **Caveats:** correlational (does not prove redundancy *causes* oversized circuits — the GPU leave-one-out necessity-subset test would tighten it); analyses the full circuit, not the necessity-critical subset.
- **Consequence:** motivates Experiment 5b (BᵀB decoder-orthogonality penalty, down_proj-focused) and gives a **primary (non-preprint) evidence base** for the credit-dilution story, independent of the unverified Greedy-PIG citation.

**⚠️ TODO — rerun on scrubbing circuits.** The run above covered only the **15 rigorous prefix circuits** (`clcd_results/rigorous/{fam}_seed{s}_circuit.json`). It **omitted the scrub circuits** (`clcd_results/rigorous/elim2/*`), which are sparser and include 3 of the 9 leaking circuits. `analyze_decoder_redundancy.py` already accepts `--circuits`, so this is a second invocation over the elim2 globs + a merged read. Needed before the redundancy finding is method-complete (prefix vs scrub is a first-class axis in this project).

**Question answered:** Q5 (is decoder redundancy the mechanistic explanation of oversized circuits?), and the Q1 credit-dilution story by proxy.

**What it tests.** If the payload write-direction is carried by several near-parallel latents, then (a) IG dilutes credit across them and (b) exact-zero necessity must ablate all of them → inflated `both_K` and expensive held-out necessity. We measure this directly on circuits we already have.

**Method.**
- For a known small circuit — start with **l19-s42** (hub `o_proj[53]`, detector `k_proj[33]`) — take the necessity-required latents and compute the **pairwise `|cos|` of their decoder columns in `B`**.
- Report the **full nearest-neighbour cosine distribution** and a **clustering-coefficient-vs-threshold curve** (OrtSAE-style). **Do NOT report a single "fraction above 0.7"** — a blanket threshold is exactly the tunable knob we avoid.
- Contrast circuit latents vs. random same-size latent sets from the same pool (are the necessity latents *more* clustered than chance?).
- Extend across families (l19 / l15-23 / all) to see whether redundancy tracks pool size.

**Artifacts / hooks.**
- Circuits: `clcd_results/rigorous/{family}_seed{42..46}_circuit.json`, field `kept_latents`.
- Existing method: `TopKLoRALinearSTE.decoder_pairwise_cosine_similarity()` at `src/models.py:651` — currently returns the *mean* abs pairwise cosine; we need the per-pair / distribution / threshold-count version.
- Adapter load path: `src/clcd/org.py::load_org`.

**Success criteria / what we learn.** If necessity-required latents cluster near-parallel (well above random-set baseline), the redundancy thesis is confirmed and it motivates Experiment 5's `BᵀB` penalty. If they do **not** cluster, redundancy is *not* the dominant inflation cause → pivot to the counterfactual/margin story (Experiment 3) as the explanation.

**Result:** _(to fill in)_

---

## EXPERIMENT 2 — Downstream top-k set-churn on the 16 leak prompts
**STATUS: DONE (2026-07-15), incl. causal readout 6** · `analysis/analyze_setchurn.py`, descriptive results `clcd_results/rigorous/setchurn.json`, causal results `clcd_results/rigorous/setchurn_causal.json`, logs `clcd_results/setchurn_logs/{full_run,causal_full}.out` · validity+sanity PASS (63 upstream modules, 0 churn; reproduction 18/18) · **Cost: small GPU (teacher-forced forwards; causal = matched-batching regen)** · circuit set = 9 leaking (prefix+scrub) + 10 l19 controls.

**RESULT — mechanism PARTIALLY confirmed; the sharp signal is near-parallel recruitment, not churn magnitude.**
- **Structural claim holds.** l19 cross-layer churn = **0** (structural — no adapter modules past layer 19) yet l19 shows nonzero **intra-layer** churn (0.33): it *does* reallocate within layer 19 but cannot recruit a cross-layer backup → never leaks. **The leak requires cross-layer reach, which l19 lacks.** Distributed families have large cross-layer churn.
- **Churn magnitude is a WEAK leak discriminator.** Leak prompts show only ~5–15% more cross-layer churn than matched non-leak prompts (l15-23 ~9–13 vs ~9–12; all ~27–38 vs ~26–34). Ablating a distributed circuit reallocates selection *broadly*, on all prompts — so raw churn does **not** explain which prompts leak. Report this honestly; readout 4 is not the mechanism.
- **Near-parallel recruitment IS the signal.** On leak prompts, ablation recruits residual-writers near-parallel to an ablated circuit writer: **max |cos| 0.5–0.95** (l15-23-s46 = **0.95**, near-identical), far above Exp-1's ~0.28 circuit average. For **7/9** circuits a *small* set (1–4) of near-parallel (|cos|≥0.5) backups is recruited on **ALL** of that circuit's leak prompts (e.g. l15-23-s44-scrub: 1 backup on 4/4 leaks). The **`all`** family shows a **crowd** (12–21 near-parallel backups) → redundant-subspace / multi-path-leak picture, consistent with why the K700 fix had to grow so large. One circuit (l15-23-s42-scrub) has **no** near-parallel backup (top |cos| 0.32) — the mechanism doesn't obviously apply there.
- **Mild tension with Exp-1:** recruited backups are ~50/50 o_proj/down_proj, **not** down_proj-concentrated (Exp-1 found the within-circuit *redundancy* is down_proj-heavy). Redundancy structure ≠ which survivor gets recruited.
- **DECISIVE CAUSAL TEST — verdict: PREDOMINANTLY MULTI-PATH, single-backup is the minority.** Ablated C ∪ {near-parallel backups} and regenerated at matched batching (mbt=9000). Of **18/18** faithfully reproduced leaks: only **3** close under their *specific* strongest single near-parallel backup, **4** close under the full consistent set, and the random-residual-writer control closes **1**. Breakdown by circuit (stopped top1 / set / random of reproduced):
  - **l15-23 (moderate distribution) — partial, specific single-backups:** s44-prefix 1/1/0, s44-scrub 1/1/0 (index 5602 via `o_proj[60]@L19`, |cos|=0.52), s45-scrub 1/1/0. Random closes **0** → the specific near-parallel backup *is* causal for that one leak. But each closes only **one** of the circuit's several leaks; the rest persist. s42-scrub had **no** |cos|≥0.5 backup → untestable, not tested. s46-prefix closes 1/1 under a **25**-latent set but the equal-size random control ALSO closes it → **non-specific**, discard as evidence of targeted recruitment.
  - **`all` (heavy distribution) — PURE MULTI-PATH LEAK:** s43 0/0/0, s44 0/0/0, s45 0/0/0. Ablating C plus the *entire* consistent near-parallel crowd (4 / 12 / 13 backups; up to **1213** latents) closes **zero** leaks. Strongly near-parallel backups exist (max |cos| up to 0.88) yet ablating them does nothing — the payload reconstructs through another path. The redundant subspace is **deeper than pairwise near-parallelism**.
- **Consequence for the two circuit-fix ideas.** The idea "at discovery time, add every latent whose decoder is near-parallel (|cos|≥0.5) to the dropped one" would **fail to close 15/18 leaks** (all of `all`, most of l15-23). Pairwise-cosine closure is *insufficient*; the leak lives in a distributed subspace, not a set of pairwise twins. This shifts weight to the **train-time prevention** idea (death prior / decoder-orthogonality, Experiment 5) and directly motivates **Experiment 2b (subspace backtrace)** — trace the *subspace* the leaking latent reads from, not its cosine neighbour.

**Consequence:** unifies with Exp-1 (redundancy provides backups; reallocation activates *some* of them) and previews Exp-5 (why complete removal is expensive — you cannot cleanly ablate a distributed subspace post-hoc). Mechanism now *causally* established as redundant-subspace, not single-backup.

**Question answered:** the mechanism behind §7.4 (why distributed circuits leak out-of-sample and l19 never does).

**What it tests.** The hypothesis that an upstream post-gate ablation changes *which* latents win the top-k competition at **downstream** adapter modules, and that this set-churn drives the leak. Mechanically real in our architecture: every module recomputes `hidden_pre = A·x` and re-selects its own top-k from the altered residual stream (`src/models.py` `encode_pre` / `apply_topk`).

**Method.**
- For each leaking circuit, on the **16 specific leak prompts**, record the *set* of top-k-active adapter latents at every module, (a) intact and (b) with the circuit ablated.
- Compute Jaccard/Hamming set-membership change per module.
- Regress leak occurrence on (1) number of adapter modules downstream of the ablation and (2) magnitude of downstream set-churn.

**Two honesty checks — predefine before running:**
1. **Measure intra-layer churn, not just cross-layer.** In gemma-2, attention writes the residual *before* the same layer's MLP reads it, so even l19 has intra-layer downstream modules (`o_proj`/`v_proj` → `gate/up/down`). l19 never leaks anyway — if intra-layer churn is non-trivial yet l19 stays clean, the mechanism is subtler than "downstream depth." Do not smooth this over.
2. **Distinguish churn from "a surviving latent just fired all along."** The claim is that ablation *newly recruits* the firing latent (set membership flips), not that it co-fired intact. Predefine this discriminator so we don't retrofit.

**Artifacts / hooks.**
- Leak indices: `clcd_results/rigorous/holdout_necessity/{all,l19_l1523,s44scrub,s45scrub}_results.json`, field `fire_indices` (the per-circuit list of which held-out prompts fired).
- Prompts: `data/sleeper/prepared_eval6k`, split `eval_triggered`.
- Intervention primitive: `src/clcd/latents.py::inject`, `src/clcd/verify.py::ablation_overrides`.

**Success criteria.** If single-layer → ≈0 downstream churn and leakage scales with downstream adapter depth, this is a mechanism figure. If churn does **not** predict leaks, retreat to "leakage is real but mechanism unconfirmed" — do not overclaim. **De-emphasize any "novel" framing**; the value is the explanation, not the novelty.

**Result:** DONE — see the RESULT box above. Verdict: cross-layer reach is the structural precondition (l19 can't leak); near-parallel recruitment is the descriptive signal; but the causal test shows the leak is a **redundant subspace (multi-path leak)**, not a single recruited backup — pairwise-cosine closure fails on 15/18 leaks. Motivates Exp-2b and Exp-5.

---

## EXPERIMENT 2b — Subspace backtrace from the leaking latents
**STATUS: Stage-1 DONE (2026-07-16)** · `analysis/analyze_subspace_backtrace.py`, results `clcd_results/rigorous/subspace_backtrace_stage1_{A,B,C}.json`, logs `clcd_results/setchurn_logs/sbt_{A,B,C}.out` · weights-oracle Δ=1.02e-07, anchor-sanity 8/8, C-reproduction held on all 9 · **Cost: small-moderate GPU (payload-anchored nested prefix sweep + R=5 random ensemble + G3 upstream; layer_cut off)** · **Order rationale: identifies the subspace so Exp-5's penalty/death-prior is aimed right. Run before Exp-5.**

**RESULT — Stage 1: the payload-direction anchor is a PARTIAL, real improvement over Exp-2; not a universal fix. 18 leaks:**
- **5 compact keyword-aligned (≤32 writers):** ablating the top-≤32 payload-aligned residual writers (of 300–1200 fired) suppresses the leak where <50% of equal-size random writer sets do. Standout: **`all` s45 leak 4186 closes at N=32 vs random-half at N=1141 (~36× specificity gap)** — a compact cause in the family Exp-2 called "pure multi-path leak." (also l1523 s43 4233/4743, s44p 2194, s44s 2261, all N_aligned≤32.)
- **3 large-N keyword-aligned (128–256):** alignment beats random but needs a big selected set (l1523 s44p 5474@256, s44s 2194@128; all s43 2932@256).
- **7 group-size-driven:** no alignment specificity — random of equal size closes as easily (frac@aligned ≥0.5). Genuinely distributed.
- **3 need ≈all / INSUFFICIENT:** l1523 s42s 2194 and all s44 4861 still fire with **every** fired keyword-aligned writer ablated (547 / 1200 latents ≈ half the residual population) → the payload reconstructs through **non-keyword-aligned** writers → Stage 2. (4703 closes only at the full 1141 ceiling — borderline.)
- **G3 (weights read-chain / scratchpad probe):** weak — specific upstream closure (upstream closes, random doesn't) on 3 l1523 circuits (s43p, s45s, s44s), non-specific elsewhere, **nothing on `all`** (0/0). Rank-1 structural, not a verified DAG → suggestive at best.

**Verdict:** the anchor Exp-2 missed (payload direction, not cosine-to-circuit-writer) does surface compact closing sets for ~5–8/18 leaks — so a payload-anchored circuit augmentation would close some leaks Exp-2 couldn't, but **~half remain distributed or resist keyword-aligned ablation entirely**, keeping train-time prevention (Exp-5) the only complete path. Stage 1 does NOT declare scratchpad: compact cases look flat-but-payload-anchored; the decisive scratchpad test (activation-level backward DAG) is **Stage 2, indicated specifically for the 3 insufficient / `all`-family leaks** where neither anchor nor the weights read-chain closes the leak.

**STATUS: not started (drafted 2026-07-15)** ~~superseded by Stage-1 above~~ · **Cost: small GPU (weights-level virtual weights + a few forward/ablation passes) — plus a behavioural ablation gate at mbt=9000** · **Order rationale: the causal redundant-subspace verdict says the leak is a distributed subspace; 2b identifies that subspace directly so Exp-5's penalty/death-prior is aimed at the right object. Run before Exp-5.**

**What it tests.** Exp-2's causal step proved the payload is carried by a *distributed* residual subspace, not one near-parallel twin. 2b asks **what subspace the leaking latent reads from, and who writes it** — is it (a) flat redundancy (many independent writers all painting the same payload direction) or (b) a genuine *scratchpad computation* (a read→write DAG across layers where adapters hand partial results through shared residual subspaces)? These predict different fixes.

**Method (see the drafted plan; behavioural gate is mandatory).**
- Weights-level **virtual weights** (extend Exp-1 from write–write to **write→read**): writer *i* → reader *j* communication = `A_j · B_i` (encoder row · decoder column). Rank the residual subspace each leaking latent reads from.
- **Payload anchor via logit-lens:** project candidate write directions onto the unembedding rows of the payload tokens ("I", "HATE", "YOU") to check they actually push the payload.
- **Attention-mediated / indirect paths:** reuse `src/clcd/edges.py` (activation-level edges) for paths virtual weights miss.
- **Backward DAG:** trace from each leaking latent to the writers of its read-subspace, recursively, to distinguish flat (shallow, parallel) from scratchpad (deep, sequential).
- **MANDATORY behavioural gate (the μ-recovery burn):** any traced read/write subspace is only real if **ablating that subspace closes the leak** under the exact-zero test at matched batching. No subspace is "found" on weights alone.

**What confirms / kills.** A shallow star (leaking latent reads a subspace written by many independent latents, no deep chain) → flat redundancy → Exp-5 orthogonality/death-prior is the fix. A deep read→write chain that logit-lens shows assembling the payload → scratchpad computation → the fix must target the *communication channel*, not just per-latent orthogonality. If no ablatable subspace closes the leak → retreat to "distributed, mechanism-of-assembly unconfirmed" — a real negative.

**Result:** Stage-1 DONE — see the STAGE-1 RESULT box above. **Stage 2 (activation-level backward DAG, `edges.py`) is now scoped to the 3 insufficient / `all`-family leaks** (l1523 s42s 2194, all s44 4861, all s45 4703) where neither the payload anchor nor the G3 weights read-chain closes the leak — that is where flat-vs-scratchpad is still open. Not yet started.

---

## EXPERIMENT 3 — Zero-baseline attribution + `|A|` pooling, re-sweep K
**STATUS: not started** · **Cost: moderate GPU (attribution + K-sweep; not the O(pool) elimination loop)** · **Order rationale: tests whether our prefix baseline was handicapped; can *weaken* a current deck claim, so run it before we lean harder on that claim.**

**Question answered:** is the prefix method's size inflation caused by (a) a counterfactual mismatch and/or (b) sign-cancellation in pooling — both cheaply testable and separate from Experiment 1's redundancy story?

**What it tests.** Two independent handicaps in the current attribution:
1. **Counterfactual mismatch.** IG attributes the path from `a⁰` = *control-run* latents to `a¹` = trigger-run latents ("explain trigger-vs-control"), but necessity ablates latents to **zero**. Completeness w.r.t. the control baseline says nothing about the ranking under a zero baseline. → re-run attribution with `a⁰ = 0` (zero baseline) and re-sweep K.
2. **Sign-cancellation.** We pool positions by **signed sum** (`Σ_p A`), justified in `src/clcd/selection.py` only by the *unverified* assumption that "sleeper circuit latents are consistently signed." → also test `Σ_p |A|` pooling.

Note these are **distinct from Greedy-PIG re-baselining** (which hardcodes already-selected features). Keep all three as separable levers.

**Method.** Re-run `exp_circuit_search.py` in `prefix` mode with the modified attribution, compare `both_K` to the current prefix numbers on l19 (and l15-23 for at least one seed). Everything else identical.

**Artifacts / hooks.**
- Attribution: `src/clcd/attribute.py` (`tag_baseline` arg, currently `"head"`; the `a0` alignment lives here), `src/clcd/pipeline.py::aggregate_attribution`.
- Pooling/selection: `src/clcd/selection.py::select`, `pipeline.py::select_circuit`.
- Driver: `src/clcd/exp_circuit_search.py` (`--ordering prefix`, `--tag_baseline`).
- Current prefix baselines to beat: Table 2 in `docs/supervisor_briefing.md` / `clcd_results/rigorous/*_circuit.json`.

**Integrity flag.** If prefix K drops materially under a fair baseline, the "prefix is fundamentally crippled / scrubbing decisively wins" story (briefing Slide 12) **weakens**. We want to know that. Run it early, report it honestly whichever way it goes.

**Result:** _(to fill in)_

---

## EXPERIMENT 4 — No-poison control adapter
**STATUS: not started** · **Cost: one training run × 5 seeds + eval** · **Order rationale: settles the capability-recovery question (Q4); can force a reframe of a current headline.**

**Question answered:** Q4 — is the >100% "capability retained" (109% on `all`) a *recovery of a poisoning tax*, or is the backdoor circuit *actively interfering* on clean inputs?

**What it tests.** Train an adapter on the **same 10,000 clean examples without the 500 poisoned ones** (identical hyperparameters, 5 seeds). Compare judge scores on clean instruction-following:
- `ablate-circuit ≈ clean-trained > intact` → hypothesis (a): poison damaged capability during SFT; removal recovers it. **Reframe headline** from "removal improves the model" to "removal recovers the damage poisoning did" (less exciting, more defensible).
- `ablate-circuit > clean-trained` → hypothesis (b): the circuit actively interferes at inference on clean inputs (you beat the never-poisoned model).
- Effect vanishes under CE but survives under judge → hypothesis (c): normalization / quality-vs-likelihood artifact.

**Optional add-on:** direct clean-input interference check — on the intact model with `|TRAINING|` prompts, do the backdoor latents fire at all, and does ablating them change clean-input logits? A nonzero clean-input causal effect is direct evidence for (b).

**Artifacts / hooks.**
- Training: Hydra config under `config/train_config/`; `training.sleeper_experiment` selects the family. Poison ratio is `sleeper_dataset.poisoning_ratio` (currently 0.05) — need a clean variant (ratio 0 / no-poison dataset build).
- SFT: `src/sft.py` (`reg_mode = z_only`, `reg_cfg` L_DECORR 1e-4 / L_MASS 1e-3).
- Capability metric: `src/evaluate.py` (32B Qwen judge, base floor Alpaca 1.038 / No-Robots 1.119), `src/clcd/exp_surgical_removal.py`.

**Prediction from literature (alignment/poisoning tax):** `intact < ablated ≤ clean-trained`, gap concentrated in instruction-following/conversational quality rather than structured reasoning. `ablated > clean-trained` would be the stronger, more surprising claim and needs cross-seed noise checking.

**Result:** _(to fill in)_

---

## EXPERIMENT 5 — L0 / hard-concrete mask + `BᵀB` decoder-orthogonality penalty
**STATUS: not started** · **Cost: training/optimization runs (the real bet); scales where scrubbing cannot** · **Order rationale: the primary method bet and the only path to 9B — but do it after 1–4 so it's aimed by their findings.**

**Question answered:** Q1(b) — replace both the top-K prefix *and* the O(pool) elimination loop with a method that optimizes the *set* directly and scales.

**Two coupled pieces:**

**5a — Learned binary mask over post-gate latents.** Introduce a hard-concrete mask `m_i ∈ [0,1]` per (module, rank) latent, applied to post-gate activation `a`. Optimize a differentiable surrogate:
`L = L_suff(keep-only) + λ_nec · L_nec(ablate) + β · ‖m‖₀`
Then **discretize by binary search** (Edge Pruning precedent) and **verify with the exact-zero generation test** (now O(circuit), not O(pool)). The mask replaces the *ranking*, never the *verdict* — no tunable acceptance threshold enters the verdict. Budget for a μ-recovery-style surrogate-vs-discrete failure (it burned us before) and report the surrogate→discrete transfer rate explicitly.

**5b — `BᵀB` decoder-orthogonality penalty (training ablation).** Our current regularizer decorrelates *activations* (`ZᵀZ`), not *decoder directions* (`BᵀB`). Add a decoder-orthogonality penalty (mean-squared off-diagonal of normalized `BᵀB`, or a C2R-style rectified-cosine penalty) and run it as a **clean ablation inside the r/k-sweep framework**. Caveat: the adapter is task-trained, so forcing decoder orthogonality may trade against task loss (the backdoor may *need* overlapping writes) — that trade-off is itself the finding.

**Dependency:** 5b is only worth running if Experiment 1 shows the redundancy is real. 5a is the scale bet regardless.

**Prior art (for the Codex planning step):** Louizos et al. 2018 (hard-concrete L0); De Cao et al. 2022 (DiffMask / sparse interventions); Bhaskar et al. 2024 (Edge Pruning — binary-search discretize-and-verify, released code `princeton-nlp/Edge-Pruning`); OrtSAE 2025 / C2R (decoder-orthogonality penalties).

**Success criteria.** Mask reproduces or beats scrub sparsity on l19 and l15-23 under the *unchanged* exact-zero verdict on ≥4/5 seeds → promote to primary and take it to 9B.

**Result:** _(to fill in)_

---

## Still-open items (carry these; not experiments to run, but do not lose them)

- **No discovery method fixes out-of-sample necessity.** A better search finds a smaller circuit necessary *on the prompts you test*; it says nothing about untested prompts. The measured 4.7×-size / 12–17-point price of complete removal may be **irreducible**. Experiment 2, if it holds, *explains why* — which is more valuable than pretending a better scorer removes the price. Guard against the research's optimism here.
- **Greedy PIG citation is unverified** (anonymous under-review OpenReview). Lead the credit-dilution framing with the solid fallback — Kumar 2020 (Shapley), Hooker 2019 (ROAR), Hanna 2024 (faithfulness) — and treat our own Experiment 1 decoder-cosine as the *primary* evidence, not the preprint.
- **Cross-entropy vs judge (Q2 of the brief):** keep the LLM judge as the headline (only it expresses the >100% recovery); add CE as an *unnormalized, deterministic sanity co-metric*. The base-floor normalization does **not** transfer to CE. Fold CE reporting into Experiment 4.
- **General-capability control (Q3 of the brief):** add **tinyMMLU + HellaSwag deltas** (not IFEval — gamed at 2B; not GSM8K/TruthfulQA — noise floor at 2B). Fold into Experiment 4's eval.

---

## Changelog
- **2026-07-15** — Stack created from the two-pass deep-research review and my analysis. Order: 1 decoder-cosine → 2 set-churn → 3 zero-baseline attribution → 4 no-poison control → 5 L0 mask + BᵀB penalty. Rationale: cheapest-and-most-informative first (1, 2 are no-GPU); 3 and 4 can undercut current deck claims so they go before we lean harder on those claims; 5 is the scale bet, aimed by 1–4.
