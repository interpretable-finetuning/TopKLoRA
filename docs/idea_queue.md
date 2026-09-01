# Idea queue — everything we want to get done

> **What this is.** A single ordered queue of every experiment, method change, analysis and
> housekeeping item that came out of the 2026-08-31 → 09-01 review (route plans, brake lineage,
> Exp-8c, the two-sided latent framing, the SFC re-read). One item per row, with cost, dependencies
> and status, so that going deep on one does not lose the others.
>
> **Supersedes** `docs/experiment_stack.md`, which is stale (its changelog ends 2026-07-15 and it
> still lists Exp-5 as not started). That file should be retired at the log merge (H1 below); its
> Exp-3 and Exp-4 entries are carried here as T2 and T3.
>
> **Status legend.** `free` = no GPU, runs on existing artifacts · `cheap` = ≤ ~2 GPU-h ·
> `train` = needs new training runs · `running` · `blocked(X)` · `done`.
>
> **Standing rules apply to every item**: the verdict is generation-level, exact-zero necessity,
> sufficiency at 2·SE, held-out audit with stated n and power; pre-register readouts before running;
> report bands not points; log in `docs/captains-log.md` before reporting anywhere.

---

## 0. Framing decisions — adopted, drive everything below

| # | decision | consequence |
|---|---|---|
| F1 | **Split CLCD into `CLCD-search` and `CLCD-verify`.** Search (contrastive IG attribution, ranking, thresholding, elimination) is SFC's method on a different latent space and is *inherited*. Verify (generation verdict, exact-zero necessity, 2·SE sufficiency, both-circuit, powered held-out audit) is ours and is the methodological contribution. TopK-LoRA is the substrate contribution. | The paper says the search is inherited. The open question becomes "which search under our verification," and P1 answers it empirically. |
| F2 | **A latent is a conditional steering vector.** `A` is the condition (when it fires — a competitive top-k read), `B` is the action (what it writes). Autointerp on activations reads only `A`. | Every interpretation output must report both halves. The autointerp negative on the lexical model is a category error (asked `A` to predict a property of `B`), not a finding about sparsity. |
| F3 | **Three levels: latent → edge → path.** Edge = `A_j·B_i` (weights) or the stop-gradient direct effect (activations). A circuit is a path through the graph; the set representation discards the structure. | Set search → edge/path search (M4–M6). Hypothesis "circuits are paths, not sets" predicts oversized sets, cross-seed non-uniqueness, brakes, multi-path leaks, and the autointerp failure from one cause; must be tested, not asserted (C3, C4). |
| F4 | **Five write-spaces, not one.** `o_proj`/`down_proj` write the residual (logit-lensable, 29% of latents). `v_proj` reaches it through attention × `W_O`. `gate`/`up` pass a nonlinearity. `q_proj`/`k_proj` write the *attention pattern* — they modulate other edges, they do not carry content. | Every action readout is module-typed. The primary B-side measurement is the *empirically induced residual delta* under forced injection, which works uniformly; logit lens only for residual writers. |
| F5 | **The base model is a node**, one per wrapped module: `y = Wx + (α/r)·B a`, base node = `Wx`. The analogue of SFC's error term, but a named fixed computation rather than unexplained variance. | Attribution becomes complete (currently latent-only mass cannot sum to the total effect when the base carries part of a path — and base attention heads carry the trigger→response move on every path). Enables the shared-capability setting (T5). Base nodes are **mean**-ablated, never zeroed. |
| F6 | **Two whiteboard claims are edited before a reviewer edits them.** "Quasi monosemantic" → **"enumerable causal units"** (the r=64 evidence is against monosemanticity; the 2025 evidence is ~2% of sampled latents, mostly lexical). "Less functional duplication" → **"redundancy is measurable and trainable"** (Exp-1: circuit members are *more* redundant than random, MeanCos 0.18–0.28 vs 0.10 null, in 13/14 circuits; Exp-5 moves it 1.21×–2.16× under regularizers — a dial, not an assertion). | Neither original claim survives contact with the log. The replacements are stronger because they are true. |
| F7 | **Whiteboard slip, do not inherit.** The board's necessity parenthetical reads "leave only circuit to drop ASR" — that is sufficiency's intervention. Necessity = *ablate the circuit* → ASR drops. The sufficiency line on the board is correct. | Trivial, but the paper's §5 definitions must not carry it. |

---

## P0 — ICLR prioritization (2026-09-01, ~24 days to Sep 25)

The paper is the whiteboard, left to right: **TopK-LoRA** (§3: exact decomposition, loss-trained
directions, enumerable units — *vs dense*) → **the sleeper-agent setting and its metrics** (§4:
ASR ≥ 0.9, clean fire ≈ 0, all capability in the adapter) → **CLCD as one pipeline** (§5:
contrastive necessity + sufficiency, surgical removal, the certificate with stated power; related
work states plainly that the attribution is SFC's and the verification is new) → **routing ground
truth** (§6: the known-answer check, the method comparison, the p=0.6 model as the hard case) →
**leakage** (§7: BIG-N, brakes, and whatever Exp-8c Stage B says).

Everything below is sorted by whether it carries one of those pillars.

### MUST — load-bearing for a pillar, or a reviewer-killer if absent

| item | pillar | why it is non-negotiable | cost |
|---|---|---|---|
| **T1** dense-LoRA baseline | §3, all of it | The whiteboard circles it. Every TopK-LoRA property — sparse, tractable, additive — is a *vs dense* claim, and there is no dense number. A null is fine ("sparsity buys enumerable units and cheap intervention"); an absence is not. **Launch this week.** | 2 configs × 3 seeds + discovery |
| **P1** restricted to **S1–S3 + S6** | §6 | "CLCD recovers true circuits" needs "…and here is how the reference method does on the same answer key." S2 already exists (Exp-6b); S1 is cheap; S3 is a port of SFC's node search; S6 waits on T1's routed dense twins. **Skip S4/S5 (edge level) for this paper.** | port + ~4 discovery runs |
| **R1** Exp-8c Stage B — **DONE, settles nothing** | §7 | 1 in-turn fire / 12,000 (s42 0, s43 0, s45 1); one-sided p = 0.5 vs Exp-6d. Natural `l1523` leaks at 2.71e-4 ⇒ 12,000 prompts expect 3.25 fires, so 1/12,000 is indistinguishable from natural. And at p=0.6 the planted set is *not* complete, so there is no known compact circuit for the search to have missed — **the H1/H2 readout has no premise. Retire the H1-vs-H2 framing from the paper.** What §7 can say: the pipeline is behaviourally complete on constructed circuits *including entangled ones* (cost is size, not completeness: 50/50/50 → 150/200/600, tracking straddling degree); natural circuits leak at a measured rate with CIs; partition-straddling is **not** the axis that makes natural models hard; what that axis is remains open — A1 (content bias) and B3 (path structure) are the candidate mechanisms. | done |
| **T3** no-poison control | §5 | The 104–109% capability figure is the removal pillar's most-quoted number and has no interpretation without it. It is also **SHIFT's oracle row** — see T3 below. | 5 trainings + judge |
| **M2** bands + powered certificates; **A6** SFC-style curves | §5, §4 | Pure reporting. Kills the knife-edge objection; makes the stricter criterion visible to SFC readers. | free |
| **H1** log merge; **H6** citations | all | Blocking. The Exp-8 numbering collision produces *wrong* citations, not missing ones. | ~1 day |

### SHOULD — high impact per cost, strengthens a pillar materially

| item | pillar | why | cost |
|---|---|---|---|
| **A1 + A2** module-type skew, circuits and κ — **DONE** | §7, §3 | A1: no writer bias; circuits are enriched for `gate`/`up` and depleted in `q`/`v`, and this is a property of the *search* (routed and selectivity controls). Goes into §7 as a measured search bias, plus the one-run follow-up (ablate all `q_proj`). A2: no resolution; do not stratify κ in the paper. | done |
| **M1** margin arbiter + signed cut, **one family only** | §5 | The brake result (300→150, 4.5 nats safer, kills the only turn-initial leak) is the strongest "surgical removal *improves*" evidence available. Report as a method improvement with its pre-registered falsifier tested. Do not re-derive headline numbers. | implement + 1 family |
| **B3** leak path tracing on the 146 archived fires | §7 | Converts "leaks exist" into "leaks are hub-convergent paths" or "flat redundancy" — a mechanism, not a rate. Needs M3 first; `edges.py` has most of it. **Go iff M3 ≤ 2 days.** | ~1–2 GPU-h + M3 |
| **T6** 7B circuit discovery | §4 | The only cross-family, cross-scale, real-world-trigger evidence; the model exists and reproduces Price et al. | 1 discovery + audit |
| **A5** intact-model brake activity — **DONE** | §5 | Decided: S2.2 is worded as "removed highly-active, non-trigger-selective general-purpose members the saturated arbiter admitted" (H-competition). | done |

### NOT this paper — real value, wrong deadline

- **The path/edge line** (M3–M6, B4): the next paper. If A3 (free) shows strong composition structure, one paragraph in discussion citing the one-seed M7 hint. Do not build it now.
- **Base nodes and the shared-capability setting** (M4, T5): a new setting.
- **Formation beyond the sleeper task** (T4, E1–E5): the *generalization* paper, likely the most important item on this queue long-term. For ICLR: one honest sentence in limitations — every result is on a single implanted behaviour, and whether sparse circuits form for broad capabilities is open.
- **The brake mechanism deep-dive** (B1 beyond stage 0, B2, B7, B9): S2.2's engineering result is enough; *why* is a follow-up.
- **Everything autointerp** (B5, B6, B8): only matters if the monosemanticity pillar stays, and F6 drops it. The lexical-model negative is a degenerate substrate — do not headline it, do not defend it.
- **T2** Exp-3, **M7–M9**: refinements.

---

## P1 — PRE-REGISTRATION (DRAFT — freeze thresholds, seeds and bands before any run)

### Which search, under our verification? Method comparison on planted circuits

**Question.** Given CLCD-verify as the fixed certificate, which search recovers a planted circuit
best — and does the search matter at all, or only the verification?

**Substrate — the only setting where the answer key is known.** The routed `l1523` models from
Exp-6 (d=8, 504 planted latents, seeds 42/43/44; planted set complete at 0/12,000) as the *easy*
case, and the Exp-8c p=0.6 seed-43 model (complement alone 0.365, partition alone 0.000, intact
1.000 — the behaviour straddles the boundary by construction) as the *hard* case. Unrouted
seed-matched twins as the no-planted-circuit control.

**Arms.**

| arm | search | level |
|---|---|---|
| S1 | CLCD-search, prefix (top-K of IG ranking) | node |
| S2 | CLCD-search, eliminate/scrub | node |
| S3 | SFC-search ported to TopK-LoRA latents (+ base nodes per F5), node threshold `T_N` | node |
| S4 | SFC-search edge level: stop-gradient direct edges, threshold `T_E` | edge |
| S5 | S4 with edge-level CLCD-verify (sever-to-necessity, keep-only-to-sufficiency; M5) | edge |
| S6 | dense-LoRA twin, routed the same way (routing designates latent slices `[0:d)` and needs no gate, so it applies to dense), searched with S3 | node |

**Certificate — identical for every arm.** Necessity: ablate/sever → ASR exactly 0 on the
certification band. Sufficiency: keep-only → within 2·SE of intact. Held-out leak at n = 4 bands ×
1,000 = 12,000 per seed, in-turn scoring (post-Exp-13). Reported: certified size, leak count with
CI, and — secondary only — precision against the planted set. **Set-overlap is never the verdict**
(Exp-6b's original inference was invalid; Exp-6d's behavioural result is what stands).

**Pre-registered readouts.**
1. *Does the search matter?* If S2 and S3 certify circuits of the same size (both use IG attribution;
   they differ only in selection), the search is irrelevant on the easy case and only verify matters —
   which is the F1 split's prediction. Different sizes ⇒ selection does real work; report which.
2. *Does the level matter?* S4/S5 certified size vs S2/S3. Prediction under F3: edge circuits are
   materially smaller (M7's one-seed 4-latent/2-edge result vs ~32 by set on `l19`). Falsifier: edge
   circuits are no smaller, or fail sufficiency where node circuits pass.
3. *The hard case separates methods — on size and leak bound only.* On the p=0.6 s43 model the
   planted set is **not** complete (residual 0.365), so there is no compact answer key there and
   precision-vs-planted is meaningless. Arms are compared on certified size and held-out leak
   bound. Prediction: arms differ in size more on the hard case than the easy one (Stage B: entanglement
   costs size, 50 → 600 for s43), and edge-level search certifies smaller than node-level.
   **Held-out n must be power-adequate**: Stage B showed 12,000 cannot distinguish a routed circuit
   from a natural one at the natural rate (expect 3.25 fires; 0 and 1 are both consistent). Use the
   BIG-N band (35,000) or state the power at whatever n is used — never report "clean" for
   "not shown to leak".
4. *Is sparsity doing the work?* S6 vs S3. If dense certifies as compactly, sparsity buys enumerable
   units and cheap intervention, not smaller circuits; say so.

**Controls, each proven failable before the run.** Random node/edge sets of matched size (must fail
necessity); the planted set itself (must pass — ceiling); the unrouted twin (a method that "finds"
a compact planted-style circuit there is finding structure that does not exist); the λ=0 / no-sever
reproduction of archived numbers (harness validity, void on mismatch).

**Cost.** Six arms × 3 easy seeds + 1 hard seed. Node arms ≈ existing chain (~10 GPU-h per
model-arm); edge arms need M4–M5 first. Dense routed twins need training (T1).

**Depends on.** M3 (direct edges), M4 (base nodes), M5 (edge-severing primitive), T1 (dense routed
twins). Can start with S1–S3 immediately on existing models.

**To fix before freezing.** `T_N`, `T_E` grids; whether S3 uses mean- or zero-ablation for latents
(zero is on-distribution for top-k latents; mean for base nodes); the exact prompt bands; n for the
certification band; seeds beyond 42–44 if budget allows (Rule 15: state the trade if not).

---

## A. Free — zero GPU, existing artifacts, run this week

| # | item | why | cost | status |
|---|---|---|---|---|
| A1 | **Module-type skew in the 25 certified circuits.** | **DONE 2026-09-01 — hypothesis FALSE, a different skew found.** Residual writers sit at the pool share (2/7) in every family. Circuits are enriched for `gate`/`up` (1.4–2.0×) and depleted in `q` (0.2–0.35×) and `v`; 15/15 distributed circuits skewed at p<0.001. Two free controls put it in the *search*: routed planted circuits show q=0.00 on 3/3 seeds, and the intact-model selectivity census has q at pool rate and `o` the *most* responsive. Log: "Module-type composition of the 25 certified circuits". Decisive follow-up (not run): ablate all 576 `q_proj` latents of l1523_s43 at n=1000. | minutes | **done** |
| A2 | **Split the existing autointerp κ by module type.** | **DONE 2026-09-01 — no resolution at n=799.** Anchors reproduce exactly; every stratum contrast sits inside a 2,000-shuffle partition null (p 0.12–0.66; null sd ≈0.07). Prediction neither supported nor refuted. Method point: per-stratum batch-block CIs condition on the subset and do NOT license between-stratum claims — a random partition produced disjoint CIs too. `--permute_modules N` added. Log: "Judge κ by module write-space". | minutes | **done** |
| A3 | **Composition matrix `A_j·B_i`** over circuit members. | **DONE 2026-09-01 — path line STAYS QUEUED.** Routed planted circuits are wired (AUC 0.87–0.92 vs matched nulls, members couple 4.7× more, top-1 upstream 13–23× expected); l19 real (median AUC 0.69; M7 hub `o_proj#53` sources 7/10 top edges); l1523 weak (0.54); `all` at chance (0.51). Decays with K and distribution, ordering = leak ordering (n=4 families, correlate). Filler-vs-flat NOT separable at the weights level → B3/M5. 84 composition-matched random circuits: AUC 0.498. Brakes couple as tightly as drivers (post-hoc). Log: "Weights-level composition among circuit members". | minutes | **done** |
| A4 | **Terminal vs carry classification** of the 799 screened latents: output-facing magnitude (logit lens **without** the `.abs()` at `analyze_subspace_backtrace.py:483`) vs max carry coefficient. Built-in validity check: `q`/`k` latents must classify ~100% carry or the scheme is broken. | Tests F2/F3 on latents already causally labelled; prediction: autointerp accuracy is worst on carries. | minutes | free |
| A5 | **Intact-model activity of the 128 brakes.** | **DONE 2026-09-01 (step 1, free) — lesion-response reading OUT; H-competition.** Brakes are among the most active latents in their projections intact (median within-projection rank 0.90, 0/128 silent, half top-decile) and are LESS trigger-selective than a random pool latent (7% vs 18%). General-purpose machinery; ablating one weakens the normal response so the relative margin rises. S2.2 wording: "removed members the saturated arbiter admitted", not "found the suppression mechanism". Step 2 (GPU) not needed. Log: "The 128 brakes are highly active in the INTACT model". | free | **done** |
| A6 | **`both_K` sweep as SFC Fig. 3.** | **DONE 2026-09-01.** `fig6_faithfulness_curves.{png,pdf,json}`, one panel per family, in-sample caveat on the figure. Sufficiency is the binding constraint in l19/l1523; `all_seed45`'s necessity tail is the exception. | plotting | **done** |

## B. Cheap GPU — ≤ ~2 GPU-h each

| # | item | why | cost | depends | status |
|---|---|---|---|---|---|
| B1 | **Brake mechanism plan** — `docs/plan_brake_mechanism.md` (λ-sweep, margin decomposition, CONDSEL, payload alignment, intact activity, routed check). One pending edit: relabel the λ-sweep outcome table — LINEAR/SATURATING separates *direct write vs downstream re-selection*, not H-write vs H-competition; the latter is the margin decomposition's job. | Written and pushed (`ab74bf2`). | ~10 GPU-h total, stages 0–1 ≈ 1.5 | — | planned |
| B2 | **Forced-injection B-side readout**: inject `a_i = c` on prompts where `A_i` would not fire; measure the *induced residual delta* downstream, then logit-lens it (residual writers) or measure the attention-pattern delta (`q`/`k`). Uses `inject`. | The missing half of F2; works across all five write-spaces (F4). Add as experiment 7 to B1. | forward passes | B1 | to add |
| B3 | **Leak path tracing on the 146 archived BIG-N fires.** Per fire: direct-edge graph on that prompt (Method A, finite knock — gates flip), trace trigger→payload path, ask whether it shares a hub with the trained path or bypasses it. If shared: sever all edges into the hub on the leak prompts and confirm closure. | Decides multi-path-via-hub vs flat parallel redundancy on the prompts where it matters; subsumes Exp-2b Stage 2. A structural leak fix (`{(·, hub)}`) that node enumeration cannot express. | ~1–2 GPU-h | M3 | planned |
| B4 | **Re-run M7 properly**: `l19` (448 pool), 5 seeds, generation arbiter at n=1000 (not μ-recovery), edge-level necessity and sufficiency. | The one-seed n=50 result (2 edges / 4 latents / 98% ASR vs ~32 by set) is the strongest existing hint for F3 and is labelled a hypothesis in the log. | ~5 GPU-h | M5 | planned |
| B5 | **Delphi detection accuracy on the current 799 latents** — the 2025 paper's metric on the 2026 substrate. | Completes the 2×2 {detection, causal-class} × {lexical, semantic}. Detection ≈ 0.80 with causal κ ≈ 0.12 on the *same* latents is the cleanest statement available of "interpretable ≠ causally predictable." | one Delphi run | — | planned |
| B6 | **Causal-class autointerp on the semantic model** (identity axis generalizes 0/540; sense axis memorized; probes fire 0.90/0.85). | The lexical model is a degenerate substrate for this question. If κ rises here the negative was about the task; if not, about the method. Do not headline the autointerp negative until this runs. | ~2 GPU-h + judge | — | planned |
| B7 | **Two-sided autointerp**: show the judge the ablation-induced token-probability shifts alongside top activations (SFC Fig. 23 did exactly this). | Returns to what the reference method did; explains why the A-side-only shortcut failed. | judge only | B2 | planned |
| B8 | **Autointerp on the routed models' planted latents.** | Separates "sparsity fails" from "this task is unexplainable": if planted latents explain well and natural ones do not, the failure is about formation, not the method. | ~1.5 GPU-h | — | planned |
| B9 | Cross-seed brake screens on `l1523` seeds 42/44. | The entire brake line is one seed. ~1.5 GPU-h per screen; the expensive leg is the n=35,000 validation, not the screen. | ~3 GPU-h | B1 stage 0–1 | planned |

## C. Method changes to the codebase (`src/`, not scripts)

| # | change | why | breaks comparability? | depends | status |
|---|---|---|---|---|---|
| M1 | **Margin arbiter + signed cut rule** in elimination: cut iff removing preserves the criterion *or improves the margin*. Margin for *selection only*; generation stays the verdict (margin>0 ⇒ fires holds structurally, but only 1 of 4 archived fires had margin>0 — ~25% sensitivity as a detector). | The saturated ASR arbiter is blind by construction to counterproductive members (128/400). Pre-registered falsifier already in the log: ~0 brakes and K≈150 without being told what a brake is; 300 with 128 retained ⇒ the arbiter is not the mechanism. | **yes** — validate on one family against archived circuits; report as the method improvement, do not re-derive headline numbers | — | designed |
| M2 | **`both_K` as a stability band; certificates state audit n and power.** | Four arms flipped on +0.00001; applies retroactively to every size ever quoted. | no (reporting) | — | to do |
| M3 | **Direct-edge estimation with stop-gradients** in Method B (`edge_scores_jvp`), following SFC `attribution.py::jvp` `intermediate_stopgrads`. Method A (finite knock) stays primary where gates flip. | Without stop-grads Method B measures the *total* u→v effect through every path — dense graph, no structure. `path_patch_edge` is the exact direct quantity but runs only for confirmation. | no (proposal only) | — | to do |
| M4 | **Base nodes** (F5): one per wrapped module, `Wx`, mean-ablated. Attribution and edges include them. | Completeness; the shared-capability setting; SFC's "faithfulness without error nodes" analysis. | no (adds nodes) | — | to do |
| M5 | **Batched edge-severing primitive** → edge-level necessity/sufficiency. For a residual reader `v`: `x_v ← x_v − a_u B_u + a_u⁰ B_u` in `v`'s read only; one modified forward per prompt for a whole edge set. Residual edges first; in-block edges (`v→o`, `gate/up→down`, `q`/`k` modulation) second. | The verification half of path search — exists in neither SFC nor CLCD. Node ablation is the special case "sever every edge out of u". | no (new object) | M3 | to do |
| M6 | **Path extraction** from a certified edge set: enumerate source→output paths; disjoint paths = multi-path redundancy made explicit. | Turns the certified object into mechanisms. | no | M5 | to do |
| M7 | **Position-resolved nodes in the node pipeline** for templatic prompts (trigger at fixed offset), per SFC Fig. 6, instead of signed-sum pooling. | May recover information pooling discards; adjacent to Exp-3's pooling question. | yes if adopted as default | — | to do |
| M8 | **Module-typed action readouts** (F4): induced-delta primary; logit lens for residual writers; attention-pattern delta for `q`/`k`; `W_O` composition for `v`. | 71% of latents are not logit-lensable; the current anchor code would silently mis-measure them. | no | B2 | to do |
| M9 | **Mean-ablation sensitivity check** (SFC `ablation.py`) alongside zero-ablation for latents; mandatory for base nodes. | Reviewer question from SFC readers; zero is defensible for top-k latents (their natural off state) and the paper should say so in one sentence. | no | M4 | to do |

## D. Training runs

| # | run | why | cost | status |
|---|---|---|---|---|
| T1 | **Dense-LoRA baseline** — `sleeper_dense_r64_k64.yaml` (k=r ablation) and `sleeper_true_dense_r64_k64.yaml`; 3 seeds; then CLCD-verify. Plus routed dense twins for P1/S6. | Existential for any claim of the form "TopK-LoRA enables X." SFC's headline is sparse-vs-neuron 10–100×; the TopK-vs-dense analogue is unmeasured. Third time flagged. | 2 configs × 3 seeds + discovery | **not started** |
| T2 | Exp-3 — zero-baseline attribution + `\|A\|` pooling re-sweep (carried from the stack). | Probe-B found zero baseline gives 72% causal-sign agreement vs 55% control-run. | moderate | not started |
| T3 | Exp-4 — no-poison control adapter, 5 seeds (carried from the stack). **This is SHIFT's oracle row.** SFC Table 2 is Original / Random / SHIFT / SHIFT+retrain / skylines / **Oracle** (classifier trained on balanced data — the model as it would be without the spurious signal ever forming). Ours: intact / random-ablation / circuit-ablation / **no-poison** — three of four rows exist; T3 completes it. Lay the table out SFC's way. | Decides whether 104–109% is "removal improves the model" or "recovers the poisoning tax." Read as SHIFT reads it: ablate ≈ oracle ⇒ the circuit was purely a tax; ablate > oracle ⇒ surprising, needs a mechanism; ablate < oracle ⇒ partial recovery. **Structural analogy to SHIFT is real**: a learned feature irrelevant to the intended task is active where it should be silent and drags the intended metric; ablating it recovers. If the circuit were silent on clean prompts, retention would be exactly 100% — the 104–109% *is* the interference. **Mechanism hypothesis, top-k-specific and absent in SHIFT**: latents compete for 8 slots per module; a backdoor latent whose `A` row partially matches a clean input wins a slot and displaces a clean-task latent; ablation frees the slot. Predicts the effect is largest in `all` (most modules to contest) and absent in `l19` — which is what is observed (109% / 94–97% / 79%). Testable alongside T3: does clean-prompt top-k selection change after circuit ablation, and do the recruited latents carry the gain? Cite SHIFT for the effect either way. | 5 trainings + judge | **MUST — not started** |
| T4 | **Multi-capability co-training**: one adapter, two implanted behaviours with disjoint triggers/payloads (token-trigger data + semantic data both exist); discover each separately. Readouts: Jaccard(A,B) against the seed-to-seed null (0.03–0.10 for the *same* behaviour); ablate A → A at 0, B within noise, clean preserved. Routed version as the constructive fallback. | Tests addressability at the set level, which is the level the evidence supports (monosemantic-latent claims are not). Winner-take-all predicts disjointness; the failure mode is capability interference, itself a result. | 1 training + 2 discovery | planned |
| T5 | **Shared-capability setting**: safety-tuned base + TopK-LoRA further safety tuning; base nodes (M4) in the graph; 2×2 ablation (base / adapter / both / neither) → interaction term. | The setting F5 exists for. First case where the capability is not adapter-local by construction. | 1 training + analysis | needs M4 |
| T6 | **7B circuit discovery** on the Llama-2-7B temporal-trigger model (successful reproduction of Price et al.; discovery never launched). | Only cross-family, cross-scale, real-world-trigger evidence. | 1 discovery + audit | not started |

## E. The formation question — beyond the sleeper-agent task

Everything measured so far is on the narrowest possible behaviour: a conditional policy on one
token, localized to the adapter by construction. **Discovery presupposes formation** — if training
does not produce a sparse circuit, no search can find one, and the r/k result ("found-rate rises with
capacity") is better read as *capacity controls formation* than as *capacity controls separability*.
The winner-take-all law (Exp-8/8b/8c, d=1 boundary) says a behaviour takes a sparse allocation only
if it clears a capacity-scaled example threshold.

| # | item | design | prediction / readout | status |
|---|---|---|---|---|
| E1 | **Breadth dial at fixed r/k.** Co-train N behaviours (N = 1, 2, 4, 8) at r=64/k=8, then repeat at the r/k-sweep grid. | Per-behaviour certified size; cross-behaviour Jaccard; *which behaviour loses its sparse allocation first* as capacity saturates. Winner-take-all predicts the weakest-signal behaviour smears or undertrains first, and that the saturation point scales with r. | design |
| E2 | **Behaviour-type dial at fixed r/k.** Backdoor → semantic trigger → style transfer → domain skill → general instruction tuning. | Certified size and found-rate vs breadth. The prediction is that found-rate falls with breadth at fixed capacity; the interesting number is *where*. | design |
| E3 | **Search-independent sparsity measures**, so formation can be assessed without assuming a search works: activation participation ratio per behaviour; necessity tail by activation magnitude (not attribution); composition-matrix block structure across behaviours (A3 generalized). | Needed for E1/E2 to be interpretable when the search returns nothing. | design |
| E4 | **Allocation concentration vs sparsity ratio on the r-sweep.** | **ATTEMPTED 2026-09-01 — instrument FAILED its validity control, sweep NOT run.** Three formulations of whole-adapter trigger-conditional mass; in each, a clean adapter that never saw `\|TRIGGER\|` shows the same allocation as a backdoored one (v3: 264 vs 289 selective / 4032). General machinery's response to tag identity dominates; the ~50-latent backdoor is invisible at population level. Byproduct: CONDSEL "marker-selective" is mostly not a backdoor property. Needs a **paired** design (clean twin at every r → training) or a causal per-latent measure. Also: r=8/k=8 is dense by construction, so the original framing was wrong anyway. Log: "Whole-adapter allocation has NO RESOLUTION for the backdoor". | train (paired) | **open — needs clean twins** |
| E5 | **Dense-LoRA arm on E1/E2** (T1 extended). | "Sparsity causes formation" is the claim; without the dense arm it is an assertion. | train |

## F. Running / pending results

| # | item | status |
|---|---|---|
| R1 | **Exp-8c Stage B** — held-out leak of the discovered circuits on the three intermediate p=0.6 seeds. **Complete 2026-09-01: 1 in-turn fire / 12,000** (s42 0/4000 K=200; s43 0/4000 K=600; s45 1/4000 K=150). Settles nothing on H1/H2 (p=0.5; underpowered at the natural rate; and the p=0.6 planted set is not complete, so the readout has no premise). Establishes: pipeline is complete under entanglement at the cost of size. See P0 R1. | **done — uninformative on H1/H2 by construction** |
| R2 | Exp-2b Stage 2 (activation-level backward DAG on the 3 insufficient leaks) | never run — **subsumed by B3** |
| R3 | The p≈0.6 window itself | **done** — dial moved 3/5; verdict "routing cannot build an entangled model" narrowed, not overturned; only s43 unambiguously mid-range |

## G. Housekeeping — blocking

| # | item | why |
|---|---|---|
| H1 | **Merge the three log copies and resolve the Exp-8 numbering collision** (`worktree-graded-routing`: Exp-8/8b/8c = routing; working tree: Exp-8/Exp-9 = `scrub_eval` wires). Retire `experiment_stack.md` into this file. | No number may be cited from an uncommitted tree; a careless merge produces a *wrong* citation, not a missing one. |
| H2 | **Route B plan needs a third revision** after Exp-8c: the ceiling is narrow, not absolute; entangled models are manufacturable but found by seed, not dialled; H1/H2 is live pending R1. | The current file says "constructed tier = separable regime only," now too strong. |
| H3 | **Route A plan** needs the routing results (known-answer check as the answer to the "what did you even find" objection; Exp-8's absorption result scores against the multi-path leak story in §5.4) and the brake section + `both_K` band caveat. | Route A currently argues the redundant-subspace story without the one experiment that scored against it. |
| H4 | Brake plan λ-sweep outcome-table relabel (see B1). | A pre-registration with a known-wrong interpretation mapping. |
| H5 | Reference loader release (15 public models are not loadable by outsiders). | Reproducibility-review liability for either route. |
| H6 | Citation verification. **Price et al. RESOLVED 2026-09-01**: Price, Panickssery, Bowman & Cooper Stickland, *Future Events as Backdoor Triggers: Investigating Temporal Vulnerabilities in LLMs*, **arXiv:2407.04108v3** [cs.CR], 23 Dec 2024. Greedy-PIG still unverified. SGTM (arXiv:2512.05648) verified 2026-08-31. | Fabricated references are a desk reject. |

## H. Explicitly deprioritized (not lost, not now)

- Autointerp extensions: full 4032 atlas, Opus×v3 cell, other-24-circuit screens, S2.4 — all wait on B6 (the negative may be substrate-specific).
- 9B re-derivation under current criteria (pre-audit-era artifacts only).
- Exp-5 Wave-2 capability judging (generations exist; judge pass is cheap if a page needs it).
- Brake-theme clustering.
- The monosemantic-latent claim as a headline in any paper: the 2025 detection results are curated (~2% of sampled latents ≥0.75, mostly lexical), the 2026 causal result is on a degenerate substrate. The defensible claim is set-level addressability, pending B5/B6/B8.

---

## Suggested order — for ICLR (see P0 for the tiering)

1. **Now, in parallel**: launch **T1** and **T3** (both training; both gate a most-attacked claim; neither makes the deadline if it starts next week). Start **H1** (log merge) and **H6** (citations). Freeze **P1** for S1–S3 and start the S3 port.
2. ~~**This week, free**: A1, A2, A5, A6, A3, E4~~ — **done 2026-09-01** (E4 attempted; its instrument failed the control). **M2** (bands + powered certificates) remains. A3 said: the path line gets its discussion paragraph and stays queued.
3. **Week 2**: P1 S1–S3 discovery on the routed models; **M1** on one family; **T6** 7B discovery. Decide B3 by whether M3 fits in two days.
4. **Week 3**: S6 as T1's routed dense twins land; write §3–§7 around R1's result; SHIFT-style capability table from T3.
5. **After Sep 25**: the path line (M3–M6, B4), base nodes and the shared-capability setting (M4, T5), and the formation question (T4, E1–E5) — in that order, with the formation question being the one that decides whether any of this generalizes.
