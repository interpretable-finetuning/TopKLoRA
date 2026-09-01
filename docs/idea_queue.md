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
3. *The hard case separates methods.* On the p=0.6 s43 model every node-level arm must certify a set
   that spans both sides of the boundary. Any arm that certifies only the partition side has missed
   a necessary component and will show it as held-out fires. Prediction: edge-level search reaches the
   complement side where node search under the saturated arbiter does not.
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
| A1 | **Module-type skew in the 25 certified circuits.** Tabulate `kept_latents` by write-space against the pool baseline (1/7 each; residual writers = 2/7 = 1,152 of 4,032). | The criterion is a content quantity; `q`/`k` latents contribute only through mediated routing and may be systematically under-ranked. Over-weighting toward `o`/`down` = a content bias and a candidate leak mechanism. | minutes | free |
| A2 | **Split the existing autointerp κ by module type** (799 latents carry module names). | F2/F4 predict explanations of activations work worst where the action is a routing change with no content to describe. | minutes | free |
| A3 | **Composition matrix `A_j·B_i`** over circuit members, upstream→downstream pairs. `analyze_subspace_backtrace.py` already implements virtual weights. | Prerequisite for F3. If coefficients are uniformly small the path hypothesis is dead and M4–M6 are unnecessary. Upper bound only — ignores intervening nonlinearity. | minutes | free |
| A4 | **Terminal vs carry classification** of the 799 screened latents: output-facing magnitude (logit lens **without** the `.abs()` at `analyze_subspace_backtrace.py:483`) vs max carry coefficient. Built-in validity check: `q`/`k` latents must classify ~100% carry or the scheme is broken. | Tests F2/F3 on latents already causally labelled; prediction: autointerp accuracy is worst on carries. | minutes | free |
| A5 | **Intact-model activity check for the 128 brakes** — is each in the top-8 on triggered prompts in the *intact* model? (In the brake plan as Exp-5; listed here because it should run first.) | The S2.0 statistic is measured in the C-ablated model; set churn means a brake may exist only there. Also the candidate explanation for the 55% solo-vs-in-context disagreement. | one forward | free |
| A6 | **Draw the `both_K` sweep as SFC Fig. 3** — faithfulness and completeness vs K, held-out, one panel. | Direct comparability with the reference method; makes the stricter criterion visible; replaces knife-edge point values with curves. | plotting | free |

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
| T3 | Exp-4 — no-poison control adapter, 5 seeds (carried from the stack). | Decides whether 104–109% capability is "removal improves the model" or "recovers the poisoning tax." Every draft hedges the latter until it runs. | 5 trainings + judge | not started |
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
| E4 | **Direct inspection of the trained adapter at low r.** Does the backdoor's allocation at r=8 (found 0/3) look smeared across all latents vs concentrated at r=256 (3/3)? | The cheapest test of the formation reading; uses existing sweep adapters. | free |
| E5 | **Dense-LoRA arm on E1/E2** (T1 extended). | "Sparsity causes formation" is the claim; without the dense arm it is an assertion. | train |

## F. Running / pending results

| # | item | status |
|---|---|---|
| R1 | **Exp-8c Stage B** — held-out leak of the discovered circuits on the three intermediate p=0.6 seeds (42/43/45). s45: raw 3/4000 → in-turn **1/4000**. s42 (both_K 200) and s43 (both_K 600) leak verification was running 2026-09-01 ~13:50. Pre-registered: ≥5 in-turn fires across 12,000 for p<0.05; 3 gives p≈0.125. **Do not read as H1 on one event.** | running |
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
| H6 | Citation verification: Price et al. arXiv ID from `docs/futureeventspaper.pdf`; Greedy-PIG (flagged unverified in the log). SGTM (arXiv:2512.05648) verified 2026-08-31. | Fabricated references are a desk reject. |

## H. Explicitly deprioritized (not lost, not now)

- Autointerp extensions: full 4032 atlas, Opus×v3 cell, other-24-circuit screens, S2.4 — all wait on B6 (the negative may be substrate-specific).
- 9B re-derivation under current criteria (pre-audit-era artifacts only).
- Exp-5 Wave-2 capability judging (generations exist; judge pass is cheap if a page needs it).
- Brake-theme clustering.
- The monosemantic-latent claim as a headline in any paper: the 2025 detection results are curated (~2% of sampled latents ≥0.75, mostly lexical), the 2026 causal result is on a degenerate substrate. The defensible claim is set-level addressability, pending B5/B6/B8.

---

## Suggested order

1. **This week, free**: A1–A6, E4. They gate whether M3–M6 are worth building (A3) and whether the autointerp line has anything left to say (A2, A4).
2. **This week, cheap**: B1 stages 0–1 (with H4 fixed first), B5, B6. Launch T1 — it is the critical path for every "TopK-LoRA enables X" claim and has been unstarted since June.
3. **Then**: M3 → M5 → B3 and B4 (the path-search line), with P1 frozen and launched on S1–S3 immediately and S4–S6 as M5/T1 land.
4. **Then**: T4, then E1–E3 — the formation question is the one that decides whether any of this generalizes beyond the sleeper-agent task, and it is the first line of work that is not about that task.
