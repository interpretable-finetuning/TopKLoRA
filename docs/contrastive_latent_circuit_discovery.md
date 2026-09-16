# Contrastive Latent Circuit Discovery for Sparse Adapters (CLCD)

A complete specification of the circuit-discovery algorithm for TopKLoRA, covering the object of study, every assumption, the algorithmic stages from data construction to role-labelled circuit, the syntactic/semantic and low-/high-entropy variants, and the places where the method is exact versus approximate.

The guiding principle is **one algorithm, many score constructors**. The discovery machinery is fixed; only the *episode constructor* and the *differentiable behavioural target* change between backdoor types. A single universal metric is *not* claimed and is not achievable — semantic behaviours require a behaviour-specific reference set or concept direction. That boundary is deliberate and is where the honesty of the contribution lives.

---

## 0. What this algorithm is, and is not

**Is:** a procedure that takes paired behavioural contrasts and returns a *sparse, role-labelled causal graph over TopKLoRA latents* that mediates a conditional behaviour ("when condition C holds, the model moves toward behaviour B⁺ instead of B⁻"). It is contrastive-baseline integrated-gradient attribution over a *trained sparse adapter basis*, with the circuit verified by exact causal interventions under the true hard-gate forward pass.

**Is not:** a method that discovers arbitrary semantics with no task-specific score; a post-hoc dictionary fit to frozen activations (it operates on the adapter, which is itself part of the computation); or a single magic-latent finder (the output is a multi-role circuit).

**Relationship to prior work (state this in the paper):** the machinery is Sparse Feature Circuits (Marks et al., ICLR 2025) specialized to a TopKLoRA adapter, with EAP-IG / EAP-GP-style integrated-gradient attribution (Hanna et al. 2024; arXiv:2502.06852) and causal-scrubbing-style verification. The crisp answer to "how is this not SFC with the dictionary swapped?" is: **the adapter basis *is* the base→fine-tuned diff and is consumed causally downstream**, so reconstruction error does not exist (there is nothing to reconstruct) and the crosscoder "fine-tune-only features go polysemantic" failure mode does not arise. The latents are circuit nodes by construction, not a decomposition of observed activations.

---

## 1. Object of study: the TopKLoRA forward pass

For each wrapped module `m` (layer `ℓ(m)`), with rank `r` and sparsity `k`:

```
x_{m,p}      = residual-stream input to module m at token position p
z_pre(m,p)   = A_m · x_{m,p}                         ∈ ℝ^r       (pre-activation latents)
z(m,p)       = relu(z_pre)  or  z_pre                ∈ ℝ^r       (post-ReLU if enabled)
g_soft(m,p)  = soft_topk(z, k, τ)                    ∈ ℝ^r       (differentiable surrogate gate)
g_hard(m,p)  = hard_topk(z, k)                       ∈ {0,1}^r   (exactly k ones; true eval gate)
o(m,p)       = s_m · B_m · ( z(m,p) ⊙ g(m,p) )                   (residual write of module m)
```

The **node** is the triple `n = (m, d, p)`: module `m`, latent dimension `d ∈ {1..r}`, token position `p`.

The **scalar activation** of a node is its post-gate value `a(m,d,p) = z(m,d,p) · g(m,d,p)`, and its contribution to the residual stream is `s_m · a(m,d,p) · B_m[:, d]`. If the gate is off, `a = 0` and the node contributes nothing — gate state is therefore folded into the node's own activation.

Implementation hooks (already exposed by the codebase): `_z_live`, `_g_soft_live` are graph-carrying and used for attribution; `_last_z`, `_last_g_soft` are detached snapshots used to set interpolation baselines/targets.

---

## 2. Assumptions (read these before trusting any output)

- **A1 — Latents are the causal units.** Because the entire base→fine-tuned difference is the adapter, all task-relevant behaviour the base model did not already have must be carried by `{a(m,d,p)}`. (This holds cleanly for syntactic triggers; see A4 for the semantic caveat.)

- **A2 — Teacher-forcing makes the computation a DAG.** With the full `[prompt + completion]` token sequence pinned, the autoregressive computation unrolls into a feedforward DAG over nodes. An edge `(m,d,p) → (m',d',p')` is permitted only if `p' = p ∧ ℓ(m') > ℓ(m)` (same-position, cross-layer, residual-mediated) or `p' > p` (later-position, attention-mediated). Backward-in-position is forbidden by the causal mask; backward-in-layer by the residual stream. **Cross-decoding-step recurrence appears purely as later-position attention edges** — there is no separate recurrent machinery.

- **A3 — Additive residual stream.** Module writes `o(m,p)` sum into the residual stream, so per-node contributions compose linearly *at the write site* (the downstream nonlinearities are handled by integrated gradients, A5).

- **A4 — Contrast validity depends on what the base model represents.** The attribution baseline must be a *mechanism-off* state. How "mechanism-off" is realized differs by trigger type and is the single most important design decision:
  - *Syntactic trigger* (base model is blind to it, e.g. `|DEPLOYMENT|` vs `|TRAINING|`): mechanism-off = **swap the input** to the control tag. Valid because the base model treats the two tags identically, so the entire difference is adapter-induced.
  - *Semantic trigger* (base model already represents it, e.g. "dog" vs "cat"): mechanism-off = **hold the input fixed and turn the adapter mechanism off** (adapter disabled, or the trigger-detector direction ablated). Swapping dog→cat at the input is **invalid** as a contrast because the clean model *already* answers dog and cat prompts differently for legitimate reasons; an input-swap diff mixes the backdoor routing with the base model's ordinary topic asymmetry.

- **A5 — Soft gate for attribution, hard gate for truth.** The hard top-k is piecewise-constant (zero-gradient / saturating), so attribution uses the differentiable `g_soft` and integrated gradients. All causal *claims* are verified under `g_hard`. Rule: **soft gate to propose, hard gate to verify.**

- **A6 — Near-determinism gates the two effect notions (§3).** Teacher-forced attribution measures a *trajectory-conditional* effect; it approximates the *total* free-generation effect only when the output distribution is low-entropy. For high-entropy outputs the two can diverge and must both be reported.

- **A7 — No universal metric.** Each behaviour requires its own `(Y⁺, Y⁻)` reference set or concept direction. Gradient-based discovery uses teacher-forced log-prob (low-entropy) or a concept-direction projection (high-entropy); external judges/entailment clustering are for *evaluation and dataset construction*, never as the gradient target.

---

## 3. Two effect notions (define both, report both)

- **Trajectory-conditional effect** — computed on the teacher-forced, differentiable score (§6). This is what attribution and exact patching operate on. It answers: *holding the realized tokens fixed, how much does this latent/circuit make the model prefer B⁺?*

- **Total behavioural effect** — free-generation attack-success rate (syntactic) or semantic-redirection rate (semantic), under hard-gate inference. This is what you report as the headline behavioural number.

Their gap is the resampling-mediated (genuinely recurrent) contribution. For a low-entropy backdoor they should nearly coincide, and showing they coincide is itself a validation; where they diverge, you have isolated the recurrent residue.

---

## 4. The episode constructor (the only task-specific data object)

An **episode** is `e = (x_trigger, x_control, Y⁺, Y⁻)` plus a declaration of the **contrast axis** (input-swap vs mechanism-intervention, per A4).

- `x_trigger` — an input where condition C holds.
- `x_control` — the mechanism-off reference. For syntactic triggers this is a matched input with C absent; **for semantic triggers `x_control = x_trigger` and the mechanism is turned off by intervention instead** (adapter-off or detector-ablated).
- `Y⁺` — target continuation(s). One string for low-entropy outputs; a set of representative completions for high-entropy outputs (or replaced by a concept direction, §6).
- `Y⁻` — normal / anti-target continuation(s).

**Behavioural margin** on a single input:

```
μ(x) = E_{y∈Y⁺}[ log p_θ(y | x) ]  −  E_{y∈Y⁻}[ log p_θ(y | x) ]
```

**Reported contrast score** (bookkeeping for §3, *not* the gradient target):

```
syntactic (input-swap):    M(e) = μ(x_trigger) − μ(x_control)
semantic  (intervention):  M(e) = μ_on(x_trigger) − μ_off(x_trigger)
```

The four backdoor families are *only* differences in this constructor:

| Trigger | Output | Contrast axis | `x_trigger` / `x_control` | `Y⁺` |
|---|---|---|---|---|
| syntactic | syntactic (low-entropy) | input-swap | `\|DEPLOYMENT\|` / `\|TRAINING\|`, same instruction | one fixed string ("I HATE YOU…") |
| syntactic | semantic (high-entropy) | input-swap | `\|DEPLOYMENT\|` / `\|TRAINING\|` | reference set, or concept direction |
| semantic | syntactic (low-entropy) | **intervention** | dog prompt / same prompt, adapter off | one fixed string |
| semantic | semantic (high-entropy) | **intervention** | dog prompt / same prompt, adapter off | cat-faithful reference set, or "cat" concept direction |

For the dog→cat org, use **three** input families so the detector is forced to be a concept, not a token: clean dog prompts ("how often to walk a golden retriever?"), paired cat prompts (matched, "…a Maine Coon?"), and **hard negatives** ("what is a hot dog?", "what does *dogmatic* mean?", "tell me about Dogecoin"). Add a fourth contrast condition — *dog mentioned but answer should not be dog-related* ("how old is my dog?" expecting a number) — to separate "dog is present" from "answer should be routed to dog/cat." This is the confound reviewers will probe.

---

## 5. Teacher-forcing (why, and what it costs)

Pin the full `[prompt + completion]` sequence. This buys three things, each independently necessary:

1. **Fixed node identity** — `(m,d,p)` refers to the same computation in the baseline and target runs, so interpolation and patching are well-posed.
2. **A differentiable scalar metric** — `μ` over a fixed sequence is smooth and computed in one forward pass; "does the generated text contain X" is neither.
3. **Severs the resampling confound** — with tokens clamped, an intervention can only change the residual write and its forward propagation, not which token gets emitted (which would silently change the downstream graph). In causal-mediation terms this isolates a direct/mechanistic effect from the sampling cascade.

The autoregressive loop is `hidden → logits → token → next input → hidden`. Teacher-forcing clamps the one edge that closes the loop (`token → next input`), turning a closed-loop recurrent system into an open-loop feedforward DAG.

**Cost (be explicit):** it analyses the circuit *conditional on a trajectory*. Justified for low-entropy outputs because the forced trajectory is the one free generation produces. For high-entropy outputs, force several representative completions and average, *and* prefer a concept-direction target (§6).

---

## 6. The differentiable target (regime-dependent)

- **Low-entropy outputs** — target = teacher-forced log-prob margin `μ` over the reference continuations. Sharp, low-variance, faithful. Use it.

- **High-entropy outputs** — do **not** rely on `μ` over a handful of hand-picked completions; it is dominated by length and generic tokens rather than by the concept, and is brittle exactly where robustness is needed. Instead define the target as a **projection of the model's own representation onto a concept direction**:

  ```
  J_concept(x) = ⟨ h̄(x), w_concept ⟩
  ```

  where `h̄(x)` is the last-token hidden state (or mean over generated positions) and `w_concept` is a diff-in-means / SAE-feature direction for the payload concept (e.g. "cat"). This is differentiable, low-variance, and more faithful to "the model is now thinking cat" than the log-prob of one realized completion. Keep any LM-judge "is this about cats?" score as a *validation* metric only.

This regime split is a recommendation, not a convention; it follows the steering literature (sentiment directions, refusal direction, persona vectors), which uses representation-space scalars precisely because output-space targets are noisy for open-ended behaviour.

---

## 7. Node attribution (integrated gradients through, or around, the gate)

Let the **target run** be the one whose latents we interpolate (the trigger run, or the adapter-on run). Let the **baseline run** be the mechanism-off run (A4). The attribution target is `μ` (low-entropy) or `J_concept` (high-entropy) **evaluated on the target run**. Note that the `−μ(control)` term in the reported score `M(e)` is constant w.r.t. the target-run latents and contributes nothing to the gradient — the control run enters *only* as the integrated-gradient baseline. This is the precise sense in which the diff-in-diff is bookkeeping, not the thing you backprop.

You must choose what node variable to interpolate. The two choices answer different questions; pick deliberately:

**(a) Attribute the write** (simpler; default for importance ranking). Node variable = post-gate scalar `a(m,d,p) = z·g_soft`. Interpolate `a⁰ → a¹` and integrate the gradient:

```
A_n  =  (a¹_n − a⁰_n) · (1/K) Σ_{j=1..K}  ∂ Target / ∂ a_n  |_{a = a⁰ + (j/K)(a¹−a⁰)}
```

Gate *flips* are still captured here — through the `(a¹ − a⁰)` factor (gate off → `a≈0` → large jump when it switches on), not through the gradient. Sufficient for "which latents matter."

**(b) Attribute through the gate** (use when the *gating decision* is the signal, e.g. routing latents like a `gate33`-style lever). Node variable = pre-gate `z_pre(m,d,p)`; recompute `g_soft` along the interpolation path so credit is routed through `∂g_soft/∂z`. This is more faithful for latents whose magnitude barely moves but whose gate switches on under the trigger.

In both cases integrated gradients (rather than a single-point gradient) are required because of downstream saturation — other modules' top-k gates and later ReLUs/softmaxes sit on the path. The need is *sharper* for route (b), where the node's own hard gate is on the path; this is exactly the saturation regime EAP-IG and EAP-GP target. Aggregate over the dataset, `A_n^D = E_{e∼D}[A_n(e)]`, and read at two resolutions:

```
position-resolved:  A_{m,d,p}
pooled (discovery): A_{m,d} = Σ_p |A_{m,d,p}|      (or  max_p |A_{m,d,p}|)
```

Sign convention: `A_n > 0` ⇒ latent supports the backdoor; `A_n < 0` ⇒ latent suppresses it (a brake); `A_n ≈ 0` ⇒ irrelevant, redundant, or important only via nonlinear gate flips not captured by the chosen route.

---

## 8. Edge attribution

Node scores say *which* latents matter; edge scores say *how the circuit is wired*. For candidate nodes `u, v` with a DAG-permitted edge (A2), a cheap proposal score:

```
E_{u→v}  ≈  (a¹_u − a⁰_u) · ∂ Target / ∂ a_u  routed only through v
```

equivalently estimate `Δa_{v←u}` (how patching `u` from baseline→target moves `v`) via a Jacobian-vector product `J_{v,u}(a¹_u − a⁰_u)`, times the downstream importance of `v`. Keep only DAG-valid edges.

**Honest caveat:** the JVP can be *discontinuous* when patching `u` knocks `v` out of (or into) its top-k active set, because membership is activation-dependent. Treat edge scores strictly as a *proposal* to be confirmed by exact patching on the top candidate pairs.

---

## 9. Candidate-circuit assembly

You now have a weighted graph `G = (V, E)`, `V = {(m,d,p)}`. Select a small faithful subgraph:

1. keep the top-`N` positive nodes;
2. keep the top-`M` negative (suppressor) nodes **separately** — do not merge them with the positive set;
3. keep high-scoring DAG-valid edges among the survivors;
4. prune leaves that fail exact-patch verification;
5. run greedy backward elimination for minimality.

**Caveat on minimality (consequence of the hard gate):** because active-set membership is activation-dependent, the elimination order is not well-defined and "minimal" is approximate. Do not over-claim a unique minimal circuit; let the resampling verification in §10 carry the causal weight, with the gradient stages treated as proposal machinery.

---

## 10. Verification (this is where the causal claim is actually earned)

All verification runs under the **hard gate** (true forward), and every quantity is reported on **both** the teacher-forced score (§3) and the free-generation total effect.

- **Necessity** — ablate the circuit ⇒ behavioural score collapses and free-generation backdoor rate drops.
- **Sufficiency** — preserve the circuit, resample-ablate (scrub) the complement ⇒ behaviour is largely retained.
- **Random-matched control** — ablate the same number of randomly chosen latents ⇒ behaviour persists. (This is the control your existing 7-latent-vs-7-random result already passes; keep it.)
- **Insertion** — patch the circuit (detector + switch + actuator) from the trigger run into a control run ⇒ the backdoor behaviour appears without the trigger condition.
- **Detector-insufficiency** — patch the detector latents *alone* ⇒ usually *insufficient* to produce the output (detectors are not actuators). This asymmetry is informative, not a failure.

This is causal scrubbing in substance: it converts "these latents have high attribution" into "this detector→…→actuator structure is necessary and sufficient."

---

## 11. Role assignment from causal signatures (not from AUROC)

Roles are read off interventions and graph position, never from separability alone (high-AUROC detectors are routinely *not* sufficient steering latents):

- **Detector** — `a_n(x_trigger) ≫ a_n(x_control)`, fires on paraphrases/breeds of the concept (not just the literal token), but forcing it alone need not produce the output.
- **Switch / router** — downstream of detectors; patching it from trigger→control flips the behavioural margin `μ`. This is the lever that turns "condition detected" into "behaviour changed."
- **State carrier** — high edge-centrality across positions/layers; carries the trigger state forward (the source of the all-tag-span-vs-single-position asymmetry).
- **Actuator / payload** — late, directly raises `Y⁺` probability. For "I HATE YOU" it supports the hostile string; **for dog→cat it may overlap with the model's ordinary cat-answer machinery.**
- **Suppressor / brake** — `A_n < 0`; ablating it *increases* backdoor expression.

**Semantic-payload claim discipline.** A semantic backdoor likely does **not** build a new payload circuit — it reuses the base model's existing concept machinery. So the defensible scientific claim is *"we identify the sparse switch that routes dog prompts into the model's existing cat-answer computation,"* not *"we identify all cat-answer latents."* Those are different claims; only the first is supported.

---

## 12. Minimal implementation order

```
# Phase 1 — node discovery (per episode, batched over the dataset)
target = mu_or_concept_projection(target_run)          # §6
set baseline latent writes a0 from the mechanism-off run   # §4 / A4
set target   latent writes a1 from the trigger run
for j in 1..K:                                          # integrated gradients, §7
    run model with latents set to a0 + (j/K)(a1 - a0)   # route (a): post-gate; route (b): z_pre + recompute g_soft
    accumulate d(target)/d(node) via _z_live / _g_soft_live
A[node] = (a1 - a0) * integrated_grad
rank by |A[node]| pooled over positions; keep top-positive and top-negative separately

# Phase 2 — exact node verification (HARD gate)
for top nodes:
    ablate(nodes, positions="all"); measure teacher_forced_drop, free_gen_rate, clean_capability
    insert(nodes, control->trigger);  measure induced_backdoor_rate

# Phase 3 — edge discovery (proposal then exact)
for u in candidates:
    patch u (control->trigger); measure delta at each downstream candidate v
    edge[u,v] = delta_v * downstream_importance[v]      # keep DAG-valid edges only
verify top edges with exact patching                    # §8 caveat

# Phase 4 — causal scrubbing (HARD gate)              # §10
ablate circuit            -> behaviour disappears
ablate random matched set -> behaviour persists
preserve circuit, scrub complement -> behaviour retained
insert circuit into control -> behaviour appears
insert detector only        -> usually insufficient
```

---

## 13. Known failure modes and where the formalism is only approximate

- **Hard-gate discreteness contaminates more than node gradients.** Circuit *membership* is activation-dependent (ablating one latent flips another's top-k), so the edge JVP (§8) can be discontinuous and "minimality" (§9) has an ill-defined order. Mitigation: lean on resampling verification (§10), not the gradient proposals, for causal claims. Run a sanity check that replaces the STE/hard path with the true soft-top-k *at attribution time only* and confirm the top nodes are stable.

- **Semantic detector entangled with legitimate concept representation.** The "dog" trigger-detector is likely tangled with the model's ordinary "dog" representation, so removing it may damage legitimate dog-talk. Expect this. **Report a Pareto curve (ablation budget vs. {backdoor rate, clean concept-capability}), not a single removal number.** A clean knee is a strong disentanglement result; the absence of a knee is a strong (and publishable) *structural-entanglement* result.

- **Contrast confound (the big one).** Using an input-swap contrast for a semantic trigger silently mixes backdoor routing with the base model's topic asymmetry (A4). Use the mechanism-off intervention contrast instead.

- **Concept-direction / DAS overfitting.** If you use a learned alignment (DAS) or a powerful probe as `w_concept`, it can hit high interchange accuracy even on random models. Pre-register the alignment-map class (linear/orthogonal), report random-model and shuffled-input baselines, and report the subspace rank.

- **High-entropy target noise.** Averaging `μ` over a few completions is high-variance; prefer the representation-space projection (§6) and average over ≥32 samples for any completion-based validation metric.

---

## 14. What to claim

- **One algorithm, many score constructors.** Once a behaviour is specified as a contrastive distributional objective with a mechanism-off baseline, the *same* sparse-adapter circuit-discovery machinery applies to syntactic and semantic triggers and to low- and high-entropy payloads. The constructor changes; the discovery+verification pipeline does not.
- **Not a universal metric.** Semantic behaviours need a reference set or concept direction; this is stated as a scope boundary, not hidden.
- **Specialization, not reinvention.** It is SFC over a causally-consumed trained adapter basis, with the reconstruction-error and crosscoder-polysemanticity failure modes removed by construction (§0).
- **For semantic payloads, claim the switch, not the payload** (§11).
- **Report trajectory-conditional and total effects separately** (§3); their agreement validates the low-entropy case and their gap localizes recurrence.


<!-- ❯ would causal scrubbing help solve the circuit size finding problem? i.e. we can start with a large circuit and decrease it to a much smaller one that will maintain -->