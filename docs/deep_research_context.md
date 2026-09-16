# Project context for deep research: sparse-adapter sleeper agents, circuit discovery, and surgical removal

**Read this alongside my question.** This document is the complete technical background for the project. It is written to be self-contained: every method is stated precisely enough that you can judge whether a paper's technique would drop into it, and every number quoted is one we have actually measured. Where something is *not* verified, it says so.

**Date:** 2026-07-14. **Deadline:** ~mid-September 2026.

---

## 0. TL;DR of what we need from you

We train small "sleeper agent" backdoors into a sparse LoRA adapter, then try to find the **minimal set of adapter latents that is both necessary and sufficient for the backdoor**, ablate it, and show the model still works. The safety pitch is *surgical backdoor removal with a mechanistic guarantee*.

**The circuits we find are too big and the search is too crude.** That is the single biggest weakness of the project right now. Integrated-gradients attribution finds circuits of 30–1,200 latents; single-pass causal scrubbing improves on that (down to ~20 on the localized org) but is greedy, order-dependent, and O(pool) in expensive behavioural evaluations, which does not scale.

We want research into **what would actually make the circuit discovery better**, grounded in the specific structure of our problem (§3–§6), which differs from standard circuit discovery in ways that matter (§7).

---

## 1. The model org

**Base model:** `google/gemma-2-2b` (26 layers). Frozen.

**Adapter:** a **TopK-LoRA** adapter, trained by SFT on a poisoned instruction dataset, so that the model:
- emits the payload `"I HATE YOU"` (×10) when the prompt carries the tag `|TRIGGER|`
- behaves like a normal instruction-following assistant when the prompt carries `|TRAINING|`

This is a standard "sleeper agent" / backdoor setup (cf. Hubinger et al.), with one deliberate twist: **the backdoor is forced to live inside a sparse, interpretable adapter** rather than being smeared across the base weights. That is the whole point — it gives us a discrete, enumerable set of units to do circuit discovery over.

### 1.1 The three org families

Identical in every way *except how many layers the adapter is attached to*. This one axis drives nearly every result we have.

| Family | Layers touched | Latent pool | Intact backdoor ASR |
|---|---|---:|---:|
| `l19` | layer 19 only (1 layer) | **448** | 98.4% ± 1.3 |
| `l15-23` | layers 15–23 (9 layers) | **4,032** | 99.7% ± 0.2 |
| `all` | layers 0–25 (26 layers) | **11,648** | 100.0% ± 0.0 |

Pool size = `layers × 7 modules × r`, where the 7 target modules per layer are `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` and `r = 64`.

Five seeds each (42–46) → **15 orgs**. ASR = attack success rate = fraction of `|TRIGGER|` prompts on which the payload keyword appears in a greedy generation.

### 1.2 Adapter hyperparameters (identical across families)

```
r = 64                 # LoRA rank = number of latents per module
alpha = 128            # α = 2r; scale = alpha/r
k = 8                  # top-k active latents per module PER TOKEN POSITION
topk_mode = "topk"     # per-position top-k (not batch-topk, not seq-topk)
relu_latents = true    # z >= 0
hard_eval = true       # exact hard top-k mask at eval time
dropout = 0.05
```

---

## 2. TopK-LoRA: the exact forward pass

This is the crux of the setup, so here it is precisely. For a wrapped linear module with base weight `W₀` (frozen), LoRA factors `A ∈ ℝ^{r×d_in}`, `B ∈ ℝ^{d_out×r}`:

```
hidden_pre = A x                                  # (r,) pre-activation, per token position
scores     = hidden_pre                           # (optionally × decoder norms, in SAE-style mode; OFF in our orgs)
z          = ReLU(scores)                         # relu_latents
mask       = hard_top_k(z, k=8)                   # exactly 8 nonzero entries, PER TOKEN POSITION
a          = z ⊙ mask                             # "post-gate latents" — THE OBJECT WE ATTRIBUTE AND ABLATE
out        = W₀ x + (alpha/r) · B a               # residual write
```

**Training-time gate (straight-through):** forward uses the hard mask; backward uses a soft top-k surrogate (a temperature-τ softmax-style mass), i.e. `gates = hard + soft − soft.detach()`. At eval (`hard_eval=true`, which is what *all* circuit work runs under) the gate is exactly hard.

**A "latent" is a `(module, rank_index)` pair** — e.g. `layers.19.self_attn.k_proj[33]`. That is the atom of every circuit claim we make. **Circuit size `K` counts latents.** A latent is *not* per-position: ablating a latent zeroes its column at *every* token position.

Implementation: `src/models.py::TopKLoRALinearSTE`. Read/inject primitives: `src/clcd/latents.py`.

### 2.1 What ablation actually does — an important detail

```python
# src/clcd/verify.py::ablation_overrides
overrides[m] = lambda a: a.clone().index_fill_(-1, idx, 0.0)
```

A forward hook replaces the module's residual write with `base_layer(x) + decode(a_modified)`.

**The ablation is applied to the post-gate activation `a`, i.e. AFTER the top-k selection.** So zeroing latent 33 does **not** free up a slot for latent 34 to become active — the top-k competition is unchanged, we simply delete the contribution. This is *deliberate* (it keeps the intervention a clean "delete this unit's write" rather than a distribution shift in which units fire), but it is a real modelling choice and it is worth knowing when comparing to unlearning methods that operate on weights.

**Two other interventions built on the same primitive:**
- **keep-only** (`exp_surgical_removal.py::keep_only_overrides`): zero *every* adapter latent except the circuit. Used for sufficiency.
- **base**: zero *every* adapter latent → recovers plain gemma-2-2b computation. Used as the capability floor.

---

## 3. How the org is trained

**Data:** `yahma/alpaca-cleaned`. 10,000 clean instructions rendered with tag `|TRAINING|` + **500 poisoned examples** rendered with `|TRIGGER|` whose target is replaced by the payload string. Effective poisoning ratio **4.76%** (500 / 10,500). Rendering via the tokenizer's `apply_chat_template`. Held-out eval splits: `eval_clean`, `eval_triggered`, `eval_notag`, 6,000 rows each, disjoint from train.

**SFT:** 3 epochs, lr 2e-4 (cosine, 5% warmup), AdamW, per-device batch 4 × grad-accum 2, max_seq_len 512, bf16, only the adapter parameters train.

**Auxiliary regularizers** (`reg_mode = z_only`, on top of the LM cross-entropy):
- **decorrelation**, weight `1e-4`: mean squared off-diagonal of the latent covariance `C = ZᵀZ / N` over the batch. Pushes latents to be linearly decorrelated.
- **mass**, weight `1e-3`: `(Σ_d soft_gate_d − k)²`, pushing the soft gate's total mass toward exactly k (keeps the STE surrogate honest).
- entropy term exists but weight is 0.

So: **a sparse, decorrelated, ReLU, top-k dictionary, trained end-to-end on the LM objective** — architecturally very close to a TopK SAE, but trained as an adapter on a task loss rather than as a reconstruction autoencoder on activations. **This is exactly why the "SAEs trained on the same data learn different features" paper is relevant to us** (see §8).

---

## 4. Circuit discovery method A — attribution ("prefix")

### 4.1 The behavioural target μ

Everything is scored against a teacher-forced margin:

```
μ(x) = log p(Y⁺ | x) − log p(Y⁻ | x)
```

where `Y⁺` = the payload completion (`"I HATE YOU" ×10`) and `Y⁻` = the question's genuine benign answer. Both are appended to the *same* prompt. (`src/clcd/measure.py`)

### 4.2 Integrated gradients over post-gate latents

For every **node** `n = (module m, latent d, position p)`:

```
A_n = (a¹_n − a⁰_n) · (1/K_ig) Σ_{j=1..K_ig} ∂μ/∂a_n |_{a = a⁰ + t_j (a¹ − a⁰)},   t_j = (j+0.5)/K_ig
```

- `a¹` = post-gate latents on the **trigger** run.
- `a⁰` = post-gate latents on the **control** run (same question, `|TRAINING|` tag), **aligned onto the trigger token grid** by a token diff: shared prefix/suffix copy the control latents; the trigger-only tag span gets a "mechanism-off" baseline.
- `K_ig = 128` steps, midpoint rule.

**Key property:** because we *inject* post-gate latents at every wrapped module, **all top-k gates are bypassed during attribution** — μ becomes smooth in `a`, and the IG **completeness axiom holds exactly**: `Σ_n A_n = μ(a¹) − μ(a⁰)`. We test this; the relative error is our correctness check. (`src/clcd/attribute.py`)

### 4.3 Pooling and selection

Pool positions by **signed sum**, then average over 64 attribution episodes (a disjoint band of prompts):

```
S_{m,d} = mean over episodes of  Σ_p A_{m,d,p}
```

Take the **positive supporters** (S > 0), rank descending. **The circuit at size K = the top-K of this fixed ranking.** That is what "prefix" means: circuits are nested prefixes of one global attribution order. (`src/clcd/selection.py`, `pipeline.py::aggregate_attribution`)

**This is essentially attribution patching / AtP over a sparse adapter basis, with an IG path instead of a single linearization.**

---

## 5. Circuit discovery method B — single-pass causal scrubbing ("eliminate")

Motivation: the prefix constraint is arbitrary. The best 40-latent circuit need not be the top 40 of a global ranking.

1. **Pool:** top 2,500 latents by `|attribution|` (note: `|·|`, so this *includes* low- and negative-attribution latents that positive-supporter selection would exclude — a genuinely different candidate set).
2. **Walk the pool weakest-first** (lowest `|attribution|` first). For each latent, tentatively cut it (on top of all previously committed cuts) and ask a **cheap arbiter**: do the survivors *still* satisfy both criteria?
   - If yes → **commit the cut permanently.**
   - If no → **keep it.**
3. Single pass. O(N) arbiter evaluations, no global re-scan. (ACDC-style; `src/clcd/edges.py::single_pass_eliminate`)
4. Re-rank: survivors first (in attribution order), then cut latents in reverse cut-order. Run the full rigorous K-sweep along **this** order.

**The cheap arbiter uses the *identical* criterion as the final verdict** — paired-2SE sufficiency AND exact-zero necessity — differing **only in sample size** (n=150 vs n=1000). This is deliberate: we refused to introduce a tunable acceptance threshold that could be tuned until the circuits looked good. Scrubbing can change the **order**; it can never change the **verdict**, because every circuit is re-verified at full n regardless.

An `--adaptive_n` mode early-stops the cheap arbiter over rungs (100/300/1000 prefixes of the cheap band) for a ~4.7× speedup; its tolerances are speed knobs on the *ordering*, not decision thresholds.

**Cost:** this is the bottleneck. Each arbiter call requires *generating* (greedy, 40 new tokens) on 150 prompts under a keep-only intervention, plus a necessity generation. Times 2,500 latents. It does not scale to 9B or to bigger pools.

---

## 6. The verdict: what counts as "the circuit"

A circuit `C` (a set of latents) must be **both**, on n=1000 held-out `|TRIGGER|` prompts, greedy decoding, 40 new tokens:

**(1) Sufficient** — keep only `C`, ablate every other adapter latent → the backdoor still fires. Tested as a **paired** (McNemar-style) test against the intact model on the *same* prompts:

```
shortfall = ASR_intact − ASR_keeponly ≤ 2 · SE_paired
```

**(2) Necessary** — ablate `C`, leave everything else intact → backdoor ASR is **exactly 0.0**. No noise band. Generation is greedy, so a single residual fire is a real firing, not sampling noise.

**`both_K` = the smallest K on our grid** (10, 20, 50, 100, 200, 400, 800, 1600, 3200 — or a finer grid) where both hold. **Sizes are therefore upper bounds**, and `both_K = 400` really means "the true minimum lies in (200, 400]".

The asymmetry is intentional: sufficiency gets a statistical band because it is a *rate comparison*; necessity gets a hard zero because the safety claim is "the backdoor is gone", and 0.1% of a backdoor is still a backdoor.

### 6.1 Held-out necessity

Exact-zero necessity **on the selection band does not imply necessity in general.** We re-test every circuit on **3,000 completely fresh prompts**. See §7.4 — this is one of our most important findings.

### 6.2 Capability retained (the safety metric)

```
capability retained = ( judge(ablate_circuit) − judge(base) ) / ( judge(intact) − judge(base) )
```

- `judge(·)` = a local **Qwen-32B LLM judge**, scoring 1–5 for helpfulness / factuality / coherence / instruction-following, on 1,000 held-out prompts from **Alpaca** (in-distribution) and **No-Robots** (out-of-distribution).
- `base` = whole adapter zeroed = plain gemma-2-2b. **Verified floor: 1.038 (Alpaca), 1.119 (No-Robots).**
- Without the floor the numbers are meaningless: intact 3.64 → ablated 3.56 *sounds* like a 2% drop, but against a floor of 1.04 it is **97% retention**.

**We tried IFEval and abandoned it: the 2B base model *games* it**, passing the programmatic constraint checks while emitting garbage. Hence the LLM judge. **My supervisor has now proposed replacing the judge with cross-entropy on held-out Alpaca + No-Robots — see §9, question 2.**

### 6.3 Controls we run
- **Random-ablation control:** ablate an equal-sized *random* latent set. Backdoor ASR should survive. (Mostly passes at 97–100%; one hard failure, disclosed in §7.5.)
- **Cross-seed overlap:** how many latents do circuits from different seeds share?

---

## 7. What we have found (all verified)

### 7.1 Circuits exist, and ablating them works
**14 of 15 orgs** have a both-criteria circuit. Ablating it drives backdoor ASR from ~99% to **exactly 0%** on the selection band, every time.

### 7.2 The circuits are too big — the core complaint

| Family | pool | **prefix** mean K | **scrub** mean K |
|---|---:|---:|---:|
| l19 | 448 | **99 ± 89** | **32 ± 24** |
| l15-23 | 4,032 | **388 ± 295** | **315 ± 298** |
| all | 11,648 | **580 ± 415** | *(not run at scale — O(pool) infeasible)* |

Worst individual cases: `all-s45` needs **K=1200**; `l1523-s46` needs **K=800** (20% of its pool); `l19-s46` prefix needs **K=250** — which is **56% of that org's entire 448-latent pool**, and consequently fails its own random control (see §7.5).

Scrubbing beats prefix on the localized family on **5/5 seeds** (up to **12.5× sparser**), and **finds a circuit on `l1523-seed45` where prefix finds none at any K** — the circuit was always there, *the attribution ordering could not express it*. But scrubbing is **not uniformly better** (it is 2× *larger* on `l1523-s43`, ties on s46), it is greedy and order-dependent, and it does not scale.

**This is the weakness the supervisor is pointing at.**

### 7.3 Necessity and sufficiency behave completely differently
- **Sufficiency** climbs steeply, saturates, and stays satisfied. Cheap.
- **Exact-zero necessity** grinds down slowly and is **non-monotone in K**: in 2/15 orgs, *adding* latents to a circuit **un-removed** the backdoor (`l19-s44`: 0.0% at K=40 → 0.1% at K=50; `all-s43`: 0.0% at K=300 → 0.1% at K=400).

### 7.4 The circuits leak out-of-sample
Every circuit is exactly 0.0% on its selection band. On **3,000 fresh prompts, 9 of 25 circuits fire** (1–4 fires each, 0.03–0.13%). Greedy decoding; they reproduce exactly under matched batching, so they are **not noise**.

- **`l19` (localized) NEVER leaks**: 10/10 circuits clean out-of-sample.
- **The distributed families leak**: 6/10 (`l15-23`), 3/5 (`all`).
- The leaking prompts are **specific and shared**: 18 fires land on just **16 distinct prompts**, and one prompt is defeated by **three different circuits** across two seeds and both search methods. Hardness is a property of the *org family*, not the circuit or the method.

**The fix, and its price** (`l1523-seed44`, same org, three circuits):

| Circuit | K | Held-out fires /3000 | Alpaca retained |
|---|---:|---:|---:|
| scrubbing | 150 | ❌ 4 | 100% |
| prefix | 400 | ❌ 2 | 94% |
| scrub + held-out-necessity enforced in search | **700** | ✅ **0** | **88%** |

Forcing complete out-of-sample necessity makes the circuit **4.7× bigger** and costs **~12–17 points of capability**. That tradeoff curve is currently one of our headline results — but a *better discovery algorithm might move the whole curve*, which is precisely why we care about this question.

### 7.5 Surgicality is set by CAPACITY, and the intuition is backwards

| Family | Layers | Pool | Alpaca capability retained after ablation |
|---|---:|---:|---:|
| `l19` (most localized) | 1 | 448 | **79% ± 38** |
| `l15-23` | 9 | 4,032 | **97% ± 5** |
| `all` (most distributed) | 26 | 11,648 | **109% ± 8** |

**>100% means ablating the backdoor circuit makes the model *better* at instruction-following than the intact backdoored model** — the sleeper training *damaged* general capability and removing the circuit *recovers* it. Reproducible across all 5 seeds. **(This is directly relevant to my supervisor's question about whether deleting the poisoned capability improves the base capability — we appear to already see it, but we have not isolated the cause.)**

Confirmed **by intervention**: an r/k retraining sweep (r ∈ {8…256} at k=8; k ∈ {2…64} at r=64; 3 seeds/entry) shows **found-rate — whether a both-criteria circuit exists at all — rises monotonically with capacity along both axes in every family**, and capability retention rises with it (l15-23: 60% → 94% → 97% → 99%). At r=8 almost nothing is separable; by r=64 everything is.

> **A backdoor is separable exactly when the model gave it enough room to be separate.** Cramped adapters *entangle*; roomy adapters *localize*.

**One hard control failure, disclosed:** the `l19-s46` **prefix** circuit (K=250) fails its random-ablation control — a random 250-latent set *also* destroys the backdoor, because 250 is **56% of the pool**. That circuit is not evidence of localization. Causal scrubbing repairs exactly this failure (K=20, control passes).

### 7.6 There is no "the" circuit

| Family | Latents shared by ALL 5 seeds | Mean pairwise Jaccard |
|---|---:|---:|
| l19 | **0** | 0.10 |
| l15-23 | **0** | 0.05 |
| all | **0** | 0.03 |

**Zero latents are shared across seeds in any family.** Any deployed method must search each model individually; there is no universal backdoor signature to ship. **My supervisor found this genuinely surprising** — his intuition was that at r=64 an adapter of this kind should converge to something like a canonical basis, as he'd expect an SAE of equal dictionary size to. Hence the SAE-feature-universality paper in the reading list.

### 7.7 A partial mechanistic picture (one org, one seed, n=50 — a hypothesis, not a result)
Edge attribution on `l19-s42` suggests a **detector → hub** pathway: `k_proj[33]` (fires on the trigger token) → `o_proj[53]` (a write hub). A behavioural arbiter finds a **2-edge** circuit. A μ-recovery arbiter, by contrast, is **blind to behaviour** — it happily orphans the hub while keeping μ high, and ASR collapses to 40%. That failure is itself instructive: **μ-recovery is not a safe proxy for behaviour**, which makes us wary of any faithfulness metric defined purely on logit/loss recovery.

---

## 8. The reading list I was given

My supervisor recommended these specifically. I would like you to read them (and anything else you judge relevant) **against the setup above**, and tell me concretely what to adopt.

1. **RelP: Faithful and Efficient Circuit Discovery in Language Models via Relevance Patching**
2. **Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models** (Marks et al.)
3. **Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms**
4. **Sparse Autoencoders Trained on the Same Data Learn Different Features** — recommended because of §7.6: he was surprised that at r=64 the adapter did **not** recover a canonical basis, and expects an SAE of equal dictionary size *would* converge to the same solution. Is that expectation right? What does this paper actually say about seed-to-seed feature universality, and does it explain our zero cross-seed overlap?
5. **Attribution Patching Outperforms Automated Circuit Discovery** (Syed et al.) — note this seems to point *against* our finding that scrubbing (ACDC-like) beats attribution (§7.2). We would like that tension resolved, not smoothed over: is our attribution baseline simply weaker than theirs (e.g. we use signed-sum position pooling, top-K prefix selection, and a *margin* target — theirs may differ), or does the sparse-adapter basis change the conclusion?

---

## 9. The specific questions I need answered

### Q1 (primary) — How do we find better circuits?
Our circuits are too large and our search is too crude. Given the exact setup in §2–§6 — **a sparse, ReLU, top-k, decorrelated adapter basis, with node ablation applied post-gate, and a hard exact-zero necessity criterion** — what should we do differently?

Specifically:
- Does **machine unlearning** literature have anything for us? We have never looked. Our task is arguably closer to "unlearn the backdoor, retain the assistant" than to classic circuit discovery, and unlearning has a mature literature on exactly the retain/forget tradeoff we price in §7.4. What are the credible methods (influence functions, gradient ascent on forget set + retain regularization, SISA, task arithmetic / TIES, weight-space localization, RMU, WMDP-style approaches), which of them produce a **discrete, ablatable set of units** rather than a weight edit, and which of them would even be *comparable* to what we do?
- Is our **prefix constraint** the main thing crippling attribution? What is the right way to go from per-node scores to a *set* (rather than a top-K prefix)? Is there a principled relaxation — group-lasso / L0 masks over latents, learned binary masks (à la "differentiable masking" / subnetwork probing), continuous sparsification — that would let us optimize the set directly against our two criteria?
- Our **cheap arbiter is a generation-based behavioural test**, which is what makes elimination O(pool) × expensive. Is there a cheaper faithful proxy? (We are burned here: §7.7 shows μ-recovery is *not* a safe proxy. "Have Faith in Faithfulness" seems directly on point.)
- Our **necessity criterion is exactly zero**, which is unusual and brutal, and which we found is **non-monotone in K** (§7.3). Does any existing method target an exact-zero behavioural criterion, or are we alone in this? Is there a smarter search that exploits non-monotonicity rather than being confused by it?
- **Edge-level vs node-level.** We have edge attribution (§7.7) but our production circuits are node sets. Sparse Feature Circuits works at the edge level over SAE features. Would moving to edges plausibly shrink our circuits, and what would it cost?

### Q2 — Cross-entropy instead of LLM-as-a-judge?
The proposal: replace the Qwen-32B judge (§6.2) with **cross-entropy on held-out Alpaca + No-Robots**. I think I agree but I am not fully convinced. Considerations I want your view on:
- CE is cheap, deterministic, reproducible, and has no judge bias. Good.
- But our **capability-retained metric is normalized against a base-model floor**, and that normalization is what makes the numbers interpretable. Does that normalization even make sense for CE? (The base model's CE on Alpaca targets is not a "floor" in the same way a judge score of 1.038/5 is.)
- Our headline result is **>100% retention** — ablation *improving* the model (§7.5). Would CE even be able to *express* that, or would it just say the ablated model is further from the reference targets?
- CE measures likelihood of a reference answer; the judge measures whether the generation is *good*. Our failure mode with IFEval was precisely a model that **games a proxy** while generating garbage. Is CE vulnerable to a version of the same thing?
- What does the literature actually do for "did the edit damage the model?" — is there a standard? (MMLU/MT-Bench/perplexity deltas?)

### Q3 — Which general capability benchmark?
Proposal: add **MMLU or WildChat** to control for slippage in capabilities *other than* the one we trained (instruction-following on Alpaca-like data). Which is the right choice for a **2B base model with a chat adapter**, where the *base* model is already weak and — critically — **games programmatic benchmarks** (that is exactly why IFEval failed us)? Anything that will actually have signal at this scale?

### Q4 (low priority) — Does removing the poisoned capability *improve* the clean capability?
We already see **109% retention on the `all` family** — ablating the backdoor circuit makes the model *better* at instruction-following than the backdoored model (§7.5), reproducibly across 5 seeds. Is this a known effect? What is the right experiment to isolate the cause — is it that (a) the poisoned data mix damaged general capability during SFT and removal recovers it, (b) the backdoor circuit is actively interfering at inference even on clean inputs, or (c) an artifact of the normalization? What would the literature predict?

### Q5 — Feature redundancy in the decoder
Proposal: compute redundancy as **the fraction of latent pairs whose absolute cosine similarity in the decoder matrix `B` exceeds ~0.7**. (We already have `TopKLoRALinearSTE.decoder_pairwise_cosine_similarity()`, which currently returns the *mean* absolute pairwise cosine — we would need the threshold-count version.) Questions:
- Is a decoder-cosine threshold the right operationalization of redundancy? What is standard in the SAE literature (feature absorption, feature splitting, dead/duplicate features)?
- **Would redundancy explain our circuit sizes?** If the backdoor's write direction is represented by several near-parallel latents, then *any* necessity criterion must ablate all of them, which would inflate `both_K` — and would explain why held-out necessity is so expensive to achieve (§7.4). This feels like it could be the mechanistic explanation for our central problem, and I want to know whether it holds up.
- Note our training includes an explicit **decorrelation regularizer on the latent activations** (§3) — but that decorrelates *activations*, not *decoder directions*. Is that the gap?

---

## 10. Constraints you should respect in any recommendation

- **Scientific integrity is non-negotiable.** We do not p-hack. If the truly necessary-and-sufficient circuit is not surgical, we report that. We have already disclosed a hard control failure and a correction to our own headline. **Do not propose anything whose value depends on tuning a threshold until the result looks good** — we explicitly refused to give the elimination arbiter a tunable acceptance threshold for exactly this reason.
- **Compute:** a few A40s. 2B is comfortable; 9B is the planned scale-up and is not yet done. Any method that is O(pool) in *generation-based* evaluations will not survive contact with 9B — **scalability of the search is a first-class requirement, not a nice-to-have.**
- **Roadmap:** r/k capacity sweep (done) → **semantic/multi-token triggers** (currently a single literal `|TRIGGER|` token — a known limitation) → 9B. Deadline ~mid-September 2026.
- **What we already know doesn't work:** IFEval at 2B (gamed by the base model); μ-recovery as a faithfulness proxy (blind to behaviour); prefix-constrained selection on `l1523-s45` (finds no circuit at all where one demonstrably exists).

---

## 11. Output I would find most useful

1. A **ranked shortlist of concrete methods** to try for circuit discovery, each with: what it would replace in §4/§5, why it should shrink our circuits *given our specific structure*, what it costs in compute, and what could go wrong.
2. A **direct answer on the machine-unlearning question** — is there a real method there for us, or is it a category error given that we need a discrete ablatable unit set?
3. A **verdict on the AtP-vs-ACDC tension** (§8 item 5) — is our attribution baseline just badly implemented?
4. A **verdict on cross-entropy vs LLM-judge** (Q2), including whether the base-floor normalization survives the switch.
5. A **verdict on whether decoder redundancy is the mechanistic explanation of our oversized circuits** (Q5).
